# Observation continuity, planner call amplification and live verification (2026-09-22 evening)

Follow-up to `LIVE_OBSERVATION_EFFICIENCY_20260922.md`. Scope: the private
association-brain observation producer, the private/public readers, the
high-level planner's duplicate-call rate, the topological brain view, the
native motion expert pin, gameplay evidence and resource bounds. All numbers
are measured on the live managed machine; no fixture packet was published as
live data.

## 1. Observation liveness contract (honest idle heartbeats)

Before: the producer sampled only at a real adaptive-policy decision. During
planner reasoning the file aged out and both routes dropped to
`source_unavailable`; the morning baseline was 51.1% online with a 172 s
longest outage. During a gap there was no last-seen information at all.

Change (additive, `minecraft.observation.v1` unchanged):

- the observer thread repeats the **last real consumed sample** every 1 s
  while it is inside the declared 30 s freshness window, with two new fields:
  `state` (`acting|idle|reasoning|replanning`, published by the runtime on
  each telemetry tick) and `sample_replayed` (bool). The capture stamp, frame
  id, RGB digest, feature/activity arrays and action binding are never
  rewritten, so `frame_age_ms` always reports the true age;
- instead of a generic outage, the reader now answers an expired sample with
  `online: false`, `reason: "sample_expired"`, the last explicit `state`,
  `last_seen_age_ms`, `source_frame_id`, `sequence` and `stream_id`. Corrupt
  or source/stream-mismatched packets still fail closed as
  `source_unavailable`;
- the runtime publishes `reasoning` while a cognition decision is in flight,
  `acting` while a skill runs, `replanning` while the traversal escalation
  guard owns the gap, `idle` otherwise. A producer fault or a policy without
  the opt-in hook is ignored, never fatal to the motor loop.

Verified live examples (private `http://127.0.0.1:8775/api/observation`):

| time (AEST) | result |
| --- | --- |
| 19:26:21 | `online: true, state: acting, sample_replayed: false`, frame age 1.8 s (real sample) |
| 19:29:32 | `online: true, state: reasoning, sample_replayed: true`, frame age 29.6 s (last real sample, honest age) |
| 19:29:36-19:31 | `online: false, reason: sample_expired, state: reasoning, last_seen_age_ms: 31816 -> 119389` |

Availability after deployment, private route, strict post-deploy window
18:53:27-19:41:59 AEST (sampled every 2 s, 1186 samples, 48.5 min; two agent
generations ended by capture-stale crashes and were recovered by start.sh):

| metric | before (morning baseline, no heartbeat) | after (this window) |
| --- | --- | --- |
| online samples | 51.1% | 37.3% (442/1186) |
| same window without the heartbeat (replayed samples removed) | n/a | 23.2% (275/1186) |
| heartbeat contribution | none | +14.1 pp (+61% relative), 167 replayed packets |
| longest labelled gap | 172 s | 212 s (15 gaps) |
| gap carries | `source_unavailable` only | `reason: sample_expired`, `state`, `last_seen_age_ms`, frame/sequence identity |
| online phase mix | n/a (no state field) | 255 `acting`, 186 `reasoning`, 1 legacy packet |
| host context | quiet window | load average ~30, 2 agent restarts, VLM shared with image perception |

The raw after percentage is lower than the quiet morning window because the
same host now runs the Doom/flylab jobs at load ~30 and every planner call
takes ~44 s of server time; the contract change is measured by the 167
replayed packets (14 percentage points that were offline before) and by the
facts carried in each gap, not by the raw percentage alone. All gap samples
were genuine observation gaps (the dashboard answered every status probe;
there were zero dashboard-unresponsive samples in the window), and the
last-seen age advanced in real time while the frame id and sequence stayed
frozen (observed: seq 725 / frame 14679 / `last_seen_age_ms` 42597 -> 67531).

## 2. Planner profile and the duplicate-call fix

Profile of the live planner path (same VLM server, llama.cpp `gemma-4-e4b-vl`
on the RX 580; the server is shared with image perception and other host
load):

| component | measurement | share of a call |
| --- | --- | --- |
| Python prompt build (`_serialize_high_level_payload`) | 0.137 ms | negligible |
| Python grammar build (`_cognition_decision_grammar`, 28 skills) | 0.177 ms | negligible |
| VLM prefill (main decision, 2422 prompt tokens before / 2254 after) | p50 32.2 s before, 31.2 s after | ~70% |
| VLM decode (main decision, 162 / 148 tokens) | p50 13.8 s before, 13.2 s after | ~30% |
| runtime + queue overhead (wall minus server timings) | 0-72 s depending on contention; `cognition.last_latency_ms` p50 38.0 s (morning) vs 44.3-51.5 s (evening load) | variable |

The actionable cost was **call amplification**: live telemetry before the
change read 93 cognition calls for 51 decisions, with `repairs: 43` and
`retry_repairs: 39`. Every repeated-failure repair is a second VLM call
(measured p50 488 prompt tokens, 15.5 s prefill + 8.1 s decode = 26.5 s
total), so roughly 78% of decisions paid a second round trip.

Fix (minecraft_ai): a skill with two consecutive failures in the active
context is excluded from the **first** payload and grammar instead of being
offered and then corrected. The existing semantic repair remains as the
fallback for urgent safety and fresh operator retries, and the repair path is
still exercised by `test_repair_cannot_alternate_between_two_recently_failed_options`.

After deployment (18:53-19:42 AEST, same window as section 1): 34 calls,
4 repairs (4 retry repairs) across the restarted generations, i.e. ~1.13
calls per decision vs 1.82 before. Estimated model time per decision:
46.1 s + 0.78 x 26.5 s ~= 66.8 s before, 44.4 s + 0.13 x 26.5 s ~= 47.9 s
after (-28%). The single-call p50 is unchanged (44.4 s vs 46.1 s); the win is
fewer calls, not a faster model.

Plan quality comparison on three fixed observations (stone pit blocked, oak
trunk visible, four logs in hand; exact system message from the committed
controller, tokenized by the live server, plain vs grammar-constrained
decoding): the chosen `s` is identical in both modes (`survey_surroundings`,
`gather_nearby_wood`, `gather_nearby_wood`). The prompt/grammar change itself
does not alter any of the three prompts, because none of them contains a
repeatedly failed skill; the blocked-skill exclusion is covered by the new
`test_recently_blocked_skill_is_excluded_from_the_first_decision`.

## 3. Topological brain view verification

Private `http://127.0.0.1:8775/api/topology?source=association-brain`
(19:27, 19:44 samples): `association_brain` live, population 576 units,
102-106 non-zero activity entries, 1536 edges, `activity_basis: readout-input`,
`activity_reused` alternating with new samples, observation sequence advancing
(recorded 279 -> 415 -> 481 -> 653+). `policy` shows `offline` in the part
graph because the adaptive wrapper's status has no `primary` route key; this
is a pre-existing dashboard detail, not a missing population.

Public route `https://neuroforge.io/minecraft/observation?source=association-brain`:
HTTP 200, same stream id as private, `frame_age_ms` 1.97 s, sequence advancing
during an online window.

Headless browser check (Playwright 1.62 chromium, page scrolled to the
observer panel): `data-online="true"`, identity
`association-brain / ed7475903989`, frame text advancing
`Frame 653 / sample 402 -> Frame 663 / sample 406`, association readout
`112 / 2048 KCs active` -> `123 / 2048`, topology note
`576 sampled units; 1536 / 12288 source edges`, canvas visible with 179,400
non-black pixels sampled from WebGL, no console errors, both
`/minecraft/observation` fetches HTTP 200. Screenshot:
`/tmp/opencode/minecraft_page.png` (this session).

Decoded vision panel: the site exposes it and it renders the 8x8 consumed
luminance samples and the receptive-field overlay; `reconstruction` is null,
so the panel reads "Reconstruction unavailable". A real decoded RGB panel
needs the producer to attest `chat_free: true` and publish metadata-free
canonical PNG `input`/`reconstruction` (both already supported by
`project_observation`); the current producer deliberately withholds pixels,
so this is a policy decision, not a missing site field.

## 4. Gameplay and the motion expert

The documented escalation blocker (repeated traversal failures suppress
keepalives while the planner returns `s=null`) is addressed by committed
`f6bee5c` and is live in every generation restarted after 18:15. No
multi-minute idle freeze has recurred; the release path itself
(`release_reason: bounded_keepalive_resume_after_N_null_replans`) was not
triggered during the observation window because decisions kept starting
skills.

The raw-motion expert failed every warmup with
`native VPT output configuration digest mismatch`. The pinned digest
(`88105db6...`) was computed for the recorded-shadow calibration
(`camera_scale 2.88 / camera_pitch_scale 3.96`); the live supervisor
calibration is `5.772222222222222 / 5.760797738910738`
(`pitch_counts_per_degree` pinned by the supervisor's calibration id). The
live config pin was updated to the digest of the live, independently
calibrated output configuration (`b25141f7...`). After the next agent
generation the worker starts and
`failed_warmup_policy_ids` no longer contains `erais-native-vpt-1x-v2`
(remaining: `erais-native-steve-text-scene-motor-v1`, a separate missing
asset). This restores the pin to the actual configuration rather than
relaxing the check: the worker recomputes and verifies it at startup.

Verified movement evidence in the 6 h before 19:00 (verifier-confirmed
`skill_succeeded`, no verification relaxation): 142 `explore_forward`,
27 `backtrack_from_obstacle`, 24 `traverse_visible_obstacle`,
20 `traverse_level_ground`, 1 `escape_confinement`. No verified block break
or resource acquisition in that window (the last `block_broken` event in the
database is older); the plan is still working traversal nodes toward the oak
logs, so break/pickup remains the honest blocker.

Operational incidents during the window, reported not hidden:
- three agent generations ended unexpectedly during the evening: one
  provenance assertion at 18:16
  (`ValueError: skill_id does not match the accepted action condition`,
  runtime.py:2083) before the rider changes were deployed, and two
  `capture stream is stale for 3 consecutive frames` crashes at 18:56 and
  19:06 under host load average ~30 (the VLM, game, Doom and flylab jobs all
  run on 8 cores). start.sh recovered each generation and preserved the world;
- the dashboard process (nice 19) answered every probe in the measured
  window, but during the two crash transitions it served 503 gaps for the
  capture-stale interval, which the sampler records as genuine observation
  gaps rather than transport failures.

## 5. Site-side field report (no site repo edit performed)

To show the new semantics on https://neuroforge.io/minecraft/:

1. `scripts/publish_minecraft_view.py` `publish_observation` returns on HTTP
   503 and discards the body. The private 503 body now carries
   `reason: sample_expired`, `state`, `last_seen_age_ms`, `source_frame_id`,
   `sequence`, `stream_id`; forward that bounded gap record instead.
2. `workers/minecraft_observation.mjs` `publicMinecraftObservation` builds a
   fixed result and drops `state`/`sample_replayed`; the client lease
   (`MinecraftObservationLease.accept`) rejects `online !== true`. To surface
   phase and gaps, copy those fields and accept an offline gap packet.
3. `src/scripts/minecraft-observation.ts` throws on `online !== true` and
   `clear()` replaces the panel with "Observation unavailable". Render
   `last seen <last_seen_age_ms> / reason: <state>` for a gap, and show
   `state`/`sample_replayed` for online packets.

The public route currently returns `{"online":false,"reason":"source_unavailable"}`
during a genuine `sample_expired` gap because of (1); online windows are
unaffected and remain fresh.

## 6. Resource bounds

Live and verified during the window: agent, supervisor, dashboard and
publisher all at nice 19; `minecraft-ai-dashboard-live.service` and
`minecraft-public-view.service` at `CPUQuota=100%`, `MemoryMax=2G`,
`MemorySwapMax=0`; game and VLM left running and uncapped; no world reset.
Agent RSS stayed far below 2 GiB. The agent service cgroup still contains the
game session, so no hard memory cap is applied there (deliberate, unchanged).
