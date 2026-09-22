# Live observation publication, planning latency and gameplay evidence (2026-09-22)

Scope: private exporter -> `/api/observation` -> public publisher -> site, the
high-level planner prompt budget, the adaptive interaction policy's movement
handling, and observation-side resource bounds. All numbers below are measured
on the live managed machine; no fixtures were published as live data.

## 1. Live observation chain

Contract (committed, unchanged): the association-brain producer samples the
real readout at the actual decision, writes `minecraft.observation.v1` with the
consumed RGB digest and a sampled/bounded neural population, the operator
allowlists it, and the outbound publisher re-validates freshness before posting
to the site. Samples older than 30 s are `source_unavailable` by contract.

Verified live at 16:00-16:02 AEST:

| endpoint | result |
| --- | --- |
| `http://127.0.0.1:8775/api/observation?source=association-brain` | HTTP 200, `online: true`, stream `40ec5b9784f940c2bf442d9ce89435bf`, sequence `53 -> 327` advancing, `frame_age_ms` 1.7-3.0 s, `consumed_rgb_sha256` present |
| `https://neuroforge.io/minecraft/observation?source=association-brain` | HTTP 200, same stream/sequence as the private packet, fresh (`frame_age_ms` ~3.0 s) |
| `http://127.0.0.1:8775/api/topology` | association-brain population: 576 units, 102 non-zero activity entries, 1536 edges, `activity_basis: readout-input`, detail `1 calls - 2112 units - 135/2048 KC busy` |

The site blocks the default `Python-urllib` user agent (403); curl and the
committed publisher user agent receive the packet. The publisher process is the
one that matters and it was already running with `--observations
--observation-source association-brain`.

Availability is inherently intermittent: the producer only samples while the
adaptive policy actually acts (traversal/experiment skills). Planner calls take
tens of seconds, and during those gaps no association readout exists to publish
honestly. No idle packet is fabricated.

## 2. Planning prompt budget and latency (measured)

The live planner was the dominant cost: `gemma-4-e4b-vl` (llama.cpp, Vulkan)
prefill plus decode, and the runtime discards decisions older than its request
deadline. Baseline live observations showed 50-140 s cognition calls, decisions
discarded at the 60 s deadline, and multi-minute motor freezes.

Changes (minecraft_ai):
- compact system prompt: the 51-key perception enum is expressed once in
  brace-compacted form (the sampler grammar still enforces the literal keys);
  redundant wording removed, all behavioural constraints preserved;
- skills payload: description cap 120 -> 80 chars, unused `effects` dropped;
- `cognition_request_timeout_ms` 60 s -> 120 s so one main call plus one
  bounded semantic repair can complete instead of being silently discarded.

Fixed-observation prompt measurement (same controller code path, tokenized by
the live server):

| observation | before (system/payload/directive) | after | reduction |
| --- | --- | --- | --- |
| stone pit blocked | 929 + 1064 + 113 = 2106 | 635 + 952 + 113 = 1700 | -19.3% |
| oak trunk visible | 929 + 1110 + 111 = 2150 | 635 + 1006 + 111 = 1752 | -18.5% |
| 4 logs in hand | 929 + 1040 + 109 = 2078 | 635 + 936 + 109 = 1680 | -19.2% |

Plan quality comparison on the same three fixed observations (constrained
decoding, same server): the chosen skill and the plan sequence are preserved
(`s=None` pit replan with `[survey_surroundings, traverse_visible_obstacle,
explore_forward]`; `s=explore_forward` for the visible oak; plank-first crafting
plan with logs in hand). Only the wording of the free-text direction changed.

Server-side prefill for those calls (concurrent live load, so wall times are
noisy; token counts are deterministic):

| window | prompt tokens | prefill | per-token |
| --- | --- | --- | --- |
| before | 2125 / 2125 / 2097 | 28.3 s / 34.7 s / 29.1 s | 13.3-16.3 ms |
| after | 1719 / 1771 / 1699 | 26.0 s / 28.4 s / 26.9 s | 15.1-16.0 ms |

Live cognition telemetry: 31.6 min baseline vs a 14.5 min post-change steady
window that excludes two launcher recoveries (see the monitor captures in the
task transcript). Sampling interval 2 s.

| metric | before | after |
| --- | --- | --- |
| cognition last-latency p50 / p90 / max | 59.1 s / 117.4 s / 138.1 s | 38.0 s / 61.1 s / 62.0 s |
| cognition calls per minute | 0.60 | 0.35 (fewer replans per run) |
| motor-frozen sample fraction | 59% | 49% |
| active-skill coverage | 39.5% | 49.7% |
| private observation online | 41.5% | 51.1% |
| longest private outage | 444 s | 172 s |
| private frame age p50 / p95 | 1.49 s / 2.51 s | 1.54 s / 2.37 s |
| private sequence rate (median) | 2.03 Hz | 2.08 Hz |
| agent CPU median / RSS median | 47.0% / 165 MiB | 45.3% / 149 MiB |
| dashboard CPU median / RSS median | 33.1% / 182 MiB | 36.1% / 185 MiB |
| publisher CPU median / RSS median | 3.0% / 33 MiB | 3.0% / 29 MiB |

Full-window caveat: two `start.sh` launcher recoveries happened in the
post-change window when the runtime main loop stopped advancing capture frames
for >5 s and readiness telemetry aged past its 5 s bound. Both recoveries
preserved the running world (`Managed Bedrock session is already running`) and
the agent resumed; the steady window above excludes both. The baseline window
had no such freeze. A stack watcher was armed after the second event and the
stall did not recur in the following 10 minutes, so it remains intermittent and
not yet attributed. This is reported rather than papered over.

## 3. Gameplay evidence and the concrete blocker

State before: the player was confined on a stone plateau/pit; the adaptive
policy held forward into a wall while the traversal verifier correctly reported
`locomotion.stalled`, and `explore_forward` often failed with
`controller.starvation` because the policy never commanded locomotion inside
the bounded startup window.

Policy fix (erais `minecraft_experiment.py`, no verification change):
- a movement choice that executes with near-zero aligned scene flow is charged
  intrinsic cost and marked blocked; the next decisions prefer turning/backing;
- two adjust turns clear the blocked flag so movement is retried instead of
  looking forever;
- traversal options with a movement contract command locomotion within their
  startup window (at most one adjusting look first).

Evidence after the fix (same verifiers, no relaxation):
- verified `skill_succeeded` events for `explore_forward` (3) and
  `traverse_level_ground` (1) within the first ten minutes of the reloaded
  agent, i.e. the traversal verifier confirmed repeated quiet-camera luma
  displacement;
- `explore_forward` failures changed from `controller.starvation` (no movement
  commanded) to `locomotion.stalled` (movement commanded, route physically
  blocked);
- trajectory action distribution now contains `w`, `space+w` and `+/-45` yaw
  turns instead of look-only holds;
- live policy telemetry exposes `blocked_movement` and `escape_turns`.

Not yet achieved: verified block break or resource acquisition. The last
verified `block_broken` event predates this session (2026-09-13); the player is
on open stone terrain with forest visible at the horizon and the plan is
working through survey/traverse/explore nodes. This remains the honest blocker.

## 4. Resource bounds

Applied live and committed as drop-ins under `systemd/`:

| unit | Nice | CPUQuota | MemoryMax | MemorySwapMax |
| --- | --- | --- | --- | --- |
| `minecraft-ai-dashboard-live.service` (private route) | 19 | 100% | 2G | 0 |
| `minecraft-public-view.service` (outbound producer) | 19 | 100% | 2G | 0 |

The agent and supervisor processes were reniced to 19 live. The agent service
cgroup also contains the running game session; a hard `MemoryMax` there could
OOM the live game, so it was deliberately not capped. Measured agent RSS stayed
at 127-165 MiB (well under 2G) across the session, and the VLM service was not
touched.

## 5. Honest limitations

- Observation availability is duty-cycle bound by planner latency, not by the
  transport chain.
- The public observation route rejects the default `python-urllib` user agent;
  this is site-side and was not modified.
- The measured wall-clock prefill numbers were taken while the live agent was
  also calling the shared local model; only the token counts are load-free.
