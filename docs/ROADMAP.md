# Implementation Roadmap

The roadmap is ordered by dependency and release gates. Later phases must not bypass earlier safety and observability requirements.

## Latest execution checkpoint — 2026-09-12

- [x] Preserve exact namespaced block/tool identities when looking up mining knowledge. Incoming mod-identifier support exposed normalization collisions between hyphens, slashes, dots and underscores; 21 actual-guard regressions now prevent one block/tool's observed breaks or rules from authorizing another, while keeping bare natural-name and vanilla-prefix compatibility. This protects learned-rule attribution; it does not verify live modded mining.
- [x] Add an explicit opt-in external-worker terminal checkpoint transport. READY binds private session/checkpoint metadata; the parent retains the last flushed inference barrier, consumes outstanding replies without acting, validates one bounded ACK and waits for normal child exit under one deadline. Fake-pipe/schema regressions cover identity, deadline, failed write/reap and terminal admission rules; legacy reset/stop stays unchanged. This is staged parent support, not a qualified native checkpoint or a live reload.
- [ ] Wire the qualified native drain into planned agent retirement without a shutdown-induced policy reset or early group signal. Release real inputs first, share a total budget with recording cleanup, retain the outer launcher barrier and exact supervisor generation, and report checkpoint/recorder/containment outcomes separately. Emergency containment must remain independent. The preserved agent has not been reloaded for this transport change.
- [x] Prevent typed plan progress from diverging after an unrelated successful skill. Live telemetry showed the displayed index at 2 while the graph cursor stayed at 0; the regression reproduces a later matching success rewinding the index to 1. Bound nodes now require their matching skill, while explicitly allowed legacy prose advances both representations together after existing goal/GUI gates. Twenty-one focused regressions preserve terminal deduplication, failure/cancellation and plan-neutral handling. This corrects source bookkeeping, not resource acquisition or completed operator intent; an already-diverged live plan is not retroactively relabeled.
- [ ] Qualify a bounded response to gathering that emits no useful input. The retained run `8a80f148d2c447e98e12cf008b69d287` timed out after 90.046 seconds: 519 supervisor-accepted records contained no key-down or button-down, 517 empty packets and two camera changes. The hotbar remained empty and prohibited inventory predictions were suppressed. This is not evidence that attack was released too early; do not lengthen mining holds, unmask inventory or force attack to conceal the failure.

## Current Bedrock execution checkpoint — 2026-09-08

This is the immediate gameplay critical path, not a claim that whole phases below are complete.

- [x] Restore the live private-display client, dashboard and recorded agent execution after the stopped stack. The launcher’s documented crash acknowledgement completed its own idle-prefix/kernel checks; no reboot, GPU reset, driver change or memory-setting change was used. Physical camera/body-pose qualification remains open.
- [x] Put the existing world on Hard while retaining Survival and cheats disabled. Server startup reports `Difficulty: 3 HARD`; its saved difficulty is 3. After the world-server-only restart, an exact configured-server click produced a fresh player connection/spawn and a retained normal survival-HUD image. The client, GPU model and subsequent agent-reload supervisor generation were preserved. This is a world/rejoin result, not successful feeding, combat or escape.
- [x] Observe autonomous death-screen recovery in the actual Hard world. Five retained death-to-world pairs (trajectory steps 167–172 and 174–177, 2026-09-08 04:38 UTC session) show three zombie deaths and two spider deaths followed by accepted Respawn clicks and normal-world frames. Returning outdoors was respawning, not escaping or successfully defending itself. Step 173 was a duplicate no-action success caused by an older death fact surviving a newer playable HUD; the source correction publishes `scene.death=False` only from the existing trusted normal-HUD branch. Regression tests retain ambiguous/UI death evidence and admit subsequent genuine deaths. After the correction, steps 371–378 in the later live recording show four distinct zombie deaths, each with an accepted 75 ms Respawn click, a normal-HUD frame and `scene.death=False`. No accepted locomotion occurs between those four recovery pairs: repeated mob deaths remain a survival failure, not successful defense or escape.
- [x] Retire obsolete traversal escalation only after a genuinely observed death acquires respawn GUI ownership. The previous latch survived death and could suppress ordinary movement after returning to a different collision scene. The 73-test recovery checkpoint preserves the latch for unverified death, unrelated UI recovery and failed ownership; the respawn transaction still blocks WORLD input until a trusted playable return. In the 06:26 UTC recording, retained steps 10859–10860 show a Drowned death, accepted Respawn click and normal HUD; step 11045 records accepted forward input about 47.3 seconds later. That traversal still ended stalled. This verifies a natural recovery-to-movement-emission chain, not successful navigation, improved survival, an isolated causal effect or a native world-memory reset.
- [x] Reject the retained BedrockConnect ServerList as playable creative terrain. The former gray-pixel ratio accepted its sky/menu background; creative admission now requires the existing pinned hotbar rail/grid/selection geometry and negative UI/death checks. Native-scale, title-owned caption OCR recovers the exact configured server label. Unsupported creative layouts abstain; full survival detection is unchanged.
- [x] Restore complete terrain-query transaction ownership and keep the traversal escalation latch from authorizing disposable wandering. `cb713d8` passes the full 2,068-test suite and terminal CI; the useful-observation-to-escape live gate is still open.
- [x] Preserve capable external motor workers' world memory across skill handoffs. `840b7bb` negotiates optional, worker-generation-bound `reset_scopes`; ordinary cleanup requests `actions`, while explicit caller-owned `reset_world()` requests `world`. Legacy workers retain their reset command and late predictions remain retired. The 114 focused tests qualify the public protocol, not native model activation, a wired automatic world-boundary detector or learning effectiveness.
- [x] Drain a matching retired option's error without discarding persistent worker memory. Keep malformed/mismatched records fatal and preserve exactly-once accounting for an expired external timeout. The first correction exposed a missed transport-counter regression in CI; the combined 149-test transport/policy suite now retains that original assertion.
- [x] Constrain generated planner perception questions to the existing 51-key grounding vocabulary in both schema and grammar, including bounded JSON repair. Live calls had repeatedly invented unsupported position/exit keys and started no vision work. `2c3d57c` passes 145 focused tests and is loaded through an agent-only reload preserving client, supervisor and calibration. Empty questions and action abstention remain valid; a useful live question/answer/action chain is still required.
- [x] Carry an explicit planner target description into its existing requested-perception transaction, including format repair and the matching cognition follow-up. Missing descriptions are explicitly unresolved in the question; no target is inferred from reasoning, past skills or plans. Typed requests cannot acquire unrelated chat evidence from descriptor wording. The initial 406 focused regressions preserve scene ownership, preemption, action restrictions and the obstacle-only idle-stall fallback. A further 444-test checkpoint carries the frozen descriptor into the matching positive observation's next planner context, independently of arbitrary track labels; expiry, preemption, replacement and clear revoke it. This remains descriptive request context, not an observed identity, action permission, verified target recognition or acquisition.
- [x] Retain an explicitly supplied target description in the existing bounded semantic-repair context. Eight new regressions preserve absent/empty/escaped descriptions, operator action restrictions and the existing two-call budget, without copying the old referent onto a revised target. The live agent has issued unspecified target questions, but retained telemetry does not establish that any particular model response lost its descriptor in repair; this closes a confirmed source omission, not a recognition or gameplay gate.
- [x] Retain explicitly requested named-target companions (`near`, `mineable`, `kind`, `dx`, `dy`) beside their accepted same-query, same-time `target.visible=True` anchor during the existing scene-owned follow-up. Preserve negative/zero values, confidence and evidence references; keep the existing expiry, preemption and bounded action grace without another observation call. Focused regressions qualify source behavior only: this does not establish refreshed geometry, useful live recognition or successful mining.
- [x] Reject target questions with no explicit referent before perception admission or crafting cleanup. Reject the entire mixed request and simultaneous skill, retaining the model's original replan choice, goal and action restrictions; do not adopt its plan or acknowledge an unresolved operator instruction. Existing operator retries and valid non-target questions remain unchanged. Regression checks include exact bound-decision receipts and the one-shot stall fallback. This avoids spending another vision turn on an undefined target, without inventing a target or claiming useful observation, mining or escape.
- [x] Separate explicit inventory-inspection candidate admission from literal direct-input parsing after a no-logs crafting failure. Active ordinary inventory instructions (including acknowledged instructions) and fresh corrections with unrelated actuator prohibitions may make `open_inventory` available to deliberation; crafting remains blocked, and quoted, negated, conditional or query-only wording gains no new exception. Archived or acknowledged corrections and directives hidden behind newer authority do not qualify. Keep the literal fast path, action restrictions, nullable decisions, perception requests and acknowledgment lifecycle unchanged. Regression coverage qualifies source behavior, not live command fulfillment or a repaired possession prerequisite.
- [x] Give the selected unresolved operator instruction/correction a stable deliberation opportunity: cancel and release only a source-proven disposable keepalive, then require a fresh capture before cognition. Suppress optional reorientation and replacement wandering while that directive remains queued/delivered; questions, feedback, acknowledged work and real recovery retain existing behavior. Forty focused regressions preserve execution-revision rejection, single cancellation, scene recovery and literal dispatch after one capture without a model wait. The previous `de7548` live trial was delivered at +58.52 seconds but produced no operator-owned execution within 120 seconds; real locomotion/recovery failures occurred during that window. Its legacy adapter retained no exact future-discard receipt, so snapshot rejection is a source-supported diagnosis rather than an observed per-request result. This change does not prove ordered completion or protect all post-acknowledgment continuations.
- [x] Recognize the exact canonical `close_open_inventory` plan node, which the existing prose matcher incorrectly rejected after replacing underscores with spaces. Twenty-four regressions retain existing prose matching, goal/outcome/plan-neutral gates and terminal deduplication. A recorded open/close pair now advances the matching canonical nodes once in regression; recovery context is deliberately unchanged. The previous live close remaining scene-owned was not itself the cause of the bookkeeping failure. This exact-ID repair does not create an operator continuation, alter action/success rules or establish a paid completion receipt.
- [x] Preserve the strict crosshair classifier's parsed block/confidence and exact query/crop identity as diagnostic completion metadata. `2823fd0` passes 192 focused tests: explicit unknown, null block and missing/zero confidence remain distinct without publishing terrain facts or adding inference. In the current Hard-world generation the first real classifier call completed in 37.87 seconds without usable block evidence; its older raw pair was not retained. No escape or mining success is claimed.
- [x] Give the next eligible planner context one bounded historical outcome when a completed headroom inspection establishes no clearable target. The 250 focused regressions preserve query/crop/run attribution, distinguish a published explicit abstention from generic unusable evidence, and exclude the record after operator/context changes or during successor execution. This is working-context feedback, not a new terrain fact, retry policy, durable learned skill or demonstrated escape.
- [x] Observe a query-bound soft-block break and visible resource possession in Hard survival. Query `752679d148e548789d0dcdb48336e60d` returned dirt at 0.90; run `5c9fdcb75bda4d18ae9a70722ce2f3d0` held supervisor-accepted attack for about 2.664 seconds without camera or locomotion. Retained trajectory frames show damage, a cavity and a dirt icon; the later full-resolution hotbar shows two dirt. The following bounded jump/forward retry changed the local view but ended beside dirt. This is human-reviewed acquisition evidence, not an automated dirt-count receipt, a completed escape or a retained replay of the classifier's original full-resolution crop.
- [x] Qualify one bounded exact-window H.264/AAC encode with a dedicated game-only audio monitor, without changing the GPU workload or recording desktop/microphone audio. The ten-second local 720p24 clip decoded successfully, but its audio was silent. Audible game audio, persistent video delivery and a public video service remain unqualified; the working public viewer is still the JPEG fallback.

- [x] Implement and review the headless Weston virtual-only seat input path, including exact display-to-Xwayland-to-compositor ownership and inherited host-transport/module-override rejection. The disposable session passed 19 checks covering holds/releases, relative raw/core mouse input, confinement, accelerated capture and live admission. These source/display results do not qualify Wine or the preserved game; see [scope and evidence](INPUT_ISOLATION.md).
- [ ] Deliberately migrate the preserved game to the headless session, then qualify Win32 clipping, camera response and actual gameplay movement. Display-level tests do not close this live gate.

- [x] Guarded private-LAN operator viewing and commands; agent output uses the private game display. Excluding host-origin input from that display is a separate, still-open qualification below.
- [x] Repair local hostname advertisement collisions without restarting the game; verify the dashboard, live frame, message history, and readiness over the physical LAN interface. Access from a second physical device remains unverified.
- [x] Deterministic inventory open/close: live transitions verified without restarting Minecraft.
- [x] Recognize the observed populated wide inventory without relying on recipe-tile colors. Require fixed header, search-bar and border chrome together with the existing independent right-panel checks; retain the old dark-slot/compact paths and confidence. Forty regressions cover missing regions, negative controls and NumPy/fallback parity. The retained full-resolution inventory is recognized and retained world/title controls remain rejected; this does not qualify arbitrary UI scales, item identity or ordered command completion.
- [x] Exercise that inventory recognition in the live Hard world. On `8ebdda3`, ordinary trial `871a143a9df943aba29d56214c662420` produced operator-owned open run `5946fd683ef042839fc556dce088e842`, accepted E sequence 478 and visible inventory at trajectory step 479; the fast observer published `scene.inventory_overlay=True` and `scene.playable=False` at 0.995 confidence. Open succeeded in 631 ms. Scene-recovery close `484ea23bcba84ab890ae361ff492c738` emitted E sequence 480 and succeeded in 421 ms; the retained full-resolution successor shows the normal survival HUD. No movement or attack occurred during the GUI sequence and no second inventory cycle occurred in the first 120 seconds. Delivery took 54 ms, but execution began about 94 seconds later after the protected prior gather finished and slow cognition completed. The close remained scene-owned, leaving the operator plan at its close step: this is verified GUI functionality, not full instruction completion. Root retirement/archive afterward is cleanup, not gameplay success.
- [x] Observe the ordered inventory portion and matching canonical plan advancement in a fresh live trial. On `20f0ea3`, instruction `1302d8d576104b549b817bb15d96b6ff` released disposable walking once; the next accepted positive input was open E sequence 51 about 53.25 seconds later. Operator open `0e2b662a5db445e39de2d58b5dea619f` produced visible, trusted inventory evidence; scene-recovery close `5cb13b5e73de4d6c9ab4a1a9b8f56d9f` emitted E sequence 53 and returned to a retained normal-HUD frame. The plan advanced 0 → 1 → 2. The first 120 seconds contained exactly two E presses, no attacks, no movement while waiting or during the GUI sequence, and no repeated inventory cycle. The close keeps its actual recovery attribution. This qualifies physical sequencing and plan bookkeeping, not complete retirement of the instruction or a paid receipt.
- [ ] Complete one-time operator-command retirement and durable return to autonomous intent. Trial `1302d8d576104b549b817bb15d96b6ff` retained a literal `"null"` third plan step and its operator goal after the verified close, even though optional locomotion resumed. Root retirement/archive restored ordinary autonomous operation afterward; that cleanup is not agent-completed intent retirement. Determine the next narrow plan/completion correction before adding paid admission. Acknowledgment, a no-op close, a synthetic terminal label or retrospective relabeling of recovery cannot serve as a paid completion receipt.
- [x] Recognize the lower-center AFK notice with narrowly cropped OCR and implement one screenshot-bound menu click followed by fresh normal-HUD verification. The complete new autonomous branch passed its next natural occurrence: after 12 missing-HUD checks, one AWAY-to-IN_WORLD click restored navigation and agent readiness without a game/GPU restart. Failed wake-key attempts were retained, not shipped as fallback inputs.
- [x] Keep machine-local adapters, settings, runtime artifacts and model payloads excluded from public Git; existing CI tests reject tracked files in those categories while preserving generic adapter code and public model metadata. This complements source review, not a guarantee that arbitrary source text contains no private material.
- [x] Add an optional request-aware cognition interface with immutable semantic snapshots, durable operator revisions, exact attempt accounting and final publication checks. Expired repair work, superseded intent and bound decisions requiring crafting cleanup abstain before side effects; legacy crafting remains unchanged. The 163 new regression cases qualify this generic interface, not custom-model activation, online learning or live gameplay.
- [x] Agent-only reload: live game, GPU server, input calibration and supervisor preserved.
- [x] Reject runtime startup before model workers when its initial lease renewal fails, and honor cancellation between startup phases, including after warming telemetry. Confirmed pause/stop rejection remains normal shutdown; genuine lease expiry remains a fault. Thirteen pure regressions include actual vision-worker thread-start failure and downstream cleanup. This does not maintain the lease during earlier process/database/capture assembly, which remains open before heavyweight native initialization.
- [x] Add an explicitly configured local planning factory to the canonical agent process, preserving supplied movement/calibration and resource ownership. Pure regressions qualify existing-lease renewal during construction, sticky cancellation, protected fields and cleanup without supervisor mutations. Native imports/constructors remain cooperatively cancellable; the earlier process/database/capture lease gap and actual custom-model activation remain open.
- [x] Add an optional deny-only fresh-capture continuation check for local runtime adapters. The immutable bundle binds the exact capture to its synchronous fast facts, excluding older merged semantics and clearing after capture/inference failure. Twenty-four focused regressions plus 534 existing startup, recovery, factory, perception and geometry tests preserve ordinary stop/release cleanup, stale-frame precedence, and input authority. Default behavior remains unchanged; no private model, pilot activation, automatic world reset or learning claim is included.
- [x] Support an optional per-client lifetime worker-start limit for bounded pilots. Exhaustion rejects before replacement shared-memory allocation, and a failed spawn consumes an attempt; close/reset do not refund it. The 165-test policy/transport checkpoint preserves unlimited default restarts and fail-closed input release, with startup verification/liveness/counts exposed in status. This prevents a second process start, not excess inference/memory use or a live native qualification.
- [x] Complete one bounded private raw-motor pilot through the existing generic adapter. The 08:38 UTC recording contains two exact native-attributed supervisor-accepted camera-only predictions, with 339 recorded steps and zero drops. One raw reply was late. Separately, the policy counter records three selections, not three accepted predictions; the traversal ended in controller starvation. The ordinary teacher configuration was exactly restored without replacing the game, GPU process, supervisor or calibration. No native locomotion, adaptive learning, full-native activation or performance gain is claimed.
- [x] Expose optional metadata-only skill start/terminal hooks with exact accepted decision origin captured at actual execution start. Twenty-four pure regressions cover later-decision attribution, failed-model fallback, recovery parent IDs, detached payloads and duplicate/mismatched terminal evidence. Consumers own durable joins and learning; these hooks do not establish supervisor-accepted input, online learning or live competence.
- [x] Exact traversal-stall recovery transaction implemented and tested. Live reorientation/query ran, but the initial VLM answer abstained; escape is not yet demonstrated.
- [x] Version-pinned oak hotbar counts 0–16 and pre-break-to-post-collection exact +1 verification implemented. Hotbar evidence is not a whole-inventory count; unknown items, including spruce logs, abstain.
- [x] Recognize all nine selected hotbar positions using the pinned vanilla selection-frame asset. Raw live replay recovers the previously rejected fourth-slot geometry; unknown spruce remains unknown to the oak-only counter.
- [x] Turn an empty initial replan into canonical visual prerequisites when an explicit requested skill or failed run identifies what to observe. This closes one no-op path; live perception-to-action completion remains unverified.
- [x] Permit exact vanilla leaf blocks in guarded hand-safe clearance, using the existing 2.5-second maximum verification lease. This is not a leaf-drop or live escape claim; stone/tool and gather-log restrictions remain unchanged.
- [x] Admit tiny natural-lighting drift only with a tighter structural match: keep crosshair dHash ≤4/RGB error ≤1, additionally permit dHash ≤2/RGB error ≤2. A stationary 20.57-second raw pair measured dHash 2/RGB 1.833 with unchanged camera updates and no held inputs; existing colour-swap and dHash-3/RGB-2 rejection tests remain intact.
- [x] After a headroom query abstains, retain the existing traversal escalation latch until cognition can decide. Re-enabling disposable walking immediately was repeatedly invalidating every slow planner answer on the next obstacle failure. Explicit operator and safety routes remain independent.
- [x] Treat entirely unsupported perception questions like empty ones when a requested/failed skill supplies canonical prerequisites. Live `23a2f38` finally consumed planner answers, but invented `scene.horizon_visible` and `scene.nearby_objects_types` never started a query; regression cases now cover those names.
- [x] Apply the same prerequisite fallback to skill-bearing decisions with unsupported questions. Live `3df483a` exposed `scene.horizon_visibility` bypassing the null-only repair. It also completed one canonical `scene.playable` query in 23.4 seconds, but without a positive fact or subsequent action.
- [x] Apply canonical prerequisite repair to null decisions with unsupported questions even when the model omits the replan flag. Baseline replay reproduces `fresh_facts.target_block_type` escaping repair; the corrected contract preserves pure idle decisions. This is a control-flow fix, not verified mining or escape.
- [x] Admit one runtime-attributed `obstacle.ahead` observation after an autonomous pure-idle decision and two distinct typed locomotion stalls in the same unfinished plan. Preserve action abstention, operator intent, scene ownership and the escalation latch; a consumed run ID is not retried after unknown/stale output or timeout. Regression coverage passes; the live failure-to-observation-to-accepted-action gate remains open.
- [x] Failed crafting cleanup is plan-neutral; a verified no-logs failure persists a retry prerequisite until new positive wood evidence or one fresh explicit retry. This is permission to re-audit, not a retained inventory count.
- [x] Experimental compact crosshair-query format fits the existing output budget, preserves crop provenance, and bounds numeric output. It is not promoted to live use: pinned daylight/dark-dirt probes still produced wrong block identities. Recognition accuracy and autonomous escape remain separate live gates below.
- [x] Make the focused crosshair probe produce useful, correctly grounded live answers within the small model's output budget. On the live 1920x1054 dirt-overhang frame, the strict two-field probe returned `dirt` at 0.90 confidence in 36.96s; source/current crop dHash matched and RGB-grid drift was 0.318 against the 1.0 admission limit.
- [x] Implement an observation-only visible-surface survey for body-clearance investigation: one nullable underside/riser/side-face point, exact WORLD provenance, and no motor tracks or action permissions. Retained-image probing is executable; this is not a measured body collision or a local 3D map. See [body-clearance scope and qualification](BODY_CLEARANCE.md).
- [ ] Qualify correct overhead, footstep and mirrored side-edge obstruction selection, with open/high-ceiling/occluded controls, before promoting survey targeting into live recovery.
- [x] Stop learned world controls on the observed centered storage-error dialogs, including the compact low-space acknowledgement. The fast interlock publishes only negative scene safety facts, never an error diagnosis or training labels. Scaled chrome/terrain controls and both retained live frames qualify this narrow detection.
- [ ] Demonstrate repeatable cold startup and world transfer without an intermediate retry. The earlier 30-second geometry preflight and current-boot crash guard failures remain failed attempts. A later launcher-approved cold start and actual world rejoin now work without a reboot, but one transfer-caption OCR failure still returned before the world finished joining. A live client or longer timeout alone does not establish reliable startup or camera delivery.
- [ ] Qualify camera-origin delivery before using the pitch accumulator as body-relative pose. Retained upright trunk edges indicate upward viewing despite a downward estimate; a later ordinary homing sequence again reported zero while the image remained upward-looking. Accepted mouse counts and a completed open-loop calibration are not measured physical orientation.
- [x] Invalidate an established camera origin before recalibration lease issuance (which itself releases backend inputs) and recheck pause/emergency after final settling and lease cleanup, before publishing success. Issue-time failure, interrupted return, backend failure, final-settle and cleanup regressions preserve the measured sensitivity/profile but leave origin validity false; invalid requests and pre-existing interlocks preserve the prior origin. This corrects state publication, not physical horizon delivery.
- [x] Distinguish controller starvation from action-backed collision stalls. The existing bounded no-input startup guard now reports `controller.starvation`, retains its actual cause, and does not nominate a physical obstacle recovery or count toward the two-stall observation fallback. Regression coverage includes release, memory attribution and fallback exclusion; live movement remains a separate gate.
- [x] Permit one context-local exploration-choice rotation after a keepalive receives three distinct supervisor-accepted current-run predictions that select no keys/buttons. Bounded evidence excludes cold waits, prediction holds, retired/foreign provenance, suppressed actions and rejected input. Operator, scene, execution and plan changes discard the hint; starvation classification and persistent failure statistics remain unchanged. The 358-test recovery/core checkpoint qualifies this one-choice mechanism, not improved live locomotion or an obstacle observation.
- [x] Preserve run-lifetime startup movement evidence across progress-window resets, and exclude typed controller starvation from failure-derived canonical perception questions. Explicit operator/model requests remain independent; regression cases retain later genuine stalls and new-run starvation.
- [x] Qualify the optional external raw-motion adapter through the actual temporal client without actuators, including matching replies, retired-request drainage and fresh option context after reset. Private deployment configuration and measurements remain outside this repository. This is delivery/reset qualification, not live movement, survival competence, online learning or a universal latency guarantee.
- [x] Implement and regression-test a pause-preserving agent-reload resume capability. It requires the exact paused supervisor generation, revoked lease, retired agent and clear persistent intent; it neither clears pause/emergency nor recovers/stops FAILSAFE. The user-authorized supervisor update has now installed it without restarting Minecraft or the GPU server. The measured calibration profile was preserved and the isolated camera origin deliberately re-established, not copied as a verified pose.
- [x] Require a successful worker-generation handshake after failed warmup, reject semantic/GUI output from raw-only external workers, and bound per-call reply reads while retaining partial-response ownership. Regression checks cover rejected startup reuse, fact publication, zero-budget fragmentation and positive deadlines. This is admission correctness, not live controller activation.
- [x] Preserve immutable inference-source capture ID/time through predictions, held actions and accepted trajectory provenance, separately from the control-time capture and blackboard IDs. Delayed-reply, reset/retirement and legacy-replay regressions pass. Metadata does not retain missing source pixels or establish a learning/translation result.
- [x] Use one private X connection for ordered relative camera motion, key/button events and final synchronization. Chat settle delays now follow flushed events; unconditional releases never park or warp the pointer. Wire-level relative-motion and cleanup regressions pass; physical camera delivery remains the separate open gate above.
- [x] Remove implicit private-pointer parking from WORLD motion/button admission: unknown/outside routing now rejects without an absolute warp. Explicit screenshot-bound GUI cursor placement remains separate. This removes an uncommanded motion source; it does not establish which input boundary caused the retained camera failure below.
- [x] Revoke motor authority when lease binding, renewal, accepted-action refresh or input release fails; attempt backend capability clearing even after release failure without masking the original fault. Successful renewal preserves held input. Fault-injection tests prove logical containment, not physical release after an X-server failure.
- [x] Reconcile acknowledged external input release with policy, mining and outcome bookkeeping without resetting worker memory or crafting phase. Unknown acknowledgements block positive actions; retired predictions are drained without replay, the next fresh desired hold emits a new press, and traversal waits for a post-release image. Pure regressions pass; live interruption-to-resumed-key delivery remains unqualified.
- [x] Authenticated malformed actions revoke previous held input; pause/disarm/fault/stop state and every owned shutdown cleanup progress even if physical release raises. The first error remains attributable. These are fault-injection results, not a claim that an unavailable input server physically released a key.
- [x] Make capture/GUI geometry resolution read-only and reject changed, clipped or incomplete frame geometry. Confine measured window fitting to a new isolated launch before publishing its session, with existing-session/host paths excluded. Regression coverage and live capture pass; a fresh game launch has deliberately not been exercised on the running world.
- [x] Reject render-only keyboard focus inside the game subtree and restore the verified private input parent, with interlock and focus readback. Live recovery then opened the pause menu with one Escape tap and returned to normal HUD with one screenshot-bound Resume Game click. This verifies that UI sequence, not sustained locomotion or universal key delivery.
- [x] Classify exact pointer-route failures separately from generic motor errors and retain that classification through supervisor retirement. The persistent launcher now holds without navigation, calibration or game-restart escalation until an externally recovered healthy generation passes readiness. Regression tests cover 50 held polls/500 simulated seconds, unreadable status, unavailable startup dependencies and failed status publication. Existing launchers do not acquire changed shell functions in place; the current launcher remains explicitly held until a safe handoff.
- [x] Run the registration-free, window-free Win32 ownership probe before further input experiments: 50 samples at 100 ms retain Minecraft foreground/active/focus/capture while the reported clip and actual cursor disagree. This diagnoses a confinement mismatch, not its precise implementation cause or a repaired controller.
- [x] Reproduce the logical/physical clipping split in two disposable, GPU-free Wine sessions: a single successful clip request stays physically unenforced with Explorer-owned X focus, but maps and clamps with application-owned X focus. Retained traces and asynchronous sample counts are documented in [the Wine A/B evidence](WINE_CLIP_REPRO.md). This is a reproduced mechanism, not an installed Wine fix or live game qualification.
- [x] Build the clipping-local Wine correction and qualify it against matched rebuilt controls in four fresh disposable sessions. Exact loaded driver/core identities were verified: baseline reproduces the failure, candidate restores physical confinement under Explorer focus, and ordinary-focus control remains intact. The separate 15-case predicate regression rejects foreign foreground ownership. This closes the synthetic clipping gate only; no live replacement, camera response or game migration is claimed.
- [ ] Exclude host-origin pointer, keyboard, modifiers and activation changes from the private game session, then qualify simulated input and retained-image response while the host is used independently. A different display name or XTEST acceptance alone does not establish this boundary.
- [ ] Complete the end-to-end input audit and retained-image qualification: camera response, held-key continuity after stale frames, geometry/capture consistency, authenticated malformed actions and supervisor cleanup faults. Do not promote accepted input into physical translation or flawless delivery.
- [ ] Demonstrate a fresh accepted movement prediction after a retired worker response is drained by its owner, then verify actual translation rather than infer it from accepted keys.
- [ ] Demonstrate autonomous soft-block clearance and escape from the observed dirt pocket.
- [ ] Demonstrate three consecutive verified log acquisitions without manual intervention or game restart; extend item recognition only from calibrated evidence if the local tree variant requires it.
- [ ] Demonstrate a verified log-to-planks inventory transformation, then use that run to select the next progression upgrade.

Retain failed trajectories as failures. Do not promote a successful block break into resource possession, or test coverage into a completed live gameplay milestone.

Input audit (`bf60511`): one supervisor-owned `dx=0, dy=+8` camera-only pulse
produced a large lateral/rotational view change in the retained images, not
the intended small vertical response. No key/button or second/reverse pulse
was sent. Across stationary retirement, lease-issue and settling controls,
200 tracked terrain features had median displacement below 0.005 pixels,
95th percentile below 0.052 and maximum below 0.307. The same supervisor,
game client geometry and sensitivity profile were retained through the pulse.
This isolates an action-boundary delivery failure, not its internal cause:
focus restoration, implicit parking and relative event handling were not
individually observed. The accepted pitch counter is not physical pose truth.
The preceding non-actuating attempt was rejected by a colour-drift gate;
retained-image tracking later showed its geometry was effectively stationary.
Neither failed attempt is discarded or counted as camera qualification.

A later isolated `dx=+8, dy=0` probe retained the supervisor, game, sensitivity
profile and stationary pre-action controls. Passive observation recorded one
XInput raw/transformed `(8,0)` event and one corresponding core client motion
from `x=1887` to `1895`, with `y=441` unchanged and no focus/crossing event.
The retained game image nevertheless stayed effectively stationary. Review
found this experiment was observer-contaminated: selecting core motion on the
inner drawable can stop propagation before Wine's already-interested parent.
The trace establishes the XTEST raw request, but its lack of camera response
does not establish a normal delivery failure or a root cause inside Wine.
Future core observation must not introduce a new event-propagation endpoint.
No reverse pulse, key/button press, calibration change or game restart followed.
Preceding non-actuating trials rejected ambiguous tracking or insufficient
scene features; their receipts remain failures, not successful experiments.

The corrected parent-only observer subsequently recorded exactly one raw/core
`(8,0)` motion, no extra X warp or focus/crossing event, while the retained view
changed sharply from downward ground to the tree canopy/horizon. All four
pre-action controls retained 200 stationary terrain features (maximum drift
below 0.077 pixels). This narrows the remaining camera fault below the observed
X event path, without proving a particular Wine/game mechanism. A later
autonomous pointer-routing rejection and failed startup homing required guarded
recovery. The keyboard-focus fix and observed menu sequence restored normal
startup/readiness with Minecraft and the GPU process unchanged. Camera
qualification remains open; homing completion is still not physical pose truth.

The later route failure recurred after the keyboard recovery: the private X
pointer reached `(804, 0)`, outside the drawable whose top is at `y=26`.
Repeated startup navigation correctly found the world, but homing rejected
the same unchanged pointer route. The outer launcher was held before its
generic failure-count path could restart the still-running game.

A separately compiled, window-free, registration-free Win32 query helper
then found a persistent mismatch: `GetClipCursor` returned the point rectangle
`(565, 376, 565, 376)` (the earlier Resume-click location), while both
`GetCursorPos` and private X queries returned `(804, 0)`. This was measured
first with the AFK overlay and separately with normal-world HUD, retaining
before/after images and unchanged focus, geometry and neutral inputs. In the
normal-world run, 31 Windows samples across about 15 seconds agreed. This is
logical clipping versus physical pointer disagreement, not yet proof of the
exact camera-delta corruption mechanism.

The pinned WineGDK focus handlers reapply virtual-desktop clipping on desktop
FocusIn, but one explicit private desktop-focus restoration did **not** fix
this live state: subsequent query samples retained the same mismatch. That
negative result is retained. A separate no-input GUI raw-input-sink smoke
exited cleanly with zero raw events and no observed X focus/pointer/geometry
change; it does not establish the game's receipt of any future raw packet.
No clipping setter, hidden pointer warp, raw-input packet forgery, GPU reset
or game restart was used. The input qualification and live playback gates
remain open; the public preview is withheld while the agent is not ready.

The follow-up five-second, registration-free/window-free probe includes
`GetGUIThreadInfo(game_tid)`, not calling-thread `GetCapture`. All 50 samples
retained the discovered Minecraft window as foreground, active, focused and
captured, with flags zero and clip `(565, 376, 565, 376)`. Both Win32 and X
cursor queries instead returned `(0, 83)`. All 53 read-only X snapshots kept
the same Wine-desktop focus, zero held-input mask, drawable geometry
`(0, 26, 1920, 1054)` and pointer hit-chain through the game. Thus the logical
clip is not physically enforced; the actual cursor APIs agree with each other.
The unique visible window title identified the read-only query target;
`QueryFullProcessImageNameA` returned success with an empty image name, so the
earlier executable-identity checks failed closed before sampling. This is not
an executable-identity authorization for an injector.

Before/after images retained normal-world HUD. Feature tracking kept 198/200
correspondences, median displacement 0.010 pixels and 95th percentile 0.115,
but a 20.67-pixel outlier prevents claiming a strict stationary-scene input
qualification. No input was injected and no raw registration or cursor/focus
setter followed this probe. The user separately reports that ordinary host
mouse usage changes the private game pointer/camera: the nested Wayland
backend still forwards its parent seat. A separate X display therefore does
not by itself prove host-to-game input exclusion. That correction is now an
explicit open gate, independent of Wine clipping and motor-policy quality.

A subsequent read-only XRes client-PID query proves that the focused Wine
Desktop X window belongs to Wine's `explorer.exe /desktop` process, while both
the game's input parent and render child belong to the Minecraft process.
Focus and pointer remained unchanged during that query. This supports the
pinned Wine `grab_clipping_window()` foreign-X-focus early-return hypothesis:
Windows foreground/capture ownership and the driver's process-local X focus
test are distinct. It is not direct instrumentation of the game's Xlib context
or evidence that a source-level correction has been installed.

At the next natural AFK episode the existing launcher performed one
AWAY-to-IN_WORLD UI action and started the updated runtime, retaining the
same Minecraft/GPU processes and measured sensitivity profile. No manual
movement or native-model switch was used for that recovery. Startup now
describes homing as command completion with an unverified physical horizon;
the operator dashboard no longer labels the counter as verified physical pose.

Diagnostic handoff defects are also retained: releasing an older outer
watchdog before replacement readiness caused a warm-up replacement, and a
later diagnostic process launched the agent with the policy interpreter,
which lacked Xlib. The failed child was contained and the normal launcher
restored readiness with Minecraft/GPU unchanged. Future probes must use the
application interpreter for agent launch and confirm readiness before handing
back to the outer watchdog. These operator errors are not model failures.

Latest live observation (`16da921`): the generic external raw-motion route
produced attributable, supervisor-accepted forward inputs through the unchanged
skill/safety guards. The bounded traversal ended in `locomotion.stalled` and its
terminal reset released all locomotion keys. A preselected camera-free forward
interval was effectively stationary in the retained images: accepted-input
execution passed, but translation, escape and collision geometry did not.
Surrounding no-input attempts now retain `controller.starvation` rather than
manufacturing terrain failures. The live agent and recording were operational
at that checkpoint; a later AFK overlay caused the launcher to retire the agent
and supervisor despite the game remaining alive. The notice was subsequently
dismissed via a screenshot-gated menu click; normal navigation, replacement
agent readiness and recording recovered with the same game/GPU processes.
The saved baseline configuration was used for this recovery; the earlier
private canary remains retired. Detailed deployment receipts stay private.

A later natural AFK episode did not repeat that successful click recovery:
two separate one-click navigation attempts failed their normal-HUD gate. The
launcher was contained before escalation. A desktop-focus click variant also
failed; one bounded supervisor-owned sneak tap subsequently restored the HUD.
The view changed markedly upward across backend attachment, lease issuance,
the tap and cleanup despite zero commanded camera deltas. This does not isolate
which boundary moved the view. Do not promote the click path as universally
reliable or treat command integration as observed camera orientation. The
planned small camera-pulse comparison never reached actuation; its failed
dependency preflight was retained, not counted as a physical experiment.

The private-X pointer route now avoids absolute recentering when fresh pointer
queries reach the exact game client inside its bounds. A single no-action
lease-issue/revoke bracket retained the same attached supervisor and game:
200 selected static terrain features had zero median displacement across both
boundaries, with less than 0.005 pixels maximum across the retained comparison
frames. Pointer coordinates and commanded-camera accounting were unchanged.
This is a stationary observed bracket, not a measured warp count, a controlled
before/after comparison, physical-horizon qualification or gameplay success.
A subsequent agent stopped after 119 accepted records on pointer-routing
rejection. Synthetic reproduction exposed an overly strict requirement that
coordinates stay identical across separate X queries; in-client recentering
now remains admissible, while each sample must still satisfy same-screen,
client bounds and the actual bounded hit chain. The exact live rejection
branch was not instrumented, so its cause is not asserted. Scoped recovery
replaced the faulted supervisor and agent, preserving Minecraft, the GPU and
the measured sensitivity profile; fresh readiness and recording passed.

The trace also exposed a training-data limitation: the retained trajectory frame
is the observation at action acceptance, not necessarily the asynchronous
prediction's original input. The new generic annotation now carries source
capture ID/time from the submitted request, while recording the control-time
capture ID separately from the blackboard ID. Legacy records remain unknown;
lookups must stay within their capture/recording generation. Missing source
images are not retained by this metadata patch, and compact trajectory JPEGs
are not pixel-exact worker inputs. Do not use an adjacent frame as invented
training truth. A subsequent fixed 303-step baseline prefix verified immutable
source metadata on all 48 predictions and 91 holds, with eight resets remaining
null and no recording drops. Its first directional prediction retained the
same request/source through the hold and key release, then cleared them on
terminal reset. This passes accepted-input attribution, not original-pixel
retention, physical translation or a successful learning example.

Historical live observation (`b5a551c`): the user approved returning to autonomous
play; prior operator directives were archived without deleting their history.
A six-minute canary recorded 48 accepted motor steps, zero recording drops and
two distinct `locomotion.stalled` runs. Four cognition-owned `obstacle.ahead`
jobs completed with unknown answers; the separate existing headroom query was
rejected as stale. Even the final obstacle job, with frame dHash distance zero,
returned unknown after 44.17 seconds. Consumed decisions already requested
perception and replanning, so the runtime pure-idle fallback marker remained
unused: its live acceptance gate has **not** passed. No post-observation escape,
new acquisition or crafting success is claimed; the visible 10 dirt and 15
spruce logs are pre-existing stock.

A later read-only audit of the same agent's 193 contiguous recorded steps found
no locomotion key presses in either latest run: `b404fae` (37 steps) and
`177a7b0` (21 steps). Both three-second failures used `locomotion.stalled`, but
the verifier also emits that code for insufficient commanded movement. These
are controller-starvation evidence, not proof of a terrain collision. No worker
stdout was consumed externally and no in-flight shared frame was overwritten to
force resubmission. The old idle-stall fallback eligibility remains unchanged.

The earlier action-backed run `2c2f07287ebf43efb167b16b3b698641` also lacks
a useful measured translation baseline in its first clean forward interval:
steps 42–46 span 1.175 seconds with accepted forward input and no camera delta,
but retained WORLD-feature displacement has median 0.021 pixels and maximum
0.448 pixels. The camera-only control has median 11.152 pixels. These are useful
negative controls, not verified collision geometry or a local 3D map; see
[the retained-frame qualification](BODY_CLEARANCE.md#retained-motion-qualification--2026-09-05).

The external worker replay used an immutable, verified import root and no
adaptation, resume or persistent learner state. Activation was initially deferred: a reload
safety review found that ordinary supervisor `resume` clears durable operator
pause and may retire a FAILSAFE generation. Checking pause before that IPC is
not an atomic guard against a new pause. The existing live agent was left
running until the user approved the supervisor update described above. A
pause-preserving supervisor capability is required before future agent-only reloads;
do not use ordinary `resume` as a workaround or restart the whole game stack.

Deadline misses remain failed delivery attempts even when safely contained.
Keep unsuccessful private deployment replays alongside successful ones; do not
rerun until a convenient timing pass or publish proprietary implementation,
model identity, configuration or performance records with the generic adapter.

Deployment incident: an intent-lock/IPC timeout during the first reload let the
outer launcher replace the supervisor; Minecraft and the GPU model survived.
The corrected agent-only reload to `b5a551c` preserved the replacement
supervisor, game, GPU process and calibration, and readiness/recording passed.
Do not hold the operator-intent lock across `resume` IPC: its server handler
takes that lock. Keep the outer watchdog suspended until a guarded replacement
agent is established; no emergency or operator pause may be cleared implicitly.

## Phase 0 — Public extraction and safety foundation

**Goal:** a clean standalone project that cannot accidentally expose private/proprietary dependencies.

Deliverables:

- standalone package namespace and CLI;
- Apache-2.0 licensing and third-party notice process;
- public-boundary audit for private URLs, personal paths, credentials, internal names and proprietary adapters;
- supervisor state machine;
- `run/status/pause/resume/stop/logs` control protocol;
- capability-leased motor interface;
- zero live-input default;
- platform/architecture capability detection;
- deterministic test harness and fake Minecraft backend.

Exit gate:

- repository contains no private Neuroforge/ERAIS integration;
- safety fault tests pass against fake backend;
- stop is independent of cognition.

## Phase 1 — Instance-scoped capture and input

**Goal:** control one Minecraft instance without taking over the operator's desktop.

### Java

- detect installed/running versions and launcher profiles;
- create minimal client-side bridge for bounded key/mouse semantics;
- local authenticated IPC;
- instance-scoped frame capture where available, otherwise window capture;
- chat event bridge that does not expose privileged world state in strict mode;
- CI builds for supported Java/loader versions.

### Bedrock

- process/window discovery;
- platform capture adapters;
- isolated-session backend where possible;
- focused compatibility backend with explicit opt-in when isolation is unavailable.

Exit gate:

- operator can use another application while Java agent moves continuously;
- stop/pause releases all held controls;
- target-instance swap fails closed.

## Phase 2 — Versioned game knowledge compiler

**Goal:** remove hard-coded progression knowledge.

Implement `GameVersion` and provenance graph.

Importers:

- exact-version Java recipes/tags/loot/advancement data extracted from version artifacts/data packs;
- normalized versioned open data adapter (e.g. minecraft-data) as secondary coverage/cross-check;
- Bedrock vanilla behavior/resource data adapters where redistribution/access permits;
- wiki retriever with revision/source/version metadata;
- manual override/errata layer.

Graph domains:

- crafting/cooking/smithing/brewing;
- drops/loot;
- tool/harvest requirements;
- dimensions/structures/biomes;
- trades/bartering;
- portals;
- advancements/achievements;
- equipment/enchantment constraints.

Build graph query operations:

- `requirements(target)`;
- `ways_to_obtain(target)`;
- `prerequisite_closure(goal)`;
- `progression_paths(goal, state)`;
- `explain_fact(fact)` with provenance.

Exit gate:

- generated dependency plans for representative early/mid/endgame items match exact target version;
- version mismatch tests fail loudly.

## Phase 3 — Perception blackboard

**Goal:** stable real-time state estimates instead of frame-by-frame VLM narration.

- 20-30 Hz capture pipeline;
- frame ring buffer;
- visual encoder and temporal tracker;
- HUD/hotbar/GUI/chat region parsers;
- crosshair target estimator;
- terrain affordance/drop-edge estimator;
- asynchronous task-conditioned VLM queries;
- confidence decay and semantic reconciliation;
- event detection: damage, death, inventory change, block break, GUI transition, target loss.

Exit gate:

- blackboard state stays useful during VLM latency/outage;
- p95 visual-to-blackboard fast-path latency meets hardware profile target.

## Phase 4 — Skill runtime and verifier

**Goal:** replace macros and static recipe actions with closed-loop options.

- typed `SkillSpec` schema;
- precondition/effect predicates over blackboard + knowledge graph;
- parameter binding;
- skill executor;
- local recovery graph;
- success/failure verifier;
- contextual outcome database;
- candidate/trusted/deprecated lifecycle;
- skill composition and dependency resolution.

Seed robust hand-engineered/behavioral skills only as bootstrap:

- look/scan;
- approach target;
- unstuck;
- mine visible block;
- collect drop;
- select hotbar item;
- eat;
- place block;
- basic GUI/crafting interaction;
- retreat/shelter.

Exit gate:

- high-level planner requests semantic skills only;
- no strategic code emits individual key presses.

## Phase 5 — Human motor policy v1

**Goal:** replace heuristic movement/aim with a learned behavioral prior.

Data pipeline:

- ingest legally usable human gameplay/action datasets;
- support user-recorded demonstrations;
- align frames/actions/goals;
- action tokenizer for keys, buttons and mouse movement;
- trajectory quality filters;
- automatic goal relabeling where reliable.

Model:

- compact visual encoder;
- recurrent/temporal action-history trunk;
- goal/skill conditioning;
- multi-head action output;
- termination/anomaly heads.

Training:

- behavior cloning;
- held-out human trajectory evaluation;
- task fine-tuning;
- DAgger-style correction collection.

Exit gate:

- learned policy exceeds heuristic baseline on movement/mining/basic survival;
- 20-30 Hz deadline sustained on minimum supported local profile.

## Phase 6 — Hybrid hierarchical planner

**Goal:** human-like long-horizon planning grounded in actual game dependencies.

- goal/state formalization;
- prerequisite DAG/AND-OR expansion;
- HTN/partial-order plan representation;
- skill binding;
- resource/time/risk cost estimator;
- opportunity insertion;
- local recovery vs global replan policy;
- plan validation against exact game version;
- high-level multimodal model used for intent, ambiguity and novel decomposition, never as sole feasibility authority.

Exit gate:

- complete progression from new world through representative tech milestones without static goal recipes;
- materially fewer full replans than reactive baseline.

## Phase 7 — Memory and lifelong skill learning

**Goal:** improve across sessions rather than repeatedly rediscovering behavior.

- working memory;
- episodic what/where/when store;
- spatial landmark/route graph;
- procedural skill store;
- social/promises store;
- failure/remedy memory;
- relevance retrieval combining causal, spatial, semantic, temporal and goal signals;
- skill proposal/refinement loop;
- held-out promotion evaluation;
- regression tests by game and motor-policy version;
- trajectory distillation into tactical/motor policy.

Exit gate:

- repeated tasks become faster/more reliable without unbounded prompt/context growth;
- skill regressions are detected after version updates.

## Phase 8 — Advancement/achievement curriculum and roles

**Goal:** general progression plus player-selectable specialization.

- versioned official advancement/achievement model;
- derived technology capability graph;
- curriculum scheduler;
- archetype schema and built-ins;
- custom role loader;
- role-weighted standing goals and knowledge domains;
- role-specific evaluation suites.

Built-ins:

- generalist;
- farmer;
- trader;
- builder;
- redstone engineer;
- fighter;
- mob farmer;
- explorer;
- speedrunner;
- boss hunter;
- Nether specialist;
- shopkeeper example.

Exit gate:

- same base agent exhibits measurably different priorities and competent behavior under different role profiles.

## Phase 9 — Social agent and in-game wiki

**Goal:** useful teammate rather than silent automation.

- chat OCR/event ingestion per edition;
- player identity and dialogue state;
- promises/shared projects;
- proactive progress reports;
- task negotiation/delegation;
- version-aware knowledge Q&A;
- source/provenance display;
- interrupt policy so conversation does not freeze motor control.

Exit gate:

- agent can answer exact-version questions, accept a collaborative project, continue playing while inference occurs, and report completion/failure.

## Phase 10 — Motor experts and world model

**Goal:** scale competence without destructive interference.

- task-conditioned expert routing;
- shared trunk + navigation/mining/combat/building/GUI/etc experts;
- contextual expert selection metrics;
- short-horizon learned dynamics predictor;
- prediction-error anomaly signal;
- constrained offline/online RL and self-play improvements.

Exit gate:

- expert system improves heterogeneous task score without degrading established competencies.

## Phase 11 — Cross-platform one-command release

**Goal:** installable by ordinary users.

- `pipx`, standalone binary/installer or equivalent bootstrap path;
- automatic OS/arch/GPU detection;
- model profile selection (`tiny`, `balanced`, `quality`);
- checksum/license-aware downloads;
- Java bridge installer;
- update/migration system;
- diagnostic bundle;
- Windows/Linux/macOS CI;
- x86-64/ARM64 matrix where dependencies allow;
- packaged role/knowledge updates.

Exit gate:

```text
install -> doctor -> run -> pause/resume -> stop
```

works from a clean supported machine without manual source edits.

## Metrics dashboard

Every release tracks:

- frame-to-action latency p50/p95/p99;
- motor Hz deadline miss rate;
- VLM semantic refresh latency;
- task success by capability;
- advancement/achievement coverage;
- tech progression time;
- skill success/context calibration;
- recovery success vs global replans;
- deaths/resources lost;
- wiki version accuracy;
- social task completion;
- memory benefit ablations;
- operator focus-isolation violations (target: zero);
- stop latency and held-input violations (target: zero).

## 8 September 2026 owner priority: continuous native hard-survival learning

See [implementation handoff](CONTINUOUS_NATIVE_HANDOFF_20260908.md). Prioritise the full continuous ERAIS organism, asynchronous processing across time scales, evidence-backed multimodal Fracture integration, hard-survival learning and verified goal progression. This adds acceptance criteria; existing roadmap items remain in force.
