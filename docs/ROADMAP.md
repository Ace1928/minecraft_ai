# Implementation Roadmap

The roadmap is ordered by dependency and release gates. Later phases must not bypass earlier safety and observability requirements.

## Current Bedrock execution checkpoint — 2026-09-05

This is the immediate gameplay critical path, not a claim that whole phases below are complete.

- [x] Guarded private-LAN operator viewing and commands; game input remains isolated.
- [x] Repair local hostname advertisement collisions without restarting the game; verify the dashboard, live frame, message history, and readiness over the physical LAN interface. Access from a second physical device remains unverified.
- [x] Deterministic inventory open/close: live transitions verified without restarting Minecraft.
- [x] Agent-only reload: live game, GPU server, input calibration and supervisor preserved.
- [x] Exact traversal-stall recovery transaction implemented and tested. Live reorientation/query ran, but the initial VLM answer abstained; escape is not yet demonstrated.
- [x] Version-pinned oak hotbar counts 0–16 and pre-break-to-post-collection exact +1 verification implemented. Hotbar evidence is not a whole-inventory count; unknown items, including spruce logs, abstain.
- [x] Recognize all nine selected hotbar positions using the pinned vanilla selection-frame asset. Raw live replay recovers the previously rejected fourth-slot geometry; unknown spruce remains unknown to the oak-only counter.
- [x] Turn an empty initial replan into canonical visual prerequisites when an explicit requested skill or failed run identifies what to observe. This closes one no-op path; live perception-to-action completion remains unverified.
- [x] Permit exact vanilla leaf blocks in guarded hand-safe clearance, using the existing 2.5-second maximum verification lease. This is not a leaf-drop or live escape claim; stone/tool and gather-log restrictions remain unchanged.
- [x] Admit tiny natural-lighting drift only with a tighter structural match: keep crosshair dHash ≤4/RGB error ≤1, additionally permit dHash ≤2/RGB error ≤2. A stationary 20.57-second raw pair measured dHash 2/RGB 1.833 with unchanged camera updates and no held inputs; existing colour-swap and dHash-3/RGB-2 rejection tests remain intact.
- [x] After a headroom query abstains, retain the existing traversal escalation latch until cognition can decide. Re-enabling disposable walking immediately was repeatedly invalidating every slow planner answer on the next obstacle failure. Explicit operator and safety routes remain independent.
- [x] Treat entirely unsupported perception questions like empty ones when a requested/failed skill supplies canonical prerequisites. Live `23a2f38` finally consumed planner answers, but invented `scene.horizon_visible` and `scene.nearby_objects_types` never started a query; regression cases now cover those names.
- [x] Failed crafting cleanup is plan-neutral; a verified no-logs failure persists a retry prerequisite until new positive wood evidence or one fresh explicit retry. This is permission to re-audit, not a retained inventory count.
- [x] Experimental compact crosshair-query format fits the existing output budget, preserves crop provenance, and bounds numeric output. It is not promoted to live use: pinned daylight/dark-dirt probes still produced wrong block identities. Recognition accuracy and autonomous escape remain separate live gates below.
- [x] Make the focused crosshair probe produce useful, correctly grounded live answers within the small model's output budget. On the live 1920x1054 dirt-overhang frame, the strict two-field probe returned `dirt` at 0.90 confidence in 36.96s; source/current crop dHash matched and RGB-grid drift was 0.318 against the 1.0 admission limit.
- [ ] Demonstrate autonomous soft-block clearance and escape from the observed dirt pocket.
- [ ] Demonstrate three consecutive verified log acquisitions without manual intervention or game restart; extend item recognition only from calibrated evidence if the local tree variant requires it.
- [ ] Demonstrate a verified log-to-planks inventory transformation, then use that run to select the next progression upgrade.

Retain failed trajectories as failures. Do not promote a successful block break into resource possession, or test coverage into a completed live gameplay milestone.

Latest live observation (`5c258c1`): the agent-only reload preserved Minecraft,
the supervisor, isolated input, and GPU server. The player remains beneath a
leaf canopy; mining acquisition times out and traversal stalls. Two focused
answers were rejected with crosshair RGB drift 1.013 and 1.456 against the 1.0
limit after about 20 seconds. No successful decision-owned perception-to-action
cycle was observed. A visible six-log stack predates the recorded accepted
attacks and is not counted as an autonomous acquisition. The next execution
gate remains actual guarded block clearance and movement, not activity counts.

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
