# Intelligence Implementation Plan

## Bedrock-first persistent autonomous Minecraft agent — complete implementation specification

This document is the detailed engineering authority for upgrading the existing Bedrock-first runtime into a high-competence, persistent, human-like, continually improving Minecraft agent. It is intentionally implementation-specific: every major research idea is mapped to current repository code, proposed interfaces, persistence, training, runtime integration, tests, and release gates.

The goal is not to imitate one published agent. The target combines the strongest ideas from native human-interface imitation, goal-conditioned visuomotor control, cross-level action routing, hierarchical neuro-symbolic planning, experience-corrected knowledge, multimodal/episodic memory, persistent autonomous goal generation, skill discovery, world-model learning, and adaptive expert consolidation.

Bedrock Edition on Linux through BedrockOnLinux/WineGDK remains the reference runtime. Java support is optional and must never dictate the core design.

---

## 0. Non-negotiable system constraints

1. **Bedrock-first, pixels + human-style input.** Strict mode receives visual/audio/chat/UI observations and sends bounded keyboard/mouse semantics only. No hidden game-state oracle is allowed in the production agent.
2. **Safety remains outside cognition.** `supervisor.py`, `safety.py`, `emergency.py`, and the scoped Bedrock input backend remain independent of planner/model health.
3. **The exact-version graph is authoritative mechanics.** Learned experience may supplement confidence, costs, locations, action validity, and context, but may not silently overwrite exact game facts.
4. **The high-level model never emits raw OS input.** Physical actions remain below typed motor/action interfaces.
5. **Learning is data/model/skill based, not arbitrary self-editing source code.** New skills are typed declarative objects or learned policies and must pass evaluation gates before promotion.
6. **All claims are benchmarked.** Names such as `sota`, `optimal`, `human-like`, `trusted`, and `preferred` must correspond to measurable evidence, not comments or hand-authored heuristics.
7. **No public-boundary contamination.** Remove other-project names, internal endpoints, personal paths, credentials, compiled Python artifacts, and unrelated branding from source and documentation.
8. **No host-global live input fallback.** A backend that cannot prove input isolation must remain unavailable for autonomous live control.

---

## 1. Current repository baseline and immediate blockers

The current architecture has valuable substrate but the intelligence path has recently accumulated temporary shortcuts. These must be corrected before training data is collected, because otherwise those shortcuts become labels and assumptions in the learned system.

### 1.1 Keep

- `safety.py`: motor capability lease and bounded `MotorAction` contract.
- `supervisor.py`: independent lifecycle and motor gate.
- `platforms/bedrock_x11.py`: Bedrock X11 capture/input boundary, provided host-display isolation is enforced.
- `knowledge/model.py`: immutable `GameVersion`, typed exact-version graph, provenance.
- `planning.py`: AND/OR acquisition expansion as a deterministic mechanics layer.
- `perception.py`: typed blackboard/fact/track abstraction and monotonic frame model.
- `skills.py`: declarative skill identity/lifecycle concept.
- `storage.py`: restart-safe SQLite/WAL base.
- `social.py`: durable requests/promises/projects base.
- `memory.py` and `spatial.py`: memory taxonomy and place-memory concepts.
- `agent_process.py`: clean composition point for policy and services.

### 1.2 Correct before advanced work

#### `motor.py`

`DynamicSotaMotorPolicy` is still deterministic handcrafted control: periodic scans, timed jumps, fixed mouse gain, hand-authored combat rhythm, and heuristics. Rename it to `BootstrapMotorPolicy`; remove the `HeuristicMotorPolicy = DynamicSotaMotorPolicy` marketing alias; use it only as a baseline/fallback/data collection policy.

#### `perception_service.py`

`_extract_fast_visual_features()` currently identifies blocks, sky, underwater state, and obstacles with sampled RGB thresholds. Move this into `BootstrapFastPerception`; never use those facts as ground truth labels. Replace it with learned fast perception in Phase 2.

#### `cognition.py`

`AutonomousCognitionEngine` is a hand-authored priority chain rather than hierarchical cognition. It also currently references `Promise.fulfilled` and `Promise.description`, while `social.Promise` exposes `status` and `summary`. Replace the entire class with the executive architecture in this document; retain only an explicit `BootstrapCognitionPolicy` for deterministic smoke tests.

#### `runtime.py` / `skill_editor.py` / `skills.py`

Current runtime skill adaptation references APIs/statistics that do not align with the present classes (`consecutive_failures`, `evaluate_and_promote`, recovery synthesis signature). Consolidate promotion and adaptation into one evidence-driven `SkillLifecycleManager`; do not allow runtime code to manufacture an “adapted” skill merely from two failures.

#### `tech_tree.py`

The new file hard-codes a wood→stone→iron milestone tree, which contradicts the repository's exact-version-graph design. Replace it with `ProgressionModel`, derived from versioned achievements/advancements, item/tool capabilities, exact dependency graph queries, current inventory/world evidence, and role/custom objectives. Hard-coded milestones may remain only as benchmark fixtures.

#### `spatial.py`

`PlaceRecord` currently requires exact XYZ coordinates. Strict visual Bedrock operation must not depend on privileged coordinates. Introduce `PoseBelief` with optional metric pose + covariance and make the topological graph operate without exact XYZ. Exact coordinates may exist only when explicitly supplied by a permitted test/debug adapter.

#### `cli.py` / `bedrock_session.py` / `agent_process.py`

The current CLI defaults `bedrock launch` to a direct host-display mode and `agent_process.py` permits host capture. Reverse this. Managed isolated session is the production default. A direct host display must be debug-only, explicit, and unable to arm autonomous input. `stop_bedrock_session()` must correctly handle all modes without violating its own isolation check.

#### Repository hygiene

Remove committed `__pycache__`/`.pyc`, expand `.gitignore`, remove unrelated identity/demo strings, and remove misleading “OPERATIONAL & OPTIMAL”/“SOTA” demo claims. All CI gates must be green before the model/data work below is treated as a baseline.

### Phase 0 exit gate

- Ruff, strict mypy, tests green on supported matrix.
- no committed generated bytecode;
- no contradictory direct-host live-control path;
- no unresolved type/schema mismatches in runtime cognition/skill handling;
- baseline heuristic code explicitly named `Bootstrap*`;
- exact-version knowledge remains the only production mechanics authority.

---

## 2. Final runtime architecture

The final system uses multiple independently scheduled timescales.

```text
PERSISTENT IDENTITY / VALUES (Sys3)
  ValueProfile + role + autobiographical continuity + long projects
                         |
                         v
MOTIVATION / DRIVE MODEL
  survival | social obligation | competence | curiosity | creation | progress
                         |
                         v
GOAL PORTFOLIO
  focus | background | maintenance | commitment | opportunity | suspended
                         |
                         v
STRATEGIC EXECUTIVE (Sys2)
  candidate strategies -> verify -> cost -> counterfactual -> commit
          |                                  ^
          v                                  |
ExactVersionGraph <---- QueryFusion ---- ExperienceGraph
          |                                  ^
          v                                  |
PERSISTENT PLAN GRAPH ------------------ execution outcomes
  HTN + AND/OR + partial order + contingencies
                         |
                         v
TACTICAL EXECUTIVE (4-10 Hz)
  option stack | local repair | interrupts | place-event retrieval
                         |
                         v
CROSS-LEVEL ACTION ROUTER (Sys1)
  SKILL | GROUNDED | LATENT | MOTION | GUI | RAW
                         |
                         v
TEMPORAL VISUOMOTOR POLICY (20-30 Hz)
  visual/action history + goal + target + HUD + memory + latent world state
                         |
                         v
BOUNDED MotorAction -> supervisor lease -> isolated Bedrock instance

FAST PERCEPTION (20-30 Hz) ----> PerceptualState / Blackboard
ASYNC ACTIVE VLM (event/adaptive) ----^       |
SHORT-HORIZON WORLD MODEL (5-20 Hz) ----------+

EXPERIENCE LOOP
trajectory -> episode -> memory/beliefs -> skill discovery -> train -> eval -> promote
```

### Runtime scheduling target

| Process | Target cadence | Hard/soft deadline |
| --- | ---: | --- |
| capture | 20-30 Hz | hard |
| fast visual encoder | 20-30 Hz | hard |
| raw/latent motor inference | 20-30 Hz | hard |
| tactical option monitor | 4-10 Hz | soft-real-time |
| world-model prediction | 5-20 Hz | soft-real-time |
| active VLM | adaptive 0-5 Hz | asynchronous |
| plan monitor/local repair | event driven; typical 0.2-2 Hz | asynchronous |
| strategic planner/critic | event driven; seconds | asynchronous |
| goal executive | 10-60 s plus interrupts | asynchronous |
| reflection | skill/goal/milestone/session boundaries | asynchronous |
| consolidation/training | offline/low-load | no live deadline |

No slow process may block capture, lease renewal, or motor release.

---

## 3. Phase 1 — instrumentation, trajectories, and evaluation before intelligence changes

### Research basis

VPT shows that native keyboard/mouse competence benefits from large-scale behavioral imitation. Optimus-2 explicitly learns relationships among goals, observations, current actions, and action history. MCU shows why broad task evaluation must exist before “general competence” claims.

### New files

```text
src/minecraft_ai/trajectory.py
src/minecraft_ai/episodes.py
src/minecraft_ai/eval/__init__.py
src/minecraft_ai/eval/tasks.py
src/minecraft_ai/eval/evaluator.py
src/minecraft_ai/eval/metrics.py
src/minecraft_ai/eval/bedrock_worlds.py
src/minecraft_ai/datasets/__init__.py
src/minecraft_ai/datasets/schema.py
src/minecraft_ai/datasets/shards.py
```

### `trajectory.py`

Implement:

```python
class TrajectoryStep(BaseModel):
    trajectory_id: str
    step_index: int
    captured_ns: int
    frame_ref: str
    frame_hash: str
    visual_embedding_ref: str | None
    previous_action: MotorAction | None
    action: MotorAction
    action_level: ActionLevel
    behavior_token: int | None
    skill_run_id: str | None
    skill_id: str | None
    goal_id: str | None
    plan_node_id: str | None
    blackboard_snapshot_ref: str
    place_event_id: str | None
    reward_signals: dict[str, float]
    event_ids: tuple[str, ...]
    correction_of_step: int | None
```

`TrajectoryRecorder` must observe actions at the supervisor-approved motor boundary so recorded actions equal actions actually accepted, not merely proposed actions.

Store video/frame data in append-only sharded files (WebDataset/Parquet + compressed images/video chunks), not SQLite blobs. SQLite stores metadata, shard paths, hashes, episode/goal/skill joins, and indexing.

### Human demonstration mode

Add CLI:

```text
minecraft-ai record-human --role generalist --label freeplay
minecraft-ai record-human --task mine_tree
minecraft-ai record-human --task combat_zombie
```

The human demonstration recorder runs against the isolated Bedrock session and records synchronized frames + host/user input as observed inside that session. It must also capture sensitivity, FOV, resolution, Bedrock version, platform, and launcher profile because those affect the control distribution.

### Dataset source metadata

Every external/imported source has:

```python
DatasetSource:
    source_id
    source_type
    license
    redistribution_allowed
    training_allowed
    edition
    game_versions
    provenance_url
    checksum
```

Do not import VPT/MineStudio/ROCKET/STEVE weights or datasets until their specific license/redistribution/training constraints are recorded.

### Bedrock-native benchmark

Adapt MCU's category idea rather than copying Java assumptions. Define tasks across:

- navigation;
- perception/grounding;
- resource acquisition;
- inventory/GUI;
- crafting/smelting;
- survival;
- combat;
- exploration;
- building;
- memory/return-to-place;
- social/collaborative tasks.

For strict agent evaluation, the agent sees only allowed observations. A **separate evaluator channel** may use controlled test-world instrumentation or post-hoc save/world inspection to score success, because evaluation ground truth is not an agent observation.

### Baseline metrics

Record before replacing heuristics:

- task success;
- median/p95 completion time;
- deaths/damage;
- path inefficiency;
- mouse/key action count;
- stuck rate;
- skill success by context;
- full replan count;
- VLM queries/task;
- frame-to-action p50/p95/p99;
- motor deadline miss rate;
- focus-isolation violations;
- held-input-after-stop violations.

### Phase 1 exit gate

- full Bedrock trajectory can be replay-inspected frame/action aligned;
- ≥20 representative benchmark tasks executable by the harness;
- frozen baseline report produced automatically;
- deterministic fixture/replay tests do not require live Bedrock in CI.

---

## 4. Phase 2 — learned fast perception and dense perceptual state

### Research basis

MP5 demonstrates the value of task-conditioned active perception rather than treating perception as static narration. VistaWise shows that specialized Minecraft visual models can be useful with relatively modest dedicated data. ROCKET-2 shows the value of spatially grounded target representations such as masks/cross-view goals.

### Current integration point

Keep `PerceptionBlackboard`, `FrameState`, `PerceptionFact`, `Track`, and the asynchronous `ActiveVLMWorker`. Replace RGB sampling in `RealtimePerceptionService._extract_fast_visual_features()` with a pluggable learned fast path.

### New interfaces

```python
class FastPercept(BaseModel):
    frame_id: int
    visual_embedding_ref: str
    ego_motion: tuple[float, ...]
    hud: HudState
    hotbar: HotbarState
    gui: GuiState
    tracks: tuple[DenseTrack, ...]
    terrain: TerrainAffordance
    crosshair: CrosshairState
    hazards: tuple[HazardEstimate, ...]
    uncertainty: float

class FastPerceptionModel(Protocol):
    def infer(self, frame: CapturedFrame, history: PerceptionHistory) -> FastPercept: ...
```

Add to `perception.py`:

- `HudState`: health/hunger/armor/air/status effects with uncertainty;
- `HotbarState`: selected slot + per-slot visual embedding/class distribution;
- `GuiState`: world/inventory/crafting/chest/furnace/trade/chat/etc;
- `DenseTrack`: stable track ID, class distribution, mask/box, motion, range proxy;
- `TerrainAffordance`: traversability grid, drop-edge probability, obstacle height estimate, water/lava/hazard probability;
- `CrosshairState`: target track/object/mask, distance proxy, interaction affordance;
- `PerceptionHistory`: compact temporal features, not full serialized frames.

### Model decomposition

Do not force one detector to solve everything. Use a shared small visual encoder plus heads:

1. HUD/hotbar crop heads;
2. GUI-mode classifier;
3. object/block/entity detection or open-vocabulary grounding head;
4. segmentation/target mask head;
5. motion/ego-motion head;
6. traversability/hazard head;
7. event detector from feature deltas.

Use the VLM only when confidence is low or semantic interpretation is novel/goal-dependent.

### Adaptive active perception

Replace fixed `semantic_hz` as the only trigger. Add `PerceptionQueryScheduler` with budget and triggers:

- target lost;
- plan assumption uncertain;
- unknown GUI;
- novel object/structure;
- repeated motor failure;
- world-model surprise;
- player message requiring visual context;
- strategic planner requests one or more named facts.

Each query declares `output_keys` and a staleness deadline. Late answers cannot overwrite newer high-confidence fast facts without reconciliation.

### Training data

Generate labels from:

- user demonstration trajectories;
- VLM pseudo-labels followed by human correction;
- synthetic GUI crops and known item icons where licensing permits;
- temporal consistency/self-supervision;
- action effects (collision, block break, GUI transition) as weak supervision.

### Phase 2 exit gate

- bootstrap RGB heuristics disabled in quality profile;
- 20-30 Hz fast percept on target hardware;
- benchmarked HUD/hotbar/GUI/object/terrain accuracy;
- motor operation continues during VLM outage;
- perception confidence is calibrated enough for plan/skill gating.

---

## 5. Phase 3 — temporal Bedrock visuomotor foundation policy

### Research basis

- VPT: native 20 Hz keyboard/mouse human-interface behavioral prior from large-scale video/action data.
- STEVE-1: efficiently conditions a VPT prior on open-ended text/latent goals using behavioral cloning and hindsight relabeling.
- Optimus-2: explicit Goal-Observation-Action conditioning, action-guided behavior encoding, and fixed-length history behavior tokens.

### `motor.py` target

Retain the public `MotorPolicy` protocol for compatibility but introduce a stateful model-facing interface:

```python
class TemporalMotorObservation(BaseModel):
    frame_id: int
    visual_embedding_ref: str
    history_ref: str
    action_history_ref: str
    skill_embedding_ref: str | None
    subgoal_embedding_ref: str | None
    target_embedding_ref: str | None
    hud_vector: tuple[float, ...]
    hotbar_vector: tuple[float, ...]
    gui_embedding_ref: str | None
    place_context_ref: str | None
    world_latent_ref: str | None

class MotorPrediction(BaseModel):
    keyboard_logits_ref: str
    button_logits_ref: str
    hotbar_logits_ref: str
    mouse_distribution_ref: str
    chunk_length_logits_ref: str
    termination_probability: float
    failure_probability: float
    target_lost_probability: float
    uncertainty: float
```

### Model topology

Initial practical target:

```text
fast image encoder (shared with perception when possible)
        |
action-guided fusion
        |
temporal trunk (Transformer-XL / recurrent transformer / state-space equivalent)
        |
fixed-length behavior-history tokens
        |
+ goal/skill/target conditioning
        |
heads:
  keyboard multi-label
  mouse 2D distribution/codebook
  buttons
  hotbar
  termination
  failure/anomaly
```

Do not require the planner model at 20 Hz.

### Action representation

- keys/buttons: categorical/multi-label state transitions;
- mouse: learned 2D codebook or mixture distribution, not fixed hand-authored gain;
- hotbar: categorical 0-8 + no-change;
- action duration/chunk: learned discrete duration bucket;
- maintain exact held-state semantics so supervisor can release safely.

### Training ladder

1. **Foundation initialization.** Evaluate legally usable VPT/STEVE/ROCKET/MineStudio weights as initialization or teacher models. Prefer transfer/distillation over rebuilding 70k-hour VPT scale.
2. **Bedrock BC.** Train on synchronized Bedrock human trajectories.
3. **Goal relabeling.** Derive short-horizon goal labels from future visual/semantic state; use MineCLIP-like or multimodal embedding similarity.
4. **Goal-conditioned BC.** Condition on text/skill/target embeddings.
5. **DAgger/correction.** Agent acts; operator takeover or corrected segment is recorded as a paired correction.
6. **Failure-correction contrastive objective.** Treat failed and corrected trajectory pairs as explicit training examples.
7. **Offline RL.** Use conservative/advantage-weighted updates on frozen datasets after reliable reward/evaluator signals exist.
8. **Constrained online/self-play.** Only inside dedicated disposable benchmark worlds, with regression gates.

### Losses

```text
L = w_key * BCE(key states)
  + w_mouse * NLL(mouse distribution/code)
  + w_button * BCE(buttons)
  + w_hotbar * CE(slot)
  + w_term * BCE(termination)
  + w_fail * BCE(failure)
  + w_goal * contrastive(goal, behavior)
  + w_bc * behavior cloning
  + optional w_adv * advantage-weighted objective
  + optional w_corr * failure/correction contrastive loss
```

### `policy_service.py`

Add a dedicated inference process so Python cognition cannot stall the 20-30 Hz model.

Use shared-memory frame/embedding rings and a small local control socket/pipe. Expose:

```python
PolicyRequest(frame_slot, history_slot, goal_ref, deadline_ns)
PolicyResponse(prediction, model_version, inference_ns)
```

Deadline miss returns no new action; it must never repeat an unsafe stale action indefinitely.

### Phase 3 exit gate

Learned policy beats `BootstrapMotorPolicy` on frozen navigation, target approach, mining, eating, simple combat, simple GUI, and placement tasks while meeting realtime deadline.

---

## 6. Phase 4 — cross-level action routing instead of one action space

### Research basis

OpenHA found that no one action abstraction is best for all Minecraft tasks. CrossHA trains one model to dynamically choose heterogeneous action spaces and reports state-of-the-art performance across 800+ Minecraft tasks. This becomes a core architecture feature, not a late optimization.

### New file: `action_router.py`

```python
class ActionLevel(StrEnum):
    RAW = "raw"
    MOTION = "motion"
    LATENT = "latent"
    GROUNDED = "grounded"
    GUI = "gui"
    SKILL = "skill"

class RoutedAction(BaseModel):
    level: ActionLevel
    payload: dict[str, ...]
    horizon_steps: int
    expected_termination: str | None
    uncertainty: float

class CrossLevelPolicy(Protocol):
    def route(self, state: TacticalState) -> RoutedAction: ...
```

### Semantics

**RAW** — direct native mouse/keyboard action for reflexes, PvP, ledge/fall recovery, precise aim.

**MOTION** — learned temporally extended chunks such as circle-strafe, sprint-jump traversal, backoff-while-tracking.

**LATENT** — behavior tokens learned from trajectory segments; semantics emerge from data.

**GROUNDED** — target/mask/track-conditioned actions such as `approach(track)`, `mine(mask)`, `interact(region)`.

**GUI** — inventory/crafting/trade cursor grammar.

**SKILL** — semantic `SkillSpec` option with longer horizon and verifier.

### Router training

Cold start labels can be derived from successful trajectories and option segmentation. Train the router with success, completion time, action count, uncertainty, and switching cost. Then use multi-turn policy optimization similar in spirit to CrossHA/GRPO after a safe simulator/evaluator loop exists.

Router reward should include:

```text
+ task/skill progress
+ successful local recovery
+ efficiency
- death/damage/resource loss
- unnecessary high-level calls
- action-level thrashing
- deadline misses
- control uncertainty
```

Use hysteresis/minimum dwell for non-emergency level switching, but always allow immediate drop to RAW for safety reflexes.

### Phase 4 exit gate

Cross-level controller beats each fixed action-space baseline across a heterogeneous Bedrock suite and demonstrates efficient high-level travel plus precise low-level recovery/combat.

---

## 7. Phase 5 — hierarchical skill runtime and automatic skill discovery

### Research basis

Voyager demonstrates the value of an accumulating temporally extended skill library with feedback/self-verification. Open-World Skill Discovery (ICCV 2025) shows an annotation-free Skill Boundary Detection approach using action-prediction error spikes, improving both short-horizon policies and hierarchical agents.

### Upgrade `skills.py`

Replace the simple contract with richer typed models while maintaining migration compatibility:

```python
class SkillEffect(BaseModel):
    predicate: str
    probability: float
    expected_delta: float | None

class SkillCostModel(BaseModel):
    duration_mean_s: float
    duration_std_s: float
    resource_delta: dict[str, float]
    health_delta_mean: float

class FailureMode(BaseModel):
    failure_id: str
    predicate: SkillCondition | None
    probability: float
    recoveries: tuple[str, ...]

class SkillSpec(BaseModel):
    skill_id: str
    version: int
    stage: SkillStage
    parameter_schema: dict[str, ...]
    preconditions: tuple[SkillCondition, ...]
    probabilistic_effects: tuple[SkillEffect, ...]
    success_conditions: tuple[SkillCondition, ...]
    termination_model_ref: str | None
    initiation_model_ref: str | None
    policy_ref: str | None
    behavior_token_ids: tuple[int, ...]
    child_skill_graph: tuple[...]
    cost_model: SkillCostModel
    failure_modes: tuple[FailureMode, ...]
    compatible_game_versions: tuple[str, ...]
    compatible_policy_versions: tuple[str, ...]
```

### Contextual competence

Replace global success counts as the primary selection signal with a model of:

```text
P(success | skill, policy_version, biome/context, equipment, target class, GUI, hazard class)
E(duration | context)
E(resource/health cost | context)
P(failure mode | context)
```

Start with hierarchical Bayesian/Beta-Binomial buckets; later replace/augment with learned competence predictor embeddings.

### `execution.py`

Replace single active skill with `OptionStack`:

```python
ExecutionFrame:
    option_id
    parent_id
    parameters
    started_ns
    expected_effects
    local_plan_node
    allowed_interrupts

SkillExecutor:
    stack: list[ExecutionFrame]
```

Support:

- nested/composed skills;
- learned initiation/termination;
- local interruption and resume;
- local recovery before global replan;
- verifier events;
- action-level downgrade for precision;
- context-key generation automatically from percept/equipment/policy version.

### Automatic discovery: `behavior_tokens.py`

Offline pipeline:

```text
trajectory
 -> unconditional next-action prediction error
 -> boundary candidates
 -> temporal segment embeddings
 -> cluster / VQ codebook
 -> behavior tokens
 -> infer initiation/effects/termination from before/after states
 -> candidate SkillSpec
 -> held-out evaluation
 -> lifecycle promotion
```

Do not automatically promote a discovered skill because it clusters well. Promotion requires generalization over varied contexts and regression comparison against parent/base policy.

### Skill lifecycle

```text
candidate
 -> experimental (offline evidence)
 -> trusted (held-out worlds/context suite)
 -> preferred (beats alternatives with confidence)
 -> deprecated (regression or replaced)
 -> retired
```

A version update, motor-model update, or Bedrock version change can invalidate confidence and require requalification.

### Phase 5 exit gate

At least several useful options are discovered from unsegmented demonstrations, instantiated as typed skills, and outperform equivalent unsegmented/raw execution in long-horizon tasks.

---

## 8. Phase 6 — exact mechanics + experience-corrected world knowledge

### Research basis

XENON (ICLR 2026) shows that LLM prompting alone often fails to correct incorrect planning knowledge. Its Adaptive Dependency Graph and Failure-aware Action Memory algorithmically update dependency/action knowledge from observed successes/failures.

### Keep `knowledge/model.py` authoritative

Do **not** let the LLM or experience alter exact recipe/drop/version facts.

### New file: `experience.py`

```python
class BeliefKind(StrEnum):
    DEPENDENCY = "dependency"
    ACTION_EFFECT = "action_effect"
    FAILURE_PATTERN = "failure_pattern"
    SKILL_COMPETENCE = "skill_competence"
    RESOURCE_LOCATION = "resource_location"
    ROUTE = "route"
    PLAYER_PREFERENCE = "player_preference"

class ExperienceBelief(BaseModel):
    belief_id: str
    kind: BeliefKind
    subject: str
    predicate: str
    object: str | float | int | bool
    context_key: str
    support_count: int
    contradiction_count: int
    confidence: float
    created_ns: int
    updated_ns: int
    provenance_episode_ids: tuple[str, ...]
```

### XENON-style updates

**Dependency observation:** when a target state/item is successfully obtained, record the actual observed prerequisite/resources/actions as an experience dependency. Do not overwrite exact graph; use this to correct planner assumptions about which known method actually worked in the current UI/controller/context.

**Failure-aware action memory:** maintain success/failure evidence for `(subgoal/target, action/skill, context)`. Repeated failures reduce empirical validity; successes restore it. Distinguish controller unreliability from mechanics contradictions using tolerance/confidence.

### Query fusion

New `KnowledgeFusion` service exposes:

```python
mechanics_requirements(target)
known_acquisition_methods(target)
contextual_method_score(method, world_state)
known_resource_locations(resource)
known_failure_patterns(action, context)
explain_plan_assumption(...)
```

The exact graph answers “can this mechanic work in this version?” Experience answers “how well has this method worked here, with this agent, equipment, route, and policy?”

### Replace `tech_tree.py`

Create `progression.py`:

```python
ProgressionState:
    achievements
    capabilities
    material/tool tiers
    equipment readiness
    food/base security
    farming/trading/redstone/enchanting capability
    dimension readiness
    boss readiness
    role capability
    skill competence
```

Derive candidates from exact graph + achievements + current evidence + role/custom projects. Do not maintain a single manually curated linear age tree as production authority.

### Phase 6 exit gate

Injected wrong priors/action preferences are corrected by experience; planner stops repeatedly attempting empirically invalid methods while still respecting exact-version mechanics.

---

## 9. Phase 7 — persistent verified hierarchical planning

### Research basis

Metagent-P supports planning → verification → execution → reflection with neural-symbolic hierarchical representation and reports fewer long-term replans. XENON demonstrates robust planning from corrected knowledge. The existing `DependencyPlanner` already provides useful AND/OR mechanics expansion and should become one component, not the complete planner.

### Extend `planning.py`

Add:

```python
class PlanNodeKind(StrEnum):
    GOAL = "goal"
    METHOD = "method"
    SUBGOAL = "subgoal"
    SKILL = "skill"
    OBSERVE = "observe"
    RESEARCH = "research"
    WAIT = "wait"
    CONTINGENCY = "contingency"

class PlanNodeStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    ACTIVE = "active"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

class PlanHypothesis(BaseModel):
    hypothesis_id: str
    goal_id: str
    nodes: tuple[PlanNode, ...]
    causal_edges: tuple[PlanEdge, ...]
    resource_requirements: dict[str, float]
    expected_time_s: float
    expected_success: float
    expected_risk: float
    expected_resource_cost: float
    uncertainty: float
    expected_information_gain: float
    contingencies: tuple[str, ...]

class PlanRun(BaseModel):
    plan_id: str
    chosen_hypothesis_id: str
    current_nodes: tuple[str, ...]
    completed_nodes: tuple[str, ...]
    blocked_nodes: tuple[str, ...]
    revision: int
```

### Planning sequence

1. **Goal state formalization.** Convert goal to explicit predicates, constraints, success/abort conditions.
2. **Candidate generation.** Produce multiple plausible strategy skeletons using exact graph, experience, spatial memory, available skills, and optional LLM decomposition.
3. **Symbolic validation.** Reject impossible version/mechanics constraints.
4. **Experience validation.** Apply method/route/skill empirical validity and failure evidence.
5. **Costing.** Use contextual competence, travel, equipment, time-of-day, health/hunger, inventory reserves, promises/deadlines, risk, and uncertainty.
6. **Counterfactual/world-model estimate.** When available, simulate uncertain short action/option segments.
7. **Critic pass.** Identify unsupported assumptions and request perception/research before commitment when value of information is high.
8. **Commit persistent plan graph.** Persist it; do not reconstruct from scratch every cognition turn.
9. **Tactical execution.** Activate READY nodes and bind best compatible skills/action levels.
10. **Local repair.** Retry/recover/substitute skill/method locally.
11. **Partial replan.** Reopen only affected graph region when possible.
12. **Global replan.** Only when goal assumptions/feasibility meaningfully change.
13. **Reflection.** Update beliefs, memory, competence, and goal history from outcome.

### Candidate plan objective

Use a configurable utility model, initially:

```text
J(P) = value
     + information_gain
     + competence_gain
     - time_cost
     - risk_cost
     - resource_cost
     - uncertainty_cost
     - goal_switch_cost
```

Every term must be logged so decisions are auditable.

### Tactical executive: new `tactical.py`

Runs 4-10 Hz and owns:

- current PlanRun;
- option stack;
- local interrupt policy;
- route/place retrieval;
- plan-node verification;
- action-level router state;
- short-horizon world-model alerts;
- failure classification.

It should not call the strategic LLM for ordinary target loss/stuck recovery if local recovery exists.

### Phase 7 exit gate

Long-horizon benchmark tasks show materially fewer global replans and lower wasted action counts than the current reactive `CognitionDecision -> skill` loop.

---

## 10. Phase 8 — persistent identity, motivation, and human-like goal portfolios

### Research basis

PEPA proposes a three-system persistent-autonomy architecture: personality-aligned autonomous goal synthesis/reflection, deliberative planning, and sensorimotor execution. The important lesson for this project is not roleplay; it is that internally generated goals require a persistent organizing prior and episodic feedback.

### Replace flat curriculum as primary autonomy mechanism

Keep `curriculum.py` for training/competence-frontier scheduling, not day-to-day “personality.” Add `executive.py`.

### `executive.py`

```python
class ValueProfile(BaseModel):
    survival: float
    helpfulness: float
    exploration: float
    craftsmanship: float
    achievement: float
    efficiency: float
    curiosity: float
    sociality: float
    resource_stewardship: float
    risk_aversion: float
    aesthetic_preference: float

class DriveState(BaseModel):
    safety_pressure: float
    hunger_pressure: float
    equipment_need: float
    social_obligation: float
    unfinished_commitment: float
    competence_drive: float
    exploration_drive: float
    information_need: float
    creation_drive: float
    progression_drive: float

class GoalBucket(StrEnum):
    FOCUS = "focus"
    BACKGROUND = "background"
    MAINTENANCE = "maintenance"
    COMMITMENT = "commitment"
    OPPORTUNITY = "opportunity"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"

class GoalCandidate(BaseModel):
    goal: Goal
    value_links: dict[str, float]
    drive_links: dict[str, float]
    expected_value: float
    expected_duration_s: float
    expected_risk: float
    expected_resource_cost: float
    feasibility: float
    uncertainty: float
    information_gain: float
    competence_gain: float
    social_obligation: float
    commitment_strength: float
    success_conditions: tuple[GoalPredicate, ...]
    abort_conditions: tuple[GoalPredicate, ...]

class GoalPortfolio:
    focus: GoalCandidate | None
    background: ...
    maintenance: ...
    commitments: ...
    opportunities: ...
    suspended: ...
    archived: ...
```

### Goal arbitration

Score candidate goals using transparent components:

```text
value alignment
+ drive pressure
+ commitment strength
+ social obligation
+ progression value
+ competence/frontier value
+ information gain
+ opportunity
- time
- risk
- resource cost
- uncertainty
- switching cost
```

Use **hysteresis**: a candidate must exceed current focus by a configurable margin to switch, except hard interrupts.

Hard/priority interrupts:

- immediate danger;
- critical hunger/health/air;
- high-priority player request/commitment;
- deadline;
- goal infeasible;
- major rare opportunity above threshold.

This prevents goal thrashing and creates human-like project persistence.

### `roles.py`

Convert roles from direct standing-goal lists into presets over `ValueProfile`, reserve policy, risk preference, knowledge preference, social style, and skill affinity. Standing goals may seed portfolio candidates but must not be the entire autonomous goal system.

### Goal synthesis

The LLM may propose novel candidates (“improve the western farm”, “build a safer Nether route”), but candidates are converted into typed goal state and must pass feasibility/value/constraint checks. The LLM never directly sets unbounded priority.

### Phase 8 exit gate

In long unattended runs, the same base agent exhibits stable role-consistent behavior, maintains multi-session projects, temporarily handles maintenance/interrupts, and resumes prior work without constant external instructions.

---

## 11. Phase 9 — multimodal, tactical, spatial, and autobiographical memory

### Research basis

MrSteve shows that low-level execution benefits from Place Event Memory rather than giving all memory only to the planner. M2PA supports multiple memory systems for long-horizon/lifelong behavior. PEPA supports episodic reflection for persistent autonomy.

### Upgrade `memory.py`

Keep `MemoryKind`, but add typed indices/links rather than relying primarily on text + tag scoring.

```python
MemoryLink:
    source_id
    target_id
    relation
    confidence

EpisodeMemory:
    episode_id
    temporal bounds
    goal/plan/skill links
    event summary
    outcome
    embedding_ref

ProceduralMemory:
    skill/policy references
    context competence
    corrections

AutobiographicalEntry:
    narrative_id
    time range
    project/relationship links
    salience
    summary
    follow_up_goal_ids
```

### Place Event Memory

Upgrade `spatial.py`:

```python
class PoseBelief(BaseModel):
    metric_xyz: tuple[float, float, float] | None
    covariance: tuple[float, ...] | None
    heading: float | None
    dimension: str
    source: str  # visual_odometry, dead_reckoning, explicit_debug, etc.

class PlaceEvent(BaseModel):
    place_event_id: str
    visual_embedding_ref: str
    semantic_embedding_ref: str | None
    pose: PoseBelief
    topological_place_id: str
    observed_entities/resources: tuple[str, ...]
    event_types: tuple[str, ...]
    observed_ns: int
    goal_ids: tuple[str, ...]
    outcome: str | None
```

Topological routing must work on connectivity and learned transition cost even when metric XYZ is unknown.

### Visual odometry / dead reckoning

Estimate relative motion from visual features + commanded mouse/keys + collision evidence. Periodically loop-close using landmark embeddings. Metric coordinates are optional; topological identity is primary.

### Retrieval

Use hybrid retrieval:

```text
semantic similarity
+ visual similarity
+ causal/action similarity
+ spatial/topological proximity
+ goal overlap
+ entity overlap
+ temporal recency
+ outcome relevance
+ importance
- contradiction penalty
```

Do not send all retrieved memory to the LLM. `TacticalMemoryRetriever` supplies compact route/place/skill facts directly to the tactical controller.

### Memory conflict management

Store contradictory evidence rather than blindly averaging embeddings. Reconcile by context/version/time and confidence. Never let a stale high-similarity memory silently replace a newer contradictory world fact.

### Autobiographical consolidation

At session/day/project boundaries:

```text
episodes -> salient events -> project summary -> belief updates -> identity continuity -> follow-up goals
```

This is for continuity and planning, not cosmetic roleplay.

### Phase 9 exit gate

Memory ablation demonstrates better resource return, route reuse, repeated-task speed, project resumption, and lower rediscovery cost with memory enabled.

---

## 12. Phase 10 — split cognition into executive, planner, critic, and reflector

### Current issue

`HighLevelController.decide()` currently asks one model call to choose goal, skill, speech, perception, research, and replan. Replace the monolithic transaction.

### New `cognition.py` service architecture

```python
GoalExecutive
StrategicPlanner
PlanCritic
Reflector
SocialReasoner
ResearchReasoner
```

They may initially share the same local model endpoint/model weights, but have separate typed inputs/outputs and event triggers.

### `GoalExecutive`

Produces/updates `GoalCandidate`s and GoalPortfolio decisions. It receives values/drives, commitments, autobiographical summary, progression gaps, opportunities, and current focus — not raw frame spam.

### `StrategicPlanner`

Produces strategy hypotheses and tool/query requests. It calls deterministic planner/knowledge/spatial APIs rather than inventing recipes.

### `PlanCritic`

Checks:

- unsupported assumptions;
- exact-version conflicts;
- missing resources/tools;
- high-risk steps;
- low-confidence skills;
- stale place/resource memories;
- missing perception/research whose information value exceeds cost.

### `Reflector`

Triggered after meaningful episodes. Outputs typed proposed updates:

```python
Reflection:
    prediction_errors
    belief_update_candidates
    failure_classification
    memory_salience
    skill_discovery_candidate_ids
    goal_value_update
    follow_up_goal_candidates
```

A deterministic validator applies permissible changes. Reflector does not mutate exact game facts or source code.

### Phase 10 exit gate

High-level model-call count drops for routine action while strategic task success improves; plan critic catches seeded invalid assumptions; reflection produces useful belief/skill/memory updates without corrupting exact mechanics.

---

## 13. Phase 11 — event bus and true multi-rate scheduler

### New file: `events.py`

```python
class EventType(StrEnum):
    BLOCK_BROKEN = ...
    RESOURCE_OBTAINED = ...
    DAMAGE_TAKEN = ...
    PLAYER_MESSAGE = ...
    GOAL_COMPLETED = ...
    GOAL_BLOCKED = ...
    TARGET_LOST = ...
    SKILL_FAILURE = ...
    PLAN_INVALIDATED = ...
    NOVEL_PLACE = ...
    WORLD_MODEL_SURPRISE = ...
    PROMISE_DUE = ...
    GUI_CHANGED = ...
    INVENTORY_CHANGED = ...
    DEATH = ...
```

Every event has priority, source, timestamp, frame/episode links, and payload schema.

### Refactor `runtime.py`

Replace one 20 Hz method that also polls cognition with coordinated services:

```text
RealtimeLoop
  capture -> fast perception -> world model -> tactical -> policy -> action

AsyncEventLoop
  semantic perception
  plan monitor
  cognition processors
  social
  research
  reflection

OfflineLearningLoop
  dataset finalization
  skill discovery
  training/evaluation/consolidation
```

Use bounded queues with latest-state semantics for frames and explicit durable events for significant transitions. Do not queue unbounded screenshots or stale model requests.

### Phase 11 exit gate

Motor/capture timing remains stable under intentionally slow/failed VLM/planner services, and cognition is primarily event-triggered rather than blind fixed polling.

---

## 14. Phase 12 — short-horizon predictive world model and later imagination

### Research basis

Dreamer 4 demonstrates that a scalable action-conditioned world model can learn Minecraft mechanics from mostly unlabeled video and train a diamond-obtaining agent entirely from offline imagination. Reproducing full Dreamer 4 immediately is not required; start with a tactical predictive model that becomes progressively more capable.

### New file: `world_model.py`

V1:

```python
WorldLatent
WorldPrediction:
    next_latent
    collision_probability
    block_break_probability
    target_loss_probability
    gui_transition_probability
    health_delta_distribution
    inventory_change_probability
    uncertainty
```

Input:

```text
recent visual latent sequence + action sequence + current target/skill context
```

### V1 losses

- latent next-state prediction;
- action-conditioned contrastive future representation;
- event prediction;
- inverse action prediction as representation regularizer;
- uncertainty/calibration.

### Tactical use

Prediction error emits `WORLD_MODEL_SURPRISE`, increasing perception attention and triggering failure/belief review when appropriate.

### Growth path

1. one-step/250 ms event prediction;
2. 1 s latent rollout;
3. 5-10 s option-conditioned rollout;
4. score candidate motion/skill chunks;
5. offline imagination for policy improvement;
6. only later attempt larger Dreamer-style long-horizon imagination training.

### Phase 12 exit gate

World-model surprise predicts meaningful control failures/novel transitions better than naive frame-difference baseline and improves local recovery or candidate action selection in ablation.

---

## 15. Phase 13 — adaptive specialist experts and parametric internalization

### Research basis

- Optimus-3 uses task-level MoE to reduce interference across heterogeneous Minecraft tasks.
- MoEC (ACL 2026) provides a stronger direct controller blueprint: subgoal-indexed non-parametric expert memory, context-conditioned routing, failure-triggered expert growth, and redundancy-aware consolidation with lightweight expert heads.
- PEAM (2026) internalizes selected failure→correction experiences into physically isolated multimodal MoE-LoRA adapters and uses a parameterization-worthiness mechanism to decide what experience should become parametric memory.

### New file: `experts.py`

```python
class ExpertDescriptor(BaseModel):
    expert_id: str
    family: str
    subgoal_keys: tuple[str, ...]
    context_centroid_ref: str
    policy_head_ref: str
    model_version: str
    competence_summary: dict[str, float]
    parameter_count: int
    created_from_episode_ids: tuple[str, ...]
    active: bool

class ExpertRegistry:
    experts
    router_memory
    redundancy_graph
```

### Architecture

Use a frozen/shared visual-temporal backbone initially, with small experts for:

- navigation/traversal;
- mining/gathering;
- combat;
- building;
- GUI/crafting;
- exploration/search;
- later role/rare-context specialists.

Routing is two-stage:

1. hard/semantic subgoal compatibility;
2. context similarity + competence + uncertainty + cost.

### Expert growth

On repeated failure after verifying that the strategic plan/method is valid:

1. classify failure as controller/context mismatch;
2. collect correction/success trajectories;
3. train candidate lightweight expert/head/LoRA on frozen backbone;
4. evaluate against shared/base + nearby experts;
5. add only if statistically beneficial.

### Redundancy consolidation

Periodically compute behavioral/representation/competence similarity between experts. Merge/prune experts whose behavior and context coverage are redundant while protecting specialists with unique competence.

### PEAM-style parameterization worthiness

Not every memory becomes weights. Score an experience/skill by:

```text
reuse frequency
+ retrieval cost saved
+ stable success evidence
+ context recurrence
+ correction information
- rarity/noise
- regression risk
- parameter cost
```

Only high-value repeated skills are internalized.

### Phase 13 exit gate

Expert system improves heterogeneous task score without degrading a frozen regression suite; failure-triggered growth creates useful specialists; consolidation reduces redundant experts without measurable regression.

---

## 16. Phase 14 — building/spatial creativity subsystem

### Research basis

MineAnyBuild exposes the gap between visual spatial understanding and executable architecture planning, with 4,000 spatial planning tasks across spatial understanding, reasoning, creativity, and commonsense.

### New package: `building/`

```text
building/spec.py
building/style.py
building/planner.py
building/materials.py
building/construction.py
building/verifier.py
```

### `BuildSpec`

```python
BuildSpec:
    project_id
    semantic_description
    constraints
    dimensions/scale
    palette_preferences
    style_embedding_ref
    functional_requirements
    reference_image_refs
```

### Planning stages

1. inspect existing site/style;
2. synthesize architectural concept;
3. represent structure as relational/voxel region graph;
4. validate support/access/clearance constraints;
5. estimate materials through exact graph;
6. create material acquisition subplans;
7. create scaffold/access plan;
8. topologically order build regions;
9. execute via grounded/place skills;
10. visual verification after each region;
11. repair deviations;
12. update project memory and player report.

Do not require exact privileged coordinates; support local visual anchors and topological site frames.

### Phase 14 exit gate

Agent can construct and repair novel multi-stage builds from semantic instructions, not merely replay block coordinates, and is evaluated on a Bedrock-native spatial planning/build suite inspired by MineAnyBuild.

---

## 17. Phase 15 — social/team intelligence integrated into planning

### Upgrade `social.py`

```python
class PlayerModel(BaseModel):
    player_id: str
    display_name: str
    inferred_preferences: dict[str, float]
    known_projects: tuple[str, ...]
    likely_current_goal: str | None
    shared_knowledge_refs: tuple[str, ...]
    trust/confidence fields

class SharedMentalModel(BaseModel):
    project_id: str
    shared_goal: str
    task_allocations: dict[str, str]
    known_resources: dict[str, int]
    assumptions: tuple[SharedAssumption, ...]
    progress: dict[str, float]
```

### Message pipeline

`ChatLine` -> `ChatEvent` -> intent/parser -> request/promise/project/team update -> GoalPortfolio/PlanGraph.

Examples:

- “finish the roof” creates/updates a project goal, not just a chat response;
- “I'll get spruce, you do stone” updates shared task allocation and prevents redundant resource gathering;
- “stop doing that” is a high-priority social interrupt with project-plan consequences.

### Proactive communication policy

Communicate when:

- accepting/rejecting/clarifying a commitment;
- major project milestone completed;
- blocked waiting on player/resource;
- plan materially changes;
- task completed/failed;
- rare useful discovery relevant to a shared project.

Avoid narrating every action.

### Phase 15 exit gate

Collaborative benchmark demonstrates durable request→promise→plan→execution→progress report lifecycle and useful task division without freezing motor control during dialogue inference.

---

## 18. Storage migration and durable schemas

Bump `SCHEMA_VERSION` with explicit migrations rather than refusing all older schemas.

Add tables/indexes for:

```text
trajectories
trajectory_shards
trajectory_steps_index
episodes
events
place_events
spatial_places
spatial_edges
experience_beliefs
belief_evidence
goal_candidates
goal_history
goal_portfolio_state
plan_runs
plan_nodes
plan_edges
plan_revisions
reflections
skill_versions
skill_context_stats
behavior_tokens
expert_registry
expert_evaluations
model_versions
benchmark_runs
benchmark_task_results
project_narratives
player_models
shared_mental_models
```

Large tensors/frames/video/models live outside SQLite with content hashes and atomic manifests.

Every learned artifact stores:

```text
artifact/version id
git commit
training dataset manifest hashes
game versions
model base/checkpoint
hyperparameters
metrics
created timestamp
license/provenance
```

This is required for reproducible promotion/rollback.

---

## 19. Configuration and CLI changes

### `config.py`

Add structured profiles:

```python
PerceptionConfig
MotorPolicyConfig
ActionRouterConfig
WorldModelConfig
ExecutiveConfig
MemoryConfig
TrainingConfig
EvaluationConfig
ExpertConfig
```

Support hardware profiles:

```text
tiny
balanced
quality
research
```

Capabilities determine which models load; do not pretend all hardware supports the same 30 Hz model.

### CLI

Add:

```text
minecraft-ai record-human
minecraft-ai dataset inspect
minecraft-ai dataset validate
minecraft-ai eval run
minecraft-ai eval compare
minecraft-ai policy train
minecraft-ai policy evaluate
minecraft-ai policy promote
minecraft-ai policy rollback
minecraft-ai skills discover
minecraft-ai skills evaluate
minecraft-ai skills list
minecraft-ai beliefs inspect
minecraft-ai beliefs explain
minecraft-ai goals list
minecraft-ai goals history
minecraft-ai plans inspect
minecraft-ai memory inspect
minecraft-ai experts list
minecraft-ai experts consolidate
minecraft-ai world-model train
minecraft-ai benchmark report
```

All promotion commands require evaluation evidence. Live runtime auto-loads only `promoted` artifacts.

---

## 20. CI, replay tests, and hardware qualification

### CI categories

1. schema/type/unit tests;
2. deterministic trajectory replay tests;
3. planner property tests on exact-version graph fixtures;
4. experience correction tests with seeded false priors;
5. memory retrieval/conflict tests;
6. goal portfolio/hysteresis tests;
7. option stack/local repair tests;
8. model artifact manifest validation;
9. offline tiny-model smoke inference;
10. public-boundary scan;
11. no generated artifacts committed;
12. existing safety/fault tests.

### Hardware qualification

On the actual Bedrock Linux machine:

- isolated session retains input when operator changes host focus;
- operator types/mouses elsewhere with zero agent leakage;
- 20/30 Hz frame/action deadlines;
- VLM and high-level model can stall without motor deadline failure;
- agent process kill releases controls;
- policy service crash releases/halts safely;
- suspend/resume;
- Bedrock/Wine crash;
- session/window replacement;
- emergency stop while keys/buttons held.

The learned policy cannot be marked production-capable until these pass again with the policy service inserted.

---

## 21. Evaluation hierarchy and promotion rules

Use layered frozen suites so improvements cannot hide regressions.

### Level A — primitive control

Aim/turn, move, jump, sprint, swim, edge recovery, target following, cursor movement.

### Level B — atomic skills

Mine, collect, eat, place, open/use, basic combat, select item, inventory movement, crafting.

### Level C — compositional tasks

Gather+craft, travel+retrieve, cave mining, shelter, farming, trading, basic build.

### Level D — long horizon

Iron/diamond progression, Nether preparation, village project, multi-resource build, boss preparation.

### Level E — memory/autonomy/social

Return to remembered resource, resume multi-session project, maintain stock/food, fulfill promise, divide work, self-generate useful goal.

### Level F — lifelong learning

Compare hour 1 vs later checkpoints on both new and previously mastered tasks. Require improvement on adaptation suite without statistically meaningful regression on frozen suite.

### Promotion rule

No model/skill/expert becomes preferred from a single aggregate score. Require:

- minimum sample count;
- confidence interval or bootstrap significance;
- no critical safety regression;
- no unacceptable drop on protected competencies;
- latency/memory budget;
- artifact provenance complete.

---

## 22. Exact implementation order

This order is dependency-driven. Do not jump directly to world-model/MoE work before data/eval and a competent shared motor prior exist.

### Milestone M0 — clean and freeze baseline

Files: `cli.py`, `motor.py`, `perception_service.py`, `cognition.py`, `runtime.py`, `skill_editor.py`, `tech_tree.py`, `spatial.py`, `.gitignore`, CI.

Deliverable: green, honestly named, Bedrock-isolated baseline.

### M1 — trajectory/evaluation substrate

Files: new trajectory/dataset/eval packages + storage migration + CLI.

Deliverable: synchronized Bedrock dataset + frozen benchmark baseline.

### M2 — fast learned perception

Files: `perception.py`, `perception_service.py`, model service support.

Deliverable: 20-30 Hz dense FastPercept; heuristic RGB path only as tiny/debug fallback.

### M3 — learned temporal motor V1

Files: `motor.py`, `policy_service.py`, `training/motor/*`, `agent_process.py`.

Deliverable: learned Bedrock policy beats bootstrap on Level A/B.

### M4 — cross-level action routing

Files: `action_router.py`, `behavior_tokens.py`, `execution.py`.

Deliverable: adaptive RAW/MOTION/LATENT/GROUNDED/GUI/SKILL routing beats fixed action-space variants.

### M5 — hierarchical skill system + discovery

Files: `skills.py`, `skill_editor.py` replaced by lifecycle manager, `execution.py`, discovery training/eval.

Deliverable: discovered options generalize and promote under evidence gates.

### M6 — ExperienceGraph and version-derived progression

Files: `experience.py`, `progression.py`, `planning.py`, `knowledge/*`, storage.

Deliverable: experience corrects action/dependency assumptions; no production hard-coded tech tree.

### M7 — persistent plan graph and tactical executive

Files: `planning.py`, `tactical.py`, `events.py`, `runtime.py`.

Deliverable: persistent multi-step plans, local repair, partial replan.

### M8 — persistent autonomous executive

Files: `executive.py`, `roles.py`, `curriculum.py`, `cognition.py`, storage.

Deliverable: values/drives/goal portfolio/hysteresis/commitments/autobiographical continuity.

### M9 — memory and topological world continuity

Files: `memory.py`, `spatial.py`, `tactical.py`, storage.

Deliverable: Place Event Memory directly improves tactical execution and navigation.

### M10 — metacognitive split and event-driven cognition

Files: `cognition.py`, `events.py`, `runtime.py`.

Deliverable: GoalExecutive/Planner/Critic/Reflector; slow reasoning no longer polled as one generic decision.

### M11 — world model

Files: `world_model.py`, training pipeline, tactical integration.

Deliverable: calibrated surprise + useful short-horizon action/option prediction.

### M12 — adaptive experts / parametric memory

Files: `experts.py`, policy service, training/consolidation pipeline.

Deliverable: MoEC/PEAM-inspired specialist growth and redundancy consolidation with protected regression suite.

### M13 — building intelligence

Files: `building/*`.

Deliverable: semantic multi-stage novel construction and repair.

### M14 — full social/team intelligence

Files: `social.py`, `executive.py`, `planning.py`, cognition/social reasoner.

Deliverable: shared projects, task allocation, persistent player models, proactive but restrained communication.

### M15 — lifelong closed loop

Deliverable:

```text
PLAY
 -> record
 -> detect events/failures/surprise
 -> segment episodes/skills
 -> update memory + ExperienceGraph
 -> create correction pairs
 -> offline train
 -> benchmark
 -> promote/rollback
 -> consolidate repeated competence into experts
 -> PLAY BETTER
```

A key acceptance target is that a long-running agent measurably improves from accumulated experience while frozen previously mastered tasks do not regress.

---

## 23. Research-to-code mapping

### VPT — OpenAI, 2022

Use: human-native keyboard/mouse behavioral prior, 20 Hz action framing, IDM/video-pretraining philosophy.

Code: `trajectory.py`, `motor.py`, `training/motor/`.

Reference: https://openai.com/index/vpt/

### STEVE-1

Use: low-cost text/latent goal conditioning of a VPT-style behavior prior; hindsight relabeling.

Code: goal relabeling + temporal motor policy conditioning.

Reference: https://arxiv.org/abs/2306.00937

### Optimus-2 — CVPR 2025

Use: Goal-Observation-Action policy, action-guided behavior encoder, compressed historical behavior tokens.

Code: `TemporalMotorObservation`, temporal trunk, behavior history, MGOA-style dataset schema.

Reference: https://openaccess.thecvf.com/content/CVPR2025/html/Li_Optimus-2_Multimodal_Minecraft_Agent_with_Goal-Observation-Action_Conditioned_Policy_CVPR_2025_paper.html

### OpenHA / CrossHA — 2025/2026

Use: heterogeneous action spaces; no single abstraction is universally optimal; learned dynamic action-level switching.

Code: `action_router.py`, `CrossLevelPolicy`.

References:
- https://arxiv.org/abs/2509.13347
- https://openaccess.thecvf.com/content/CVPR2026/html/He_Training_One_Model_to_Master_Cross-Level_Agentic_Actions_via_Reinforcement_CVPR_2026_paper.html

### ROCKET-2 — 2025

Use: spatially grounded visual goals/masks and efficient visuomotor target conditioning.

Code: target embeddings/masks in fast perception and grounded action level.

Reference: https://arxiv.org/abs/2503.02505

### Open-World Skill Discovery — ICCV 2025

Use: prediction-error-based Skill Boundary Detection from unsegmented video.

Code: `behavior_tokens.py`, offline skill discovery.

Reference: https://openaccess.thecvf.com/content/ICCV2025/html/Deng_Open-World_Skill_Discovery_from_Unsegmented_Demonstration_Videos_ICCV_2025_paper.html

### XENON — ICLR 2026

Use: Adaptive Dependency Graph and Failure-aware Action Memory; algorithmic correction from success/failure rather than LLM-only reflection.

Code: `experience.py`, KnowledgeFusion, planner method validity.

Reference: https://openreview.net/pdf/e1cf5c6b8fb15055b0a9a794be234b32494b693f.pdf

### Metagent-P — ACL Findings 2025

Use: planning-verification-execution-reflection, neural-symbolic validation, metacognitive monitoring/local correction.

Code: persistent plan graph, critic, reflector.

Reference: https://aclanthology.org/2025.findings-acl.1169/

### M2PA — ACL Findings 2025

Use: multi-memory planning and lifelong experience integration.

Code: memory separation and hybrid retrieval.

Reference: https://aclanthology.org/2025.findings-acl.1191/

### MrSteve — ICLR 2025

Use: Place Event Memory supplied to low-level exploration/execution, not planner-only memory.

Code: `spatial.py`, tactical memory retrieval.

Reference: https://openreview.net/pdf/9812fe85d096855db342c62bcb1b1cd0c42e6bd5.pdf

### PEPA — 2026

Use: persistent autonomous values/personality → internally generated goals → reflection across Sys3/Sys2/Sys1 timescales.

Code: `executive.py`, role/value presets, autobiographical consolidation.

Reference: https://arxiv.org/abs/2603.00117

### Dreamer 4 — 2025

Use: scalable action-conditioned world-model learning and later offline imagination; start with tactical predictive model.

Code: `world_model.py`.

Reference: https://arxiv.org/abs/2509.24527

### Optimus-3 — 2025/2026

Use: task-level expert routing to reduce heterogeneous-task interference.

Code: expert families/shared trunk.

Reference: https://arxiv.org/abs/2506.10357

### MoEC — ACL 2026

Use: subgoal-indexed non-parametric expert memory, context routing, failure-triggered expert growth, redundancy-aware consolidation.

Code: `experts.py`, ExpertRegistry, expert growth/consolidation.

Reference: https://aclanthology.org/2026.acl-long.1027.pdf

### PEAM — 2026

Use: selective parameterization of repeated experience, failure-correction contrastive internalization, isolated expert adapters.

Code: parameterization-worthiness and expert training/consolidation.

Reference: https://arxiv.org/abs/2605.27762

### MP5 — CVPR 2024

Use: goal-conditioned active perception and modular perception/planning/execution communication.

Code: adaptive `PerceptionQueryScheduler`.

Reference: https://openaccess.thecvf.com/content/CVPR2024/html/Qin_MP5_A_Multi-modal_Open-ended_Embodied_System_in_Minecraft_via_Active_CVPR_2024_paper.html

### VistaWise — EMNLP 2025

Use: low-cost domain detector training and cross-modal domain knowledge as evidence that Minecraft-specific fast perception need not require frontier-scale data.

Code: fast detector/grounding training.

Reference: https://aclanthology.org/2025.emnlp-main.1111/

### MCU — ICML 2025

Use: broad atomic/compositional evaluation, task diversity, general evaluator philosophy.

Code: `eval/` and benchmark promotion gates.

Reference: https://proceedings.mlr.press/v267/zheng25j.html

### MineAnyBuild — NeurIPS 2025

Use: building/spatial-planning evaluation dimensions and executable architecture-plan framing.

Code: `building/`.

Reference: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8048e2b9b631b15e1158f36275d8fc11-Abstract-Datasets_and_Benchmarks_Track.html

### MineStudio

Use: research integration/evaluation reference and candidate legally usable VPT/STEVE/GROOT/ROCKET checkpoints. Do not add as a hard runtime dependency without license/environment review.

Reference: https://github.com/CraftJarvis/MineStudio

---

## 24. Definition of “complete”

The intelligence upgrade is complete only when all of the following are true:

- Bedrock isolation/safety hardware gates pass;
- fast perception is learned, calibrated, and realtime;
- learned temporal motor control beats the bootstrap controller;
- cross-level action routing beats fixed action-space baselines;
- skills are closed-loop, contextual, compositional, discoverable, and evidence-promoted;
- exact-version mechanics are preserved and experience beliefs are separately corrected;
- persistent plan graphs execute with local repair and partial replanning;
- values/drives/goal portfolio sustain coherent multi-session autonomous projects;
- tactical Place Event Memory improves actual execution;
- autobiographical/project continuity survives restarts;
- strategic cognition is split into executive/planner/critic/reflector roles;
- world-model prediction produces measurable tactical benefit;
- specialist experts grow from repeated failure/correction and redundant experts consolidate;
- building handles novel semantic multi-stage construction;
- social requests/promises/shared plans alter real planning and execution;
- offline learning improves competence without catastrophic regression;
- all promoted artifacts are reproducible, versioned, benchmarked, and rollback-safe;
- the agent is measurably better after accumulated experience than at initialization while retaining previously mastered competencies.

That is the target system: not a scripted bot with an LLM attached, but a Bedrock-native persistent agent whose fast control, memory, plans, goals, skills, and experts form one measurable learning system.