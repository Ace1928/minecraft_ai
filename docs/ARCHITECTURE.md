# Target Architecture

## 1. System contract

Minecraft AI is a layered embodied-agent system. Its central invariant is that slow cognition may decide **what** to do, but only a bounded fast controller decides **how** to move the avatar at motor frequency.

The architecture is deliberately model-agnostic. Local VLMs, language models, learned motor policies and future model families are adapters behind stable protocols.

## 2. Runtime processes

```text
operator
  |
  v
supervisor -------------------------------+
  | heartbeat / capability leases         |
  +--> capture service                    |
  +--> perception service                 |
  +--> cognition service                  |
  +--> knowledge service                  |
  +--> memory service                     |
  +--> skill service                      |
  +--> motor service ---------------------+--> scoped input backend --> Minecraft instance
  +--> social/chat service                |
  +---------------------------------------+
```

The **supervisor is not part of the agent's cognition**. It owns process lifecycle, input capability leases, watchdogs, pause/stop state, and forced key/button release.

### Optional external temporal worker

`PolicyConfig.provider: external` admits a trusted local raw-motion worker through
the existing temporal client. No external implementation or private bundle ships
in this repository. First admission is raw motion only; goal-conditioned and
scene-model replacement require separate qualification.

The configured Python module receives argv, not a shell command. `source_path`
is its import root (for example an immutable checkout's `src` directory), placed
first in `PYTHONPATH`. Pin the source revision and model digest, verify licensing,
and keep mutable learning/resume state out of an initial canary. This is a
trusted plugin boundary, **not an operating-system sandbox or attestation**.

Workers use `minecraft-ai.temporal-policy.v1`: one BGRA shared frame and one
JSONL inference request in flight. The ready record must match architecture,
model digest, model version and `goal_conditioned: false`; the worker must verify
the artifact it actually loads. `reset` has no acknowledgement. A retired reply
must drain through its owning client before shared pixels are reused; it cannot
restore the former skill's permissions, camera delta or target. A complete
matching prediction must arrive inside the parent's absolute deadline, not just
report a short compute duration. Responses are limited to 64 KiB per line.

`transport_pending` distinguishes an outstanding retired reply from an active
option; `retired_responses` and `last_response_age_ms` expose owner-side delivery.
Neither means movement occurred. Admission requires a real client/worker replay
with fresh request/frame/episode attribution, followed separately by a guarded
live trace through the supervisor. Accepted inputs, observed displacement and
successful gameplay remain different evidence gates.

Before an automated agent-only reload, require supervisor status
`agent_reload_resume_supported: true`. After confirmed retirement, use
`resume-for-agent-reload` with that exact `session_id`; it permits only an
unarmed cleanup pause with no remaining agent descriptor or persistent stop/pause
intent. It never grants a new operator resume or retires a faulted supervisor.
Older generations must not be probed after stopping the agent: leave them
running until an authorized supervisor update can install this capability.
Ordinary `resume` is an operator action, not the automated reload substitute.

### Required failure behavior

If any of the following occur, the motor capability lease is revoked and all held inputs are released:

- supervisor heartbeat expires;
- capture target disappears or changes identity;
- input backend loses instance scoping;
- frame timestamps become stale;
- motor policy misses its deadline repeatedly;
- operator pauses/stops the agent;
- cognition requests an unsupported/unsafe motor primitive;
- process crashes or IPC validation fails.

## 3. Three control timescales

### L0: reflex and motor, 20-30 Hz

Deadline-bound local controller. No network call and no general-purpose LLM call is allowed on this path.

Inputs:

- current compact visual representation;
- 0.5-2 seconds of visual/action history;
- current skill/goal embedding;
- target/object hints from the perceptual blackboard;
- health/hunger/hotbar/UI state estimates;
- previous action and predicted motion;
- optional recurrent state.

Outputs:

- movement key state;
- jump/sprint/sneak;
- mouse delta;
- attack/use;
- hotbar selection;
- skill termination probability;
- local failure/anomaly probability.

Target latency: p95 < 35 ms and p99 < 50 ms on supported hardware profiles.

### L1: perception, skill execution and tactical control, 2-10 Hz

Maintains scene tracks, crosshair target, terrain affordances, task-specific perceptual queries, skill progress, recovery policies, and short-horizon navigation.

This layer decides whether a skill should continue, recover, terminate, or escalate. It does not rebuild the strategic plan on every frame.

### L2: cognition, planning, conversation and research, event-driven / 0.1-2 Hz

Handles:

- strategic goal choice;
- hierarchical planning;
- player requests and dialogue;
- wiki/game-knowledge retrieval;
- long-horizon reflection;
- role/archetype policy;
- creation and refinement of skill specifications;
- build design and project decomposition;
- explicit replanning after meaningful state changes.

## 4. Perceptual blackboard

The agent should never repeatedly ask a large VLM to redescribe the whole screen. Perception writes typed, timestamped facts into a blackboard.

Example schema:

```text
FrameState
  frame_id
  timestamp
  target_window
  gui_mode
  player.health
  player.hunger
  player.air
  player.pose_estimate
  selected_slot
  hotbar[]
  crosshair.object
  crosshair.distance
  crosshair.break_progress
  tracks[]
  terrain.walkable_regions
  terrain.drop_edges
  terrain.water/lava
  chat_lines[]
  confidence[]
```

### Perception stack

1. **Fast visual encoder/tracker** at capture rate.
2. **HUD/UI parsers** for stable screen regions.
3. **Object/terrain trackers** for continuity between semantic refreshes.
4. **Task-conditioned VLM** asynchronously answering narrow questions such as `find nearest visible oak log`, `is crosshair on target`, or `what changed after action`.
5. **Semantic reconciler** that merges model observations with tracked state and confidence decay.

All facts carry provenance and age. Planning must not treat stale low-confidence perception as truth.

## 5. Motor policy

The production motor controller should be a goal-observation-action conditioned policy trained from human gameplay before reinforcement learning/self-improvement.

Recommended training sequence:

1. behavior cloning from human video/action pairs;
2. goal relabeling / language conditioning;
3. DAgger-style correction data from agent failures;
4. offline RL on successful/failed trajectories;
5. constrained online improvement in test worlds;
6. optional task-expert routing once sufficient data exists.

### Motor experts

A shared visual/action-history trunk may route into experts such as:

- locomotion/navigation;
- mining/resource gathering;
- combat;
- building/placement;
- inventory/crafting GUI;
- farming/animal handling;
- exploration/parkour;
- Nether/End traversal.

Experts are an optimization, not a public API: skills request capabilities, not expert IDs.

### Human-style requirement

Human-style means the policy is trained on human trajectories and constrained to ordinary gameplay controls. It does **not** mean adding cosmetic random mouse jitter. Competence, smooth corrections, realistic reaction latency and consistent action history should emerge from the behavioral prior.

## 6. Skills: closed-loop options, not macros

A skill is a reusable policy contract:

```text
SkillSpec
  id
  name
  version
  description
  preconditions[]
  parameters{}
  expected_effects[]
  success_condition
  failure_conditions[]
  policy_ref
  max_duration
  recovery_skills[]
  provenance
  evaluation_stats
  context_success_model
```

Examples:

- `approach_visible_target(target)`
- `fell_tree(log_count)`
- `mine_vein(block_family, target_count)`
- `craft_item(item, count)`
- `bridge_to(target)`
- `fight_melee(target_class)`
- `return_to_landmark(name)`
- `build_wall(anchor, dimensions, palette)`

Skills terminate based on observed state changes, not elapsed key sequences.

### Skill creation loop

```text
novel subgoal
  -> retrieve similar skills + game knowledge
  -> high-level controller writes candidate SkillSpec
  -> sandbox executor tries candidate
  -> verifier measures result
  -> critic classifies failure
  -> repair candidate
  -> repeated success across varied contexts
  -> promote to skill library
  -> distill successful trajectories into motor/tactical policy
```

Promotion requires held-out evaluation. A single success never creates a trusted permanent skill.

### Skill lifecycle

`candidate -> experimental -> trusted -> preferred -> deprecated/retired`

Every skill stores successes and failures by context, game version and motor-policy version. Regression after an update automatically lowers confidence.

## 7. Planning

Planning uses a hybrid hierarchical system rather than a static recipe table or a single free-form LLM response.

### Planner inputs

- current world/player state;
- desired goals;
- role/archetype utility weights;
- achievement/advancement state;
- technology tier state;
- versioned dependency graph;
- available skills and their contextual success/cost models;
- episodic/spatial memory;
- promises/shared projects;
- risk/resource budgets.

### Planning sequence

1. high-level cognition proposes desired state/goals;
2. symbolic graph planner expands prerequisites;
3. skill planner binds graph operations to executable skills;
4. temporal/resource estimator scores alternatives;
5. validator rejects impossible/obsolete steps;
6. executor runs until an event invalidates assumptions;
7. local recovery handles routine failure before strategic replanning.

The planner should support HTN/AND-OR/DAG structure, partial-order plans and opportunistic insertion.

## 8. Versioned game dependency graph

Hard-coded `GOAL_RECIPES` must disappear.

Each `GameVersion` is identified by:

```text
edition: java | bedrock
version_id
protocol/data/resource version where available
loader/mod profile if relevant
source manifest hashes
```

The knowledge compiler produces nodes and typed edges for:

- items, blocks, entities, biomes, structures, dimensions;
- crafting, smelting, blasting, smoking, stonecutting, smithing and brewing;
- tags/ingredient alternatives;
- tools/harvest requirements;
- drops/loot sources;
- mob drops and conditions;
- villager/piglin/barter trades;
- structure/biome availability;
- advancement/achievement prerequisites;
- enchantment/equipment constraints;
- portals and dimension transitions;
- optional community-derived strategy edges with separate provenance.

A plan for `obtain diamond pickaxe` is therefore solved against the exact version graph, not authored manually.

### Source priority

1. exact-version first-party/machine-readable game data where legally and technically accessible;
2. well-maintained normalized open datasets with exact version mapping;
3. official documentation;
4. Minecraft Wiki retrieval for explanation/strategy;
5. community strategy sources only when explicitly enabled.

Conflicting facts remain separate records until reconciled; provenance is never discarded.

## 9. Advancement, achievement and technology progression

Progress is multi-axis rather than a single scripted tech tree.

`ProgressState` tracks:

- official advancements/achievements completed;
- material/tool technology capabilities;
- dimensions unlocked/reliably reachable;
- combat/boss milestones;
- agriculture/trading/redstone capability;
- learned skill competency;
- role-specific milestones;
- custom user projects.

The curriculum scheduler balances:

- prerequisite value;
- unexplored official progression;
- role utility;
- skill-learning value;
- current opportunities;
- user-specified goals;
- survival/resource risk.

Official progression is a guide and coverage target, not a command to ignore user goals.

## 10. Roles and archetypes

Roles are declarative overlays rather than bespoke agent code.

```yaml
id: builder
standing_goals:
  - maintain_safe_material_reserve
  - improve_home_base
utility_weights:
  construction: 1.0
  resource_gathering: 0.7
  exploration: 0.35
risk_tolerance: 0.35
preferred_skills:
  - build_wall
  - scaffold
  - palette_gather
knowledge_domains:
  - blocks
  - architecture
  - lighting
```

Built-ins should include farmer, trader, builder, redstone engineer, fighter, mob farmer, explorer, speedrunner, boss hunter, Nether specialist and generalist. Users can compose or create arbitrary roles such as shopkeeper.

## 11. Social/player interaction

Chat is an event stream, not a separate chatbot bolted onto the agent.

Social memory tracks:

- player identity/name;
- requests;
- promises;
- shared projects;
- permissions/ownership conventions;
- recent dialogue;
- known preferences only when learned in-world or explicitly configured.

The dialogue controller can:

- answer version-specific game questions using cited knowledge;
- negotiate/divide tasks;
- report progress;
- ask for clarification when necessary;
- interrupt lower-priority activity for urgent player requests;
- create durable shared goals from promises.

Motor control continues independently while dialogue inference runs.

## 12. In-game wiki

The knowledge service exposes a version-locked retrieval API:

```text
lookup_fact(query, GameVersion)
explain(query, GameVersion)
plan_requirements(goal, GameVersion)
source_trace(fact_id)
```

Responses identify edition/version and source date. If a wiki statement cannot be confidently mapped to the running version, the system says so rather than silently applying current-version information.

Online wiki access is optional. Frequently used exact-version facts are cached locally with source URLs/revision IDs and invalidated when the selected game version changes.

## 13. Memory

Separate stores are maintained for:

- **working memory:** seconds/minutes of current execution;
- **episodic memory:** what happened, where and when;
- **spatial memory:** landmarks, routes, hazards, bases, portals;
- **semantic memory:** versioned game facts and learned world facts;
- **procedural memory:** skills and recovery procedures;
- **social memory:** players/promises/projects;
- **failure memory:** repeated action/context failures and remedies.

Retrieval combines semantic, causal, spatial, temporal, goal and success relevance and includes redundancy suppression.

## 14. World model and anomaly detection

A lightweight short-horizon dynamics predictor may estimate expected next visual/state embeddings from recent state + action. Prediction error is a cheap signal for collision, missed input, target loss, unexpected GUI state, damage, displacement or environmental change.

It is advisory: deterministic safety and observation-based verification remain authoritative.

## 15. Edition adapters

Stable interfaces:

```text
EditionAdapter
  detect_version()
  find_instances()
  capture(instance)
  chat_events(instance)
  knowledge_sources(version)
  input_backends(instance)
```

### Java

Preferred backend: a minimal open-source client-side bridge scoped to a single instance. It receives bounded semantic key/mouse commands over authenticated local IPC and feeds them into the client's normal input path. It exposes no world-state oracle in strict mode.

This keeps the operator's OS mouse/keyboard independent while preserving gameplay-equivalent controls.

### Bedrock

Use platform-specific window capture and the strongest safe isolation available. When true per-instance input is unavailable, run in an isolated desktop/session or require focus. Global desktop injection must be explicitly opted into and visibly warned about.

## 16. Packaging and portability

Python hosts orchestration, knowledge, planning and research services. Performance-critical inference may use ONNX Runtime, llama.cpp-compatible servers, platform GPU runtimes, or separately packaged native components.

Support matrix is capability-based:

```text
OS      arch      edition      capture      scoped-input      inference backend
```

`minecraft-ai doctor` reports capabilities instead of pretending every combination is equivalent.

## 17. Single-command lifecycle

`minecraft-ai install`:

- detects OS/arch/GPU;
- creates isolated environment;
- installs compatible optional components;
- validates model licenses before downloading;
- downloads selected model profile;
- builds/syncs exact-version knowledge;
- installs/configures scoped input bridge where chosen;
- runs safety self-tests.

`minecraft-ai run` starts the supervisor first and grants motor capability only after all gates pass.

`pause`, `resume`, `stop`, `status`, and `logs` talk to the supervisor rather than cognition.

## 18. Evaluation

Every architecture change is evaluated on reproducible suites:

- motor reaction latency and frame-to-action latency;
- navigation success;
- resource acquisition;
- crafting/progression tasks;
- long-horizon advancement coverage;
- skill transfer and recovery;
- building fidelity;
- combat survival;
- player collaboration;
- wiki factual/version accuracy;
- unattended stability;
- stop/pause/focus-isolation safety.

Human-style quality should also be evaluated against human trajectories using action distribution, smoothness, correction behavior and task completion, not visual aesthetics alone.
