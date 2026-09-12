# Minecraft AI

A self-contained, open-source project for building a persistent Minecraft agent that plays through human-style visual perception and key/mouse controls while combining fast low-level motor control with slower multimodal cognition, planning, memory, social interaction, and continual skill learning.

> Status: active development with isolated live Bedrock execution. The supervisor, scoped-input backend, visual observation, asynchronous cognition, skill execution and trajectory recording are implemented. Live startup remains subject to safety/readiness checks; reliable survival and resource progression are the next gameplay milestones.

## Built foundation and direction

The running system separates fast physical control from slower planning and perception. Typed skills, observed outcomes, persistent memories and contextual recovery statistics let work span many control ticks. Recent upgrades add model-independent action restrictions, verified traversal-stall recovery, adaptive mining aim and matching-skill plan completion.

The destination is a persistent player that can follow directions, learn useful skills, manage survival, build and collaborate with people. Those are progression goals, not guarantees about today's player. See the [execution roadmap](docs/ROADMAP.md) for demonstrated behavior and the [intelligence implementation plan](docs/INTELLIGENCE_IMPLEMENTATION_PLAN.md) for the wider design. Rates in the target architecture below are design targets, not measured live throughput.

Custom native models and learning systems connect through optional motor, cognition and lifecycle interfaces. Their implementations, model assets and deployment configuration can remain outside this open foundation. Installing an adapter and demonstrating better gameplay are separate steps.

## Reference platform

The primary/default runtime is **Minecraft Bedrock Edition for Windows running on Linux through BedrockOnLinux/WineGDK**. The project should be optimized, tested, documented, and packaged for that environment first.

Java Edition remains an optional compatibility target. It must not determine default CLI behavior, safety assumptions, knowledge defaults, capture design, or input architecture.

## Design goals

- **Bedrock-first:** Bedrock Edition under Linux/WineGDK is the reference runtime and default edition.
- **Human-style play:** raw visual observations plus key/mouse action semantics; no privileged world-state commands in strict mode.
- **Two-speed intelligence:** real-time perception and a fast local motor policy underneath slower multimodal planning, reflection, conversation, and knowledge retrieval.
- **Continual skills:** create, verify, refine, compose, score, distill, and retire reusable closed-loop skills.
- **Version-aware knowledge:** derive recipes, loot, tags, achievements, item/block data, and progression dependencies for the exact Bedrock version; augment this with cited wiki retrieval.
- **Progression-aware autonomy:** reason over achievements, technology tiers, custom goals, builds, exploration, and role-specific standing goals.
- **Player interaction:** treat chat, requests, promises, shared projects, and in-game questions as first-class events.
- **Archetypes:** configurable roles such as farmer, trader, builder, redstone engineer, fighter, mob farmer, explorer, speedrunner, boss hunter, Nether specialist, shopkeeper, or custom roles.
- **One-command operation:** install/doctor/run/pause/resume/stop/status from one CLI with automatic model/runtime setup where licensing and platform permissions allow it.
- **Concurrent desktop use:** the operator must be able to use other Linux applications while the Bedrock agent plays whenever the selected backend passes isolation tests.
- **Scoped control:** headless Weston with a verified virtual seat keeps ordinary desktop input outside the Bedrock input path. Host-fed nested sessions cannot arm autonomous input; there is no automatic host-global fallback.
- **Fail-safe stopping:** a separate supervisor, heartbeat watchdog, key-release on failure, and operator-owned stop paths that the agent cannot intercept.

## Target architecture

```text
BedrockOnLinux / Minecraft.Windows.exe
          |
          | video + audio + chat/UI
          v
+----------------------------+
| real-time perception       |  10-30 Hz tracking/encoding
| + semantic blackboard      |  async VLM semantic refresh
+-------------+--------------+
              |
      +-------+--------+
      |                |
      v                v
+-----------+    +------------------+
| fast      |    | high-level       |
| motor     |    | multimodal       |
| policy    |    | cognition        |
| 20-30 Hz  |    | plan/social/wiki |
+-----+-----+    +---------+--------+
      ^                    |
      |              goal / skill intent
      |                    v
      |          +--------------------+
      +----------| skill executor     |
                 | verifier / learner |
                 +---------+----------+
                           |
                 progression + memory
                           |
                 versioned game graph
```

The high-level controller does **not** emit individual key presses. It chooses goals, plans, asks targeted perceptual questions, interacts with players, and selects/refines skills. A much smaller local policy executes those skills with rapid visual feedback.

## Planned CLI

```bash
minecraft-ai install
minecraft-ai doctor
minecraft-ai run --role builder
minecraft-ai status
minecraft-ai pause
minecraft-ai resume
minecraft-ai stop
minecraft-ai knowledge sync --version <bedrock-version>
minecraft-ai wiki "How do I make a crafter?"
```

`minecraft-ai stop` must work even when the cognition/model processes are unhealthy. Each live backend remains subject to independent stopping and scoped-input checks.

## Edition support

The core remains edition-neutral, but defaults are intentionally not neutral.

- **Bedrock Edition — default/reference target:** Linux host running the Windows Bedrock client through BedrockOnLinux/WineGDK. Runtime discovery tracks the BedrockOnLinux data root, managed Wine prefix, and `Minecraft.Windows.exe` process identity. Capture/input work should first target this environment and preserve host-desktop independence.
- **Java Edition — optional compatibility target:** may use a minimal client-side bridge for per-instance control. Java-specific code must remain behind an adapter boundary and must not be required for ordinary Bedrock operation.

### Bedrock Linux control strategy

Preferred order:

1. identify the exact Bedrock process/session and Wine prefix;
2. capture only that game surface/session;
3. inject gameplay-equivalent input inside the isolated Bedrock/Wine execution boundary where technically reliable;
4. use a dedicated headless compositor/session with a verified virtual seat, excluding host input from the game as well as game input from the host;
5. refuse autonomous actuation when that isolation cannot be established; there is no host-global input fallback.

The release criterion is behavioral: while the agent is actively moving, looking, mining, fighting, or using inventory UI, the operator must be able to type and use the host desktop without receiving agent input.

## Knowledge and progression

Every running instance resolves to an immutable `GameVersion` identity. The Bedrock knowledge pipeline builds a provenance-carrying graph from exact-version machine-readable data where available, then adds secondary normalized data and wiki explanations.

Core edge classes include crafting/cooking/smithing/brewing dependencies, loot/drop sources, tool requirements, biome/dimension/structure availability, trades, achievement prerequisites, and observed world-specific routes/resources.

Plans are generated from this graph instead of a hard-coded recipe list.

## Roles

Roles change utility weights, standing goals, curriculum, planning horizon, resource reserves, risk tolerance, preferred skills, and social behavior. They do not replace the general planner.

A custom shopkeeper, for example, can prioritize stock acquisition, safe storage, pricing/trade interactions, shop maintenance, customer chat, and replenishment while retaining ordinary survival competence.

## Safety model

The control plane is deliberately separate from cognition. The agent never owns its own emergency-stop mechanism.

See [docs/SAFETY.md](docs/SAFETY.md) before implementing or enabling live input.

## Research direction

The design combines ideas validated across Minecraft-agent research: human video pretraining, goal-conditioned low-level policies, hierarchical planning, active perception, episodic what/where/when memory, skill libraries, task-specialized experts, reflection/recovery, and dependency-aware planning. The project is a new implementation rather than a vendored copy of those systems.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md), and [docs/RESEARCH_BASELINE.md](docs/RESEARCH_BASELINE.md).

## Clean public boundary

This repository is intentionally standalone. Do not add private model adapters, private service endpoints, personal filesystem paths, internal company material, credentials, or code copied from projects whose redistribution terms have not been reviewed.

Third-party model/code integrations must be optional and must record their license and source in the integration documentation.

### Optional request-aware cognition

Custom planners can implement `BoundCognitionModel` in `models.py` without placing
their implementation or model assets in this repository. Its three optional hooks
are `complete_bound_constrained`, `admit_bound_decision`, and
`discard_bound_request`; existing `LanguageModel` adapters remain supported.
All three hooks must be callable to select request-bound mode. An adapter with
only part of that optional interface continues through its existing legacy path.

The request-aware path binds inference to an immutable semantic observation and
operator/execution revisions. Before accepting a decision, the runtime rechecks
its deadline, current authority and skill prerequisites. Expired or superseded
work cannot authorize action. Admission/discard hooks must be short metadata-only
operations: no inference, input or inference-lane acquisition. Inference itself
owns one `local_model_inference_lane()` acquisition in its worker.

This interface does not select, load or activate a custom model. Deployment,
learning state and proprietary adapter details remain external; compatibility
tests do not establish gameplay competence.

### Optional local runtime factory

A trusted, separately installed planner can opt into the canonical agent process
through local configuration only: `runtime_factory.reference` is a
`module:callable`, and `runtime_factory.startup_timeout_s` bounds construction
renewals (default 120 seconds). The default remains the existing runtime; no
custom module is imported unless configured.

The callable receives `runtime_kwargs` and a sticky `cancel_event`, returning an
`AgentRuntime`. This first integration permits replacing `high_level` only;
preserve the supplied executor/policy, camera state, perception, database, lease
and trajectory ownership. Construction must not start inference or game input.
It renews only an already-valid lease, checks pause/emergency cancellation and
joins its renewal worker before normal runtime startup. It cannot revive an
expired lease or cover the earlier process/database/capture assembly gap.

The result must supply `close_constructed_runtime() -> bool`: drain private
owners, then call `close_before_run(timeout_s=remaining_budget)`. Neither cleanup
path may change supervisor state. Return exactly `True` only after cleanup;
otherwise borrowed resources and the database remain retained until process
exit. A protected ownership mismatch takes that same retention path without
trusting the candidate's cleanup hook to close displaced original resources.
A factory whose partial construction cannot safely close must raise
`RuntimeStartupCleanupIncomplete` from `minecraft_ai.runtime_factory` instead
of an ordinary exception. Normal runtime shutdown remains its own responsibility.
Import, native constructors and arbitrary close callbacks are cooperatively
cancelled, not forcibly preempted. This is a trusted extension boundary, not a
sandbox or a claim of live model activation.

Runtime subclasses may also override `allow_fresh_capture(observation)` as a
bounded, observation-only continuation veto. `CaptureObservation` retains the
exact capture, its separate blackboard frame identity, and only the fast facts
produced synchronously from those pixels; it excludes older merged semantics.
The default returns `True`. Any other result or ordinary exception stops and
releases through existing cleanup before downstream work; a `True` result cannot
bypass normal safety or input ownership. The hook runs before initial cognition
and motor warmup, and after fresh-frame/release checks on later ticks. Startup
may reuse an existing capture; an adapter must validate its age and provenance.
It must
not perform inference, inputs, resets or resource mutation. This supports a
local bounded pilot's stop condition, not automatic world-memory reset or a
claim that the supplied observations are correct training labels.

For a bounded motor pilot, `PolicyConfig.max_worker_starts=1` permits one worker
spawn attempt for that client, including a failed spawn. Ordinary option resets
and `close()` do not renew this budget; `None` retains normal restart behavior.
Status reports the attempt count, configured limit, verified startup and process
liveness. This limits process starts, not inference time or memory consumption.

### Skill lifecycle observations

Runtime subclasses can implement `on_skill_run_started` and
`on_skill_run_terminal` to connect their own experience handling. Both are
no-op by default and run synchronously on the runtime thread: overrides must
only handle or queue metadata, without blocking, IO, inference or game inputs.
Callback exceptions are logged by type and do not interrupt normal execution.

The start hook receives a detached `SkillRun`, `SkillStartSource`, optional
`SkillDecisionOrigin` and optional `parent_run_id`, only after a new run actually
exists. A direct accepted model decision carries its exact `RequestBinding`,
selected attempt, source decision hash and final admitted decision hash.
Deterministic fallbacks (including after failed model work), legacy unbound
decisions and recovery/continuation runs have no model origin. Parent IDs do not
inherit model credit. A later same-action admission does not relabel a running
skill or emit another start.

The terminal hook receives a detached terminal run and only matching,
runtime-filtered `OutcomeVerification`, or `None`. It also reports unmatched
runs, cancellation and timeout; it does not require a database. Duplicate
suppression uses the existing 4096-run-ID window, not durable exactly-once
delivery. Consumers own run-ID joins, replay handling and any learning policy;
never associate a delayed outcome using the latest admission. A terminal record
does not prove that the supervisor accepted game input. These hooks neither
activate online learning nor change action authority.

## License

Apache License 2.0. Minecraft is a trademark and intellectual property of Microsoft/Mojang. This project is independent and is not affiliated with, endorsed by, or sponsored by Microsoft or Mojang.
