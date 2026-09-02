# Minecraft AI

A self-contained, open-source project for building a persistent Minecraft agent that plays through human-style visual perception and key/mouse controls while combining fast low-level motor control with slower multimodal cognition, planning, memory, social interaction, and continual skill learning.

> Status: architecture/bootstrap phase. Live control is intentionally disabled until the independent safety supervisor and scoped-input backends pass their release gates.

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
- **Scoped control:** prefer control isolated to the Bedrock Wine/container/session boundary. Host-global input is a compatibility fallback, never the preferred backend.
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

`minecraft-ai stop` must work even when the cognition/model processes are unhealthy. Live motor control will not be enabled until that invariant is tested on each supported backend.

## Edition support

The core remains edition-neutral, but defaults are intentionally not neutral.

- **Bedrock Edition — default/reference target:** Linux host running the Windows Bedrock client through BedrockOnLinux/WineGDK. Runtime discovery tracks the BedrockOnLinux data root, managed Wine prefix, and `Minecraft.Windows.exe` process identity. Capture/input work should first target this environment and preserve host-desktop independence.
- **Java Edition — optional compatibility target:** may use a minimal client-side bridge for per-instance control. Java-specific code must remain behind an adapter boundary and must not be required for ordinary Bedrock operation.

### Bedrock Linux control strategy

Preferred order:

1. identify the exact Bedrock process/session and Wine prefix;
2. capture only that game surface/session;
3. inject gameplay-equivalent input inside the isolated Bedrock/Wine execution boundary where technically reliable;
4. otherwise use a dedicated nested compositor/session whose input cannot leak to the host desktop;
5. use host-global desktop injection only with explicit opt-in and a persistent warning.

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

## License

Apache License 2.0. Minecraft is a trademark and intellectual property of Microsoft/Mojang. This project is independent and is not affiliated with, endorsed by, or sponsored by Microsoft or Mojang.
