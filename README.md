# Minecraft AI

A self-contained, open-source project for building a persistent Minecraft agent that plays through human-style visual perception and key/mouse controls while combining fast low-level motor control with slower multimodal cognition, planning, memory, social interaction, and continual skill learning.

> Status: architecture/bootstrap phase. Live control is intentionally disabled until the independent safety supervisor and scoped-input backends pass their release gates.

## Design goals

- **Human-style play:** raw visual observations plus key/mouse action semantics; no privileged world-state commands in strict mode.
- **Two-speed intelligence:** real-time perception and a fast local motor policy underneath slower multimodal planning, reflection, conversation, and knowledge retrieval.
- **Continual skills:** create, verify, refine, compose, score, distill, and retire reusable closed-loop skills.
- **Version-aware knowledge:** derive recipes, loot, tags, advancements/achievements, item/block data, and progression dependencies for the exact game edition/version; augment this with cited wiki retrieval.
- **Progression-aware autonomy:** reason over achievements/advancements, technology tiers, custom goals, builds, exploration, and role-specific standing goals.
- **Player interaction:** treat chat, requests, promises, shared projects, and in-game questions as first-class events.
- **Archetypes:** configurable roles such as farmer, trader, builder, redstone engineer, fighter, mob farmer, explorer, speedrunner, boss hunter, Nether specialist, shopkeeper, or custom roles.
- **One-command operation:** install/doctor/run/pause/resume/stop/status from one CLI with automatic model/runtime setup where licensing and platform permissions allow it.
- **Cross-platform core:** Windows, Linux, and macOS; x86-64 and ARM64 where Minecraft and the selected model runtime are available.
- **Scoped control:** prefer per-instance input so the agent cannot steal the operator's desktop. Global input is a compatibility fallback, never the preferred backend.
- **Fail-safe stopping:** a separate supervisor, heartbeat watchdog, key-release on failure, and operator-owned stop paths that the agent cannot intercept.

## Target architecture

```text
Minecraft video/audio/chat
          |
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
minecraft-ai knowledge sync
minecraft-ai wiki "How do I make a crafter?"
```

`minecraft-ai stop` must work even when the cognition/model processes are unhealthy. Live motor control will not be enabled until that invariant is tested on each supported backend.

## Edition support

The core is edition-neutral. Edition adapters provide version detection, authoritative data extraction, window/capture behavior, chat integration, and input backends.

- **Java Edition:** first-class target. Preferred concurrent-desktop control is a minimal client-side bridge that injects the same key/mouse semantics into one game instance without taking global OS focus. Strict mode still derives gameplay decisions from human-visible observations.
- **Bedrock Edition:** supported through version-aware data and platform adapters. Scoped background input is platform-dependent; isolated-session/focused control is used where safe per-instance injection is unavailable.

## Knowledge and progression

Every running instance resolves to an immutable `GameVersion` identity. The knowledge pipeline builds a provenance-carrying graph from exact-version machine-readable game data where available, then adds secondary normalized data and wiki explanations.

Core edge classes include crafting/cooking/smithing/brewing dependencies, loot/drop sources, tool requirements, biome/dimension/structure availability, trades, advancement/achievement prerequisites, and observed world-specific routes/resources.

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

MIT. Minecraft is a trademark and intellectual property of Microsoft/Mojang. This project is independent and is not affiliated with, endorsed by, or sponsored by Microsoft or Mojang.