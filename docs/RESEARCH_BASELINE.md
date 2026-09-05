# Research Baseline

This document records the architectural ideas we are intentionally building on. It is a design reference, not a claim that Minecraft AI reproduces any paper's implementation.

## VPT — human behavioral prior

Video PreTraining demonstrated that a Minecraft policy trained from large-scale human gameplay can learn broad native-interface behavior and, with fine-tuning, reach long-horizon progression such as diamond tools. The key design lesson is to bootstrap the low-level controller from human trajectories rather than expect sparse-reward RL to discover competent movement from scratch.

Minecraft AI adopts:

- native key/mouse action semantics;
- behavior cloning as motor-policy initialization;
- temporal action/observation history;
- task fine-tuning after a broad behavioral prior.

## Voyager — automatic curriculum and reusable skills

Voyager demonstrated the value of an automatic curriculum, a growing reusable skill library, and iterative improvement from execution feedback/self-verification.

Minecraft AI differs by representing trusted skills as typed closed-loop options rather than arbitrary generated code. Skills have explicit preconditions/effects, verification, context-specific outcome statistics and lifecycle gates.

## MP5 — active perception

MP5 uses goal-conditioned active perception within a modular multimodal agent. This supports our decision not to ask a VLM for an unconstrained full-frame description at every motor tick.

Minecraft AI adopts:

- task-conditioned perceptual questions;
- modular scheduling;
- context-aware execution;
- a persistent semantic blackboard between expensive VLM updates.

## Optimus-2 — high-level planning + GOA-conditioned low-level control

Optimus-2 explicitly separates an MLLM planner from a Goal-Observation-Action conditioned action policy and models historical observation-action behavior. This closely matches the two-speed architecture required here.

Minecraft AI adopts:

- high-level multimodal planning separated from motor control;
- goal/skill conditioning in the motor policy;
- observation-action history compression/recurrent state;
- motor learning from aligned goal-observation-action trajectories.

## Optimus-3 — task-level experts

Optimus-3 addresses interference among heterogeneous Minecraft tasks with task-level mixture-of-experts routing.

Minecraft AI treats task experts as a later scaling phase after a competent shared policy exists. Expert IDs remain internal; the rest of the architecture depends only on skill/capability protocols.

## Metagent-P — neuro-symbolic planning and metacognitive verification

Metagent-P combines hierarchical symbolic structure with planning, verification, execution and reflection. This supports a planner that uses generative models for intent/novel decomposition while symbolic game/version constraints remain the feasibility authority.

Minecraft AI adopts:

- hierarchical/neuro-symbolic plan structure;
- pre-execution feasibility checks;
- monitoring and selective replanning;
- explicit reflection after meaningful failure rather than after every motor tick.

## Versioned data and knowledge

The project should not encode a hand-maintained Minecraft tech tree. Exact-version machine-readable data should be compiled whenever possible. For Java, the vanilla data generator can produce the vanilla data pack plus registry/block/command reports. Versioned normalized datasets such as PrismarineJS `minecraft-data` are useful secondary sources/cross-checks and cover both Java and Bedrock versions. Bedrock public vanilla sample/behavior data can be used where its license and version mapping permit.

Wiki retrieval is an explanatory/strategy source, not the sole authority for recipes or progression dependencies. Every retrieved fact must retain edition, version applicability, source and revision/timestamp metadata.

## Architecture synthesis

The intended synthesis is:

```text
VPT-style human behavioral prior
          +
Optimus-style goal-conditioned motor policy
          +
MP5-style active perception
          +
Voyager-style lifelong curriculum/skills
          +
Metagent-style verified hierarchical planning
          +
versioned dependency graph
          +
persistent episodic/spatial/social memory
          +
optional task-level experts at scale
```

The combination matters more than any individual component: a powerful planner with weak motor control is ineffective, a powerful motor policy without memory/planning is shortsighted, and either system without exact-version knowledge eventually produces invalid plans as Minecraft evolves.

## Evaluation principle

No paper result is accepted as a project result. Every borrowed idea must beat the local baseline in our own reproducible evaluation harness before becoming the preferred implementation.

## September 2026 update: adaptation after input qualification

The immediate prerequisite remains measured input delivery and verified gameplay
outcomes. None of the following research establishes competence on our Bedrock
session or justifies training on actions whose physical delivery is unknown.

- **Preference Goal Tuning (PGT):** the [May 2026 paper revision](https://arxiv.org/abs/2412.02125v2)
  adapts a per-task latent goal from trajectory preferences while keeping the
  policy weights fixed. This is a candidate for bounded task adaptation without
  repeatedly fine-tuning the entire motor policy. The [released implementation](https://github.com/CraftJarvis/PGT)
  targets GROOT and the Java MCP-Reborn environment with GPU rollouts; it is not
  a drop-in Bedrock or custom-native adapter. Any local trial needs a compatible
  conditioning interface, reliable outcome labels and held-out trajectories.
- **CrossHA:** the [CVPR 2026 project](https://craftjarvis.github.io/CrossHA/)
  learns to select between motion, grounding, raw, language and latent action
  formats. This supports evaluating action abstraction as part of a policy,
  while our supervisor still owns permissions and input delivery. It does not
  establish that our current routing or custom models reproduce those results.
- **ROCKET-3:** the [research](https://craftjarvis.github.io/ROCKET-3/)
  combines cross-view goals, imitation pretraining and KL-constrained multi-task
  reinforcement learning. Its published training setup uses roughly 100,000
  tasks and 72 parallel Minecraft instances, not a plan for this desktop.
  As checked on 2026-09-06, the [official repository](https://github.com/CraftJarvis/ROCKET-3)
  contains a README and demonstration GIF with code release still promised;
  the project's Hugging Face link leads to its paper, not a released checkpoint.
  Do not treat it as an immediately installable controller upgrade.

Practical order: qualify isolated camera and held-input response, retain fresh
goal/action/outcome trajectories, verify wood acquisition and crafting, then
compare one bounded adaptation against the unchanged policy. A smaller replay
loss alone is not evidence of better survival, movement or continual learning.
