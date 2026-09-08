# Continuous native organism: implementation handoff

Owner priority, 8 September 2026. Additive to ROADMAP and INTELLIGENCE_IMPLEMENTATION_PLAN; this is requested work, not a completed capability claim.

## Outcome and immediate sequence

Prioritise a persistent ERAIS-native organism that plays and learns Minecraft in **Hard survival**, with real hunger, hostile mobs, death, recovery and meaningful long-horizon progression. Preserve Bedrock-first isolated pixels/audio and bounded input. Hard difficulty is not Hardcore/permanent world deletion. Do not reset an existing valuable world without the owner's agreement; use an isolated evaluation world where necessary.

1. Inventory the actual live path and current checkpoints. Trace observation → perception → persistent organism state → goals/planning → motor action → observed outcome → learning/checkpoint. Name any teacher, external model, heuristic or fallback. A wrapper around a teacher is not native inference.
2. Connect the full continuous ERAIS organism stack end to end; upgrade ERAIS where the integration reveals a genuine missing capability. Coordinate shared ERAIS/Fracture changes with their active owner. Preserve independent safety and input isolation.
3. Implement asynchronous processing with different time scales: deadline-bound motor/reflex control; perception updates; interruptible goal/skill planning; episodic consolidation and learning. Profile actual hardware before choosing rates. A slow planner or trainer must not stall the motor loop.
4. Prove native learning and goal progression in hard survival before promoting the live stream or paid gameplay claims. Retain an explicit development label while gates remain unmet.

## Asynchronous runtime acceptance

Use bounded queues, observation timestamps and age limits, cancellation/generation IDs, versioned immutable state snapshots, resource budgets and backpressure. Discard stale plans/actions after death, respawn, world changes or goal preemption. Do not replay a stale backlog of motor commands. Training should publish validated model versions atomically with rollback, never mutate active inference weights unsafely. Record starvation, queue depth, deadlines missed and end-to-end latency. Inject delayed/crashed workers and demonstrate safe release of held inputs and recovery.

Hunger, imminent danger and death must preempt lower-priority work. Longer goals survive appropriate interruptions but are revalidated against visible inventory/world evidence. Progress requires observed effects, not issuance of actions. Track stuck/repeated-action loops, failed preconditions and replanning decisions.

## Fracture and learning priorities

Inspect existing Fracture conversion, motor learning, temporal state and multimodal integration before adding another parallel stack. Assess sparse native conversion of licensed existing perception/action/planning systems across modalities. Preserve temporal information and modality alignment. Record donor/data licences, provenance, conversion losses, active versus total parameters, memory, latency and task quality. Compare with the unconverted teacher and current native baseline at matched resource budgets; evaluate held-out worlds/seeds. Do not call an external-model fallback a successful native conversion.

Research leads, checked 8 September 2026:
- [VPT paper and implementation](https://github.com/openai/Video-Pre-Training): behavioural-prior/data bootstrap; independently verify applicable weight/data licences and edition/action mapping.
- [STEVE-1](https://arxiv.org/abs/2306.00937): goal-conditioned short-horizon behaviour, not proof of persistent survival competence.
- [DreamerV3](https://danijar.com/project/dreamerv3/): learned world models and imagined policy improvement; assess compute and domain assumptions before adopting.
- Existing RESEARCH_BASELINE covers curriculum, active perception and hierarchical skills. Refresh primary sources and compare against current implementations.

Bio-inspired hypotheses worth testing include complementary fast episodic/slow consolidated memory, event-triggered attention, homeostatic priorities and sparse modular routing. These are established broad ideas, **not novelty claims**. Document the specific proposed mechanism, closest prior work, implementation differences, falsifiable hypothesis and ablations. Keep potentially protectable implementation disclosures in the private business research record until reviewed. Evidence of an engineering improvement does not establish patentability or freedom to operate.

## Hard-survival evidence and stream

Record game edition/version, difficulty, seed/world identity, checkpoint hashes, active native components and fallback use. Run multiple held-out episodes plus a continuous soak. Report survival time, deaths and causes, hunger management, food acquisition, shelter/night survival, combat/avoidance, resource recovery, progression milestones, repeated-failure rate, skill retention after restart and compute/latency. Compare learning-on/off and asynchronous versus prior scheduling under equivalent conditions. Include unsuccessful runs and resource costs.

The viewer should show actual current goal, recent verified milestones, interruptions, deaths/recovery, active learning status and honest model provenance. Derive these from runtime events; never invent narration of success. Stream stability is not evidence of learning.

## Handoff receipt

Repository fast-forwarded from 84aa89a to 2823fd0 on 8 September. Existing uncommitted lifecycle, motor, skills and cognition-daemon work was preserved. This handoff does not certify those changes or start/reconfigure a live world. Next owner: reconcile this sequence with the current implementation, record the first failing acceptance gate, fix it and publish reproducible evidence before advancing.
