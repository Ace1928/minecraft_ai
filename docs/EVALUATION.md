# Evidence-gated evaluation

`minecraft-ai eval tasks` prints the frozen Bedrock baseline contract.
`eval run --trajectory <directory> --task <task-id> [--evidence <json>]`
scores a recorded trajectory; `benchmark report` aggregates task-tagged recordings.
Reports are written as JSON and persisted in the state database after validation.

## Independent outcome evidence

Evidence is an evaluator-only channel, never an agent observation. For example:

```json
{
  "source": "controlled-world:movement-range/trial-1",
  "metrics": {"event.destination_reached": 1},
  "artifact_refs": ["fixture://movement-range/trial-1"]
}
```

The `trace.*`, `action.*`, `camera.*`, `latency.*`, and `safety.*` namespaces
(including the bare namespace names) belong to computed trajectory measurements.
External evidence cannot supply them, even if a particular trace has no duration
or latency measurement. Exact collisions with any existing metric are also
rejected. `event.*`, `reward.*`, and other independently named outcome metrics
remain supported; numeric evidence must be finite. Invalid evidence produces an
explicit error, not a scored report with substituted measurements.

An independent destination event plus zero recorded forward presses fails
`a_move_forward`. Missing outcome evidence remains `unscored`; emitting movement
alone does not prove physical translation. Evidence source labels and artifact
references are declarations, not cryptographic proof of their contents.

## Closed source findings — 2026-09-12

- Computed-metric overwrite closed at evidence parsing and merge boundaries.
  `tests/test_eval.py` exercises a real no-op recorded trajectory, reserved keys
  present/absent from traces, mutable evidence bypass, ordinary outcomes, missing
  outcomes and CLI rejection before report/database writes.
- Adjacent non-finite outcome values (NaN/infinity) now reject explicitly rather
  than participating in truthiness or threshold scoring.

These are source-contract regressions. They do not qualify hardware gameplay,
physical movement, or model promotion.
