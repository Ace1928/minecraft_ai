# Adaptive gameplay upgrade

## What changes immediately

Normal skill recovery now ranks the declared, currently feasible recovery skills using their persisted contextual outcomes, rather than always selecting the first declaration. Missing, retired, deprecated, infeasible, and repeatedly failed choices are excluded. If every recovery is exhausted, execution returns to replanning rather than looping indefinitely. Cold-start ties preserve the existing declared order. Emergency recovery and safety interlocks are unchanged.

Skill competence separates actual execution outcomes from controller startup starvation. A failure before the controller supplies an action remains visible in diagnostics, but is censored from success-rate estimates and consecutive gameplay-failure streaks. Contextual estimates borrow a bounded leave-context-out prior, avoiding double-counting the same attempt as both global and local evidence.

Mining uses one shared visual-progress tracker in the motor lease and outcome verifier. It can recognize spatial luminance changes around the crosshair even when the coarse image hash does not change. Uniform lighting changes are removed before measuring local progress. Stale, future, replayed, malformed, and low-confidence evidence does not count. Repeated capture timestamps cannot indefinitely renew missing-evidence grace.

A mining attempt may extend its soft deadline only after enough fresh visual samples and recent progress. The absolute mining hold limit remains in force. Visual progress is not a verified block break or a success reward. Exact or settled outcome qualification remains separate.

These are tested changes to failure mechanisms, not a claim of measured live-game skill improvement. No live Minecraft instance or user checkpoint was available during implementation.

## Pull and restart

Stop the agent before updating. Back up the state database, configuration, and active checkpoint before the first run of the new version. In each existing checkout:

```sh
git status --short
git switch main
git pull --ff-only
```

Preserve local changes rather than resetting them. Use the same Python environment and install the updated checkout with the project's existing installation procedure, for example `python -m pip install -e '.[dev]'` for a development environment. Pull the authorized ERAIS checkout as well and update its editable installation in the environment used by the native worker. Keep your current launch configuration, calibrated camera scales, and active checkpoint.

The state database migrates automatically from schema 7 to schema 8, adding `censored_failures` with a zero default. It does not delete prior observations. Old failures cannot be retrospectively classified reliably. To roll back to old code, restore the pre-upgrade database backup rather than asking old code to interpret the new schema.

The new public recovery learning is automatic as real outcomes accumulate. It does not require a new neural checkpoint. The native policy is a separate learning level: train a new candidate using the upgraded ERAIS replay/trainer, then evaluate that candidate before explicitly activating it. Pulling source alone does not improve existing neural weights.

## Native learning integration

Use the native training runbook in the authorized ERAIS checkout (`docs/runbooks/minecraft_native_learning.md`). The trainer now preserves atomic clicks, short demonstrations, padding masks, causal shard continuity, and whole-trajectory worker ownership. Its optimizer updates the actual native motor policy, and its CLI can initialize from a SHA-256-pinned existing native parent instead of always starting randomly.

Use verified run admission for autonomous successful-run replay. Do not train indiscriminately on failed self-play as positive demonstrations. Synthetic labels remain excluded unless explicitly requested. Finite loss, decreasing imitation loss, a reloadable bundle, or changing routing telemetry do not qualify a better player.

## Regression checks

```sh
python -m pytest -q tests/test_adaptive_gameplay.py tests/test_storage.py
ruff check src tests
mypy src
python -m pytest -q
```

The existing CI matrix remains authoritative for platform compatibility. Regression coverage includes slow mining progress, stale-frame refusal, absolute stop bounds, recovery changing after observed failures, exhausted alternatives, infeasible recovery, censored starvation, contextual score shrinkage, persistence, and schema migration.

## Measure the actual improvement locally

Compare the old active checkpoint and candidate on the same world/task/starting conditions with repeated trials. Record verified block breaks per attempt, pickups, time spent stalled, repeated recovery failures, timeout reasons, and safe input release. Keep task outcomes distinct from visual progress and from training loss. Inspect failure categories before collecting further demonstrations: reaching a target, maintaining aim, holding attack, verifying the break, and collecting the item are different skills and different failure sources.

A retained absolute safety limit is not a learned Minecraft strategy. This upgrade deliberately learns recovery preferences and corrects native training while keeping bounded input execution, evidence freshness, and outcome authority explicit.
