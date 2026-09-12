# Learned control upgrade: 12 September 2026

## What changes immediately after pulling

The live execution path now includes acquisition-local camera response estimation and uncertainty-aware recovery selection. These are integrated changes, not unused helper classes. Existing target/tool checks, scoped input, emergency stopping, synthetic-action attribution, and absolute acquisition/attack limits remain authoritative.

### Adaptive mining aim

`control/adaptive_aim.py` replaces the fixed-gain correction in the existing exact-operator-target mining acquisition path in `control/mining.py`. Each fresh ROCKET localization measures how much the target moved after the last camera command. Separate yaw/pitch estimates then scale subsequent corrections. A small cold-start probe remains, but later corrections are no longer permanently capped at the former 12 counts.

The estimator uses a short median response history, bounded gain growth, and a 256-count default ceiling. Stale/future/replayed observations cannot issue another correction. Target changes reset the estimator; observations taken while stopping locomotion do not calibrate it. Wrong-direction feedback falls back rather than teaching an unstable positive-feedback response. An unresponsive camera still reaches the existing acquisition timeout.

This is online system identification, not a trained neural aiming policy. It currently applies to the exact operator-grounded acquisition path, not every autonomous target-selection path. The guard still requires a later fresh, centred, authorized target before pressing attack. Camera corrections remain synthetic and do not become native-policy demonstrations by accident. Calibration is intentionally local to the attempt, not persisted across camera settings or worlds.

### Recovery exploration

`skills/recovery.py` now ranks feasible options using contextual success evidence plus an uncertainty bonus. This addresses greedy lock-in: an option with many past successes need not permanently prevent an under-observed alternative from getting a trial.

Only decisive contextual outcomes contribute to the exploration count. Cancellations and censored controller-startup failures do not count as evidence of competence. The existing skill statistics persistence remains the source of experience; no schema change is introduced by this patch. Declared recovery order breaks cold-start ties rather than prescribing every subsequent choice.

The selector's `exploration_strength` defaults to `0.35`; setting it to `0.0` in a caller restores greedy ranking for an ablation. This is a score, not a calibrated probability. Exploration cannot bypass skill preconditions, retired/deprecated status, or the two-failure contextual cap. Exhausted recovery sets return to cognition instead of retrying indefinitely. Emergency handling is not reordered.

## Pull and restart

Stop the running agent with the existing operator-owned stop path before changing its environment. Preserve local changes, back up configuration/state and keep the active checkpoint. In the existing checkout:

```sh
git status --short
git switch main
git pull --ff-only
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/test_adaptive_aim.py tests/test_mining_control.py \
  tests/test_recovery_exploration.py tests/test_adaptive_gameplay.py \
  tests/test_storage.py
```

Use the same Python environment and installed capture/input extras as the current player. Run `minecraft-ai doctor`, then restart using the existing working launcher and configuration. Do not replace scoped input with a host-global fallback to bypass a failed doctor check. A dirty or divergent checkout should be reconciled, not reset or force-pulled.

These control changes do not require replacing a checkpoint. Separately installed proprietary workers, model weights and training code remain outside this public repository. Updating a separate trainer does not automatically change the active player's neural weights.

## Verification

The focused suite above passed **166 tests** during this implementation, including **28 added cases** across adaptive aim and recovery exploration. The full CI matrix for code commit `30b3b65a9b9e1d9d89baa51013893a260524ba59` also passed, including Ruff, Mypy, Python tests and the Java bridge build.

The new tests exercise multiple simulated camera sensitivities, independent axes, fresh-evidence attack admission, stale feedback, nonfinite inputs, acquisition timeouts, greedy lock-in, censored outcomes and exhausted/infeasible recovery options. A compatibility type stub was corrected after CI exposed its obsolete fixed-aim exports.

`Gameplay regressions` CI now explicitly runs the added test files. Each run retains a JUnit receipt for 14 days and an exact source snapshot with commit/hash information for one day. This makes failures reproducible without bundling checkpoints or gameplay recordings.

## Live acceptance checks still required

Software tests and simulated feedback do not prove reliable gameplay with a particular checkpoint, sensitivity, latency or game build. Compare the unchanged checkpoint before/after this code update in repeatable, disposable-world trials. Keep task definitions and starting conditions consistent.

Measure verified block breaks per attempt, time to acquire the target, attack interruptions, stuck time, repeated recovery choices, completed recoveries and safety releases. Test hand-safe blocks, appropriate tools for harder blocks, off-centre marked targets, and ordinary obstacle recovery. Confirm that target loss, tool changes, overlays, dangerous scenes and operator stop still terminate input appropriately.

Do not equate crosshair disappearance, changing pixels, or a held mouse button with a verified block break. This patch does not replace the complete autonomous motor policy, implement full online reinforcement learning, or demonstrate a solved Minecraft player. The next evidence is a controlled local gameplay comparison, not a larger collection of hardcoded movement sequences.
