# Reactive traversal and learned mining: 12 September 2026

## Delivered runtime changes

A verified traversal stall is no longer gated by a short hardcoded list of skill IDs. This includes `traverse_level_ground`; third-party skills may opt into the same verifier with `SkillSpec(outcome_kind="traversal")`. The obstruction recovery path retains target, scene and headroom verification and returns to the original traversal skill after clearance. A retry cannot complete an unrelated typed plan step.

`survey_surroundings` and `reacquire_target` are camera-only semantic options. The executor enforces their action permissions even when a model ignores permission hints: prior walking/attack is released, forbidden presses are removed and interventions remain synthetic. Camera commands still come from the installed policy, not a periodic scan macro. A policy incapable of looking will be reported as inactive; this patch does not invent neural competence.

The new `MiningKnowledge` is used by the mining guard, terminal outcome path, runtime memory and planner context. It records separate nullable fields for block breakage, harvesting and pickup, keyed by ruleset, block and equipped tool. Model guesses, absent pickups and timeouts do not become authoritative tool rules. A timeout is censored evidence, not an unbreakable-block label or a successful break-duration sample. Collection evidence must join its exact verified parent break and cannot count that break again. Existing exact outcome verifiers remain the source of authority, so generic support for identifiers does not imply every modded item already has a calibrated pickup detector.

At least three completed breaks of the same scoped block/tool pair support an upper log-duration estimate. Explicit resolved game data can supply expected duration and capability separately. The guard uses these estimates within configured bounds. Learned or resolved breakability can support obstruction clearing without insisting that the block be harvestable. Harvest-required work still checks its own contract. Familiar vanilla rules remain a conservative cold-start fallback; they have not all been removed.

Crosshair inspection accepts bounded namespaced identifiers rather than only a fixed vanilla enumeration. Unknown blocks can be tried through an explicitly permitted mining option, with a verified equipped tool, target geometry, playable scene and bounded lease. This does not grant permission to mine arbitrary visible blocks or convert every perception adapter into a modpack reader.

Planner requests explicitly include current block/tool evidence and available observed hotbar alternatives. The information does not depend on a mining note happening to appear among a small generic memory sample. Records use the existing procedural memory and SQLite persistence path; no schema migration is added by this upgrade.

## Timing and activity

The normal configured agent now defaults to a 10,000 ms acquisition bound and a 30,000 ms absolute mining hold cap. The built-in mining option has a 45,000 ms whole-skill ceiling. Standalone `MiningLeaseGuard` defaults remain compatible with previous callers. Longer limits do not disable missing-frame, target-loss, tool-change, danger, operator-stop or visual-stagnation checks. Progress may extend a soft deadline, never the absolute cap.

An inactivity watchdog detects accepted-but-empty policies: the built-in gather option allows 8,000 ms without a legal positive/held action and the survey option allows 5,000 ms. Release packets and prohibited proposals do not count as activity. Starvation is reported separately from gameplay competence. This is a liveness check, not an observed-progress reward.

Fast control checks remain on the motor/execution path rather than waiting for the slower planner. Existing asynchronous cognition, goal/plan state and memory are retained. This patch does not add separate short-, medium- and long-horizon language models, a learned planning-frequency controller, or a hard real-time latency guarantee.

## Update locally

Stop the running agent using its operator-owned stop path. Back up its consistent SQLite state, configuration and active checkpoint; preserve local edits. From the existing checkout and current player environment:

```sh
git status --short
git switch main
git pull --ff-only
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_universal_traversal.py tests/test_mining_knowledge.py tests/test_mining_control.py tests/test_headroom_recovery.py tests/test_execution_outcomes.py
minecraft-ai doctor
```

Keep installed capture/input extras, model configuration and the working launcher. Do not reset a dirty checkout or replace scoped input with a host-global fallback. No command here changes active neural weights.

## Retain learning across restarts

Add or merge these **top-level keys** into the existing YAML configuration, not a nested `runtime:` section. Replace the example identity with one describing the exact edition/version, pack revision and relevant rules/game-mode context:

```yaml
mining_ruleset_id: "my-bedrock-version:my-pack-revision:survival"
mining_max_hold_ms: 30000
mining_acquisition_timeout_ms: 10000
```

Reuse that identity only while its semantics remain unchanged. Change it after pack/rule updates, relevant permission or game-mode changes. Tool state, enchantments and effects must be represented by the observation/rule adapter when they alter the outcome; a generic name alone cannot distinguish them.

Without an explicit ruleset identity or snapshot, the runtime intentionally creates a new session scope. Records are still persisted, but old unknown-version beliefs are not automatically reused in another session. An explicit stable identity is necessary for safe cross-restart reuse in a pixel-only installation.

## Optional game-provided rule snapshots

A separately owned adapter may export a `MiningRuleSnapshot` JSON file. Configure it with the top-level `mining_rule_snapshot` path. Its content hash supersedes the descriptive ruleset ID for memory isolation, so changing rules cannot silently reuse old beliefs. The file is bounded to 16 MiB, validates exact types and refuses duplicate canonical pairs.

The schema is:

```json
{
  "schema_version": 1,
  "ruleset_id": "exact-version-and-pack-identity",
  "provenance": "description of the actual game-data adapter and export",
  "rules": [
    {
      "block": "examplepack:example_block",
      "tool": "examplepack:example_tool",
      "can_break": true,
      "can_harvest": null,
      "expected_ms": 7000.0
    }
  ]
}
```

The values above are an illustrative schema, not Minecraft facts. Unknown fields must be null or omitted, not guessed. Supply resolved effective rules rather than unevaluated pack expressions. An adapter must account for the relevant block/tool state, effects and permissions, and preserve the deployment's strict-input/data policy. This upgrade implements the consumer contract, not an automatic Bedrock registry exporter or a universal modpack interpreter. Do not feed language-model guesses into this authoritative channel.

## Verification and remaining evidence

The source-only mining commit `b1e2cb2eac093cdf02abb96df6a30d36da7c0261` passed Ruff, Mypy and the complete Python suite in workflow run `34686336909`: **2,706 passed, 3 skipped, no failures**. The preceding traversal commit passed the same code-quality gates. Both were merged while preserving concurrent main changes.

Tests cover persistence/reload, scoped separation, exact parent joins, censored missing pickups, learned duration bounds, unknown/modded identifiers, configured timing, model-independent permissions, universal traversal stalls, inactivity and legacy guard compatibility. The complete multi-platform and Java CI remains a separate gate.

No live game was opened during this implementation. Compare before/after with the same checkpoint in a disposable world: verified breaks per attempt, successful clearance followed by traversal, time inactive, camera movement, repeated recoveries and input release. Also test an obstructing block that can be broken but not harvested, delayed pickup, an unknown pack identifier, an expired target and a world/ruleset change. Passing software tests is not a claim of a generally competent player.
