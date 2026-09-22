# First verified breaks: empty-hand clearance of hard blocks (22 September 2026)

## Outcome

After the fix in this change, the live agent produced the first verified
`block_broken` events since 2026-09-14 while the same world, game session and
VLM stayed running:

| # | Wall time (AEST) | Run id | Target | Signal | Confidence |
|---|---|---|---|---|---|
| 1 | 2026-09-22 23:06:01 | `ee0822d83d104711bedcff98be1ec03d` | `stone` | `block_broken` | 0.84 |
| 2 | 2026-09-22 23:09:35 | `5d39e6fc57e446a180635558168d959e` | `stone` | `block_broken` | 0.84 |

Both came from `mine_visible_block` runs owned by the traversal headroom
recovery, bound to fresh VLM crosshair receipts, and were verified by the
existing temporal visual verifier ("stationary sustained attack traversed
multiple damage phases and settled on a stable changed luma grid after
release"). The second event repeats the first with a different query id.

Trajectory: `bedrock-agent-20260922T125332Z-62c252164cd2` (shard-000001,
run 1 steps 405-483; shard-000002/3, run 2 steps 681-783).

No `resource_acquired` event was produced. That remaining gap is analysed at
the end of this document.

## Live state before the change

- `events` table: 49 `block_broken` total, latest 2026-09-14 03:52; zero
  `resource_acquired` events ever.
- Observation of the crosshair probe showed `stone` (0.8) and occasionally
  `unknown` (0.5). The world view is a hard-survival stone/stone-brick
  chamber/ravine with an empty hotbar.
- Agent skills in the six hours before the fix were traversal only:
  `explore_forward`, `traverse_level_ground`, `survey_surroundings`,
  `backtrack_from_obstacle`, `escape_confinement` (plus two
  `gather_nearby_wood` starvation failures).

## Trace of one full attempt (18:08:41 gather)

`gather_nearby_wood` run `0e6ee41ddd5945778c17855930060dc8`, context
`operator:238f3d1462604fa08470e1cca219eb95`, steps 1887-1923:

- Every blackboard snapshot for the run contains only bootstrap/safety facts.
  There are no `target.visible`, `target.kind`, or track entries of any kind
  (`tracks: []`).
- The learned policy emitted release-only actions for the whole run
  (`"buttons_down": []`, `"policy_action_kind": "release"`).
- The mining guard admits a gather attack only when the policy presses `left`,
  or when an exact operator/crosshair-probe target exists. Crosshair-probe
  authorization is limited to `mine_visible_block`
  (`_crosshair_probe_mining_authorized`), so gather could not acquire a target
  from the probe either.
- After 8,077 ms of accepted-but-empty input the inactivity watchdog finished
  the run: `"controller emitted no permitted active input"`,
  `failure_code=controller.starvation`.
- The plan node `gather_nearby_wood` is also refused before start because
  `visible_oak_trunk()` is false, and the native-steve/rocket workers that
  could have localized a trunk failed warmup in this session
  (`failed_warmup_policy_ids` contains both, `process_alive: false`).

Conclusion: gather cannot self-acquire a target in this world and starves by
design. The only live target path is the headroom recovery's VLM crosshair
probe.

## Why the working crosshair path still could not break anything

The recovery classifies the exact crosshair crop with the local VLM
(`gemma-4-e4b-vl`, receipt `stone` 0.8 / `stone_bricks` 0.8 / `unknown` 0.5) and
then calls `_headroom_clear_target`. That gate admitted only hand-safe soft
terrain when the hotbar was empty:

```python
if not is_hand_safe_soft_block(normalized_kind):
    tool = _selected_item(...)
    if tool is None or mining_knowledge is None:
        return None
```

- `mining_ruleset_id` is null and no rule snapshot is configured, so
  `mining_knowledge` is `None`.
- The deterministic hotbar observer publishes only `inventory.hotbar.logs`;
  `player.selected_slot` is never published at `semantic_hz=0.0`, so
  `_selected_item` returns `None`.
- Therefore every `stone`/`stone_bricks` answer was rejected. `oak_log` was
  rejected by the same gate. Even with an admitted target, `_verified_target`
  refused an empty hand on tiered blocks (`MINING_TOOL_UNVERIFIED` /
  `MINING_WRONG_TOOL`), so the guard never pressed attack.
- Timing compounded it: the VLM call takes 107-167 s in this session while
  `_headroom_deadline_ns` capped the whole transaction at 180 s. Classification
  could consume the entire transaction and the clearance would be cancelled
  before it ever held attack.

This is the observed chain break: the mining skill never received a usable
target, and when a target-like classification existed, the empty hand was
treated as incapable of breaking a visible vanilla block that hard survival
players can in fact break.

## Fix

Small, guard-preserving changes in `minecraft_ai`:

1. `control/mining.py`: new `_HAND_CLEARABLE_FAMILIES` (SOFT, LOG, WOOD,
   PICKAXE, DEEPSLATE) and `_HAND_BREAK_BUDGET_MS` (2.5 s soft, 3.6 s log/wood,
   10 s stone-class, 16 s deepslate) plus public
   `is_hand_clearable_block()`.
2. `control/mining.py` `_verified_target`: an explicit clearance run
   (`harvest_required: false`) with an actually empty hand may break a
   hand-clearable block. Exact negative game rules still win, an equipped but
   insufficient tool still fails `MINING_WRONG_TOOL`, and the visual break
   verifier remains the only source of success.
3. `control/mining.py` `_lease_duration_ms` and
   `_continuation_target_failure`: the lease holds attack for the full hand
   budget and continues while the crosshair target stays fresh, extending on
   visual progress up to the configured 30 s absolute input cap.
4. `runtime_support/helpers.py` `_headroom_clear_target`: admit
   hand-clearable families without a tool or resolved rule; obsidian, ancient
   debris, bedrock and `unknown` still abstain.
5. `runtime_support/helpers.py` `_HEADROOM_TRANSACTION_MAX_S`: 180 s -> 420 s so
   a serialized VLM answer no longer consumes the transaction that must also
   hold the guard's clearance and traversal retry.

The runtime already passed `harvest_required: false` for non-soft headroom
targets; no runtime change was needed.

## Measured live evidence

Agent reload (agent process only; Bedrock window 14680065 and the VLM stayed
up) at 22:53. The first stall after reload produced:

- Query `0560cdbb396441d287f1ae6a58ea7f89` -> `recovery.crosshair.block=stone`
  (0.8). `target.visible/target.kind/target.reference_available` were published
  from the same `vlm:gemma-4-e4b-vl:...` source.
- Track `crosshair-probe:0560cdbb396441d287f1ae6a58ea7f89` (label `stone`,
  sampling aperture at the crosshair).
- Step 405: synthetic `left` press (`action_origin=synthetic`,
  `policy_action_kind=release`), held across the damage phases.
- Step 482: synthetic `left` release.
- `skill_succeeded` and `block_broken` for run `ee0822d8...` at 23:06:01 with
  evidence keys
  `target.track_id=crosshair-probe:0560...`,
  `target.binding_source=vlm:gemma-4-e4b-vl:0560...`.
- Step 481 frame shows the block gone from the crosshair (hole opened).

The second stall then produced run `5d39e6fc...` (query
`4069aecbe1e3461eaa0a583f10041831`, `stone`) and break 2 at 23:09:35.

## Remaining gaps

1. **No verified pickup yet.** The only deterministic possession verifier is
   `inventory.hotbar.logs` (pinned oak-log template). Stone drops nothing by
   hand, and the crosshair classified `stone`, `stone_bricks` and `unknown` -
   never a log - so the existing `collect_recent_drop` path had no eligible
   break. Next concrete step: wire the already-calibrated dirt template
   (`_CLASSIC_HOTBAR_DIRT_RGB_5X13`, currently only used to keep the log count
   well-defined) into a strict `inventory.hotbar.dirt` count and generalise
   `collect_recent_drop` to the verified break's item kind; or let the now
   digging-capable agent reach an oak trunk, where the existing break ->
   collect -> hotbar +1 chain already works unchanged.
2. **VLM abstention in dark/unclear views** (`unknown` 0.5) still ends a
   recovery with no target; that is a perception limit, not a guard.
3. **Native policy workers** (`native-steve`, `native-rocket`) fail warmup in
   this environment (`native VPT output configuration digest mismatch`, ROCKET
   identity mismatch), so gather cannot localize a trunk through them. The
   hand-clearance path above does not depend on them.
4. `tests/test_persistent_launcher.py` fails on the unmodified checkout in this
   environment (subprocess timeouts); it is unrelated to this change and was
   left untouched.
