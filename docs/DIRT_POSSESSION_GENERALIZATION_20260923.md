# Item-kind drop possession: dirt as the first additional verified item (23 September 2026)

Follow-up to `HAND_CLEARANCE_BREAK_20260922.md`, which recorded three verified
hand-clearance stone breaks and named the next concrete step: wire the
already-calibrated dirt hotbar template into `inventory.hotbar.dirt` and
generalise `collect_recent_drop` / `_observe_collection_possession` to the
verified break's item kind.

## Outcome

The generalisation is implemented and unit-verified. The bounded live attempt
(30 minutes, one agent-only reload, game and VLM untouched) produced **one
verified hand-clearable break** (stone-class, the same class as before) but
**no verified `resource_acquired`**. The live blockers are measured and
recorded below; no success is claimed without a verified event.

- implementation commit: `94ac0f7` (docs commit follows this file)
- reload: session `bf9de007654d8d94a8ff431fc1d94791` ->
  `ab7510ecd8fbe3c8940f63f98265c7d7` at 09:04:50 AEST
- game preserved: Bedrock PID 1036272 (started 2026-09-22 12:28:14), window
  `14680065`, display `:2`, input instance `bedrock:1.26.51.1:x11:14680065`
- observation window: 09:03:17-09:33:20 (events) and 09:16:15-09:35:10
  (crosshair receipts), 30 min bound
- `events` table: 1 `block_broken`, 0 `resource_acquired` in the window;
  total `resource_acquired` ever is still 0

## 1. Item-kind generalisation

Strict verification is unchanged; only the item identity became a parameter.

### Perception (`perception/service.py`)

- `BEDROCK_HOTBAR_DIRT_COUNT_SOURCE` =
  `deterministic:bedrock-1.26.45.1-classic-hud-hotbar-dirt-v1:not-training-label`,
  the canonical identity of the already-calibrated
  `_CLASSIC_HOTBAR_DIRT_RGB_5X13` template. No new template was invented.
- `_classic_hotbar_slot_kind` now returns `dirt` for a pinned dirt-template
  match instead of folding it into `other`.
- `_classic_hotbar_item_counts` classifies the nine slots once and counts
  every calibrated kind (`log`, `dirt`) over the same validated grid. One
  ambiguous slot abstains for **all** kinds, so no count is ever published
  while part of the hotbar is unreadable.
- `bedrock_hotbar_dirt_count` mirrors `bedrock_hotbar_log_count`; the fast
  perception publishes `inventory.hotbar.dirt` (confidence 0.995, 250 ms TTL,
  `observed_ns == frame.captured_ns`) next to `inventory.hotbar.logs`.
  Hidden inventory is never consulted and `inventory.dirt` / `inventory.logs`
  generic keys are never produced.

### Execution (`control/execution.py`)

- `_HOTBAR_ITEM_OBSERVERS` maps each collectable item kind to its exact fact
  key and pinned source. Unknown kinds never gain evidence.
- `_BREAK_KIND_DROP_ITEMS` maps verified break kinds to the item they drop by
  hand: `dirt` -> `dirt`, `grass_block` -> `dirt`. `drop_item_kind_for_break_kind`
  maps any recognised log-family label to `log` and returns `None` for
  everything else (for example `sand`, `coarse_dirt`, `stone`, `unknown`).
- `_exact_hotbar_item_fact` requires the exact key, exact source,
  confidence >= 0.99, a non-bool non-negative integer, and freshness.
- `SkillExecutor.start(..., collection_hotbar_baseline, collection_drop_kind)`
  binds the preserved pre-break count to the verified drop kind. A baseline
  for another kind, a wrong source, or an unknown kind leaves the possession
  state unbound, so the run can only fail closed.
- `_observe_collection_possession` watches the bound kind's canonical fact and
  requires stable +1 after observed motion against the frozen baseline. The
  `OutcomeVerification` carries the exact evidence key and `target_kind`
  (`log` or `dirt`); the log-only accessor never claims a dirt increment.
- `mine_visible_block` freezes `mining_drop_baseline`/`mining_drop_kind` for
  the target's mapped item kind before the first attack; the legacy log
  baseline freeze is retained for gather.

### Runtime (`runtime.py`, `runtime_support/helpers.py`, `skills/builtin.py`)

- `_verified_collectable_break` returns the exact drop kind for a verified
  mining break, or `None`. `collect_recent_drop` now starts only for those
  kinds and transfers the matching frozen baseline.
- `_start_drop_collection` publishes `collection.recent_break` with the drop
  kind as its value and source
  `verified:<run>:block-broken:<kind>`; the collector terminal path revokes
  it. The old log-specific fact key is gone.
- `collect_recent_drop` v3 precondition is `collection.recent_break`
  (truthy, min confidence 0.99).

## 2. Live attempt sequence

1. Implementation committed at `94ac0f7`.
2. Agent-only reload with `minecraft-ai stop --transient`; the persistent
   launcher rebuilt supervisor/agent on the same game session. New generation
   `ab7510ecd8fbe3c8940f63f98265c7d7` reported `RUNNING`, `live_capable`,
   motor lease active, window `14680065` (unchanged).
3. Bounded observation units (read-only, never touching game input):
   `minecraft-dirt-observe.service` and `minecraft-dirt-receipts.service`,
   both `Nice=19`, `CPUQuota=100%`, `MemoryMax=2G`, `MemorySwapMax=0`.
   Raw JSONL evidence is retained outside the repository at
   `/tmp/opencode/minecraft_dirt_20260923/` (`events.jsonl`, `status.jsonl`,
   `receipts.jsonl`).

### Crosshair receipts (the only live target path)

| wall time (AEST) | query id | VLM block | confidence |
|---|---|---|---|
| 09:16:15 | `50648d6647bc4f7db1ae4d00dfe25127` | `stone` | 0.9 |
| 09:16:56 | `975e24a490e54c8795b1674dd486106c` | `unknown` | 0.2 |
| 09:24:11 | `a07cc7bb33a941828341a03d4eb23a74` | `stone` | 0.9 |
| 09:28:14 | `e2299b464dc84ec1ad7b19d768c16585` | `stone` | 0.9 |
| 09:35:10 | `2bbbde87b14e4bedbfe8343300c0872a` | `unknown` | 0.5 |

No `dirt` or `grass_block` classification occurred. The world view in the
window was a stone/cobblestone depression (frames 09:04, 09:08, 09:34, 09:40),
matching the 2026-09-22 description.

### Verified break (hand clearance still works live)

- run `56458f31d55a4c11aecc076835ab7c92`, skill `mine_visible_block`,
  `block_broken` at 09:24:18-09:24:20, confidence 0.84, `target_kind` `stone`
- evidence keys:
  `frame.crosshair_dhash`, `frame.crosshair_luma_grid`,
  `target.track_id=crosshair-probe:a07cc7bb33a941828341a03d4eb23a74`,
  `target.binding_source=vlm:gemma-4-e4b-vl:a07cc7bb33a941828341a03d4eb23a74`
- no `collect_recent_drop` run followed: `_verified_collectable_break`
  correctly returned `None` for `stone`, so the fail-closed mapping held live.

## 3. Exact remaining blockers

1. **No dirt/grass under the crosshair.** All five receipts were
   `stone`/`unknown`; the agent remained inside a stone/cobblestone chamber
   for the whole window. The headroom clearance path ran (one verified
   stone-class break) but had no dirt to clear.
2. **The hotbar now holds an uncalibrated item.** The 09:24 clearance
   produced 4 cobblestone in hotbar slot 0 (auto-pickup). A raw live capture
   at 09:34 classified slot 0 as `ambiguous` and slots 1-8 as `other`, so
   `_classic_hotbar_item_counts` returns `None` for **every** kind and both
   `inventory.hotbar.logs` and `inventory.hotbar.dirt` are unpublished. This
   is the intended strictness (unknown hotbar items invalidate all canonical
   counts, no hidden-inventory claims), but it means that even a hypothetical
   dirt break right now could not freeze a pre-break baseline or verify a
   pickup until the hotbar is again composed only of calibrated or empty
   slots.

Evidence for blocker 2 (read-only capture of the live window, no input):

```
IsolatedX11Capture(":2", 14680065).capture()
bedrock_hotbar_log_count(frame)  -> None
bedrock_hotbar_dirt_count(frame) -> None
geometry (594, 990); slot kinds: 0 ambiguous, 1..8 other
```

## 4. Next concrete step

1. Add a calibrated hotbar observer for cobblestone (or for the actual
   cleared block's item) using the same real-capture calibration process as
   the log/dirt templates. Until an item the agent holds is calibrated, no
   canonical count can be published while it is in the hotbar.
2. Only then re-run the bounded live attempt for dirt: the implemented
   `collection.recent_break` -> `inventory.hotbar.dirt` +1 chain needs both a
   dirt/grass classification under the crosshair and a readable hotbar.
3. Getting the agent out of the stone depression toward soil remains the
   world-side prerequisite; the existing operator goal already requests that
   route. No world reset, teleport, inventory API or new template was used.

## 5. Tests and resource bounds

- New/extended unit coverage (all fixtures, no live data):
  - perception: dirt stack counts 1-16 from the calibrated template,
    independent log/dirt counts in one hotbar, abstention with an
    uncalibrated peer icon, no generic `inventory.dirt` key;
  - execution: dirt increment verification (exact key/source/target_kind),
    cross-kind isolation, unknown drop kind fails closed, mismatched or
    noncanonical baseline rejected, noncanonical/generic/stale/multi-item
    dirt evidence rejected, pre-attack baseline freeze per target kind;
  - helpers/runtime: `_verified_collectable_break` mapping and fail-closed
    statuses, `drop_item_kind_for_break_kind` mapping, live hand-off
    parametrised over `oak_log`/`dirt`/`grass_block`.
- `pytest tests/test_perception_service.py tests/test_execution_motor.py
  tests/test_execution_outcomes.py tests/test_headroom_recovery.py
  tests/test_agent_core.py tests/test_m0_architecture.py`: 627 passed.
- Full suite: 3479 passed, 7 failed - all
  `tests/test_persistent_launcher.py` subprocess-timeout cases already
  documented as failing on the unmodified checkout in this environment.
- Resource bounds during the attempt: observer and receipt units at nice 19,
  `CPUQuota=100%`, `MemoryMax=2G`, `MemorySwapMax=0`; the dashboard and
  public-view services were left as they were; the game, GPU VLM and
  calibration were never restarted.

## 6. Operational note

During source editing one agent generation started while the perception
module was mid-edit and hit a transient `NameError` for a constant added
later in the same edit session. The launcher restarted it; the deployed
generation `ab7510ecd8fbe3c8940f63f98265c7d7` started after the final source
state and ran the whole observation window without a module error. This is
recorded for completeness, not hidden.
