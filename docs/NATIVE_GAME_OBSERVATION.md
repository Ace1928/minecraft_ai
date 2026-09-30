# Native Minecraft observation closeout

The player currently reads direct HUD interlocks and calibrated hotbar icons.
Its shared World vision capability supplies uncertain CLIP category comparisons;
it does not supply authenticated resource detections, inventory measurements or
player chat. These outputs remain unavailable. Native execution alone does not
qualify a classifier for a gameplay action or learning label.

`native_world_observer.py` defines the bounded consumer seam. Each request names
one task (`resource`, `inventory`, `chat`), exact Bedrock instance/build, active
pack catalog hash, frame ID, original capture time, dimensions, exact pixel
digest, and a deadline no more than five seconds after capture. Supplied pixels
must match every field. The current private World readiness returns an explicit
unsupported result listing missing outputs, with no facts, tracks or chat
payload. Missing/private-file failures return unavailable; expired frames return
expired. No capture, dense model, CLIP-to-target conversion, or automatic
qualification is performed. This is an implemented negative consumer contract;
positive transport and observation publication remain unfinished.

## Required shared capability profiles

A future qualified profile must be registered in the **same World capability
bank**, bound to the active owner runtime ID and an independently pinned profile
hash. It must bind the exact game executable/assets, active pack UUIDs/versions
and hashes, camera/HUD/text/render settings, pixel transform, task-specific
output schema, capability artifact hash and measured qualification receipt hash.
A boolean `qualified` flag or generic vision availability is insufficient.

- Resource output needs localized `oak_log`/`oak_trunk` tracks with exact source
  frame/crop evidence, calibrated confidence, freshness and unknown/abstention
  behavior. A tight visible trunk is distinct from leaves, planks and player
  constructions. Mining still passes the existing target/hash/lease guard.
- Inventory output needs GUI-cited recipe tracks (`craftable_planks_recipe`) and
  exact item counts, accounting for hidden slots and pack-specific icons. Craft
  success requires the observed input decrease and actual output increase.
- Chat output needs the exact visible speaker and text, per-line confidence and
  chat-region pixel evidence. Missing identity or uncertain transcription cannot
  authorize a reply. Names/full chat do not leave the machine in wiki searches.

Qualification must use independently labelled real exact-build frames, separate
from fit/calibration frames, covering rendering scale, lighting, occlusion,
motion, empty/unrecognized inputs, pack overlays and ambiguous glyphs. Measure
false positive/abstention rates, speaker and full-line transcription accuracy,
localization/count accuracy and latency/CPU/memory bounds. Synthetic font
rendering can supply training fixtures; it cannot substitute for live-frame
qualification. Any collector must consume the existing capture owner's supplied
pixels rather than opening another game-display capture.

## Exact-build font assets identified locally

The current client process loads Bedrock-on-Linux release `1.26.52.3`. Its vanilla
font lives in `data/resource_packs/vanilla/font/minecraft-ten.ttf` (SHA256
`fc34c190dbfae5fe2f27adb45a93425f502388191854a6a0c2ac2f7c594a688f`),
with `smooth/*.fontdata` and `smooth/remapping.dat`. The latter hash is
`5c90eac33806d2d28c4259f7a677c80cadd71cca8d764f991c38d2f7980e8c3e`.
The game-generated temporary atlas is `minecraft-ten00_ttf.png` (SHA256
`e47d6689ff288f9cc888cf45b2982cf2eb5b213995770a71a64406ff8dea14aa`)
with binary glyph metadata `minecraft-ten00_ttf.info` (SHA256
`68233f81b04ce20283c9e2640b8371ff4158cd5fb9b8369f29458368399d2c39`).
The atlas has no embedded build guarantee: its glyph mapping and rendering
scale must be verified against this executable before use. Old mcpelauncher
atlases and older camera/text profiles are not interchangeable evidence.
Visual inspection shows this display atlas maps lowercase letters to capital
shapes. It may be the wrong renderer for world chat. A provisional binary
mapping (eight-byte header, 256 records of four uint16 rectangle corners plus
five float32 metrics, final uint32 atlas size) gives valid rectangles throughout
the 1024-square atlas; this structural check does not establish OCR accuracy.

The small deterministic glyph/template expert is a plausible shared-bank OCR
route. It still needs a parsed mapping, exact rendering calibration and the
real-frame quality receipt before producing player-chat authority. No OCR
expert is declared qualified or active by this document.

Pending typed operator questions also retrieve bounded references in the
cognition worker. An exact configured family-pack recipe takes precedence;
unanswered pack-specific questions cannot fall back to vanilla wiki advice.
Other supported public topics use the authenticated shared search service and
its existing query filter. References enter the native World reply-only prompt
with version/confidence metadata; they never become observed facts, inventory
counts, player-chat authority or an executable game action. Refresh the pinned
catalog from the actual installed family pack before resuming after an upgrade.

The existing recipe/controller/chat path now projects Latin accents and the
empty-grid symbol into the printable-ASCII input contract (`Poke Ball`, `.`).
Catalog bytes, Unicode item names and wiki extracts remain unchanged; an
unrepresentable name refuses chat instead of losing characters. The native
World planner still owns the decision; exact recipe quantities and arrangement
come from the configured hash-pinned export, never assumed inventory or a
vanilla recipe substituted for pack mechanics.

Model-produced game replies retain private metadata binding the exact source
player/explicit-channel fact. A new speaker/text, timestamp, source or pixel
evidence reference cannot authorize an older answer. Before the existing
supervisor chat command, the runtime requires the current captured HUD with
fresh no-UI/no-death/no-drowning witnesses, blocks immediate danger and critical
health (including unreadable survival hearts), and releases/reconciles all
gameplay inputs. It rechecks those conditions
after release and counts a reply only after the leased transport confirms the
exact character count. This is source-level delivery admission, not qualified
player-chat OCR or observed in-game delivery. Slash commands and unsupported
text cannot pass this reply path.

After any server pack upgrade, export and repin the actual installed catalog
before using it. The Bedrock build number alone cannot establish that a server's
pack is unchanged. A real acceptance check must retain the before/composer/
submitted/returned-world frames and the matching exact pack answer. Until an
exact-build chat observer is qualified, the native observation seam above
remains explicitly unsupported for autonomous player-question ingestion.
The current player-message authority expires after 30 seconds. An earlier
actual native recipe probe took about 36 seconds, so this delivery path must
reject such a late answer. Lower-latency reply routing remains a measured gap;
the safety deadline has not been extended to conceal it.

## Replacing the shared World without closing the game

The native World service overlay uses `Wants` and `After` for ordered bootstrap.
It deliberately avoids `Requires`: replacing World otherwise stops the player
unit, whose shutdown hooks close the managed Bedrock client. Native readiness,
runtime identity, model leases and existing input interlocks remain mandatory.
This dependency change does not itself authorize gameplay during an outage.

For a supervised runtime handover, retain the exact wrapper PID/start identity,
suspend only that wrapper's recovery loop, and use the repository CLI's
`stop --transient` to drain the agent/supervisor and release every held input.
Verify the game process, window and session are unchanged before replacing
World. Restart the dashboard after the new owner's real health checks pass,
then resume the same wrapper so the next agent loads the new source/config.
Verify its exact owner binding and live readiness before scheduling a task.
Retain the rollback source and latest consistent learning checkpoint. If
qualification fails, restore the previous owner before resuming the wrapper;
never leave a suspended recovery loop without a recorded handover owner.

## Bounded progression before autonomous observation

The existing exact-frame operator selection can supply a manually grounded
`oak_log` reference to native ROCKET. This may support a bounded wood-gather
attempt; it is neither autonomous tree discovery nor a training label. Count
only verified break-and-pickup transactions with stable canonical hotbar +1.
The next milestone is three oak logs, then observed planks/table/tools, then
apricorns/copper and four current-pack Poké Balls. Traversal and local router
updates alone do not prove these milestones or behavioral mastery.
