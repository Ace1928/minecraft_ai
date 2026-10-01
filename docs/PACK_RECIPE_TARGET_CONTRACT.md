# Exact targets in game_forge recipe snapshots

MinecraftAI consumes the existing schema-v1 snapshot produced by
`Ace1928/game_forge/minecraft/pokemon-family-server/crafting.py`. The
`PackRecipeCatalog` stays the sole active-pack recipe answer path; this change
adds no recipe database, wiki authority, model or learned recipe facts.

The reproduced defect was target substitution. In an invented active pack,
`blue:beacon` and `amber:beacon` share the display name `Beacon`. A request for
`blue:beacon` selected the alphabetically earlier namespace. Its recipe has an
empty shell as the first output, so the answer also reported the shell instead
of the requested beacon. All seven target regressions failed before the fix.

The catalog now respects an explicit namespaced key in the target clause,
including metadata such as `blue:beacon@2`. Ingredient clauses introduced by
`with`, `using`, `from` or `for` do not replace that target. Unknown explicit
keys and equally ranked display-name collisions return no exact recipe.
Multiple explicit targets require disambiguation. Rendering selects the output
whose exact key matches the requested item; missing or duplicate output matches
are refused. Tags and metadata keep their existing distinct identities.
Keys before the recipe intent retain their identity too. Intent and ingredient
words inside a namespaced key do not split that key, and unsupported identity
suffixes cannot match a valid prefix.

Unresolved explicit pack identifiers suppress general vanilla search through
the existing `mentions_pack_content` gate. A returned recipe remains reference
evidence. It does not assert inventory or permit crafting/mining input.

## Export and pin

The game_forge export command now prints `catalog_sha256`, the hash of the exact
written JSON bytes, alongside the existing recipe `revision`. Configure the
existing `pack_recipe_catalog` absolute path and `pack_recipe_catalog_sha256`
with that receipt. These hashes have different purposes: the revision identifies
recipe contents; the file hash also covers world/version, names, warnings and
other exported fields. Re-exporting after a pack change requires a reviewed new
file pin. This pass changes no deployed pin or service.

The schema keeps all recipe outputs and ingredient `key`/`id`/`data`/`tag`
fields. Consumers must select by exact key rather than by the first output,
display name or unqualified ID. Warning-bearing snapshots remain refused.

## Lightweight offline verification

From the MinecraftAI checkout, run with the game_forge kit available:

```sh
GAME_FORGE_KIT=/absolute/game_forge/minecraft/pokemon-family-server \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  timeout 30s python3 -m unittest discover -s tests \
  -p test_game_forge_recipe_contract.py -v
```

The test builds a temporary fictional `PokemonFamily` runtime from tiny
invented JSON recipes. Only one of two installed packs is active. It exercises
the real producer's stack selection, public export, file hash, consumer,
runtime cognition context, controller and existing fresh-chat authority gate.
A stub model returns a no-action decision; exact recipe text comes from the
snapshot. No server, input backend, model weights or training process is used.
An active-pack update changes the answer and revision; the old file pin is
rejected. Wrong game versions, expired chat and a replacement session do not
retain the recipe's chat authority. The test is skipped unless `GAME_FORGE_KIT`
is explicitly supplied; the focused target tests run independently.

The finished focused run passed 98 checks in 1.59 seconds with a peak resident
set of 72,132 KiB. The producer's 15 crafting checks took 0.16 seconds and
26,576 KiB. Both ran as small single-process checks with numerical-library
thread limits. See [the verification receipt](pack_recipe_target_verification.json)
for tested source hashes, baseline failures and exclusions.

## Remaining real-world checks

The catalog still uses the existing configured `PokemonFamily` snapshot and
exact BDS version check. It does not discover which world/pack stack a desktop
client has joined, re-export automatically, or prove an in-memory snapshot is
still the live server revision. A next bounded improvement should bind the
existing producer snapshot to verified live session/active-stack identity and
invalidate old evidence when that binding changes.

That bounded contract and offline invalidation logic are now implemented in
[Active recipe scope](ACTIVE_RECIPE_SCOPE.md). The first target verification
receipt above remains a historical checkpoint; live observer acceptance is
still outstanding and configured snapshots alone no longer admit live advice.

No actual desktop join, chat typing, crafting, general model quality or skill
learning was established here. Production services deploy on the desktop;
service ownership, world updates and live acceptance remain separate.
