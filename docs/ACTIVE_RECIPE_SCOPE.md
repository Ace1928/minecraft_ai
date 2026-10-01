# Active recipe scope and invalidation

The existing pinned recipe catalog describes configuration. Matching the
client's Bedrock version does not prove which world, pack stack or server
session the player joined. The regression previously answered an invented
`blue:beacon` question without any active-world observation.

`game_forge` now adds `configured_recipe_identity` to its schema-v1 export:
`state` is always `configured_only`; `scope_sha256` hashes the world, BDS
version, recipe revision and ordered behavior-pack sources. The exact exported
file hash still covers all public metadata, names and resource-derived labels.
Neither hash, a path, world name in a client ID, nor a configuration read
attests active runtime state.

Live catalog use requires a separate host-observed `PerceptionFact` with key
`world.active_recipe_identity`, source
`game_forge:active-recipe-identity-v1`, confidence at least 0.99 and a positive
TTL at most 5000 ms. Its scalar value is a JSON string containing exactly:

| Field | Value |
| --- | --- |
| `schema_version` | integer `1` |
| `state` | `observed_active` |
| `world` | verified world name |
| `bds_version` | verified active BDS version |
| `scope_sha256` | matching configured scope hash |
| `catalog_sha256` | matching pinned file hash |
| `server_session` | observed owned server run/session identity |
| `instance_id` | exact current Bedrock capture/client identity |

The JSON is bounded to 2048 characters, strings to 256 characters and hashes
to 64 lowercase hexadecimal characters. Future observations, expired facts,
different sources, unverified states and malformed payloads stay unknown.
Known world/version/scope/file mismatches are explicit `mismatch` status.
Without a verified match, player and operator recipe questions get an explicit
scope refusal and no catalog recipe evidence. Existing offline `lookup`
remains useful for inspecting a reviewed artifact; live callers use
`lookup_live`. Old exports lacking configured identity fail closed for live use.

Exact operator recipe replies use the existing reply-only channel directly.
Player replies retain normal model planning and observed-chat admission.
The host privately binds decisions to the observed identity; that field is
absent from model output schemas. Runtime consumption rejects changed or
expired identities before operator acknowledgment, plan adoption or actions.
Chat delivery rechecks it too. Identical fresh heartbeats retain an answer;
a new server session invalidates old answers even if its catalog is identical.
A newly prepared answer may then use that new verified session.
Operator storage retries retain the identity. A bounded admission callback
rechecks it after acquiring the SQLite writer and before commit; changed scope
rolls back the status and response, leaving the question pending. Rejected
final publication restores both projected plan fields and the plan graph.
Recipe context for a different current question cannot be rebound merely by
sharing the same pack session.

The bounded tests exercise the real producer through temporary fictional
packs, public export/file pin, consumer, context, controller and admission
logic. They include stack/version/world changes, recipe updates, expiry of an
immutable cognition snapshot, malformed facts, a new session, stale context
rebinding and rejection before runtime consumption. No live game, model
weights, server, actual chat/crafting input or training is involved.

## Desktop acceptance still required

This pass defines and consumes the active observation contract; it does not
install a live observer or manufacture `observed_active` from configuration.
The desktop owner must correlate the actual client and owned server session
with verified active pack selection and a fresh export, then publish/revoke
this fact through the existing perception boundary. Reuse game_forge's guard
`run_id` where that owned-session correlation is established. The guard's
current readiness/world binding alone does not attest the active recipe stack.
The source label is provenance inside that host boundary, not a new
authentication mechanism or input permission.

Until this observer is present and the snapshot is re-exported/pinned, recipe
advice remains explicitly unavailable. No deployment configuration or world
was changed. Live acceptance requires the serving owner's resource admission;
production services continue to run on the desktop.
