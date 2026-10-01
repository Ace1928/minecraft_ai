# Recipe observation delivery and desktop activation

The agent previously had an active recipe fact contract but no delivery path
from a host observer. `RuntimeConfig.pack_recipe_observation` now opts into a
bounded local file reader in the existing agent runtime. It defaults to null
and requires the existing reviewed catalog path and exact file pin. The reader
creates no thread, capture, socket, model or service. This change does not
activate a writer or modify deployment configuration.

The actual `PublishedFrameCapture.owner` selects the client; the desktop's
list of Wine windows is not a selection. `owner_manifest` exposes the same
unchanged fields already used in the private frame cache: `pid`, `started_ns`,
`proc_start_ticks`, `command_sha256`, `display`, `window_id`, `instance_id`,
`allow_host_capture`. The reader checks that exact descriptor against the
current agent-process file and existing verified Linux process identity. A
missing owner remains unknown; it cannot fall back to a discovery candidate.

An admitted same-host writer may atomically replace a private regular file
(current user, mode 0600, no symlink), containing exactly these fields:

| Envelope field | Required meaning |
| --- | --- |
| `schema_version` | integer `1` |
| `boot_id` | current host Linux boot ID |
| `capture_owner` | exact existing selected owner manifest, including field types |
| `observed_ns` | original monotonic evidence time, after owner start, never future |
| `expires_after_ms` | integer `1..5000`, measured from that original time |
| `identity` | existing active recipe scalar's object, documented in `ACTIVE_RECIPE_SCOPE.md` |

The envelope is at most 8192 bytes; identity JSON remains at most 2048
characters. Duplicate fields, extra fields, null/unknown/configured-only
identity, wrong instance/version/world/hash/pin, changed owner/boot, stale or
future timestamps, absent files, and older replayed observations revoke the
producer's fact on the next capture cycle. Re-reading identical evidence never
extends its lifetime. Valid revocation envelopes advance the replay watermark;
equal-time ambiguity stays revoked until newer evidence arrives. Equal
timestamps cannot change identity or TTL. Capture
timeout/staleness also revokes it. The reader reuses catalog scope validation
and the existing perception merge/removal boundary; it does not create another
recipe or world-knowledge store. Telemetry reports only delivery state/reason,
without adding private owner fields. Existing consumption/admission checks
invalidate prepared responses after revocation or a server session change.

## Activation requirements for the desktop owner

1. Obtain admitted, read-only **engine evidence** of the loaded world's BDS
   version, ordered behavior-pack UUIDs/versions and effective content matching
   the reviewed export. `world_behavior_packs.json`, installation metadata,
   startup readiness and a guard heartbeat alone are configuration/session
   evidence. They cannot prove the effective loaded recipe stack. The documented
   [World.getPackSettings API](https://learn.microsoft.com/en-us/minecraft/creator/scriptapi/minecraft/server/world?view=minecraft-bedrock-stable#getpacksettings)
   returns setting values, not an ordered loaded-pack enumeration.
2. Prove the **selected captured client is currently connected** to that owned
   server run. Bind this proof to the exact owner generation and revoke it on
   disconnect, ambiguity, client replacement or server restart. A matching game
   version or a player name on a server does not establish this correlation.
3. Use game_forge's private admin `recipe_observation.owned_server_session`
   where available. It binds the actual manager-owned Popen object to Linux
   boot/process-start identity and a unique launch token. It is always partial:
   the status reports `state: unknown` and missing engine/connection/artifact
   proof. It does not publish `observed_active`. An external/unverifiable server
   has no descriptor; do not adopt a PID file as an owned launch.
4. Separately review a warning-free public `crafting.py` export outside the
   world runtime. Record its exact `catalog_sha256` and configured scope hash;
   match both to actual loaded evidence. Preserve all warnings. Configure the
   catalog path, pin and optional observation file only after that review.
5. Implement/review the host writer using the admitted evidence source, original
   evidence timestamps and atomic file replacement. Missing/ambiguous proof
   must remove the file or write null identity rather than retaining success.
   The provenance label operates inside the existing host boundary; it is not
   a credential or a permission to act.
6. Obtain serving-owner admission before deployment and actual acceptance:
   verify an exact world-specific question; observe invalidation through a
   legitimate session transition and evidence expiry; check no old answer is
   committed/delivered. Coordinate a safe transition around real players.
   Do not restart a populated world to validate this source change.

Current correlated desktop evidence has no loaded-stack/client-connection
attestation and no configured catalog/pin. Activation therefore remains
blocked; this pass intentionally cannot truthfully produce a positive live
identity. Pure fixtures qualify delivery/revocation and fictional advice only,
including the actual game_forge export-to-context/controller path. They do not
prove real crafting, model quality or learning. Production stays on the desktop.

Disabling the optional field returns to the existing unconfigured reader path;
without another valid active fact, recipe advice stays explicitly unavailable.
Normal source rollback can revert this change; no world/server/DB schema or
security setting has changed.
