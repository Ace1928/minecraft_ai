# Java Edition Scoped Input Bridge

The Java bridge is a deliberately small client-side component whose purpose is to attach a private virtual keyboard/mouse to **one Minecraft client instance**. It is not a world-state API and it is not the agent.

## Goals

- allow Minecraft AI to control one Java client without owning the operator's global desktop input;
- preserve human-style action semantics: key state, mouse deltas, mouse buttons, hotbar keys;
- expose frames and player chat only through explicitly negotiated capabilities;
- never expose block/entity coordinates, inventory internals, pathfinding or other privileged state in strict human-play mode;
- fail closed on expired lease, disconnect, invalid sequence, target mismatch or protocol error;
- release every held input locally inside the client on any failure.

## Trust boundary

```text
supervisor process
    |
    | authenticated loopback session
    | short-lived motor lease
    v
Java client bridge
    |
    | Minecraft input semantics only
    v
one Minecraft client instance
```

The bridge does not trust cognition directly. Only the supervisor can bind or renew a motor lease.

## Discovery and authentication

The bridge binds loopback only (`127.0.0.1` / `::1`) on an ephemeral port and publishes an instance-scoped endpoint descriptor containing:

- protocol version;
- host/port;
- random high-entropy session token;
- Minecraft version;
- process id;
- random instance id;
- loader/profile identifier where available;
- supported bridge capabilities.

The descriptor is stored with user-only permissions where the platform supports them. This protects against accidental cross-instance control; it is not intended to defend against a hostile process already running as the same OS user.

The Python client authenticates using the token and requires the returned `InstanceIdentity` to exactly match the instance selected by the supervisor.

## Motor lease replication

Before input is accepted, the supervisor sends a lease containing:

- lease id;
- supervisor session id;
- target instance id;
- expiry deadline;
- allowed action kinds;
- maximum sequence/action limits;
- first valid sequence number.

Both sides enforce the lease independently.

The bridge rejects and releases all held input when:

- no lease is active;
- the lease is expired;
- a lease targets another instance id;
- action sequence is stale, repeated or out of order;
- an action exceeds configured bounds;
- the authenticated connection closes;
- heartbeat/renewal expires;
- Minecraft begins shutdown;
- the client changes to a state where safe injection cannot be guaranteed.

## Input application

Implementation is version-adapter specific, but the semantic contract is stable:

- key down/up;
- mouse button down/up;
- relative mouse motion;
- bounded action deadline/sequence.

The bridge should update Minecraft's own input/key-binding state on the client thread rather than synthesizing global OS events. The exact adapter for each supported Minecraft/loader version must be tested because internal input APIs change between versions.

### Strict human-play mode

Strict mode permits only information a human client can normally perceive:

- rendered frames;
- visible UI/HUD/chat;
- input acknowledgements and bridge health.

It must not export hidden world state merely because a mod can access it.

A separate research/debug mode may eventually expose privileged state for labels/evaluation, but data from that mode must be explicitly marked and must never silently enter strict-agent observations.

## Frame path

Frames should avoid JSON/base64 copies. Preferred order:

1. shared memory/ring buffer for same-host operation;
2. local zero-copy/native buffer where a platform adapter supports it;
3. compressed loopback stream as compatibility fallback.

Control messages remain small authenticated JSON/CBOR-style records. Frame metadata includes monotonic sequence/timestamp so stale perception can fail closed.

## Chat

The bridge may emit chat text that appears to the player and may accept bounded chat-send requests. Chat is not motor authority and cannot extend a motor lease.

## Multiple instances

Every bridge instance has a different token and random `instance_id`. The supervisor chooses exactly one. Switching targets requires revoking the current lease and establishing a new authenticated bridge session.

No command is accepted solely because a window title happens to contain `Minecraft`.

## Release gate

A Java adapter cannot be marked `live_capable=true` until hardware tests demonstrate:

- the agent moves while the operator types/clicks normally in another application;
- operator input does not leak into the agent's virtual input state unexpectedly;
- agent input never appears in another application;
- closing the bridge connection while holding W releases W;
- killing cognition while holding attack releases attack;
- lease expiry releases all inputs without help from Python;
- target instance replacement/mismatch rejects control;
- pause and stop release inputs within the safety deadline;
- multiple Minecraft clients cannot be confused;
- suspend/resume and game shutdown recover safely.

These are release criteria, not optional polish.
