# Bedrock-on-Linux Runtime

## Reference runtime

The primary runtime is the Windows Bedrock client launched by **BedrockOnLinux** through WineGDK/UMU on Linux. Java is optional and does not define the default lifecycle.

Minecraft AI discovers BedrockOnLinux through its managed data layout, including `BOL_HOME`, `compatdata/pfx`, the active `content` build and running `Minecraft.Windows.exe` processes. The active Bedrock version is read from BedrockOnLinux's selected build metadata when available.

## Why a separate display exists

The agent must not compete with the operator for the Linux desktop keyboard and pointer. The production launch uses headless Weston with a dedicated virtual seat and private Xwayland. There is no host window or connection to the host compositor's input seat:

```text
host desktop DISPLAY=:0
  operator keyboard/mouse

headless Weston + virtual seat, Bedrock DISPLAY=:70 (example)
  Minecraft.Windows.exe under WineGDK
  AI XTEST keyboard/mouse
  window-scoped capture
```

`IsolatedX11InputBackend` requires a different display and a verified live headless compositor, exact virtual-seat module, and matching managed session. It rechecks that provenance before lease binding and positive input. Releases remain available when isolation checks fail. There is no silent host-input fallback.

The earlier nested Wayland/Xephyr design scoped outgoing AI input but still forwarded the operator's physical input into the private display. Those sessions remain inspectable and stoppable; autonomous input now refuses them. The persistent launcher holds an existing unqualified session instead of using failed startup attempts to restart it.

Bare headless Weston has no keyboard seat and did not retain Xwayland focus in qualification. The small packaged virtual-seat module supplies keyboard/pointer capabilities without physical devices or parent input transport. It is pinned to Weston 13.0.0. Its source, binary and loaded mapping are checked; `xwayland-*` device names alone do not establish the source of input.

This prevents ordinary host desktop mouse, keyboard and focus changes from entering the game through the compositor. It is not a security boundary against programs running as the same Linux user or root. Separate OS credentials and access controls are required for that threat model. Operator pause/stop remains outside the game input path.

## Installation

Install the Python package with the Bedrock/Linux extras:

```bash
python -m pip install -e '.[bedrock-linux,vision,knowledge]'
```

The host also needs:

- BedrockOnLinux with a licensed Bedrock installation;
- Weston 13.0.0 and Xwayland, with headless EGL/OpenGL support;
- a C compiler and `libweston-13-dev`, `libwayland-dev`, `libxkbcommon-dev`, and `libpixman-1-dev` to build the packaged virtual-seat module;
- a private `XDG_RUNTIME_DIR` for the session sockets;
- Vulkan/graphics support required by BedrockOnLinux.

Run:

```bash
minecraft-ai install
minecraft-ai doctor
```

Build the module once with `python -m minecraft_ai.platforms.weston_seat`. Artifacts are kept outside the repository in the application's private data directory. Source/version changes require rebuilding and qualification before a replacement game session is used.

`doctor` reports the selected Bedrock version, Wine prefix, running game processes, optional Python modules, emergency-stop state and managed agent/session state. `status` separately reports verified input isolation; a running process is not proof of isolation.

## Starting a session

Launch Bedrock in an isolated namespace:

```bash
minecraft-ai bedrock launch
```

The default virtual output is exactly 1920x1080. Host decorations and focus do
not change its size. View the game through the existing captured-frame dashboard.
The virtual resolution can be changed explicitly:

```bash
minecraft-ai bedrock launch --width 1920 --height 1080
```

The retained `--fullscreen/--windowed` option does not create a host window in
headless mode. A previously running nested session is preserved by `launch`;
changing its compositor requires a deliberate stop and replacement after any
state-preservation experiment is complete.

Before arming live control, confirm that the dashboard frame includes all
hearts, hunger icons, and all nine hotbar slots. A partially clipped HUD is not
a valid perception or trajectory-recording surface. `run --live` enforces this
as a fail-closed actuator interlock: it will not attach or issue a gameplay
lease until an isolated capture contains the full survival HUD or a recognized
death, away, or inventory overlay that the agent can recover from. This detector
is only a launch-safety interlock; it is explicitly not a
semantic perception source or training label.

## Private-LAN operator dashboard

The dashboard stays loopback-only by default. To let trusted devices on the
same home LAN watch the live frame and submit operator instructions, bind its
guarded LAN mode:

```bash
minecraft-ai dashboard --host 0.0.0.0 --port 8765
```

Open `http://<computer-name>.local:8765/` from another device. Avahi/Bonjour
must be running for the memorable `.local` name; the machine's private IPv4
address is the fallback. The checked-in
`systemd/minecraft-ai-dashboard-live.service` uses this guarded LAN mode.

If Avahi reports `Local name collision` while publishing the computer's LAN
address, an existing static alias can already own its reverse record. The
optional `systemd/minecraft-ai-operator-mdns.service` advertises the computer's
existing `.local` name without a reverse record, preserving those aliases.
It discovers the private IPv4 address of a physical default-route interface
and refreshes the advertisement if the address changes. Install that user
unit and enable it alongside the dashboard; it needs `avahi-publish-address`
from `avahi-utils`. This does not restart the game or change its input path.

LAN mode is deliberately not an unrestricted public listener. Every request
must come from loopback, an RFC1918 subnet, or a link-local/IPv6 unique-local
address. Numeric `Host` headers must exactly match the socket destination, and
named access is limited to `localhost`, the machine's own mDNS name, and any
explicit `--allowed-host <name>.local` aliases. Mutation requests retain the
JSON-only and same-origin checks that prevent browser form CSRF and DNS
rebinding. There is no user authentication or TLS, so expose it only on a
trusted private LAN; do not port-forward 8765.

If a host firewall is enabled, allow TCP 8765 only from the directly connected
private subnet rather than opening the port globally.

Live camera control also requires an exact-version, machine-local measured
profile at:

```text
~/.local/share/minecraft-ai/calibrations/bedrock-camera-<version>.json
```

The profile is accepted only when the active Bedrock version, in-game mouse
sensitivity, field of view, and configured per-axis policy scales match. Yaw
and pitch are measured independently: WineGDK/Bedrock must not be assumed to
map both axes through one scalar. On the first attachment to a physical game
window, the supervisor uses a one-use mouse-only lease to send pitch homing commands and
establish a command-origin estimate. Motion is paced across Bedrock input frames
to reduce event coalescing. Completion and accepted counts do not verify the
physical horizon; that retained-image qualification remains open. Reattaching to
the same physical target preserves the command origin; changing targets
invalidates it, as does starting a new calibration attempt.

For a new managed build, install the optional computer-vision extra and run
`minecraft-ai bedrock calibrate-camera` while the client is in a stable world
with a complete survival HUD. The command requires an unarmed `SAFE_IDLE`
supervisor and uses reversible, mouse-only 24/48/96-count probes in each axis.
It estimates each angular delta from the active field of view and RANSAC-filtered
image features, checks linearity, restoration error, and agreement with the
configured policy, then writes the version-bound profile and a private JSON
receipt containing probe values and frame hashes. The receipt is written before
the profile becomes visible to the persistent launcher. No screenshots are
retained. The reported full-turn and pole-to-pole spans are extrapolated from
the local gain; they are not direct 360-degree or pitch-limit sweeps. Physical
horizon qualification remains separate.

```bash
uv pip install --python .venv/bin/python -e '.[camera-calibration]'
minecraft-ai bedrock calibrate-camera
minecraft-ai bedrock camera-ready
```

If BedrockOnLinux Doctor blocks a fresh GPU launch, persistent startup stops
instead of acknowledging the incident automatically. Inspect Doctor and the
graphics driver first. Only an operator, after that review, may request the
launcher's guarded `bedrock-on-linux doctor --acknowledge-gpu-crash` action.
An already-live managed client does not need a fresh-launch Doctor check.

Observe the headless client through the captured-frame dashboard. For a configured
local BedrockConnect server, `minecraft-ai bedrock navigate` performs bounded,
screenshot-bound menu navigation (see `--help` for exact server selection). It is
not a general sign-in flow, and there is no nested host window for normal physical
keyboard/mouse interaction. Required resource downloads need a recognized dialog
heading, explanatory body, caption and wide green control. Content Log History
can be closed only after three positioned labels and its pixel-font close icon
are verified; the navigator never clears the log. Loading captions authorize
waiting only. Unknown screens stop further input and may be retained with
`--evidence-dir` in a new private directory.

When BedrockConnect lists multiple local servers, set
`MINECRAFT_AI_SERVER_NAME` to the exact intended entry in the persistent
service's local environment. The launcher passes it as `--server-name` and
allows up to 300 seconds for an initial pack-backed join. Without the setting,
the navigator still handles a single-server list but refuses an ambiguous one.
Keep the selected server name in the local unit/drop-in, since different hosts
can use different BedrockConnect menus.

The 28 September desktop diagnostic accepted the server pack, reached survival,
and opened the stock starter menu with one ordinary Compass use. The menu stayed
visible beyond 40 seconds without a selection. This establishes the desktop
render path only; it does not resolve the separately reported Switch crash,
prove autonomous gameplay, or authorize bypassing deployment-specific world
restrictions. The diagnostic client was then stopped with operator pause latched.

The same day's follow-up verified server builds 103 and 104 through fresh
required-pack downloads. Build 103 retained the full starter menu for 288 seconds;
its flushed log had no Molang or animation errors. Build 104's on-demand menu
construction preserved the nine regions and starter model. Separate screenshot-
bound clicks changed Bulbasaur to Charmander and selected Johto/Chikorita. The
top-right close click did not dismiss the menu in the retained three-second
observation, so close behavior remains unresolved. No starter was confirmed and
no movement or attack was sent. Build 104's flushed log had zero UI, Molang and
animation errors, while 16 fossil-geometry warnings remained. These desktop
checks do not establish Switch stability, Pokémon skill or autonomous safety.
The client was stopped, its wrapper thawed and operator pause retained.

Once the client is in-world or on a supported recovery
overlay, start the agent:

```bash
minecraft-ai run --live --role generalist
```

`run --live` performs these steps:

1. parse options, including the shared `--capture-source [pipewire|x11]` choice,
   before any runtime mutation, then verify emergency-stop/operator-pause latches;
2. start/reuse the independent supervisor;
3. verify the managed headless Bedrock session and virtual-seat input isolation;
4. find the Minecraft window only on that private display;
5. verify a complete survival HUD or supported recovery overlay from the isolated capture;
6. attach the isolated XTEST backend to that window;
7. validate the exact-version mouse profile and establish/preserve the command-origin estimate;
8. issue a short-lived motor capability lease;
9. enter supervisor `RUNNING` state;
10. spawn the independent realtime agent process;
11. capture the Bedrock window and begin the 20 Hz player loop;
12. renew the motor lease only while the runtime remains healthy.

## Stopping

Normal stop:

```bash
minecraft-ai stop
```

This stops the realtime agent first and then the supervisor.

Stop the managed Bedrock session separately with:

```bash
minecraft-ai bedrock stop
```

Emergency stop:

```bash
minecraft-ai emergency-stop
```

Emergency stop latches persistent state and terminates the registered realtime-agent and supervisor process groups without relying on cognition or normal agent IPC. The system refuses to start while the latch is present.

Reset only after the processes are stopped:

```bash
minecraft-ai reset-emergency-stop
```

## Capture

`--capture-source` accepts only `pipewire` (default preference) and `x11`. Unknown
values fail CLI parsing before configuration writes, supervisor startup, capture,
attachment, calibration or arming. The child parser and capture factory use the
same supported-source definition. A valid PipeWire preference selects X11 on the
headless private display; Mutter/PipeWire requires an exact host-monitor binding
and explicit host capture permission. Host capture does not authorize autonomous
host input.

`IsolatedX11Capture` resolves the selected Minecraft drawable on the private X
server. It tries window-targeted XGetImage, with scoped `mss`/root-image fallbacks,
and validates complete content geometry before returning BGRA frames. Capture
timestamps must be monotonic and stale frames are fatal to the motor runtime.

The fast path does not wait for a VLM. Semantic vision runs asynchronously and merges typed facts/tracks/chat observations into the perception blackboard.

## Input

`IsolatedX11InputBackend` uses XTEST directly against the verified private X server. Commands are ordinary gameplay semantics:

- key down/up;
- mouse button down/up;
- relative look motion;
- in-game chat typing.

Every backend action requires a current supervisor motor lease. Lease identity/expiration are checked both by `MotorGate` and by the X11 backend itself.

## Hardware qualification

CI can test lifecycle, lease behavior, planning, persistence and fail-closed contracts, but it cannot prove real Wine/X11 isolation on GitHub-hosted runners.

Before marking the Bedrock backend hardware-qualified, run on the target machine and verify at least:

- hold `W`, stop agent -> no held movement remains;
- hold attack/use, stop agent -> no held button remains;
- use a host editor while the agent moves -> no agent keystrokes appear outside Minecraft;
- move/click on the host desktop -> agent remains bound to Minecraft;
- kill realtime agent -> lease expires and input releases;
- kill supervisor -> backend loses authority and input releases;
- close/crash Minecraft -> target validation fails closed;
- kill headless Weston/Xwayland -> capture/input fail closed (retained legacy Xephyr sessions also need fail-closed teardown);
- stale/frozen capture -> runtime faults supervisor;
- malformed/replayed motor actions -> lease revokes;
- suspend/resume -> no persistent held state;
- emergency-stop while moving -> process/control path terminates;
- reboot recovery -> no automatic live re-arm.

A future `minecraft-ai hardware-test` command should automate as much of this matrix as possible, but human observation is still required for the key assertion: **agent input never leaks to the host desktop**.
