# Bedrock-on-Linux Runtime

## Reference runtime

The primary runtime is the Windows Bedrock client launched by **BedrockOnLinux** through WineGDK/UMU on Linux. Java is optional and does not define the default lifecycle.

Minecraft AI discovers BedrockOnLinux through its managed data layout, including `BOL_HOME`, `compatdata/pfx`, the active `content` build and running `Minecraft.Windows.exe` processes. The active Bedrock version is read from BedrockOnLinux's selected build metadata when available.

## Why a separate display exists

The agent must not compete with the operator for the Linux desktop keyboard and pointer. The reference implementation therefore launches BedrockOnLinux inside a dedicated nested X server (Xephyr). The host desktop and Minecraft live on different X servers:

```text
host desktop DISPLAY=:0
  operator keyboard/mouse

nested Bedrock DISPLAY=:70 (example)
  Minecraft.Windows.exe under WineGDK
  AI XTEST keyboard/mouse
  window-scoped capture
```

`IsolatedX11InputBackend` refuses to open if its display resolves to the same X server as the operator's `$DISPLAY`. There is no silent host-global input fallback.

Xephyr is the conservative reference backend because the isolation boundary is explicit and testable. A nested Gamescope/Xwayland backend can replace it for lower-copy GPU capture after it passes the same isolation and failure tests.

## Installation

Install the Python package with the Bedrock/Linux extras:

```bash
python -m pip install -e '.[bedrock-linux,vision,knowledge]'
```

The host also needs:

- BedrockOnLinux with a licensed Bedrock installation;
- Xephyr (`Xephyr` executable) for the reference isolated display;
- a working X11/Xwayland host environment capable of displaying Xephyr;
- Vulkan/graphics support required by BedrockOnLinux.

Run:

```bash
minecraft-ai install
minecraft-ai doctor
```

`doctor` reports the selected Bedrock version, Wine prefix, running game processes, Xephyr availability, optional Python modules, emergency-stop state and managed agent/session state.

## Starting a session

Launch Bedrock in an isolated namespace:

```bash
minecraft-ai bedrock launch
```

The default nested resolution is 1920x1080 and Weston is presented fullscreen
on its host output. Fullscreen presentation is important: a nominal 1920x1080
host window loses drawable pixels to shell chrome and Bedrock then clips the
bottom of its 1920x1080 backbuffer, including part of the hotbar. It can be
changed explicitly:

```bash
minecraft-ai bedrock launch --width 1920 --height 1080
```

Use `--windowed` only for manual debugging where a reduced and potentially
clipped observation surface is acceptable. Autonomous play and trajectory
recording should use the fullscreen default.

Before arming live control, confirm that the dashboard frame includes all
hearts, hunger icons, and all nine hotbar slots. A partially clipped HUD is not
a valid perception or trajectory-recording surface.

Sign in/select a world through the nested Bedrock window normally. Then start the agent:

```bash
minecraft-ai run --live --role generalist
```

`run --live` performs these steps:

1. verify the emergency-stop latch is clear;
2. start/reuse the independent supervisor;
3. verify the managed nested Bedrock session;
4. find the Minecraft window only on that nested display;
5. attach the isolated XTEST backend to that window;
6. issue a short-lived motor capability lease;
7. enter supervisor `RUNNING` state;
8. spawn the independent realtime agent process;
9. capture the Bedrock window and begin the 20 Hz player loop;
10. renew the motor lease only while the runtime remains healthy.

## Stopping

Normal stop:

```bash
minecraft-ai stop
```

This stops the realtime agent first and then the supervisor.

Stop the nested Bedrock session separately with:

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

`IsolatedX11Capture` connects to the nested X server and resolves the selected Minecraft window geometry. `mss` captures only that window region into BGRA frames. Capture timestamps must be monotonic and stale frames are fatal to the motor runtime.

The fast path does not wait for a VLM. Semantic vision runs asynchronously and merges typed facts/tracks/chat observations into the perception blackboard.

## Input

`IsolatedX11InputBackend` uses XTEST directly against the nested X server. Commands are ordinary gameplay semantics:

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
- kill Xephyr -> capture/input fail closed;
- stale/frozen capture -> runtime faults supervisor;
- malformed/replayed motor actions -> lease revokes;
- suspend/resume -> no persistent held state;
- emergency-stop while moving -> process/control path terminates;
- reboot recovery -> no automatic live re-arm.

A future `minecraft-ai hardware-test` command should automate as much of this matrix as possible, but human observation is still required for the key assertion: **agent input never leaks to the host desktop**.
