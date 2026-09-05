# Simulated input without host desktop interference

The production input path is private XTEST into Xwayland under headless Weston
13.0.0. A small virtual-seat module supplies keyboard and pointer capabilities.
It has no connection to physical devices or a parent compositor's input seat.
The game is viewed through captured frames in the operator dashboard.

The previous Wayland-nested compositor imported the host seat. Its Xwayland
pointer and keyboard devices shared the same master devices used by XTEST.
Consequently, using a separate `DISPLAY` prevented outgoing AI events from
reaching the host, but did not prevent incoming host mouse/keyboard/focus changes.

## Qualification and scope

`scripts/qualify_headless_input.py --gpu-animation`, run with the application's
Python environment, creates a disposable compositor and its own test window.
It never loads a managed game session or starts Wine. It retains a JSON report,
X input receipts and images, and cleans up only its owned processes.

The module and command passed 19 disposable-session checks: keyboard delivery
and a held-key interval, release, stable focus, exact relative raw/core motion,
button hold/release, pointer confinement, captured drawable updates, accelerated
rendering, changing GPU animation, restricted access and process cleanup, plus
live environment validation, exact Xwayland listener ownership and 30 read-only
production admission calls. A
separate interrupted run verified cleanup during a held-key phase. These are
display/input boundary results, not proof of correct Bedrock camera processing.

Bare headless Weston failed keyboard focus qualification. The virtual seat is
required; removing it is not an equivalent configuration. Its Xwayland wrappers
still have names beginning `xwayland-`, so names alone are not an isolation test.
The loaded module, executable command, process identity, source and binary
provenance are verified before admitting autonomous input. Admission also binds
the exact Xwayland child and process start time to every observed pathname and
abstract Unix listener for the selected display. A substituted display, reused
PID or listener owned by another server fails closed. Inherited parent-display
handles and Weston/dynamic-loader overrides are stripped at launch and rejected
when observed in the live compositor environment; this includes
`WAYLAND_SERVER_SOCKET` and `WESTON_MODULE_MAP`.

The 2026-09-06 disposable run measured 30 admission calls at median 2.55 ms,
nearest-rank p95 3.72 ms and maximum 4.28 ms. These are single-call observations,
not complete action-loop latency or a performance guarantee: an action can
revalidate more than once. The retained qualifier source SHA-256 was
`e278eb72bc138fc1ca45dee269a581ecd3b4e504973b0a9a2ef7338ce5d45e0d`.

Normal desktop use cannot enter this input path through the host compositor.
Programs with the same Linux credentials, privileged processes, and authorized
operator control endpoints remain outside that guarantee. Preventing deliberate
access by those actors requires a separate OS security boundary. Pause and stop
remain independent of Minecraft's virtual keyboard.

## Preserved Win32 state experiment

A coordinated window-free, registration-free five-second probe collected 50
Win32 samples and 53 X snapshots from the existing game. Minecraft retained
Win32 foreground, active, focus and capture ownership in every sample. Wine's
logical clip remained a point at `(565, 376)`, while both cursor queries returned
`(0, 83)`. X focus, client geometry, pointer hit-chain and button mask remained
unchanged in the sampled observations. The current cursor was inside the client
at its left edge; it was outside the reported logical clip.

This confirms logical clipping versus actual cursor disagreement in that
preserved session. It does not establish the exact internal cause of the large
camera response. No raw-input registration, injected pulse, focus change,
clipping setter or cursor warp was used for this probe. Failed helper startup
attempts were retained separately from the successful result.

`scripts/win32_state_probe.c` is the reusable three-second, at-most-10-Hz version.
Compile it as a GUI-subsystem helper using the matching Wine headers and loader;
it creates no window or message pump. Supply an explicit Minecraft Win32 HWND
and a new output path. It validates the title/handle and stable process/thread,
then samples `GetGUIThreadInfo(game_tid)`, foreground, clipping and cursor state.
Bracket its lifetime with the existing X queries without selecting new events.
Never use a new prefix or a different Wine build to inspect a live game.

The APIs report another thread's GUI state and the logical confinement rectangle;
see [Microsoft's GUI-thread API](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getguithreadinfo)
and [cursor clipping query](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getclipcursor).
Weston documents headless operation without physical input/output in its
[backend guide](https://wayland.pages.freedesktop.org/weston/toc/running-weston.html).

## Existing-session migration

An existing host-connected compositor cannot be changed to headless in place.
Its game client must be deliberately stopped and relaunched after the preserved
state investigation is complete. Session liveness stays separate from input
authority: the status/readiness checks reject unqualified input, while the
persistent launcher holds without navigation, calibration or automatic restart.
The world/server and model services do not need replacement for this change.

Before calling the gameplay route qualified, repeat the Win32/X state comparison
on the replacement session and measure Bedrock camera response and actual
movement from retained images. Synthetic XTEST delivery alone does not close
those gameplay gates or justify a Wine focus/clipping patch.
