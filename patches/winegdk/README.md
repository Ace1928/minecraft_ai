# Wine virtual-desktop clipping candidate

`virtual-desktop-clip-owner.patch` is a generic Wine X11 driver source patch
for WineGDK commit `75637b674e1f191e65753663c4c0c32bea05ba6e`. It contains no
model, private adapter, account, or runtime configuration.

The patch keeps ordinary process-owned X focus unchanged and adds one
clipping-local exception: Wine's virtual-desktop X window has focus and the
Win32 foreground window belongs to the current process. Failed focus queries,
missing thread/display data, and foreign foreground ownership do not admit
the exception. The existing desktop-thread exclusion and keyboard-grab refusal
remain intact; the global focus helper and `event.c` are unchanged.

This does not isolate host input by itself. It also does not authorize input,
clear safety stops, or provide an automatic installer. Nothing in the agent
loads or applies this patch automatically.

## Build and qualification boundary

Retain BedrockOnLinux native5 patch
`0005-winex11-use-client-surface-origin.patch` when building both a matched
baseline and candidate. Its patched `dlls/winex11.drv/init.c` SHA-256 is
`c0aa5c68ec1b41d521021dcb3d9aaf596ae73724efa87615148fb87a1bc7f72f`.
Omitting it would silently revert the existing client-surface geometry fix.

The relevant artifact is the native x86-64 ELF `winex11.so`, not the PE
`winex11.drv` or Winelib `winex11.drv.so`. Keep the exact pinned core libraries
and driver ABI: GDI 108, OpenGL 37, Vulkan 47. Do not replace a managed engine
or test against a live game as an installation shortcut.

The locally built experimental driver requires GLIBC 2.38 and is **not a
portable Bullseye release artifact**. Production packaging must independently
meet the project's GLIBC 2.31 ceiling; source compatibility is not binary
portability. No compiled driver is included here.

Before any deployment, use fresh disposable prefixes and isolated displays
for matched unpatched/patched driver tests. Verify the actually loaded driver
by device/inode and hash, not only its apparent pathname under a private mount
overlay. Preserve the original core-library mappings. A successful disposable
experiment is not live gameplay qualification or proof of complete host-input
isolation.

## Regression boundary

`tests/test_wine_clip_patch.py` always checks patch scope and source extraction.
When a native C compiler is available, it extracts the exact added predicate
from the patch and compiles it between the reviewed C fixtures, then runs 15
checks. The fixtures use only native stubs: no Wine, Xlib, display connection,
input registration, or injected input. Only the compile/run test is skipped
when no compiler is available.

Those checks cover ordinary and virtual-desktop ownership, invalid focus and
foreground states, a failed X query that nevertheless writes a root-window
output, and preserved desktop-thread/keyboard-grab guards. Rejections must not
reach modeled map/resize/grab operations. The surrounding call site is modeled,
not the complete Wine driver; these tests do not prove driver integration.

In particular, calling public `ClipCursor` from a background helper is not the
foreign-foreground negative control: Wine routes the physical clipping message
to the foreground thread. The negative predicate case must actually execute
with another process owning the foreground window.
