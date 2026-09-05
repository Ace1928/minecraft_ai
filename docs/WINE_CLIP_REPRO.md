# Disposable Wine clipping comparison — 2026-09-06

The installed Wine build reproduces the logical/physical cursor-confinement
split without Minecraft, model inference or a GPU. This strongly supports
X-focus ownership as a causal boundary. It does **not** establish repaired
live camera motion, host-input independence or gameplay readiness.

## Experiment and results

Two fresh Wine prefixes ran sequentially in separate headless-pixman Weston
sessions with the virtual-only seat. Each bounded user service had a one-CPU
quota, 2 GiB memory limit, zero swap and private devices; the runner refused
to launch Wine if `/dev/dri` was present. Existing game, model, compositor,
paused supervisor and stopped launcher identities remained unchanged.

The same compiled helper created its own ordinary window and established
Win32 foreground/focus. XRes verified application and Explorer ownership.
The runner selected focus only inside its newly owned X server. Each helper
then made one `ClipCursor((300,250,300,250))` request, sampled state at roughly
10 Hz for two seconds, released its clip and exited successfully. No raw-input
registration, mouse/keyboard injection, capture call or explicit cursor warp
was used. Wine's own clipping implementation may move the scratch pointer.

| Evidence | A: Explorer-owned desktop X focus | B: application-owned X focus |
| --- | --- | --- |
| Win32 foreground/active/focus | Application | Application |
| Clip request | Success; logical point retained | Success; logical point retained |
| Physical X pointer, 17 snapshots | `(100,100)` | `(300,250)` |
| Clip window | Unmapped, unchanged `(0,0,1,1)` | Viewable, `(300,250,1,1)` |
| Post-call Win32 samples | 19 retain cursor `(100,100)` | 19 report cursor `(300,250)` |
| Existing Wine trace | Physical `(100,100)`, server `(300,250)` | `grab_clipping_window` reaches clipping; cursor positions agree |

B recorded the application's processed `NotifyNormal` FocusIn before the
request. Its first X snapshot pairs with the last **pre-call** Win32 row;
only the remaining 16 snapshot/row pairs are post-call on both sides.
These are asynchronous observations, not 17 simultaneous API pairs.

Both runs report the same RpcSs-startup and synthetic ConfigureNotify
warnings. Their presence in successful B means they do not explain this
contrast. Neither run reports a keyboard-grab refusal at the clip request.

Both helpers completed and released their own clipping successfully. A's
runner nevertheless exited 1 because it closed its observer connection
*after* stopping its compositor. Its original failure receipt is retained;
the already-written measurements are not a clean whole-harness pass. Cleanup
ordering was corrected for B, which exited 0 with no cleanup errors. Both
disposable process groups were retired; neither touched the managed session.

## Interpretation and next boundary

In the inspected [pinned WineGDK mouse implementation](https://github.com/Weather-OS/WineGDK/blob/75637b674e1f191e65753663c4c0c32bea05ba6e/dlls/winex11.drv/mouse.c),
`grab_clipping_window()` returns before initializing/moving/mapping its clip
window when process-local X focus does not belong to that process. This is
consistent with A's unchanged initial geometry. The preserved live game's
unmapped window instead retains an earlier configured geometry; that does
not mean it is still physically confining the pointer.

The A/B contrast supports a clipping-local eligibility correction, not a
global relaxation of focus checks. A candidate must require legitimate
virtual-desktop X focus plus current-process Win32 foreground ownership.
The separate `keyboard_grabbed` ordering concern is **not reproduced** by
this experiment and remains a separate hypothesis. No Wine patch was installed.

Any future negative test must qualify the process executing the driver,
not merely the caller of the public API: Wine's
[cursor-message dispatch](https://github.com/Weather-OS/WineGDK/blob/75637b674e1f191e65753663c4c0c32bea05ba6e/server/queue.c#L494)
routes clipping notifications to the foreground window. A background process
calling `ClipCursor` can therefore cause the foreground process to execute
the driver. Treating that as a background-driver permission bypass would
test the wrong boundary.

## Retained local evidence

### Built candidate, not installed

The [generic source patch](../patches/winegdk/README.md) now builds as the
actual native x86-64 X11 driver, not merely a stub. Both matched builds retain
the installed native5 client-surface patch and reuse the unchanged installed
`ntdll.so` and `win32u.so`; no core, server, or Windows module was rebuilt.
The candidate incremental build recompiles only `mouse.c` and relinks.
Its source is byte-identical to the pinned source plus the published patch.

Both CPU-only, device-isolated build services completed successfully. The
baseline consumed 34.486 CPU seconds with a 219.8 MiB reported peak; the
candidate consumed 1.906 CPU seconds. Both use configuration hash
`ab0f2db5d89255dee084faea2ee746b18b6a50fb1dc69500742b4fdc1b928083`, retaining
XI2, Xi, OpenGL, Vulkan and the detected X extensions. Neither ELF contains
RPATH/RUNPATH, and their direct dependency lists match each other. The older
installed ELF additionally lists `libpthread` and `libdl`; these rebuilt
binaries are matched local controls, not bitwise copies of that installed build.

| Local artifact | SHA-256 |
| --- | --- |
| Rebuilt baseline `winex11.so` | `fb41e3ff89429fa2c6abf952dec646cb5b264cd5b3dd636403c4b975d33ff3de` |
| Candidate `winex11.so` | `5cba094bc5fc8114771923c3b10b049a28068b74e282e41dd4dfeb95e1304621` |
| Published source patch | `5f6b18a3c9bb3b49e9db002a81586789ad817abaa5ada2099de16826055a8b2a` |

The build fixture is `/tmp/minecraft-wine-driver-build-svfZwN/`. The resulting
ELF requires GLIBC 2.38, so it is not a portable Bullseye release. The exact
extracted predicate passes 15 native stub checks, including rejection when
another process owns foreground. These tests model the call-site boundary;
they do not establish the full driver's physical behavior. The separate
disposable execution result follows; live Bedrock qualification remains open.

### Matched driver execution

All four subsequent comparison services and helpers completed successfully,
with no cleanup errors. Each ran in its own fresh prefix and headless-pixman
session inside a read-only mount namespace. Only the selected X11 driver ELF
was overlaid at an installed-engine path. The writable case directory and
Wine server sockets were private; the original installed file was unchanged.

Before focus selection and again before the clip request, both the helper
and Explorer passed executable-map path/device/inode checks and process-root
hash checks for the selected driver and unchanged core/Windows modules.
This establishes which binary executed despite its apparently installed path.

| Loaded driver / X focus | Physical observations, 17 snapshots | Post-call Win32 cursor, 19 rows |
| --- | --- | --- |
| Rebuilt baseline / Explorer | Unmapped clip; pointer `(100,100)` | One transient `(300,250)`, then 18 at `(100,100)` |
| Rebuilt baseline / application | Mapped clip; pointer `(300,250)` | All `(300,250)` |
| Candidate / Explorer (P1) | Mapped clip; pointer `(300,250)` | All `(300,250)` |
| Candidate / application (P3) | Mapped clip; pointer `(300,250)` | All `(300,250)` |

The candidate P1 trace reaches `grab_clipping_window` in the helper's
Windows process/thread; no keyboard-grab refusal appears. The baseline's
transient logical cursor row does not override its independently observed
physical pointer. Sampling is asynchronous; the first X snapshot in each
case pairs with a pre-call Win32 row. No claim of simultaneous API pairs is made.

The new fixture is `/tmp/minecraft-wine-driver-repro-v2-fkvOcq/`. Its helper
source hash is `d9698ce038ec7a5d5f553eaabb37d3bd76e7432cad658d63140891230a4a0ef1`
and binary hash is `1b19799dd29a5db63b0464ff23722939499590165fe161d4f21856834a89f5ca`.

| Case receipt | SHA-256 |
| --- | --- |
| Baseline A | `57820ac72828a77942946430430417eadc5196d9345ffa8120c1150478b66927` |
| Baseline B | `70c64541e14a40bc54e6add58009379699dc454c1dcf1c05f67d531afc3b2f04` |
| Candidate P1 | `57ec2abc501b7720ef51df722ac7f999771e2fff6fec45c8d21fcf3b3c075f22` |
| Candidate P3 | `c3f2c6fd568f67e27bd9a36971b5cb58fe8ff3e8b7d66ccd70e7824fa003fddc` |

An earlier overlay fixture, `/tmp/minecraft-wine-driver-repro-ZUyrgr/`,
failed before Wine started because its private `/tmp` lacked `.X11-unix`.
That failure is retained, not reported as a Wine result or erased for retry.
The corrected experiment used a separate fixture and fresh case directories.

This qualifies the clipping correction for these disposable conditions.
It does **not** qualify sustained relative camera motion, focus transitions,
live game input, complete host-input exclusion, or a production deployment.
The actual Minecraft, model, original compositor, paused supervisor and
stopped launcher remained preserved. No Wine replacement was installed.

### Original installed-build observations

Artifacts remain under `/tmp/minecraft-wine-clip-repro-wppvRA/`; they are not
portable public fixtures. These hashes identify the original measurements:

| Artifact | SHA-256 |
| --- | --- |
| `run-a/receipt.json` | `88a10b9264a413364ff73e03ce69677b6d977fa00f45e52494bd24b71d46886b` |
| `run-b/receipt.json` | `2c6540f4b915295754194f4e6adfacb9f90d4d534d0cce77a9c5c6b35fea35bd` |
| `case-a.jsonl` | `639b11a12c1f63a36f68aa7adf9bf0668e28180772febf5959ac1f0aaaa93f99` |
| `case-b.jsonl` | `16feada4bcbecf41050af40fc7b97d0b45ff99d988876beabb97d01d1ae02d3c` |
| `run-a/wine.log` | `5446dae6ce4c38952932118f309929b29f44e61512c90b047e7676f577d1b732` |
| `run-b/wine.log` | `83dcf8a757eb60cc8fa98d7c3406fafe7bb61a066c40067a52f190b71bded1ee` |
| Helper source | `27070a5e70f7f55122d2bbffbeafa5228a8fbc584102e2a7d607fa434826022d` |
| Helper binary | `20012757675c74270caf19a32bbcec526f6d9feada0701837becf9908995bd59` |
| Installed `winex11.so` | `0be11c0993eafd4cc3a7e171b85cd980b0df2c1eb595718eadc712522c4f420a` |

The receipts also pin the installed loader, wineserver and Windows driver.
The research source revision is not asserted to be a reproducible build
attestation for those installed binaries.
