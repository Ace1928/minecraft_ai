#!/usr/bin/env python3
"""Qualify simulated input on an owned disposable headless Weston display.

Run manually with the project's Python environment; this needs Weston 13, the
virtual-seat build dependencies (or an existing verified artifact), python-xlib,
Pillow, xinput, xhost, glxinfo and stdbuf. ``--gpu-animation`` also needs glxgears.
An extracted development-package tree can be passed with ``--include-root``.

The command never loads a managed game session or starts Wine. It removes host
display handles, uses a new private runtime directory, and targets only its own
test window. JSON, text receipts and PNGs remain in the printed output directory.
This hardware qualification is deliberately excluded from default pytest runs.
It proves the private X display route, not any game's handling of those inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minecraft_ai.platforms.bedrock_session import (
    _headless_compositor_environment,
    _terminate_spawned_process_group,
    _wait_for_weston_xwayland,
    _weston_command,
)
from minecraft_ai.platforms.bedrock_x11 import require_isolated_display
from minecraft_ai.platforms.weston_seat import (
    build_headless_seat_module,
    require_loaded_headless_seat,
)


def _focus_id(connection: Any) -> int:
    focus = connection.get_input_focus().focus
    return int(getattr(focus, "id", focus))


def _pointer(connection: Any) -> dict[str, int]:
    pointer = connection.screen().root.query_pointer()
    return {
        "x": pointer.root_x,
        "y": pointer.root_y,
        "child": int(getattr(pointer.child, "id", pointer.child)),
        "mask": pointer.mask,
    }


def _events(connection: Any) -> list[dict[str, Any]]:
    result = []
    while connection.pending_events():
        event = connection.next_event()
        result.append(
            {
                "type": event.type,
                "detail": getattr(event, "detail", None),
                "event": str(event),
            }
        )
    return result


def _capture(window: Any, path: Path) -> dict[str, Any]:
    from PIL import Image
    from Xlib import X

    geometry = window.get_geometry()
    capture = window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xFFFFFFFF)
    if len(capture.data) != geometry.width * geometry.height * 4:
        raise RuntimeError("test drawable is not the expected 32-bit image format")
    Image.frombytes(
        "RGB",
        (geometry.width, geometry.height),
        capture.data,
        "raw",
        "BGRX",
    ).save(path)
    return {
        "path": str(path),
        "width": geometry.width,
        "height": geometry.height,
        "sha256": hashlib.sha256(capture.data).hexdigest(),
        "unique_bytes": len(set(capture.data)),
    }


def _find_named(window: Any, name: str) -> Any:
    if window.get_wm_name() == name:
        return window
    for child in window.query_tree().children:
        found = _find_named(child, name)
        if found is not None:
            return found
    return None


def _raw_motion_receipts(text: str) -> list[dict[str, Any]]:
    receipts = []
    blocks = re.findall(r"EVENT type 17 \(RawMotion\)(.*?)(?=EVENT type|\Z)", text, re.S)
    for block in blocks:
        device = re.search(r"device:\s*(\d+)\s*\((\d+)\)", block)
        axes = re.findall(r"^\s*(\d+):\s*([-+\d.]+)", block, re.M)
        receipts.append(
            {
                "device": list(map(int, device.groups())) if device else None,
                "axes": {int(axis): float(value) for axis, value in axes},
            }
        )
    return receipts


def _qualify(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    from Xlib import X, XK, display
    from Xlib.ext import xtest

    required = ["weston", "xinput", "xhost", "glxinfo", "stdbuf"]
    if args.gpu_animation:
        required.append("glxgears")
    for executable in required:
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is missing: {executable}")
    artifact = build_headless_seat_module(root=args.module_root, include_root=args.include_root)
    report["seat_artifact"] = asdict(artifact)
    processes: dict[str, subprocess.Popen[bytes]] = {}
    connection = None
    cleanup: dict[str, bool] = {}

    def stop(name: str) -> None:
        process = processes.pop(name, None)
        if process is not None:
            cleanup[name] = _terminate_spawned_process_group(process)

    with (
        tempfile.TemporaryDirectory(prefix="mcai-headless-runtime-") as runtime,
        ExitStack() as files,
    ):
        environment = _headless_compositor_environment()
        environment["XDG_RUNTIME_DIR"] = runtime
        environment["LC_ALL"] = "C"
        report["runtime_directory"] = runtime
        command = _weston_command(
            weston=str(shutil.which("weston")),
            wayland_socket="qualify",
            width=800,
            height=600,
            fullscreen=False,
            compositor_log=output / "weston.log",
            seat_module=Path(artifact.module_path),
        )
        report["compositor_command"] = command

        def start(name: str, command: list[str], log: str) -> subprocess.Popen[bytes]:
            stream = files.enter_context((output / log).open("wb", buffering=0))
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[name] = process
            report.setdefault("owned_processes", {})[name] = process.pid
            return process

        try:
            compositor = start("weston", command, "weston-stdout.log")
            private_display = _wait_for_weston_xwayland(compositor, output / "weston.log")
            require_loaded_headless_seat(compositor.pid, artifact)
            require_isolated_display(private_display)
            environment["DISPLAY"] = private_display
            report["display"] = private_display
            report["checks"]["loaded_module_provenance"] = True
            for executable, extra in (
                ("xinput", ["--list", "--long"]),
                ("xhost", []),
                ("glxinfo", ["-B"]),
            ):
                receipt = subprocess.run(
                    [executable, *extra],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                )
                (output / f"{executable}.txt").write_text(
                    receipt.stdout + receipt.stderr,
                    encoding="utf-8",
                )
                if executable == "glxinfo":
                    report["checks"]["accelerated_renderer"] = "Accelerated: yes" in receipt.stdout
                elif executable == "xhost":
                    report["checks"]["access_control_enabled"] = (
                        "access control enabled" in receipt.stdout
                    )
                    allowed = [
                        line.strip() for line in receipt.stdout.splitlines()[1:] if line.strip()
                    ]
                    report["checks"]["only_current_user_authorized"] = allowed == [
                        f"SI:localuser:{pwd.getpwuid(os.getuid()).pw_name}"
                    ]

            connection = display.Display(private_display)
            root = connection.screen().root
            window = root.create_window(
                0,
                0,
                800,
                600,
                0,
                connection.screen().root_depth,
                X.InputOutput,
                X.CopyFromParent,
                background_pixel=0x185E92,
                event_mask=(
                    X.KeyPressMask
                    | X.KeyReleaseMask
                    | X.PointerMotionMask
                    | X.ButtonPressMask
                    | X.ButtonReleaseMask
                    | X.ExposureMask
                    | X.StructureNotifyMask
                    | X.FocusChangeMask
                ),
            )
            window.set_wm_name("Minecraft AI disposable input qualification")
            window.map()
            connection.sync()
            time.sleep(0.6)
            window.set_input_focus(X.RevertToParent, X.CurrentTime)
            connection.sync()
            time.sleep(0.1)
            xtest.fake_input(connection, X.MotionNotify, x=300, y=300)
            connection.sync()
            time.sleep(0.1)
            report["setup_events"] = _events(connection)
            report["test_window"] = window.id
            drawing = window.create_gc(foreground=0xCBE239)
            window.fill_rectangle(drawing, 120, 140, 140, 90)
            connection.sync()
            report["capture_before"] = _capture(window, output / "window-before.png")

            keycode = connection.keysym_to_keycode(XK.string_to_keysym("w"))
            held_samples = []
            xtest.fake_input(connection, X.KeyPress, keycode)
            connection.sync()
            try:
                for _ in range(20):
                    keymap = connection.query_keymap()
                    held_samples.append(
                        {
                            "monotonic_ns": time.monotonic_ns(),
                            "focus": _focus_id(connection),
                            "pointer": _pointer(connection),
                            "held": bool(keymap[keycode // 8] & (1 << (keycode % 8))),
                        }
                    )
                    time.sleep(0.1)
            finally:
                xtest.fake_input(connection, X.KeyRelease, keycode)
                connection.sync()
            time.sleep(0.1)
            report["held_samples"] = held_samples
            held_events = _events(connection)
            report["held_events"] = held_events
            report["checks"]["held_key_and_stable_route"] = all(
                sample["held"]
                and sample["focus"] == window.id
                and (sample["pointer"]["x"], sample["pointer"]["y"]) == (300, 300)
                for sample in held_samples
            )
            report["checks"]["key_press_and_release_delivered"] = all(
                any(
                    event["type"] == event_type and event["detail"] == keycode
                    for event in held_events
                )
                for event_type in (X.KeyPress, X.KeyRelease)
            )
            report["checks"]["key_released"] = not bool(
                connection.query_keymap()[keycode // 8] & (1 << (keycode % 8))
            )

            # Raw-event selection is restricted to this disposable server.
            start("raw_monitor", ["stdbuf", "-oL", "xinput", "test-xi2", "--root"], "xi2.txt")
            time.sleep(0.2)
            before = _pointer(connection)
            xtest.fake_input(connection, X.MotionNotify, detail=1, root=X.NONE, x=8, y=-3)
            connection.sync()
            time.sleep(0.1)
            after = _pointer(connection)
            report["relative_motion"] = {"before": before, "after": after}
            report["relative_events"] = _events(connection)
            report["checks"]["relative_pointer_delta"] = (
                after["x"] - before["x"],
                after["y"] - before["y"],
            ) == (8, -3)
            xtest.fake_input(connection, X.ButtonPress, 1)
            connection.sync()
            try:
                time.sleep(0.05)
                report["checks"]["button_held"] = bool(_pointer(connection)["mask"] & X.Button1Mask)
            finally:
                xtest.fake_input(connection, X.ButtonRelease, 1)
                connection.sync()
            time.sleep(0.05)
            report["checks"]["button_released"] = not bool(
                _pointer(connection)["mask"] & X.Button1Mask
            )
            report["button_events"] = _events(connection)
            report["checks"]["button_press_and_release_delivered"] = all(
                any(
                    event["type"] == event_type and event["detail"] == 1
                    for event in report["button_events"]
                )
                for event_type in (X.ButtonPress, X.ButtonRelease)
            )
            stop("raw_monitor")
            receipts = _raw_motion_receipts((output / "xi2.txt").read_text(encoding="utf-8"))
            report["raw_motion_receipts"] = receipts
            report["checks"]["one_exact_raw_motion_receipt"] = (
                len(receipts) == 1
                and receipts[0]["axes"] == {0: 8.0, 1: -3.0}
                and receipts[0]["device"] == [2, 4]
            )

            # This small confinement rectangle belongs solely to the test window.
            clip = window.create_window(
                250,
                250,
                120,
                100,
                0,
                0,
                X.InputOnly,
                X.CopyFromParent,
                override_redirect=True,
                event_mask=X.PointerMotionMask,
            )
            clip.map()
            connection.sync()
            status = window.grab_pointer(
                False,
                X.PointerMotionMask | X.ButtonPressMask | X.ButtonReleaseMask,
                X.GrabModeAsync,
                X.GrabModeAsync,
                clip,
                X.NONE,
                X.CurrentTime,
            )
            try:
                xtest.fake_input(
                    connection,
                    X.MotionNotify,
                    detail=1,
                    root=X.NONE,
                    x=1200,
                    y=1200,
                )
                connection.sync()
                time.sleep(0.1)
                confined = _pointer(connection)
                report["confinement"] = {"grab_status": status, "pointer": confined}
                report["checks"]["pointer_confinement"] = (
                    status == X.GrabSuccess
                    and 250 <= confined["x"] < 370
                    and 250 <= confined["y"] < 350
                )
            finally:
                connection.ungrab_pointer(X.CurrentTime)
                clip.destroy()
                connection.sync()
            drawing.change(foreground=0xFE5C44)
            window.fill_rectangle(drawing, 200, 240, 180, 100)
            connection.sync()
            report["capture_after"] = _capture(window, output / "window-after.png")
            report["checks"]["changing_drawable_capture"] = (
                report["capture_before"]["sha256"] != report["capture_after"]["sha256"]
            )

            if args.gpu_animation:
                start("glxgears", ["glxgears", "-geometry", "500x400"], "glxgears.txt")
                deadline = time.monotonic() + 5
                gears_window = None
                while time.monotonic() < deadline:
                    gears_window = _find_named(root, "glxgears")
                    if gears_window is not None:
                        break
                    time.sleep(0.1)
                if gears_window is None:
                    raise RuntimeError("owned glxgears window did not appear")
                time.sleep(0.5)
                first = _capture(gears_window, output / "gpu-a.png")
                time.sleep(0.2)
                second = _capture(gears_window, output / "gpu-b.png")
                report["gpu_frames"] = [first, second]
                report["checks"]["animated_gpu_capture"] = (
                    first["sha256"] != second["sha256"] and first["unique_bytes"] > 32
                )
        finally:
            for name in list(processes):
                if name != "weston":
                    stop(name)
            try:
                if connection is not None:
                    connection.close()
            finally:
                stop("weston")
                report["process_cleanup"] = cleanup
                report["checks"]["owned_processes_stopped"] = bool(cleanup) and all(
                    cleanup.values()
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="New evidence directory; defaults to a temp dir"
    )
    parser.add_argument("--include-root", type=Path, help="Root containing extracted usr/include")
    parser.add_argument("--module-root", type=Path, help="Override the virtual-seat artifact root")
    parser.add_argument(
        "--gpu-animation", action="store_true", help="Also capture changing glxgears frames"
    )
    args = parser.parse_args()
    if args.output_dir is None:
        output = Path(tempfile.mkdtemp(prefix="minecraft-ai-headless-qualification-"))
    else:
        output = args.output_dir.resolve()
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    print(f"Evidence directory: {output}", flush=True)
    report: dict[str, Any] = {
        "schema": "headless_input_qualification_v1",
        "scope": "owned_disposable_display_only",
        "created_ns": time.time_ns(),
        "qualifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_directory": str(output),
        "checks": {},
        "gpu_animation_requested": args.gpu_animation,
        "limitations": [
            "No Wine or game behavior is qualified.",
            "No deliberate host input is injected; host independence relies on backend provenance.",
            "Same-user processes and privileged users are outside the isolation boundary.",
        ],
    }

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}; stopping owned test processes")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        _qualify(args, output, report)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    report["passed"] = (
        bool(report["checks"]) and all(report["checks"].values()) and "error" not in report
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "error": report.get("error"),
                "report": str(output / "result.json"),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
