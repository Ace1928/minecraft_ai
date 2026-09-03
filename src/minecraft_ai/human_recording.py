from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .datasets import ActionLevel, DatasetSource, DatasetSourceType, TrajectoryManifest
from .perception import FrameState
from .platforms.bedrock_x11 import IsolatedX11Capture, require_isolated_display
from .safety import MotorAction
from .trajectory import (
    ActionOrigin,
    ActionProvenance,
    TrajectoryRecorder,
    new_trajectory_id,
)


class HumanInputKind(StrEnum):
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    BUTTON_PRESS = "button_press"
    BUTTON_RELEASE = "button_release"
    MOTION = "motion"


@dataclass(frozen=True)
class HumanInputEvent:
    kind: HumanInputKind
    detail: int = 0
    dx: float = 0.0
    dy: float = 0.0
    observed_ns: int = field(default_factory=time.monotonic_ns)


_EVENT_LINE = re.compile(r"^EVENT type (13|14|15|16|17) \(")
_DETAIL_LINE = re.compile(r"^\s*detail:\s*(\d+)\s*$")
_VALUATOR_LINE = re.compile(r"^\s*(0|1):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))(?:\s+\([^)]*\))?\s*$")
_RAW_EVENT_KINDS = {
    13: HumanInputKind.KEY_PRESS,
    14: HumanInputKind.KEY_RELEASE,
    15: HumanInputKind.BUTTON_PRESS,
    16: HumanInputKind.BUTTON_RELEASE,
    17: HumanInputKind.MOTION,
}


class XInput2StreamParser:
    """Parse `xinput test-xi2` raw events without relying on pointer positions."""

    def __init__(self) -> None:
        self._kind: HumanInputKind | None = None
        self._detail = 0
        self._dx = 0.0
        self._dy = 0.0

    def feed(self, line: str) -> tuple[HumanInputEvent, ...]:
        match = _EVENT_LINE.match(line)
        if match is not None:
            completed = self._finish()
            self._kind = _RAW_EVENT_KINDS[int(match.group(1))]
            return completed
        if self._kind is None:
            return ()
        detail = _DETAIL_LINE.match(line)
        if detail is not None:
            self._detail = int(detail.group(1))
            return ()
        valuator = _VALUATOR_LINE.match(line)
        if valuator is not None and self._kind == HumanInputKind.MOTION:
            axis = int(valuator.group(1))
            value = float(valuator.group(2))
            if axis == 0:
                self._dx = value
            else:
                self._dy = value
            return ()
        if not line.strip():
            return self._finish()
        return ()

    def close(self) -> tuple[HumanInputEvent, ...]:
        return self._finish()

    def _finish(self) -> tuple[HumanInputEvent, ...]:
        kind = self._kind
        if kind is None:
            return ()
        event = HumanInputEvent(
            kind=kind,
            detail=self._detail,
            dx=self._dx,
            dy=self._dy,
        )
        self._kind = None
        self._detail = 0
        self._dx = 0.0
        self._dy = 0.0
        return (event,)


_KEY_ALIASES = {
    "shift_l": "shift",
    "shift_r": "shift",
    "control_l": "ctrl",
    "control_r": "ctrl",
    "alt_l": "alt",
    "alt_r": "alt",
    "return": "enter",
    "escape": "escape",
    "space": "space",
    "iso_left_tab": "tab",
}
_BUTTON_NAMES = {1: "left", 2: "middle", 3: "right", 4: "wheel_up", 5: "wheel_down"}


def normalize_keysym(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().lower()
    normalized = _KEY_ALIASES.get(normalized, normalized)
    if not normalized or len(normalized) > 32:
        return None
    return normalized


class X11KeyResolver:
    def __init__(self, display_name: str) -> None:
        display_module = importlib.import_module("Xlib.display")
        self._xk = importlib.import_module("Xlib.XK")
        self._display: Any = display_module.Display(display_name)

    def __call__(self, keycode: int) -> str | None:
        keysym = int(self._display.keycode_to_keysym(keycode, 0))
        return normalize_keysym(self._xk.keysym_to_string(keysym))

    def close(self) -> None:
        self._display.close()


class HumanInputAccumulator:
    """Convert raw XInput2 events into lossless 20 Hz motor transitions."""

    def __init__(self, key_resolver: X11KeyResolver) -> None:
        self._key_resolver = key_resolver
        self._lock = threading.Lock()
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._keys_down: set[str] = set()
        self._keys_up: set[str] = set()
        self._buttons_down: set[str] = set()
        self._buttons_up: set[str] = set()
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        self._last_event_ns: int | None = None
        self.events = 0

    def ingest(self, event: HumanInputEvent) -> None:
        with self._lock:
            self.events += 1
            self._last_event_ns = event.observed_ns
            if event.kind in {HumanInputKind.KEY_PRESS, HumanInputKind.KEY_RELEASE}:
                key = self._key_resolver(event.detail)
                if key is None:
                    return
                if event.kind == HumanInputKind.KEY_PRESS and key not in self._held_keys:
                    self._held_keys.add(key)
                    self._keys_down.add(key)
                    self._keys_up.discard(key)
                elif event.kind == HumanInputKind.KEY_RELEASE and key in self._held_keys:
                    self._held_keys.remove(key)
                    self._keys_up.add(key)
                    self._keys_down.discard(key)
            elif event.kind in {
                HumanInputKind.BUTTON_PRESS,
                HumanInputKind.BUTTON_RELEASE,
            }:
                button = _BUTTON_NAMES.get(event.detail)
                if button is None:
                    return
                if event.kind == HumanInputKind.BUTTON_PRESS and button not in self._held_buttons:
                    self._held_buttons.add(button)
                    self._buttons_down.add(button)
                    self._buttons_up.discard(button)
                elif event.kind == HumanInputKind.BUTTON_RELEASE and button in self._held_buttons:
                    self._held_buttons.remove(button)
                    self._buttons_up.add(button)
                    self._buttons_down.discard(button)
            elif event.kind == HumanInputKind.MOTION:
                self._mouse_dx += event.dx
                self._mouse_dy += event.dy

    def snapshot(self, sequence: int, *, duration_ms: int) -> tuple[MotorAction, int]:
        with self._lock:
            mouse_dx = max(-4096, min(4096, round(self._mouse_dx)))
            mouse_dy = max(-4096, min(4096, round(self._mouse_dy)))
            self._mouse_dx -= mouse_dx
            self._mouse_dy -= mouse_dy
            action = MotorAction(
                sequence=sequence,
                keys_down=tuple(sorted(self._keys_down)),
                keys_up=tuple(sorted(self._keys_up)),
                buttons_down=tuple(sorted(self._buttons_down)),
                buttons_up=tuple(sorted(self._buttons_up)),
                mouse_dx=mouse_dx,
                mouse_dy=mouse_dy,
                duration_ms=duration_ms,
            )
            accepted_ns = self._last_event_ns or time.monotonic_ns()
            self._keys_down.clear()
            self._keys_up.clear()
            self._buttons_down.clear()
            self._buttons_up.clear()
            self._last_event_ns = None
            return action, accepted_ns

    def discard_pending(self) -> None:
        with self._lock:
            self._keys_down.clear()
            self._keys_up.clear()
            self._buttons_down.clear()
            self._buttons_up.clear()
            self._mouse_dx = 0.0
            self._mouse_dy = 0.0
            self._last_event_ns = None


class XInput2Monitor:
    """Observe raw input only inside the managed Xwayland display."""

    def __init__(self, display_name: str) -> None:
        require_isolated_display(display_name, allow_host=False)
        if shutil.which("xinput") is None:
            raise RuntimeError("xinput is required for lossless human input recording")
        self.display_name = display_name
        self.key_resolver = X11KeyResolver(display_name)
        self.accumulator = HumanInputAccumulator(self.key_resolver)
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("XInput2 monitor is already started")
        command = ["xinput", "test-xi2", "--root"]
        if shutil.which("stdbuf") is not None:
            command = ["stdbuf", "-oL", *command]
        environment = dict(os.environ)
        environment["DISPLAY"] = self.display_name
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        self._thread = threading.Thread(
            target=self._read,
            name="minecraft-ai-human-xinput2",
            daemon=True,
        )
        self._thread.start()
        time.sleep(0.1)
        if self._process.poll() is not None:
            raise RuntimeError("xinput monitor exited before recording began")

    def snapshot(self, sequence: int, *, duration_ms: int) -> tuple[MotorAction, int]:
        if self._error is not None:
            raise RuntimeError("xinput monitor failed") from self._error
        return self.accumulator.snapshot(sequence, duration_ms=duration_ms)

    def discard_pending(self) -> None:
        self.accumulator.discard_pending()

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self.key_resolver.close()

    def _read(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        parser = XInput2StreamParser()
        try:
            for line in process.stdout:
                for event in parser.feed(line):
                    self.accumulator.ingest(event)
            for event in parser.close():
                self.accumulator.ingest(event)
        except BaseException as exc:
            self._error = exc


@dataclass(frozen=True)
class HumanRecordingRequest:
    display: str
    window_id: int
    instance_id: str
    role: str
    game_version: str
    artifact_root: Path
    state_db_path: Path
    duration_s: float
    capture_hz: float = 20.0
    label: str = "human-demonstration"
    task_id: str | None = None
    fov: float | None = None
    mouse_sensitivity: float | None = None
    shard_steps: int = 256
    queue_size: int = 512


def record_human_session(request: HumanRecordingRequest) -> TrajectoryManifest:
    if request.duration_s <= 0.0:
        raise ValueError("human recording duration must be positive")
    if not 5.0 <= request.capture_hz <= 60.0:
        raise ValueError("human recording capture_hz must be in 5..60")
    capture = IsolatedX11Capture(request.display, request.window_id, allow_host=False)
    monitor = XInput2Monitor(request.display)
    probe = capture.capture()
    trajectory_id = new_trajectory_id("bedrock-human")
    manifest = TrajectoryManifest(
        trajectory_id=trajectory_id,
        source=DatasetSource(
            source_id=f"minecraft-ai:{trajectory_id}:xinput2",
            source_type=DatasetSourceType.BEDROCK_HUMAN,
            license="operator-owned-gameplay",
            redistribution_allowed=False,
            training_allowed=True,
            edition="bedrock",
            game_versions=(request.game_version,),
        ),
        role=request.role,
        label=request.label,
        task_id=request.task_id,
        game_version=request.game_version,
        platform="bedrock-on-linux/xwayland/xinput2",
        launcher_profile="bedrock-on-linux/winegdk",
        resolution=(probe.width, probe.height),
        fov=request.fov,
        mouse_sensitivity=request.mouse_sensitivity,
        started_ns=time.time_ns(),
    )
    recorder = TrajectoryRecorder(
        manifest=manifest,
        artifact_root=request.artifact_root,
        state_db_path=request.state_db_path,
        shard_steps=request.shard_steps,
        queue_size=request.queue_size,
    )
    period_s = 1.0 / request.capture_hz
    duration_ms = max(1, min(1000, round(period_s * 1000.0)))
    started = time.monotonic()
    deadline = started
    sequence = 0
    frame = probe
    closed_manifest: TrajectoryManifest | None = None
    try:
        monitor.start()
        monitor.discard_pending()
        while time.monotonic() - started < request.duration_s:
            deadline += period_s
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            action, accepted_ns = monitor.snapshot(sequence, duration_ms=duration_ms)
            blackboard = FrameState(
                frame_id=sequence,
                captured_ns=frame.captured_ns,
                instance_id=request.instance_id,
                width=frame.width,
                height=frame.height,
            )
            recorder.record_accepted(
                action=action,
                provenance=ActionProvenance(
                    policy_id="human:xinput2-raw-observed",
                    route_id="human",
                    action_level=ActionLevel.RAW,
                    origin=ActionOrigin.HUMAN,
                ),
                supervisor_response={
                    "accepted_sequence": sequence,
                    "accepted_monotonic_ns": accepted_ns,
                    "source": "xinput2-raw-observed",
                },
                frame=frame,
                blackboard=blackboard,
            )
            sequence += 1
            frame = capture.capture()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            monitor.close()
        finally:
            try:
                capture.close()
            finally:
                closed_manifest = recorder.close()
    if closed_manifest is None:  # pragma: no cover - finally always assigns
        raise RuntimeError("human trajectory did not close")
    return closed_manifest
