from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..safety import MotorAction, MotorLease, MotorRejected


class IsolationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScreenRect:
    """One immutable rectangle in X root-window coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise IsolationError("screen rectangle dimensions must be positive")


@dataclass(frozen=True)
class HostMonitorBinding:
    """Proof that one Minecraft window exclusively occupies one host monitor.

    Host-display motor control is intentionally unavailable without this
    binding.  It records an exact RandR output and exact root-window geometry;
    every accepted action rechecks the window against these immutable bounds.
    """

    display: str
    output_name: str
    monitor: ScreenRect
    window_id: int
    bound_ns: int

    def __post_init__(self) -> None:
        if not self.display.strip():
            raise IsolationError("host-monitor display is required")
        if not self.output_name.strip():
            raise IsolationError("host-monitor output name is required")
        if self.window_id <= 0:
            raise IsolationError("host-monitor Minecraft window id must be positive")
        if self.bound_ns <= 0:
            raise IsolationError("host-monitor binding timestamp must be positive")

    def payload(self) -> dict[str, object]:
        return {
            "display": self.display,
            "output_name": self.output_name,
            "monitor": {
                "x": self.monitor.x,
                "y": self.monitor.y,
                "width": self.monitor.width,
                "height": self.monitor.height,
            },
            "window_id": self.window_id,
            "bound_ns": self.bound_ns,
        }

    @classmethod
    def from_payload(cls, raw: object) -> HostMonitorBinding:
        if not isinstance(raw, Mapping):
            raise IsolationError("host-monitor binding payload must be an object")
        monitor_raw = raw.get("monitor")
        if not isinstance(monitor_raw, Mapping):
            raise IsolationError("host-monitor binding monitor must be an object")

        def integer(mapping: Mapping[object, object], key: str) -> int:
            value = mapping.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise IsolationError(f"host-monitor binding {key!r} must be an integer")
            return value

        display = raw.get("display")
        output_name = raw.get("output_name")
        if not isinstance(display, str) or not isinstance(output_name, str):
            raise IsolationError("host-monitor binding display/output must be strings")
        return cls(
            display=display,
            output_name=output_name,
            monitor=ScreenRect(
                x=integer(monitor_raw, "x"),
                y=integer(monitor_raw, "y"),
                width=integer(monitor_raw, "width"),
                height=integer(monitor_raw, "height"),
            ),
            window_id=integer(raw, "window_id"),
            bound_ns=integer(raw, "bound_ns"),
        )


_CONNECTED_OUTPUT = re.compile(
    r"^(?P<name>\S+)\s+connected(?:\s+primary)?\s+"
    r"(?P<width>\d+)x(?P<height>\d+)"
    r"(?P<x>[+-]\d+)(?P<y>[+-]\d+)(?:\s|$)"
)


def parse_connected_outputs(text: str) -> dict[str, ScreenRect]:
    """Parse active RandR output geometry without accepting disconnected modes."""

    outputs: dict[str, ScreenRect] = {}
    for line in text.splitlines():
        match = _CONNECTED_OUTPUT.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name in outputs:
            raise IsolationError(f"duplicate connected RandR output {name!r}")
        outputs[name] = ScreenRect(
            x=int(match.group("x")),
            y=int(match.group("y")),
            width=int(match.group("width")),
            height=int(match.group("height")),
        )
    return outputs


def connected_outputs(display_name: str) -> dict[str, ScreenRect]:
    """Return current active XRandR outputs for one explicit display."""

    executable = shutil.which("xrandr")
    if executable is None:
        raise IsolationError("xrandr is required to bind a host-monitor session")
    try:
        result = subprocess.run(
            [executable, "--display", display_name, "--query"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IsolationError(f"cannot query host monitor topology: {exc}") from exc
    outputs = parse_connected_outputs(result.stdout)
    if not outputs:
        raise IsolationError("xrandr reported no active connected outputs")
    return outputs


def _window_root_rect(display: Any, window_id: int) -> ScreenRect:
    try:
        root = display.screen().root
        window = display.create_resource_object("window", window_id)
        attributes = window.get_attributes()
        geometry = window.get_geometry()
        # python-xlib's source window is the coordinate space being translated
        # *from*.  Translate root origin into the target window to obtain the
        # target's absolute root position (the reverse order negates offsets).
        translated = root.translate_coords(window, 0, 0)
    except Exception as exc:
        raise IsolationError("cannot resolve host-monitor Minecraft window geometry") from exc
    if int(attributes.map_state) == 0:
        raise IsolationError("host-monitor Minecraft window is not viewable")
    return ScreenRect(
        x=int(translated.x),
        y=int(translated.y),
        width=int(geometry.width),
        height=int(geometry.height),
    )


def _require_exact_monitor_occupancy(window: ScreenRect, monitor: ScreenRect) -> None:
    if window != monitor:
        raise IsolationError(
            "host-display motor control requires Minecraft to occupy exactly one "
            f"bound monitor; window={window.width}x{window.height}+{window.x}+{window.y} "
            f"monitor={monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}"
        )


def bind_host_monitor(
    display_name: str,
    target_window_id: int,
    output_name: str,
) -> HostMonitorBinding:
    """Create a fail-closed binding for an already-positioned Minecraft window."""

    outputs = connected_outputs(display_name)
    monitor = outputs.get(output_name)
    if monitor is None:
        raise IsolationError(
            f"RandR output {output_name!r} is not active; available={sorted(outputs)}"
        )
    try:
        display_module = importlib.import_module("Xlib.display")
        display: Any = display_module.Display(display_name)
    except Exception as exc:
        raise IsolationError(f"cannot open host display {display_name}: {exc}") from exc
    try:
        window = display.create_resource_object("window", target_window_id)
        input_window = _resolve_minecraft_input_window(display, target_window_id)
        if int(input_window.id) == int(window.id):
            identity = " ".join(
                (
                    str(window.get_wm_name() or ""),
                    *(str(value) for value in (window.get_wm_class() or ())),
                )
            ).casefold()
            if not any(token in identity for token in ("minecraft", "bedrock", "wine")):
                raise IsolationError("bound host window is not recognizably Minecraft/Wine")
        _require_exact_monitor_occupancy(
            _window_root_rect(display, target_window_id),
            monitor,
        )
    finally:
        display.close()
    return HostMonitorBinding(
        display=display_name,
        output_name=output_name,
        monitor=monitor,
        window_id=target_window_id,
        bound_ns=time.monotonic_ns(),
    )


def validate_host_monitor_window(
    display: Any,
    binding: HostMonitorBinding,
    *,
    target_window_id: int,
    input_window_id: int | None = None,
    require_focus: bool = False,
) -> None:
    """Recheck the immutable monitor boundary immediately before host input."""

    if target_window_id != binding.window_id:
        raise IsolationError("host-monitor target no longer matches its binding")
    _require_exact_monitor_occupancy(
        _window_root_rect(display, target_window_id),
        binding.monitor,
    )
    if not require_focus:
        return
    if input_window_id is None:
        raise IsolationError("host-monitor input window is unavailable")
    try:
        focus = display.get_input_focus().focus
    except Exception as exc:
        raise IsolationError("cannot inspect host-display input focus") from exc
    if not _window_is_descendant_or_same(focus, input_window_id):
        raise IsolationError(
            "host-display focus left Minecraft; motor input was released instead of "
            "stealing focus from the operator display"
        )


class _NativeRelativeMouse:
    """Issue genuine relative XTEST motion on the isolated X server.

    python-xlib's ``xtest.fake_input`` exposes the protocol's absolute motion
    request but not ``XTestFakeRelativeMotionEvent``. Bedrock uses a grabbed,
    recentered pointer, so reconstructing an absolute coordinate races that
    recenter and produces discontinuous camera jumps.
    """

    def __init__(self, display_name: str) -> None:
        x11_path = ctypes.util.find_library("X11")
        xtst_path = ctypes.util.find_library("Xtst")
        if x11_path is None or xtst_path is None:
            raise IsolationError("libX11 and libXtst are required for relative mouse input")
        try:
            self._x11 = ctypes.CDLL(x11_path)
            self._xtst = ctypes.CDLL(xtst_path)
        except OSError as exc:
            raise IsolationError(f"cannot load native XTEST libraries: {exc}") from exc
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XFlush.argtypes = [ctypes.c_void_p]
        self._x11.XFlush.restype = ctypes.c_int
        self._xtst.XTestFakeRelativeMotionEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._xtst.XTestFakeRelativeMotionEvent.restype = ctypes.c_int
        display_pointer = self._x11.XOpenDisplay(display_name.encode("utf-8"))
        if not display_pointer:
            raise IsolationError(f"cannot open {display_name} for native relative mouse input")
        self._display = ctypes.c_void_p(int(display_pointer))

    def move(self, mouse_dx: int, mouse_dy: int) -> None:
        accepted = int(
            self._xtst.XTestFakeRelativeMotionEvent(
                self._display,
                mouse_dx,
                mouse_dy,
                0,
            )
        )
        if accepted == 0:
            raise IsolationError("isolated X server rejected relative mouse input")
        self._x11.XFlush(self._display)


class _TargetedPointer:
    """Window-targeted mouse for host-monitor play without focus stealing.

    Bedrock's grabbed, recentered pointer cannot be driven through XTEST while
    the operator uses the desktop (the recenter would follow the server pointer
    and the deltas would land in whichever window owns the cursor). Motion and
    buttons are therefore injected straight into the Minecraft window with
    synthetic events and a virtual cursor position that mirrors the game's own
    recentering: each delta advances the virtual position, which is wrapped
    back to the window centre after crossing the edge, exactly like the
    captured pointer the game expects.
    """

    def __init__(self, backend: Any, event_module: Any, x: Any) -> None:
        self._backend = backend
        self._event = event_module
        self._x = x
        self._px = 0
        self._py = 0
        self._ready = False

    def _window_and_size(self) -> tuple[Any, int, int]:
        window = self._backend._input_window()
        geometry = window.get_geometry()
        return window, int(geometry.width) or 1, int(geometry.height) or 1

    def _motion_options(self) -> dict[str, Any]:
        root = self._backend._display.screen().root.id
        return {
            "time": 0,
            "root": root,
            "window": self._backend._input_window_id,
            "same_screen": 1,
            "child": 0,
            "root_x": 0,
            "root_y": 0,
            "event_x": self._px,
            "event_y": self._py,
            "state": 0,
            "detail": 0,
        }

    def move(self, mouse_dx: int, mouse_dy: int) -> None:
        window, width, height = self._window_and_size()
        # The game recenters its captured pointer to the window centre between
        # frames (standard Wine/Windows "grab + warp" relative-mouse loop), so
        # its delta baseline is always the centre. Sending absolute positions
        # that we accumulate would drift: real recenter events would interleave
        # with ours, producing giant deltas. Emit each delta relative to the
        # centre instead, exactly like the recentred pointer the game expects.
        self._px = max(0, min(width // 2 + int(mouse_dx), width))
        self._py = max(0, min(height // 2 + int(mouse_dy), height))
        motion = self._event.MotionNotify(**self._motion_options())
        window.send_event(motion, event_mask=self._x.PointerMotionMask)
        self._backend._display.sync()

    def click(self, button: int, down: bool) -> None:
        window, _, _ = self._window_and_size()
        common = {
            "time": 0,
            "root": self._backend._display.screen().root.id,
            "window": self._backend._input_window_id,
            "same_screen": 1,
            "child": 0,
            "root_x": 0,
            "root_y": 0,
            "event_x": self._px,
            "event_y": self._py,
            "state": 0,
        }
        if down:
            event = self._event.ButtonPress(detail=button, **common)
            window.send_event(event, event_mask=self._x.ButtonPressMask)
        else:
            event = self._event.ButtonRelease(detail=button, **common)
            window.send_event(event, event_mask=self._x.ButtonReleaseMask)
        self._backend._display.sync()

    @property
    def ready(self) -> bool:
        return self._ready


_KEYSYM_NAMES: dict[str, str] = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "q": "q",
    "e": "e",
    "f": "f",
    "t": "t",
    "space": "space",
    "shift": "Shift_L",
    "ctrl": "Control_L",
    "control": "Control_L",
    "alt": "Alt_L",
    "escape": "Escape",
    "esc": "Escape",
    "enter": "Return",
    "return": "Return",
    "tab": "Tab",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "`": "grave",
    "-": "minus",
    "=": "equal",
    "[": "bracketleft",
    "]": "bracketright",
    "\\": "backslash",
    ";": "semicolon",
    "'": "apostrophe",
    ",": "comma",
    ".": "period",
    "/": "slash",
}
_BUTTONS: dict[str, int] = {"left": 1, "middle": 2, "right": 3}
_SHIFTED_ASCII = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+{}|:"<>?')
_SHIFT_MAP: dict[str, str] = {
    "~": "`",
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    '"': "'",
    "<": ",",
    ">": ".",
    "?": "/",
}


def _display_identity(name: str) -> str:
    value = name.strip()
    if not value:
        raise IsolationError("isolated X display is required")
    return value.split(".", 1)[0]


def _x11_keysym_name(key: str) -> str:
    normalized = key.lower()
    return _KEYSYM_NAMES.get(normalized, normalized)


def _resolve_minecraft_input_window(display: Any, target_window_id: int) -> Any:
    """Find Wine's interactive Minecraft child beneath the capture desktop.

    BedrockOnLinux exposes the composited Wine desktop as the stable capture
    drawable, while keyboard focus belongs to the nested ``Minecraft`` window.
    Treating those as the same X window makes relative camera events appear to
    work but causes keyboard/menu actions to disappear after focus changes.
    """

    target = display.create_resource_object("window", target_window_id)
    try:
        candidates = [target, *target.query_tree().children]
    except Exception:
        return target
    for window in candidates:
        try:
            name = str(window.get_wm_name() or "")
            if "minecraft" in name.casefold():
                return window
        except Exception:
            continue
    fallback: tuple[int, Any] | None = None
    for window in candidates:
        try:
            wm_class = window.get_wm_class() or ()
            identity = " ".join(str(value) for value in wm_class).casefold()
            if "minecraft" not in identity:
                continue
            geometry = window.get_geometry()
            area = int(geometry.width) * int(geometry.height)
            if fallback is None or area > fallback[0]:
                fallback = area, window
        except Exception:
            continue
    if fallback is not None:
        return fallback[1]
    return target


def _window_is_descendant_or_same(window: Any, ancestor_window_id: int) -> bool:
    """Return whether an X focus window belongs to the Minecraft subtree."""

    current = window
    visited: set[int] = set()
    for _ in range(32):
        current_id = int(getattr(current, "id", 0))
        if current_id == ancestor_window_id:
            return True
        if current_id <= 0 or current_id in visited:
            return False
        visited.add(current_id)
        try:
            current = current.query_tree().parent
        except Exception:
            return False
    return False


def require_isolated_display(
    display_name: str,
    host_display: str | None = None,
    *,
    allow_host: bool = False,
) -> None:
    if allow_host:
        return
    host = os.environ.get("DISPLAY", "") if host_display is None else host_display
    if host and _display_identity(display_name) == _display_identity(host):
        raise IsolationError(
            f"refusing Bedrock motor/capture backend on host display {display_name!r}"
        )


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    captured_ns: int
    width: int
    height: int
    bgra: bytes


class IsolatedX11InputBackend:
    """XTEST input connected only to a dedicated Bedrock X server."""

    backend_id = "bedrock-isolated-x11-xtest"

    def __init__(
        self,
        display_name: str,
        *,
        host_display: str | None = None,
        target_window_id: int | None = None,
        allow_host: bool = False,
        host_monitor_binding: HostMonitorBinding | None = None,
        input_permitted: Callable[[], bool] = lambda: True,
    ) -> None:
        require_isolated_display(display_name, host_display, allow_host=allow_host)
        if host_monitor_binding is not None:
            if not allow_host:
                raise IsolationError("host-monitor binding requires explicit host-display access")
            if target_window_id is None:
                raise IsolationError("host-monitor binding requires an exact target window")
            if host_monitor_binding.display != display_name:
                raise IsolationError("host-monitor binding display does not match backend display")
            if host_monitor_binding.window_id != target_window_id:
                raise IsolationError("host-monitor binding window does not match backend target")
        try:
            display_module = importlib.import_module("Xlib.display")
            self._x = importlib.import_module("Xlib.X")
            self._xk = importlib.import_module("Xlib.XK")
            self._xtest = importlib.import_module("Xlib.ext.xtest")
        except ImportError as exc:
            raise IsolationError("python-xlib is required for isolated Bedrock X11 input") from exc
        self.display_name = display_name
        self.target_window_id = target_window_id
        self._input_window_id = target_window_id
        self._host_monitor_binding = host_monitor_binding
        # Isolated displays use XTEST for the complete keyboard/mouse stream on
        # their private input namespace. The retained host-display debug path
        # sends keys directly to Minecraft with XSendEvent; autonomous CLI use
        # rejects that mode because host XTEST mouse events are display-global.
        self._targeted = allow_host
        self._targeted_pointer: _TargetedPointer | None = None
        try:
            self._display: Any = display_module.Display(display_name)
        except Exception as exc:
            raise IsolationError(f"cannot open isolated X display {display_name}: {exc}") from exc
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._lease: MotorLease | None = None
        self._input_permitted = input_permitted
        self.release_count = 0
        self.live_capable = True
        if target_window_id is not None and not self.probe_target():
            self.close()
            raise IsolationError(f"target X window {target_window_id} is unavailable")
        if target_window_id is not None:
            input_window = _resolve_minecraft_input_window(self._display, target_window_id)
            self._input_window_id = int(input_window.id)
        if host_monitor_binding is None:
            if not self._targeted:
                self._ensure_input_focus()
        else:
            assert target_window_id is not None
            validate_host_monitor_window(
                self._display,
                host_monitor_binding,
                target_window_id=target_window_id,
                input_window_id=self._input_window_id,
                require_focus=False,
            )
        self._relative_mouse = _NativeRelativeMouse(display_name)

    @property
    def held_keys(self) -> frozenset[str]:
        return frozenset(self._held_keys)

    @property
    def held_buttons(self) -> frozenset[str]:
        return frozenset(self._held_buttons)

    @property
    def input_window_id(self) -> int | None:
        return self._input_window_id

    def bind_lease(self, lease: MotorLease) -> None:
        if lease.backend_id != self.backend_id:
            raise MotorRejected("motor lease backend identity mismatch")
        self._lease = lease

    def clear_lease(self) -> None:
        self._lease = None

    def _require_live_lease(self) -> MotorLease:
        lease = self._lease
        if lease is None:
            self.release_all()
            raise MotorRejected("Bedrock backend has no active motor lease")
        if lease.expired():
            self.release_all()
            self._lease = None
            raise MotorRejected("Bedrock backend motor lease expired")
        return lease

    def _keycode(self, key: str) -> int:
        keysym_name = _x11_keysym_name(key)
        keysym = int(self._xk.string_to_keysym(keysym_name))
        keycode = int(self._display.keysym_to_keycode(keysym))
        if keycode <= 0:
            raise IsolationError(f"unsupported key for X11 backend: {key!r}")
        return keycode

    def _input_window(self) -> Any:
        return self._display.create_resource_object("window", self._input_window_id)

    def _park_pointer_in_game(self) -> None:
        """Warp the real server pointer into the game client.

        XTEST click/motion events are delivered to the window under the pointer,
        not the focused window. Parking the pointer inside the game client (in
        absolute root coordinates) routes those events to the game regardless of
        which window owns the operator's focus.
        """
        try:
            window = self._display.create_resource_object("window", self._input_window_id)
            geometry = window.get_geometry()
            root = self._display.screen().root
            coords = root.translate_coords(window, geometry.width // 2, geometry.height // 2)
            root.warp_pointer(int(coords.x), int(coords.y))
            self._display.sync()
        except Exception:
            pass

    def _position_pointer_in_game(self, cursor_x: float, cursor_y: float) -> None:
        """Place the GUI cursor at one normalized captured-frame point.

        World look remains relative-only.  This path exists for visually
        grounded GUI controls whose exact current-frame region is known, and
        is deliberately unavailable on the host-display debug route because a
        root-pointer warp there would interfere with the operator's desktop.
        The semantic track is normalized to the cropped Wine game drawable,
        so reuse the capture rect rather than the surrounding Wine window.
        """
        if self._targeted:
            raise IsolationError("absolute GUI cursor targets require an isolated X display")
        try:
            window_id = self.target_window_id or self._input_window_id
            if window_id is None:
                raise IsolationError("Minecraft input window is unavailable")
            window = self._display.create_resource_object("window", window_id)
            geometry = window.get_geometry()
            width = int(geometry.width)
            height = int(geometry.height)
            if width <= 0 or height <= 0:
                raise IsolationError("Minecraft input window has invalid geometry")
            content_rect = _wine_content_rect(
                self._display,
                window_id,
                width,
                height,
            ) or (0, 0, width, height)
            x, y = _normalized_content_point(cursor_x, cursor_y, content_rect)
            root = self._display.screen().root
            coordinates = root.translate_coords(window, x, y)
            root.warp_pointer(int(coordinates.x), int(coordinates.y))
            self._display.sync()
        except IsolationError:
            raise
        except Exception as exc:
            raise IsolationError(f"cannot position isolated Minecraft GUI cursor: {exc}") from exc

    def _targeted_key(self, keycode: int, down: bool) -> None:
        """Deliver a key event directly to the Minecraft window (no focus needed)."""
        window = self._input_window()
        common = {
            "time": 0,
            "root": self._display.screen().root.id,
            "window": self._input_window_id,
            "same_screen": 1,
            "child": 0,
            "root_x": 0,
            "root_y": 0,
            "event_x": 0,
            "event_y": 0,
            "state": 0,
            "detail": keycode,
        }
        event_module = importlib.import_module("Xlib.protocol.event")
        if down:
            window.send_event(
                event_module.KeyPress(**common), event_mask=self._x.KeyPressMask
            )
        else:
            window.send_event(
                event_module.KeyRelease(**common), event_mask=self._x.KeyReleaseMask
            )
        self._display.sync()

    def _send_key(self, keycode: int, down: bool) -> None:
        """Route keys through private-display XTEST or host-only XSendEvent."""
        if self._targeted:
            self._targeted_key(keycode, down)
            return
        event_type = self._x.KeyPress if down else self._x.KeyRelease
        self._xtest.fake_input(self._display, event_type, keycode)

    def _require_positive_input_permitted(self) -> None:
        """Fail closed at the final actuator boundary for positive input."""

        try:
            permitted = self._input_permitted()
        except Exception as exc:
            raise MotorRejected("actuation interlock could not be verified") from exc
        if not permitted:
            raise MotorRejected("actuation interlock is not clear")

    def probe_target(self) -> bool:
        if self.target_window_id is None:
            return True
        try:
            window = self._display.create_resource_object("window", self.target_window_id)
            window.get_attributes()
        except Exception:
            return False
        return True

    def _ensure_input_focus(self) -> None:
        """Bedrock only consumes input while its client window has X focus.

        Isolated (non-host) displays assert focus before bursts. Host-display
        play is target-oriented (window-directed keys, parked-pointer XTEST) and
        never touches the operator's focus, so this stays a no-op there.
        """
        if self._targeted:
            return
        input_window_id = self._input_window_id
        if input_window_id is None:
            return
        try:
            current = self._display.get_input_focus().focus
            if _window_is_descendant_or_same(current, input_window_id):
                return
            input_window = self._display.create_resource_object(
                "window",
                input_window_id,
            )
            input_window.get_attributes()
            input_window.set_input_focus(self._x.RevertToParent, self._x.CurrentTime)
            self._display.sync()
        except Exception as exc:
            raise IsolationError(
                "cannot focus the isolated Minecraft input window"
            ) from exc

    def apply(self, action: MotorAction) -> None:
        self._require_live_lease()
        if not self.probe_target():
            self.release_all()
            raise IsolationError("Bedrock target window disappeared")
        self._ensure_input_focus()
        if (
            not self._targeted
            and self._host_monitor_binding is not None
            and self.target_window_id is not None
        ):
            try:
                validate_host_monitor_window(
                    self._display,
                    self._host_monitor_binding,
                    target_window_id=self.target_window_id,
                    input_window_id=self._input_window_id,
                    require_focus=True,
                )
            except Exception:
                self.release_all()
                raise
        # On an isolated display every event below is XTEST-scoped to that X
        # server. The host debug path retains window-directed keys, but is not
        # accepted by the autonomous CLI because its mouse events are global.
        try:
            tap_keys = frozenset(action.keys_down).intersection(action.keys_up)
            tap_buttons = frozenset(action.buttons_down).intersection(action.buttons_up)
            for key in action.keys_up:
                if key in tap_keys:
                    continue
                self._send_key(self._keycode(key), down=False)
                self._held_keys.discard(key.lower())
            for button in action.buttons_up:
                if button in tap_buttons:
                    continue
                button_id = _BUTTONS.get(button.lower())
                if button_id is None:
                    raise IsolationError(f"unsupported mouse button: {button!r}")
                self._park_pointer_in_game()
                self._xtest.fake_input(self._display, self._x.ButtonRelease, button_id)
                self._held_buttons.discard(button.lower())
            absolute_cursor = action.cursor_x is not None and action.cursor_y is not None
            if absolute_cursor:
                self._require_positive_input_permitted()
                assert action.cursor_x is not None and action.cursor_y is not None
                self._position_pointer_in_game(action.cursor_x, action.cursor_y)
            if action.mouse_dx or action.mouse_dy:
                relative_x, relative_y = _wine_relative_motion_delta(
                    action.mouse_dx,
                    action.mouse_dy,
                )
                self._require_positive_input_permitted()
                self._park_pointer_in_game()
                self._require_positive_input_permitted()
                self._relative_mouse.move(relative_x, relative_y)
            for key in action.keys_down:
                keycode = self._keycode(key)
                self._require_positive_input_permitted()
                self._send_key(keycode, down=True)
                self._held_keys.add(key.lower())
            for button in action.buttons_down:
                button_id = _BUTTONS.get(button.lower())
                if button_id is None:
                    raise IsolationError(f"unsupported mouse button: {button!r}")
                self._require_positive_input_permitted()
                if not absolute_cursor:
                    self._park_pointer_in_game()
                self._require_positive_input_permitted()
                self._xtest.fake_input(self._display, self._x.ButtonPress, button_id)
                self._held_buttons.add(button.lower())
            # A token present in both down/up sets is an atomic tap.  Keep the
            # release in the same accepted supervisor action so GUI clicks and
            # inventory toggles cannot remain held if the next frame stalls.
            # Flush and honor its bounded duration before release: Bedrock's
            # gameplay controls are sampled and can miss a zero-time pulse.
            if (tap_keys or tap_buttons) and action.duration_ms > 0:
                self._display.sync()
                time.sleep(action.duration_ms / 1000.0)
            for key in sorted(tap_keys):
                self._send_key(self._keycode(key), down=False)
                self._held_keys.discard(key.lower())
            for button in sorted(tap_buttons):
                button_id = _BUTTONS.get(button.lower())
                if button_id is None:
                    raise IsolationError(f"unsupported mouse button: {button!r}")
                self._xtest.fake_input(self._display, self._x.ButtonRelease, button_id)
                self._held_buttons.discard(button.lower())
            self._display.sync()
        finally:
            pass

    def type_chat(
        self,
        text: str,
        *,
        input_permitted: Callable[[], bool] = lambda: True,
    ) -> None:
        """Type one bounded ASCII chat message through the isolated input server."""
        self._require_live_lease()
        if not self.probe_target():
            self.release_all()
            raise IsolationError("Bedrock target window disappeared")
        if (
            not self._targeted
            and self._host_monitor_binding is not None
            and self.target_window_id is not None
        ):
            try:
                validate_host_monitor_window(
                    self._display,
                    self._host_monitor_binding,
                    target_window_id=self.target_window_id,
                    input_window_id=self._input_window_id,
                    require_focus=True,
                )
            except Exception:
                self.release_all()
                raise
        self._ensure_input_focus()
        if not text or len(text) > 256:
            raise IsolationError("chat text must contain 1..256 characters")
        if any(ord(char) < 32 or ord(char) > 126 for char in text):
            raise IsolationError("isolated chat actuator currently supports printable ASCII only")
        previous_keys = set(self._held_keys)
        previous_buttons = set(self._held_buttons)
        self.release_all()

        def require_permitted() -> None:
            if not input_permitted():
                raise IsolationError("chat input interlock is not clear")

        try:
            require_permitted()
            self._tap_key("t")
            self._display.sync()
            time.sleep(0.04)
            require_permitted()
            for char in text:
                require_permitted()
                self._type_ascii(char)
            require_permitted()
            self._tap_key("enter")
            # Bedrock's full chat screen retains two nested UI layers after a
            # submitted message (compose, then history). Return explicitly to
            # the world so the learned policy never receives chat pixels as a
            # playable scene. This behavior was verified on the managed client.
            time.sleep(0.04)
            require_permitted()
            self._tap_key("escape")
            time.sleep(0.04)
            require_permitted()
            self._tap_key("escape")
        except Exception:
            # Never strand the player in a half-typed chat overlay.
            if input_permitted():
                try:
                    self._tap_key("escape")
                except Exception:
                    pass
            raise
        finally:
            if input_permitted():
                for key in sorted(previous_keys):
                    self._send_key(self._keycode(key), down=True)
                for button in sorted(previous_buttons):
                    button_id = _BUTTONS.get(button)
                    if button_id is not None:
                        self._xtest.fake_input(self._display, self._x.ButtonPress, button_id)
                self._held_keys = previous_keys
                self._held_buttons = previous_buttons
            else:
                self._held_keys.clear()
                self._held_buttons.clear()
            self._display.sync()

    def _type_ascii(self, char: str) -> None:
        if char == " ":
            self._tap_key("space")
            return
        shifted = char in _SHIFTED_ASCII
        base = _SHIFT_MAP.get(char, char.lower() if char.isalpha() else char)
        if shifted:
            shift = self._keycode("shift")
            self._send_key(shift, down=True)
        try:
            self._tap_key(base)
        finally:
            if shifted:
                self._send_key(shift, down=False)

    def _tap_key(self, key: str) -> None:
        keycode = self._keycode(key)
        self._send_key(keycode, down=True)
        self._send_key(keycode, down=False)

    def release_all(self) -> None:
        try:
            try:
                self._ensure_input_focus()
            except Exception:
                # The target may already be gone during failsafe cleanup; keep
                # releasing every known token as a best-effort final sweep.
                pass
            for key in sorted(set(_KEYSYM_NAMES) | self._held_keys):
                try:
                    self._send_key(self._keycode(key), down=False)
                except Exception:
                    continue
            for button_id in sorted(set(_BUTTONS.values())):
                try:
                    self._park_pointer_in_game()
                    self._xtest.fake_input(self._display, self._x.ButtonRelease, button_id)
                except Exception:
                    continue
            self._display.sync()
        finally:
            self._held_keys.clear()
            self._held_buttons.clear()
            self.release_count += 1

    def close(self) -> None:
        try:
            self.release_all()
            self.clear_lease()
        finally:
            try:
                self._display.close()
            except Exception:
                pass


def _wine_relative_motion_delta(
    mouse_dx: int,
    mouse_dy: int,
) -> tuple[int, int]:
    """Map MineRL/VPT camera deltas onto Bedrock's relative pointer input.

    MineRL camera actions are ordered as positive pitch-down and positive
    yaw-right. Bedrock uses the same signs for relative XTEST motion. Keep these
    as deltas: absolute root coordinates passed with X.NONE are interpreted as
    large relative movements by the XTEST protocol.
    """
    return mouse_dx, mouse_dy


def _normalized_content_point(
    cursor_x: float,
    cursor_y: float,
    content_rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Map normalized captured-frame coordinates into its parent X window."""
    x, y, width, height = content_rect
    if width <= 0 or height <= 0:
        raise IsolationError("Minecraft content rectangle has invalid geometry")
    return (
        x + min(width - 1, max(0, int(round(cursor_x * (width - 1))))),
        y + min(height - 1, max(0, int(round(cursor_y * (height - 1))))),
    )


def _wine_content_rect(
    display: Any,
    target_window_id: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Resolve Wine's real Minecraft drawable relative to its desktop window."""
    try:
        target = display.create_resource_object("window", target_window_id)
        minecraft_window: Any | None = None
        for child in target.query_tree().children:
            name = str(child.get_wm_name() or "").lower()
            if "minecraft" in name:
                minecraft_window = child
                break
        if minecraft_window is None:
            return None
        candidates = list(minecraft_window.query_tree().children)
        if not candidates:
            return None
        client = max(
            candidates,
            key=lambda item: int(item.get_geometry().width) * int(item.get_geometry().height),
        )
        geometry = client.get_geometry()
        translated = target.translate_coords(client, 0, 0)
        horizontal_shift = _content_axis_reposition_delta(
            parent_size=width,
            start=int(translated.x),
            content_size=int(geometry.width),
        )
        if horizontal_shift:
            # Wine maximizes its decorated child at x=-8 while the actual game
            # drawable begins four pixels outside the desktop. Moving only the
            # decorated Minecraft window by that measured delta exposes the
            # complete game surface; it does not synthesize input or guess a
            # title-bar size. Once the drawable is contained, the normal
            # ConfigureNotify path below can fit it to the remaining viewport.
            minecraft_position = target.translate_coords(minecraft_window, 0, 0)
            minecraft_window.configure(x=int(minecraft_position.x) + horizontal_shift)
            display.sync()
            geometry = client.get_geometry()
            translated = target.translate_coords(client, 0, 0)
        available_width, available_height = _client_fit_dimensions(
            parent_width=width,
            parent_height=height,
            x=int(translated.x),
            y=int(translated.y),
        )
        if int(geometry.width) != available_width or int(geometry.height) != available_height:
            # A nested Weston surface can be resized by the host while Wine's
            # client drawable retains its previous dimensions. Resize only the
            # isolated Minecraft drawable to the exact visible client area;
            # this sends the normal X ConfigureNotify path and makes Bedrock
            # redraw its complete HUD without synthesizing game input.
            client.configure(width=available_width, height=available_height)
            display.sync()
            geometry = client.get_geometry()
            translated = target.translate_coords(client, 0, 0)
    except Exception:
        return None
    return _contained_content_rect(
        parent_width=width,
        parent_height=height,
        x=int(translated.x),
        y=int(translated.y),
        content_width=int(geometry.width),
        content_height=int(geometry.height),
    )


def resolve_host_monitor_content_rect(
    display_name: str,
    binding: HostMonitorBinding,
) -> tuple[int, int, int, int]:
    """Resolve the Wine client drawable without changing host-window geometry.

    A host-monitor ScreenCast contains the complete physical output, including
    Wine's desktop/title-bar pixels.  The agent must consume only the exact
    Minecraft drawable.  Unlike ``_wine_content_rect`` (which may repair a
    nested compositor), this host path is deliberately read-only: any clipped
    or unexpected geometry invalidates capture instead of moving or resizing a
    window on the operator's desktop.
    """
    if binding.display != display_name:
        raise IsolationError("host-monitor binding display does not match capture display")
    try:
        display_module = importlib.import_module("Xlib.display")
        display: Any = display_module.Display(display_name)
    except Exception as exc:
        raise IsolationError(f"cannot open host display {display_name}: {exc}") from exc
    try:
        validate_host_monitor_window(
            display,
            binding,
            target_window_id=binding.window_id,
        )
        target = display.create_resource_object("window", binding.window_id)
        minecraft_window: Any | None = None
        for child in target.query_tree().children:
            name = str(child.get_wm_name() or "").casefold()
            if "minecraft" in name:
                minecraft_window = child
                break
        if minecraft_window is None:
            raise IsolationError("cannot resolve Wine Minecraft window inside bound monitor")
        candidates = list(minecraft_window.query_tree().children)
        if not candidates:
            raise IsolationError("cannot resolve Wine Minecraft client drawable")
        client = max(
            candidates,
            key=lambda item: int(item.get_geometry().width) * int(item.get_geometry().height),
        )
        geometry = client.get_geometry()
        translated = target.translate_coords(client, 0, 0)
        return _contained_content_rect(
            parent_width=binding.monitor.width,
            parent_height=binding.monitor.height,
            x=int(translated.x),
            y=int(translated.y),
            content_width=int(geometry.width),
            content_height=int(geometry.height),
        )
    except IsolationError:
        raise
    except Exception as exc:
        raise IsolationError("cannot resolve bound Wine Minecraft drawable") from exc
    finally:
        display.close()


def _content_axis_reposition_delta(
    *,
    parent_size: int,
    start: int,
    content_size: int,
) -> int:
    """Return the minimal translation needed to expose one complete axis."""
    if parent_size <= 0 or content_size <= 0:
        raise IsolationError("Minecraft window has invalid geometry")
    if content_size > parent_size:
        return 0
    if start < 0:
        return -start
    overflow = start + content_size - parent_size
    return -overflow if overflow > 0 else 0


def _client_fit_dimensions(
    *,
    parent_width: int,
    parent_height: int,
    x: int,
    y: int,
) -> tuple[int, int]:
    """Return the drawable size that fills, but never exceeds, its parent."""
    if parent_width <= 0 or parent_height <= 0:
        raise IsolationError("Minecraft parent window has invalid geometry")
    if x < 0 or y < 0 or x >= parent_width or y >= parent_height:
        raise IsolationError("Minecraft client origin is outside the isolated compositor")
    return parent_width - x, parent_height - y


def _contained_content_rect(
    *,
    parent_width: int,
    parent_height: int,
    x: int,
    y: int,
    content_width: int,
    content_height: int,
) -> tuple[int, int, int, int]:
    """Require the complete game drawable before admitting a capture stream."""
    if content_width <= 0 or content_height <= 0:
        raise IsolationError("Minecraft client drawable has invalid geometry")
    if x < 0 or y < 0 or x + content_width > parent_width or y + content_height > parent_height:
        raise IsolationError(
            "Minecraft client drawable is clipped by the isolated compositor "
            f"({content_width}x{content_height}+{x}+{y} inside "
            f"{parent_width}x{parent_height}); relaunch Bedrock at 1920x1080 or larger"
        )
    return x, y, content_width, content_height


class IsolatedX11Capture:
    """Window-scoped BGRA capture from the same isolated X server as input."""

    def __init__(
        self,
        display_name: str,
        target_window_id: int,
        *,
        host_display: str | None = None,
        allow_host: bool = False,
    ) -> None:
        require_isolated_display(display_name, host_display, allow_host=allow_host)
        try:
            display_module = importlib.import_module("Xlib.display")
            self._X = importlib.import_module("Xlib.X")
        except ImportError as exc:
            raise IsolationError("python-xlib is required for X11 capture") from exc
        try:
            self._mss_module: Any | None = importlib.import_module("mss")
        except ImportError:
            self._mss_module = None
        self.display_name = display_name
        self.target_window_id = target_window_id
        self._display: Any = display_module.Display(display_name)
        self._frame_id = 0

    def _bounds(self) -> dict[str, int]:
        try:
            root = self._display.screen().root
            window = self._display.create_resource_object("window", self.target_window_id)
            geometry = window.get_geometry()
            translated = root.translate_coords(window, 0, 0)
        except Exception as exc:
            raise IsolationError("cannot resolve Bedrock window geometry") from exc
        width = int(geometry.width)
        height = int(geometry.height)
        if width <= 0 or height <= 0:
            raise IsolationError("Bedrock target window has invalid geometry")
        return {
            "left": int(translated.x),
            "top": int(translated.y),
            "width": width,
            "height": height,
        }

    def _content_rect(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        """Resolve Wine's real Minecraft client drawable inside its desktop window."""
        return _wine_content_rect(self._display, self.target_window_id, width, height)

    def capture(self) -> CapturedFrame:
        bounds = self._bounds()
        bgra_bytes: bytes = b""
        # Capture the target drawable first. Under nested Weston/Xwayland, root
        # capture can succeed while returning an all-black uncomposited buffer;
        # Wine's desktop window contains the actual Minecraft pixels.
        try:
            window = self._display.create_resource_object("window", self.target_window_id)
            raw = window.get_image(
                0,
                0,
                bounds["width"],
                bounds["height"],
                self._X.ZPixmap,
                0xFFFFFFFF,
            )
            bgra_bytes = raw.data
        except Exception:
            bgra_bytes = b""
        if not bgra_bytes and self._mss_module is not None:
            try:
                with self._mss_module.mss(display=self.display_name) as grabber:
                    shot = grabber.grab(bounds)
                    bgra_bytes = bytes(shot.bgra)
            except Exception:
                bgra_bytes = b""
        if not bgra_bytes:
            try:
                root = self._display.screen().root
                raw = root.get_image(
                    bounds["left"],
                    bounds["top"],
                    bounds["width"],
                    bounds["height"],
                    self._X.ZPixmap,
                    0xFFFFFFFF,
                )
                bgra_bytes = raw.data
            except Exception as exc:
                raise IsolationError(f"isolated X11 capture failed: {exc}") from exc
        content_rect = self._content_rect(bounds["width"], bounds["height"])
        frame_width = bounds["width"]
        frame_height = bounds["height"]
        if content_rect is not None:
            bgra_bytes = _crop_bgra(
                bgra_bytes,
                source_width=bounds["width"],
                rect=content_rect,
            )
            _, _, frame_width, frame_height = content_rect
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            captured_ns=time.monotonic_ns(),
            width=frame_width,
            height=frame_height,
            bgra=bgra_bytes,
        )

    def close(self) -> None:
        try:
            self._display.close()
        except Exception:
            pass


def _crop_bgra(
    source: bytes,
    *,
    source_width: int,
    rect: tuple[int, int, int, int],
) -> bytes:
    x, y, width, height = rect
    source_stride = source_width * 4
    if x == 0 and width == source_width:
        # The live Wine content crop removes only the top/bottom decoration.
        # Its rows are already contiguous, so one bounded slice avoids more
        # than a thousand Python-level row copies on every captured frame.
        start = y * source_stride
        return source[start : start + height * source_stride]
    row_bytes = width * 4
    output = bytearray(row_bytes * height)
    source_view = memoryview(source)
    output_view = memoryview(output)
    for row in range(height):
        source_start = (y + row) * source_stride + x * 4
        output_start = row * row_bytes
        output_view[output_start : output_start + row_bytes] = source_view[
            source_start : source_start + row_bytes
        ]
    return bytes(output)


def find_minecraft_window(
    display_name: str,
    *,
    host_display: str | None = None,
    allow_host: bool = False,
) -> int | None:
    """Find a visible Minecraft window on an already-isolated X display."""
    require_isolated_display(display_name, host_display, allow_host=allow_host)
    try:
        display_module = importlib.import_module("Xlib.display")
        display: Any = display_module.Display(display_name)
    except Exception as exc:
        raise IsolationError(f"cannot inspect isolated X display {display_name}: {exc}") from exc
    try:
        stack = list(display.screen().root.query_tree().children)
        fallback: int | None = None
        while stack:
            window = stack.pop()
            try:
                name = str(window.get_wm_name() or "")
                wm_class = window.get_wm_class() or ()
                combined = " ".join((name, *(str(value) for value in wm_class))).lower()
                attributes = window.get_attributes()
                if int(attributes.map_state) != 0:
                    if any(token in combined for token in ("minecraft", "bedrock", "wine")):
                        return int(window.id)
                    if fallback is None and window.id != display.screen().root.id:
                        fallback = int(window.id)
                stack.extend(window.query_tree().children)
            except Exception:
                continue
        return fallback
    finally:
        display.close()


def request_window_close(
    display_name: str,
    window_id: int,
    *,
    host_display: str | None = None,
    allow_host: bool = False,
) -> bool:
    """Request a normal game shutdown so BedrockOnLinux can clear GPU state."""
    require_isolated_display(display_name, host_display, allow_host=allow_host)
    try:
        display_module = importlib.import_module("Xlib.display")
        event_module = importlib.import_module("Xlib.protocol.event")
        display: Any = display_module.Display(display_name)
    except Exception as exc:
        raise IsolationError(f"cannot open X display {display_name}: {exc}") from exc
    try:
        window = display.create_resource_object("window", window_id)
        protocols = display.intern_atom("WM_PROTOCOLS")
        delete_window = display.intern_atom("WM_DELETE_WINDOW")
        event = event_module.ClientMessage(
            window=window,
            client_type=protocols,
            data=(32, [delete_window, int(time.time()), 0, 0, 0]),
        )
        window.send_event(event, event_mask=0)
        display.flush()
        return True
    except Exception:
        return False
    finally:
        display.close()
