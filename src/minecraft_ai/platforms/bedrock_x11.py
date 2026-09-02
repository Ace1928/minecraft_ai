from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Any

from ..safety import MotorAction, MotorLease, MotorRejected


class IsolationError(RuntimeError):
    pass


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
    ) -> None:
        require_isolated_display(display_name, host_display, allow_host=allow_host)
        try:
            display_module = importlib.import_module("Xlib.display")
            self._x = importlib.import_module("Xlib.X")
            self._xk = importlib.import_module("Xlib.XK")
            self._xtest = importlib.import_module("Xlib.ext.xtest")
        except ImportError as exc:
            raise IsolationError("python-xlib is required for isolated Bedrock X11 input") from exc
        self.display_name = display_name
        self.target_window_id = target_window_id
        try:
            self._display: Any = display_module.Display(display_name)
        except Exception as exc:
            raise IsolationError(f"cannot open isolated X display {display_name}: {exc}") from exc
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._lease: MotorLease | None = None
        self.release_count = 0
        self.live_capable = True
        if target_window_id is not None and not self.probe_target():
            self.close()
            raise IsolationError(f"target X window {target_window_id} is unavailable")

    @property
    def held_keys(self) -> frozenset[str]:
        return frozenset(self._held_keys)

    @property
    def held_buttons(self) -> frozenset[str]:
        return frozenset(self._held_buttons)

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

    def probe_target(self) -> bool:
        if self.target_window_id is None:
            return True
        try:
            window = self._display.create_resource_object("window", self.target_window_id)
            window.get_attributes()
        except Exception:
            return False
        return True

    def apply(self, action: MotorAction) -> None:
        self._require_live_lease()
        if not self.probe_target():
            self.release_all()
            raise IsolationError("Bedrock target window disappeared")
        for key in action.keys_up:
            self._xtest.fake_input(self._display, self._x.KeyRelease, self._keycode(key))
            self._held_keys.discard(key.lower())
        for button in action.buttons_up:
            button_id = _BUTTONS.get(button.lower())
            if button_id is None:
                raise IsolationError(f"unsupported mouse button: {button!r}")
            self._xtest.fake_input(self._display, self._x.ButtonRelease, button_id)
            self._held_buttons.discard(button.lower())
        if action.mouse_dx or action.mouse_dy:
            root = self._display.screen().root
            pointer = root.query_pointer()
            self._xtest.fake_input(
                self._display,
                self._x.MotionNotify,
                x=int(pointer.root_x) + action.mouse_dx,
                y=int(pointer.root_y) + action.mouse_dy,
            )
        for key in action.keys_down:
            self._xtest.fake_input(self._display, self._x.KeyPress, self._keycode(key))
            self._held_keys.add(key.lower())
        for button in action.buttons_down:
            button_id = _BUTTONS.get(button.lower())
            if button_id is None:
                raise IsolationError(f"unsupported mouse button: {button!r}")
            self._xtest.fake_input(self._display, self._x.ButtonPress, button_id)
            self._held_buttons.add(button.lower())
        self._display.sync()

    def type_chat(self, text: str) -> None:
        """Type one bounded ASCII chat message through the isolated input server."""
        self._require_live_lease()
        if not self.probe_target():
            self.release_all()
            raise IsolationError("Bedrock target window disappeared")
        if not text or len(text) > 256:
            raise IsolationError("chat text must contain 1..256 characters")
        if any(ord(char) < 32 or ord(char) > 126 for char in text):
            raise IsolationError("isolated chat actuator currently supports printable ASCII only")
        previous_keys = set(self._held_keys)
        previous_buttons = set(self._held_buttons)
        self.release_all()
        try:
            self._tap_key("t")
            self._display.sync()
            time.sleep(0.04)
            for char in text:
                self._type_ascii(char)
            self._tap_key("enter")
        except Exception:
            # Never strand the player in a half-typed chat overlay.
            try:
                self._tap_key("escape")
            except Exception:
                pass
            raise
        finally:
            for key in sorted(previous_keys):
                self._xtest.fake_input(self._display, self._x.KeyPress, self._keycode(key))
            for button in sorted(previous_buttons):
                button_id = _BUTTONS.get(button)
                if button_id is not None:
                    self._xtest.fake_input(self._display, self._x.ButtonPress, button_id)
            self._held_keys = previous_keys
            self._held_buttons = previous_buttons
            self._display.sync()

    def _type_ascii(self, char: str) -> None:
        if char == " ":
            self._tap_key("space")
            return
        shifted = char in _SHIFTED_ASCII
        base = _SHIFT_MAP.get(char, char.lower() if char.isalpha() else char)
        if shifted:
            shift = self._keycode("shift")
            self._xtest.fake_input(self._display, self._x.KeyPress, shift)
        try:
            self._tap_key(base)
        finally:
            if shifted:
                self._xtest.fake_input(self._display, self._x.KeyRelease, shift)

    def _tap_key(self, key: str) -> None:
        keycode = self._keycode(key)
        self._xtest.fake_input(self._display, self._x.KeyPress, keycode)
        self._xtest.fake_input(self._display, self._x.KeyRelease, keycode)

    def release_all(self) -> None:
        try:
            for key in sorted(set(_KEYSYM_NAMES) | self._held_keys):
                try:
                    self._xtest.fake_input(
                        self._display,
                        self._x.KeyRelease,
                        self._keycode(key),
                    )
                except Exception:
                    continue
            for button_id in sorted(set(_BUTTONS.values())):
                try:
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
            translated = window.translate_coords(root, 0, 0)
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
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            captured_ns=time.monotonic_ns(),
            width=bounds["width"],
            height=bounds["height"],
            bgra=bgra_bytes,
        )

    def close(self) -> None:
        try:
            self._display.close()
        except Exception:
            pass


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
