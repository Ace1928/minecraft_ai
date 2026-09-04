from __future__ import annotations

import csv
import io
import ipaddress
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .perception_service import bedrock_in_world_hud_present
from .platforms.bedrock_x11 import (
    CapturedFrame,
    IsolationError,
    require_isolated_display,
)


DEFAULT_BEDROCK_CONNECT_SERVERS = Path("/opt/bedrock-connect/custom_servers.json")


class MenuNavigationError(RuntimeError):
    """The bounded menu navigator could not prove its next safe action."""


class MenuStage(StrEnum):
    STARTUP_POPUP = "startup-popup"
    TITLE = "title"
    PLAY = "play"
    BEDROCK_CONNECT = "bedrock-connect"
    DEATH = "death"
    LOADING = "loading"
    IN_WORLD = "in-world"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OcrLine:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 100.0

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2


@dataclass(frozen=True)
class MenuObservation:
    frame: CapturedFrame
    lines: tuple[OcrLine, ...]
    stage: MenuStage

    @property
    def text(self) -> str:
        return " | ".join(line.text for line in self.lines)

    def summary(self, *, limit: int = 240) -> str:
        compact = " ".join(self.text.split())
        return compact[:limit] or "no readable text"


@dataclass(frozen=True)
class ConfiguredServer:
    name: str
    address: str
    port: int


@dataclass(frozen=True)
class MenuNavigationResult:
    server: ConfiguredServer
    actions: int
    visited: tuple[MenuStage, ...]
    elapsed_s: float

    def payload(self) -> dict[str, object]:
        return {
            "status": "in-world",
            "server": {
                "name": self.server.name,
                "address": self.server.address,
                "port": self.server.port,
            },
            "actions": self.actions,
            "visited": [stage.value for stage in self.visited],
            "elapsed_s": round(self.elapsed_s, 3),
        }


class FrameSource(Protocol):
    def capture(self) -> CapturedFrame: ...


class MenuTextReader(Protocol):
    def read(self, frame: CapturedFrame) -> tuple[OcrLine, ...]: ...


class MenuClickBackend(Protocol):
    def click(self, frame: CapturedFrame, x: int, y: int) -> None: ...


class TesseractMenuTextReader:
    """Deterministic local OCR for screenshots; no model server is involved."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        timeout_s: float = 8.0,
        input_scale: int = 3,
    ) -> None:
        selected = shutil.which("tesseract") if executable is None else executable
        if not selected:
            raise MenuNavigationError(
                "tesseract is required for fail-closed Bedrock menu recognition"
            )
        if input_scale not in (1, 2, 3):
            raise ValueError("menu OCR input scale must be 1, 2, or 3")
        self.executable = selected
        self.timeout_s = timeout_s
        self.input_scale = input_scale

    def read(self, frame: CapturedFrame) -> tuple[OcrLine, ...]:
        expected_bytes = frame.width * frame.height * 4
        if frame.width <= 0 or frame.height <= 0 or len(frame.bgra) != expected_bytes:
            raise MenuNavigationError("captured menu frame has invalid BGRA geometry")
        image = Image.frombytes(
            "RGB",
            (frame.width, frame.height),
            frame.bgra,
            "raw",
            "BGRX",
        )
        # Bedrock's pixel font is substantially more reliable in Tesseract at
        # enlarged nearest-neighbour scale (notably title anchors and
        # configured server names). Keep click geometry in original screenshot
        # coordinates when parsing.
        if self.input_scale != 1:
            image = image.resize(
                (image.width * self.input_scale, image.height * self.input_scale),
                Image.Resampling.NEAREST,
            )
        encoded = io.BytesIO()
        image.save(encoded, format="PNG", compress_level=1)
        try:
            completed = subprocess.run(
                [self.executable, "stdin", "stdout", "--psm", "11", "tsv"],
                input=encoded.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MenuNavigationError(f"menu OCR failed: {exc}") from exc
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise MenuNavigationError(f"menu OCR exited {completed.returncode}: {error[:200]}")
        return _parse_tesseract_tsv(
            completed.stdout.decode("utf-8", errors="replace"),
            coordinate_scale=self.input_scale,
        )


class NestedXTestMenuInput:
    """Absolute menu clicks constrained to a private nested X server.

    This intentionally exposes no movement keys, camera deltas, or held-input
    API and never registers with the general supervisor. Coordinates must come
    from the exact screenshot being acted upon and must still match the live
    Minecraft drawable geometry.
    """

    def __init__(
        self,
        display_name: str,
        target_window_id: int,
        *,
        host_display: str,
        input_permitted: Callable[[], bool] = lambda: True,
    ) -> None:
        require_isolated_display(display_name, host_display, allow_host=False)
        try:
            display_module = __import__("Xlib.display", fromlist=["Display"])
            self._x = __import__("Xlib.X", fromlist=["ButtonPress"])
            self._xtest = __import__("Xlib.ext.xtest", fromlist=["fake_input"])
            self._display: Any = display_module.Display(display_name)
        except (ImportError, OSError) as exc:
            raise IsolationError(f"cannot open nested XTEST menu backend: {exc}") from exc
        self.display_name = display_name
        self.target_window_id = target_window_id
        self._input_permitted = input_permitted
        try:
            self._window = _largest_minecraft_drawable(self._display, target_window_id)
            self.input_window_id = int(self._window.id)
            geometry = self._window.get_geometry()
            if int(geometry.width) <= 0 or int(geometry.height) <= 0:
                raise IsolationError("nested Minecraft menu drawable has invalid geometry")
        except Exception:
            self.close()
            raise

    def click(self, frame: CapturedFrame, x: int, y: int) -> None:
        geometry = self._window.get_geometry()
        width = int(geometry.width)
        height = int(geometry.height)
        if width != frame.width or height != frame.height:
            raise MenuNavigationError(
                "Minecraft menu geometry changed after recognition; refusing stale click"
            )
        if not 0 <= x < width or not 0 <= y < height:
            raise MenuNavigationError("recognized menu target lies outside Minecraft")
        try:
            self._window.get_attributes()
            self._window.set_input_focus(self._x.RevertToParent, self._x.CurrentTime)
            root = self._display.screen().root
            # python-xlib names this method from the destination's perspective:
            # destination.translate_coords(source, x, y). Translate the
            # screenshot/window point into private-X-server root coordinates.
            coordinates = root.translate_coords(self._window, x, y)
            root.warp_pointer(int(coordinates.x), int(coordinates.y))
            self._display.sync()
            # Bedrock can ignore a zero-duration synthetic click while its
            # title UI is animating. Give focus and the pointer a moment to
            # settle, then hold the button for a short human-scale interval.
            time.sleep(0.03)
            if not self._input_permitted():
                raise MenuNavigationError(
                    "menu input interlock is not clear; no click was sent"
                )
            self._xtest.fake_input(self._display, self._x.ButtonPress, 1)
            self._display.sync()
            time.sleep(0.075)
            self._xtest.fake_input(self._display, self._x.ButtonRelease, 1)
            self._display.sync()
        except MenuNavigationError:
            raise
        except Exception as exc:
            raise IsolationError(f"nested menu click failed: {exc}") from exc

    def close(self) -> None:
        display = getattr(self, "_display", None)
        if display is None:
            return
        try:
            self._xtest.fake_input(display, self._x.ButtonRelease, 1)
            display.sync()
        except Exception:
            pass
        try:
            display.close()
        except Exception:
            pass
        self._display = None


@dataclass(frozen=True)
class _Transition:
    target_text: tuple[str, ...]
    destination: MenuStage
    region: tuple[float, float, float, float]


class BedrockMenuNavigator:
    """Screenshot-gated finite-state navigator for the configured local world."""

    def __init__(
        self,
        *,
        capture: FrameSource,
        text_reader: MenuTextReader,
        click_backend: MenuClickBackend,
        lan_name: str,
        server: ConfiguredServer,
        timeout_s: float = 120.0,
        response_timeout_s: float = 8.0,
        poll_interval_s: float = 0.25,
        max_retries: int = 2,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        hud_detector: Callable[[CapturedFrame], bool] = bedrock_in_world_hud_present,
        input_permitted: Callable[[], bool] = lambda: True,
    ) -> None:
        if not lan_name.strip():
            raise ValueError("LAN world name is required")
        if timeout_s <= 0 or response_timeout_s <= 0 or poll_interval_s < 0:
            raise ValueError("menu navigation timing must be positive")
        if max_retries < 1 or max_retries > 5:
            raise ValueError("menu navigation retries must be in [1, 5]")
        self.capture = capture
        self.text_reader = text_reader
        self.click_backend = click_backend
        self.lan_name = lan_name.strip()
        self.server = server
        self.timeout_s = timeout_s
        self.response_timeout_s = response_timeout_s
        self.poll_interval_s = poll_interval_s
        self.max_retries = max_retries
        self.clock = clock
        self.sleep = sleep
        self.hud_detector = hud_detector
        self.input_permitted = input_permitted

    def run(self) -> MenuNavigationResult:
        started = self.clock()
        deadline = started + self.timeout_s
        self._require_input_permitted()
        observation = self._observe()
        visited: list[MenuStage] = [observation.stage]
        actions = 0

        while observation.stage == MenuStage.LOADING:
            observation = self._wait_loading(deadline)
            visited.append(observation.stage)

        while observation.stage != MenuStage.IN_WORLD:
            if observation.stage == MenuStage.ERROR:
                raise MenuNavigationError(
                    f"Bedrock reported an error screen: {observation.summary()}"
                )
            if observation.stage == MenuStage.UNKNOWN:
                raise MenuNavigationError(
                    f"unrecognized Bedrock screen; no input sent: {observation.summary()}"
                )
            transition = self._transition_for(observation)
            observation, used = self._perform_transition(
                observation,
                transition,
                deadline=deadline,
            )
            actions += used
            visited.append(observation.stage)

        return MenuNavigationResult(
            server=self.server,
            actions=actions,
            visited=tuple(visited),
            elapsed_s=self.clock() - started,
        )

    def _transition_for(self, observation: MenuObservation) -> _Transition:
        if observation.stage == MenuStage.STARTUP_POPUP:
            return _Transition(
                target_text=("skip for now", "not now", "continue", "ok", "let's go"),
                destination=MenuStage.TITLE,
                region=(0.10, 0.15, 0.90, 0.95),
            )
        if observation.stage == MenuStage.TITLE:
            return _Transition(
                target_text=("play",),
                destination=MenuStage.PLAY,
                region=(0.25, 0.20, 0.75, 0.70),
            )
        if observation.stage == MenuStage.PLAY:
            return _Transition(
                target_text=(self.lan_name,),
                destination=MenuStage.BEDROCK_CONNECT,
                region=(0.02, 0.10, 0.98, 0.98),
            )
        if observation.stage == MenuStage.BEDROCK_CONNECT:
            return _Transition(
                target_text=(self.server.name,),
                destination=MenuStage.IN_WORLD,
                region=(0.10, 0.10, 0.90, 0.95),
            )
        if observation.stage == MenuStage.DEATH:
            return _Transition(
                target_text=("respawn",),
                destination=MenuStage.IN_WORLD,
                region=(0.25, 0.52, 0.75, 0.75),
            )
        raise MenuNavigationError(
            f"no safe transition exists from {observation.stage.value}"
        )

    def _perform_transition(
        self,
        observation: MenuObservation,
        transition: _Transition,
        *,
        deadline: float,
    ) -> tuple[MenuObservation, int]:
        source = observation.stage
        for attempt in range(1, self.max_retries + 1):
            if self.clock() >= deadline:
                raise MenuNavigationError("Bedrock menu navigation timed out")
            self._require_input_permitted()
            line = _transition_click_target(observation, transition)
            x, y = line.center
            self.click_backend.click(observation.frame, x, y)
            response_deadline = min(deadline, self.clock() + self.response_timeout_s)
            while self.clock() < deadline:
                self.sleep(self.poll_interval_s)
                self._require_input_permitted()
                current = self._observe()
                if current.stage == transition.destination:
                    return current, attempt
                if current.stage == MenuStage.LOADING:
                    current = self._wait_loading(deadline)
                    if current.stage == transition.destination:
                        return current, attempt
                    self._raise_unexpected(source, transition.destination, current)
                if current.stage == source:
                    observation = current
                    if self.clock() < response_deadline:
                        continue
                    break
                # A menu transition can briefly contain only artwork or
                # partially rendered labels. Observe through that bounded
                # interval without issuing another input; fail closed if no
                # known destination appears before the deadline.
                if current.stage == MenuStage.UNKNOWN:
                    if self.clock() < response_deadline:
                        continue
                    self._raise_unexpected(source, transition.destination, current)
                self._raise_unexpected(source, transition.destination, current)
        raise MenuNavigationError(
            f"{source.value} did not transition to {transition.destination.value} "
            f"after {self.max_retries} bounded attempts"
        )

    def _require_input_permitted(self) -> None:
        if not self.input_permitted():
            raise MenuNavigationError(
                "menu input interlock is not clear; stopped before further input"
            )

    def _wait_loading(self, deadline: float) -> MenuObservation:
        while self.clock() < deadline:
            self.sleep(self.poll_interval_s)
            self._require_input_permitted()
            observation = self._observe()
            if observation.stage != MenuStage.LOADING:
                return observation
        raise MenuNavigationError("Bedrock remained on a loading screen until timeout")

    @staticmethod
    def _raise_unexpected(
        source: MenuStage,
        destination: MenuStage,
        observation: MenuObservation,
    ) -> None:
        if observation.stage == MenuStage.ERROR:
            raise MenuNavigationError(
                f"Bedrock error after {source.value}: {observation.summary()}"
            )
        if observation.stage == MenuStage.UNKNOWN:
            raise MenuNavigationError(
                f"unknown screen after {source.value}; stopped before further input: "
                f"{observation.summary()}"
            )
        raise MenuNavigationError(
            f"unexpected {observation.stage.value} after {source.value}; "
            f"expected {destination.value}"
        )

    def _observe(self) -> MenuObservation:
        frame = self.capture.capture()
        lines = self.text_reader.read(frame)
        stage = classify_menu_stage(
            frame,
            lines,
            lan_name=self.lan_name,
            server_name=self.server.name,
            hud_detector=self.hud_detector,
        )
        return MenuObservation(frame=frame, lines=lines, stage=stage)


def classify_menu_stage(
    frame: CapturedFrame,
    lines: tuple[OcrLine, ...],
    *,
    lan_name: str,
    server_name: str,
    hud_detector: Callable[[CapturedFrame], bool] = bedrock_in_world_hud_present,
) -> MenuStage:
    text = _normalized_text(" ".join(line.text for line in lines))
    compact = text.replace(" ", "")

    error_phrases = (
        "unable to connect",
        "connection failed",
        "failed to connect",
        "disconnected from server",
        "outdated client",
        "outdated server",
        "you were kicked",
        "world is unavailable",
        "server may not exist",
    )
    if any(_normalized_text(phrase) in text for phrase in error_phrases):
        return MenuStage.ERROR

    # Tesseract commonly renders Bedrock's block-font "YOU" as "TOU". Keep
    # the heading constrained to the upper screen so a chat message containing
    # "died" cannot turn into a menu action. The subsequent respawn action is
    # independently gated on the upper green button's visual rectangle.
    if any(
        line.center[1] <= frame.height * 0.45
        and frame.width * 0.20 <= line.center[0] <= frame.width * 0.80
        and _text_match_score(line.text, "you died") >= 0.80
        for line in lines
    ):
        return MenuStage.DEATH

    dismiss_phrases = ("skip for now", "not now", "continue", "ok", "let s go")
    popup_anchors = (
        "welcome to minecraft",
        "sign in with a microsoft",
        "global resources",
        "safe area",
        "privacy",
        "storage",
        "accessibility",
    )
    if any(phrase in text for phrase in dismiss_phrases) and any(
        phrase in text for phrase in popup_anchors
    ):
        return MenuStage.STARTUP_POPUP

    if (
        "serverlist" in compact
        or "connect to a server" in text
        or (
            _normalized_text(server_name).replace(" ", "") in compact
            and "manage server list" in text
        )
    ):
        return MenuStage.BEDROCK_CONNECT

    lan_compact = _normalized_text(lan_name).replace(" ", "")
    lan_label_visible = bool(lan_compact) and any(
        _text_match_score(line.text, lan_name) >= 0.84 for line in lines
    )
    play_tabs = {
        anchor
        for anchor in ("worlds", "realms", "servers")
        if any(_text_match_score(line.text, anchor) >= 0.86 for line in lines)
    }
    if lan_label_visible and (
        any(anchor in text for anchor in ("worlds", "lan games", "play"))
        or len(play_tabs) >= 2
    ):
        return MenuStage.PLAY

    if "minecraft" in text and "play" in text and any(
        anchor in text for anchor in ("settings", "marketplace", "dressing room")
    ):
        return MenuStage.TITLE

    # Bedrock 1.26's stylized logo and selected Play label are unreliable in
    # Tesseract. Accept the title without them only when the characteristic
    # wide central green Play control and at least two independently placed
    # title-menu labels are both present. The left-side "Play Now!" promotion
    # is outside this visual band and is never a target.
    if _find_green_title_play_control(frame) is not None and len(
        _matched_title_menu_anchors(frame, lines)
    ) >= 2:
        return MenuStage.TITLE

    loading_phrases = (
        "loading resource packs",
        "generating world",
        "locating server",
        "connecting to multiplayer game",
        "connecting to external server",
        "appearance is taking a long time to load",
        "proceeding with world generation",
        "loading",
    )
    loading_heading_visible = any(
        _text_match_score(line.text, target) >= 0.78
        for line in lines
        for target in ("generating world", "locating server")
    )
    if any(phrase in text for phrase in loading_phrases) or loading_heading_visible:
        return MenuStage.LOADING

    if hud_detector(frame):
        return MenuStage.IN_WORLD
    return MenuStage.UNKNOWN


def load_configured_local_server(
    path: Path = DEFAULT_BEDROCK_CONNECT_SERVERS,
    *,
    requested_name: str | None = None,
) -> ConfiguredServer:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MenuNavigationError(f"cannot read BedrockConnect servers from {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise MenuNavigationError("BedrockConnect custom server file must contain a list")
    servers: list[ConfiguredServer] = []
    for item in raw:
        if not isinstance(item, dict) or "content" in item:
            continue
        name = item.get("name")
        address = item.get("address")
        port = item.get("port")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(address, str)
            or not address.strip()
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65_535
        ):
            raise MenuNavigationError("BedrockConnect custom server entry is malformed")
        if not _is_local_server_address(address):
            continue
        servers.append(ConfiguredServer(name=name.strip(), address=address.strip(), port=port))

    if requested_name is not None:
        matches = [
            server
            for server in servers
            if server.name.casefold() == requested_name.casefold()
        ]
        if len(matches) != 1:
            raise MenuNavigationError(
                f"configured local server {requested_name!r} was not found uniquely in {path}"
            )
        return matches[0]
    if len(servers) != 1:
        names = [server.name for server in servers]
        raise MenuNavigationError(
            "exactly one local BedrockConnect custom server is required when --server-name "
            f"is omitted; found {names}"
        )
    return servers[0]


def _is_local_server_address(address: str) -> bool:
    normalized = address.strip().rstrip(".").casefold()
    if normalized == "localhost" or normalized.endswith(".local"):
        return True
    try:
        parsed = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return parsed.is_private or parsed.is_loopback or parsed.is_link_local


def _parse_tesseract_tsv(
    payload: str,
    *,
    coordinate_scale: int = 1,
) -> tuple[OcrLine, ...]:
    if coordinate_scale < 1:
        raise ValueError("OCR coordinate scale must be positive")
    grouped: dict[tuple[int, int, int, int], list[tuple[int, str, int, int, int, int, float]]] = {}
    reader = csv.DictReader(io.StringIO(payload), delimiter="\t")
    try:
        for row in reader:
            if row.get("level") != "5":
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            confidence = float(row.get("conf") or -1)
            if confidence < 30.0:
                continue
            key = (
                int(row.get("page_num") or 0),
                int(row.get("block_num") or 0),
                int(row.get("par_num") or 0),
                int(row.get("line_num") or 0),
            )
            word = (
                int(row.get("word_num") or 0),
                text,
                int(row.get("left") or 0),
                int(row.get("top") or 0),
                int(row.get("width") or 0),
                int(row.get("height") or 0),
                confidence,
            )
            grouped.setdefault(key, []).append(word)
    except (TypeError, ValueError) as exc:
        raise MenuNavigationError(f"invalid tesseract TSV output: {exc}") from exc

    lines: list[OcrLine] = []
    for words in grouped.values():
        words.sort(key=lambda item: item[0])
        left = min(word[2] for word in words)
        top = min(word[3] for word in words)
        right = max(word[2] + word[4] for word in words)
        bottom = max(word[3] + word[5] for word in words)
        lines.append(
            OcrLine(
                text=" ".join(word[1] for word in words),
                left=round(left / coordinate_scale),
                top=round(top / coordinate_scale),
                width=max(1, round((right - left) / coordinate_scale)),
                height=max(1, round((bottom - top) / coordinate_scale)),
                confidence=sum(word[6] for word in words) / len(words),
            )
        )
    lines.sort(key=lambda line: (line.top, line.left))
    return tuple(lines)


def _find_click_target(
    observation: MenuObservation,
    targets: tuple[str, ...],
    *,
    region: tuple[float, float, float, float],
) -> OcrLine:
    x0, y0, x1, y1 = region
    candidates: list[tuple[float, OcrLine]] = []
    for line in observation.lines:
        center_x, center_y = line.center
        normalized_x = center_x / observation.frame.width
        normalized_y = center_y / observation.frame.height
        if not x0 <= normalized_x <= x1 or not y0 <= normalized_y <= y1:
            continue
        for target in targets:
            score = _text_match_score(line.text, target)
            if score >= 0.86:
                candidates.append((score + line.confidence / 10_000.0, line))
    if not candidates:
        raise MenuNavigationError(
            f"expected clickable text {targets!r} was not recognized on "
            f"{observation.stage.value}: {observation.summary()}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 0.01:
        if candidates[0][1] != candidates[1][1]:
            raise MenuNavigationError(
                f"click target {targets!r} is ambiguous on {observation.stage.value}"
            )
    return candidates[0][1]


def _transition_click_target(
    observation: MenuObservation,
    transition: _Transition,
) -> OcrLine:
    try:
        return _find_click_target(
            observation,
            transition.target_text,
            region=transition.region,
        )
    except MenuNavigationError as original_error:
        if observation.stage == MenuStage.TITLE:
            target = _find_green_title_play_control(observation.frame)
            if target is None:
                raise MenuNavigationError(
                    "title screen was recognized, but the central green Play control "
                    "was not visually confirmed; no input sent"
                ) from None
            return target
        if observation.stage == MenuStage.DEATH:
            target = _find_green_respawn_control(observation.frame)
            if target is None:
                raise MenuNavigationError(
                    "death screen was recognized, but the upper green respawn control "
                    "was not visually confirmed; no input sent"
                ) from None
            return target
        if observation.stage == MenuStage.BEDROCK_CONNECT:
            target = _find_unique_server_name_prefix(
                observation,
                transition.target_text[0],
                region=transition.region,
            )
            if target is not None:
                return target
        raise original_error


def _find_unique_server_name_prefix(
    observation: MenuObservation,
    target: str,
    *,
    region: tuple[float, float, float, float],
) -> OcrLine | None:
    """Accept a unique, substantial OCR prefix of the configured server name."""

    expected_words = _normalized_text(target).split()
    if len(expected_words) < 2:
        return None
    expected_compact = "".join(expected_words)
    x0, y0, x1, y1 = region
    candidates: list[OcrLine] = []
    for line in observation.lines:
        center_x, center_y = line.center
        normalized_x = center_x / observation.frame.width
        normalized_y = center_y / observation.frame.height
        if not x0 <= normalized_x <= x1 or not y0 <= normalized_y <= y1:
            continue
        observed_words = _normalized_text(line.text).split()
        if len(observed_words) < 2 or observed_words[:2] != expected_words[:2]:
            continue
        observed_compact = "".join(observed_words)
        coverage = len(observed_compact) / len(expected_compact)
        if expected_compact.startswith(observed_compact) and coverage >= 0.55:
            candidates.append(line)
    return candidates[0] if len(candidates) == 1 else None


def _matched_title_menu_anchors(
    frame: CapturedFrame,
    lines: tuple[OcrLine, ...],
) -> set[str]:
    anchors = (
        ("settings", (0.28, 0.45, 0.72, 0.70)),
        ("realms", (0.28, 0.53, 0.72, 0.82)),
        ("marketplace", (0.28, 0.68, 0.72, 0.94)),
        ("profile", (0.00, 0.65, 0.32, 0.95)),
        ("dressing room", (0.65, 0.65, 1.00, 0.95)),
        ("social", (0.70, 0.00, 1.00, 0.25)),
    )
    matched: set[str] = set()
    for target, (x0, y0, x1, y1) in anchors:
        for line in lines:
            center_x, center_y = line.center
            if not (
                x0 <= center_x / frame.width <= x1
                and y0 <= center_y / frame.height <= y1
            ):
                continue
            if _text_match_score(line.text, target) >= 0.64:
                matched.add(target)
                break
    return matched


def _find_green_title_play_control(frame: CapturedFrame) -> OcrLine | None:
    return _find_dense_green_control(
        frame,
        region=(0.25, 0.32, 0.75, 0.55),
        minimum_width_fraction=0.24,
        minimum_height_fraction=0.05,
        description="visually confirmed title Play control",
    )


def _find_green_respawn_control(frame: CapturedFrame) -> OcrLine | None:
    """Locate the dense green upper death-screen control, conservatively.

    Bedrock's block-font Respawn label is often omitted by OCR. This detector
    searches only the known upper-button band and requires a wide, contiguous,
    densely green rectangle. It intentionally excludes the lower Game Menu
    control and returns no target when the geometry is ambiguous.
    """
    return _find_dense_green_control(
        frame,
        region=(0.25, 0.52, 0.75, 0.75),
        minimum_width_fraction=0.20,
        minimum_height_fraction=0.035,
        description="visually confirmed respawn control",
    )


def _find_dense_green_control(
    frame: CapturedFrame,
    *,
    region: tuple[float, float, float, float],
    minimum_width_fraction: float,
    minimum_height_fraction: float,
    description: str,
) -> OcrLine | None:
    expected_bytes = frame.width * frame.height * 4
    if frame.width < 64 or frame.height < 64 or len(frame.bgra) != expected_bytes:
        return None
    x_start, y_start, x_end, y_end = region
    x0, x1 = int(frame.width * x_start), int(frame.width * x_end)
    y0, y1 = int(frame.height * y_start), int(frame.height * y_end)
    source = memoryview(frame.bgra)

    def is_green(x: int, y: int) -> bool:
        offset = (y * frame.width + x) * 4
        blue, green, red = (int(value) for value in source[offset : offset + 3])
        return green >= 80 and green >= red * 1.18 and green >= blue * 1.05

    qualifying_rows: list[int] = []
    minimum_row_pixels = max(1, int(frame.width * minimum_width_fraction))
    for y in range(y0, y1):
        if sum(is_green(x, y) for x in range(x0, x1)) >= minimum_row_pixels:
            qualifying_rows.append(y)
    row_runs = _contiguous_runs(qualifying_rows)
    strong_rows = [run for run in row_runs if len(run) >= frame.height * minimum_height_fraction]
    if len(strong_rows) != 1:
        return None
    rows = strong_rows[0]

    qualifying_columns: list[int] = []
    minimum_column_pixels = max(1, int(len(rows) * 0.45))
    for x in range(x0, x1):
        if sum(is_green(x, y) for y in rows) >= minimum_column_pixels:
            qualifying_columns.append(x)
    column_runs = _contiguous_runs(qualifying_columns)
    strong_columns = [
        run for run in column_runs if len(run) >= frame.width * minimum_width_fraction
    ]
    if len(strong_columns) != 1:
        return None
    columns = strong_columns[0]
    return OcrLine(
        text=description,
        left=columns[0],
        top=rows[0],
        width=columns[-1] - columns[0] + 1,
        height=rows[-1] - rows[0] + 1,
        confidence=100.0,
    )


def _contiguous_runs(values: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values:
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def _text_match_score(candidate: str, target: str) -> float:
    candidate_compact = _normalized_text(candidate).replace(" ", "")
    target_compact = _normalized_text(target).replace(" ", "")
    if not candidate_compact or not target_compact:
        return 0.0
    if len(target_compact) <= 4:
        return 1.0 if candidate_compact == target_compact else 0.0
    if target_compact in candidate_compact:
        return 1.0
    return SequenceMatcher(a=candidate_compact, b=target_compact).ratio()


def _normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _largest_minecraft_drawable(display: Any, target_window_id: int) -> Any:
    target = display.create_resource_object("window", target_window_id)
    candidates: list[tuple[int, int, Any]] = []
    frontier: list[tuple[int, Any]] = [(0, target)]
    seen: set[int] = set()
    while frontier:
        depth, window = frontier.pop()
        window_id = int(getattr(window, "id", 0))
        if window_id <= 0 or window_id in seen or depth > 3:
            continue
        seen.add(window_id)
        try:
            geometry = window.get_geometry()
            area = int(geometry.width) * int(geometry.height)
            attributes = window.get_attributes()
            if area > 0 and int(attributes.map_state) != 0:
                candidates.append((area, depth, window))
            frontier.extend((depth + 1, child) for child in window.query_tree().children)
        except Exception:
            continue
    if not candidates:
        raise IsolationError("no viewable Minecraft menu drawable was found")
    largest_area = max(candidate[0] for candidate in candidates)
    large = [candidate for candidate in candidates if candidate[0] >= largest_area * 0.80]
    # Prefer the deepest near-fullscreen drawable: Wine's named outer window
    # can extend beyond the X root, while its child is the exact game surface.
    return max(large, key=lambda candidate: (candidate[1], candidate[0]))[2]
