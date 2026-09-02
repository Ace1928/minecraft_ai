from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import pickle
import select
import subprocess
import sys
import time
import types
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import PolicyConfig
from .motor import MotorIntent, MotorPolicy
from .perception import PerceptionBlackboard
from .perception_service import perceptual_hash_distance
from .platforms.bedrock_x11 import CapturedFrame
from .safety import MotorAction


class LearnedPolicyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keys: tuple[str, ...] = ()
    buttons: tuple[str, ...] = ()
    mouse_dx: int = 0
    mouse_dy: int = 0
    inference_ns: int = Field(ge=0)
    model_version: str


@dataclass
class PolicyServiceMetrics:
    requests: int = 0
    responses: int = 0
    deadline_misses: int = 0
    failures: int = 0
    scene_blocks: int = 0
    last_inference_ms: float = 0.0
    last_error: str | None = None


@dataclass
class TemporalPolicyClient:
    """Deadline-aware client for an isolated learned temporal policy process."""

    config: PolicyConfig
    frame_provider: Callable[[], CapturedFrame | None]
    policy_id: str = field(init=False)
    metrics: PolicyServiceMetrics = field(default_factory=PolicyServiceMetrics, init=False)
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _memory: shared_memory.SharedMemory | None = field(default=None, init=False)
    _memory_size: int = field(default=0, init=False)
    _held_keys: set[str] = field(default_factory=set, init=False)
    _held_buttons: set[str] = field(default_factory=set, init=False)
    _held_until_ns: int = field(default=0, init=False)
    _last_sequence: int = field(default=-1, init=False)
    _pending_request_id: str | None = field(default=None, init=False)
    _pending_deadline_ns: int = field(default=0, init=False)
    _pending_miss_recorded: bool = field(default=False, init=False)
    _consumed_miss_recorded: bool = field(default=False, init=False)
    _discard_pending_response: bool = field(default=False, init=False)
    _estimated_pitch_units: int = field(default=0, init=False)
    _camera_recovery_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _validate_policy_config(self.config)
        self.policy_id = f"learned:{self.config.provider}:{self.config.model_version}"

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        if sequence <= self._last_sequence:
            raise ValueError("motor policy sequence must increase monotonically")
        self._last_sequence = sequence
        if _learned_scene_blocked(blackboard):
            self.metrics.scene_blocks += 1
            return self._release(sequence)
        frame = self.frame_provider()
        if frame is None:
            return self._release(sequence)
        try:
            self._ensure_started(len(frame.bgra))
            response = self._consume_pending_response()
            if self._pending_request_id is not None:
                if time.monotonic_ns() <= self._pending_deadline_ns:
                    return self._hold(sequence)
                if not self._pending_miss_recorded:
                    self.metrics.deadline_misses += 1
                    self._pending_miss_recorded = True
                return self._release(sequence)
            if self._discard_pending_response:
                response = None
                self._discard_pending_response = False
            if response is None:
                self._submit(frame, intent, blackboard)
                return self._hold(sequence)
            output = LearnedPolicyOutput.model_validate(response["output"])
            self.metrics.responses += 1
            self.metrics.last_inference_ms = output.inference_ns / 1_000_000.0
            self.metrics.last_error = None
            if output.inference_ns > self.config.deadline_ms * 1_000_000:
                if not self._consumed_miss_recorded:
                    self.metrics.deadline_misses += 1
                self._submit(frame, intent, blackboard)
                return self._release(sequence)
            action = self._output_action(output, sequence)
            self._submit(frame, intent, blackboard)
            return action
        except Exception as exc:
            self.metrics.failures += 1
            self.metrics.last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            return self._release(sequence)

    def reset(self) -> MotorAction:
        sequence = self._last_sequence + 1
        if self._pending_request_id is not None:
            self._discard_pending_response = True
        process = self._process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"type":"reset"}\n')
                process.stdin.flush()
            except OSError:
                self.close()
        self._last_sequence = sequence
        return self._release(sequence)

    def status(self) -> dict[str, object]:
        process = self._process
        return {
            "policy_id": self.policy_id,
            "provider": self.config.provider,
            "model_version": self.config.model_version,
            "source_commit": self.config.source_commit,
            "license": self.config.license,
            "goal_conditioned": self.config.provider in {
                "minestudio-steve1",
                "minestudio-rocket2",
            },
            "grounding_mode": (
                "cross-view-object" if self.config.provider == "minestudio-rocket2" else "text"
            ),
            "temporal_memory": True,
            "research_only": self.config.research_only,
            "condition_scale": self.config.condition_scale,
            "camera_scale": self.config.camera_scale,
            "camera_max_step": self.config.camera_max_step,
            "camera_pitch_limit": self.config.camera_pitch_limit,
            "action_hold_ms": self.config.action_hold_ms,
            "estimated_pitch_units": self._estimated_pitch_units,
            "camera_recovery_active": self._camera_recovery_active,
            "process_alive": bool(process is not None and process.poll() is None),
            "requests": self.metrics.requests,
            "responses": self.metrics.responses,
            "deadline_misses": self.metrics.deadline_misses,
            "failures": self.metrics.failures,
            "scene_blocks": self.metrics.scene_blocks,
            "last_inference_ms": round(self.metrics.last_inference_ms, 3),
            "last_error": self.metrics.last_error,
        }

    def close(self) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"type":"stop"}\n')
                    process.stdin.flush()
                except OSError:
                    pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._process = None
        if self._memory is not None:
            self._memory.close()
            try:
                self._memory.unlink()
            except FileNotFoundError:
                pass
            self._memory = None
        self._memory_size = 0
        self._pending_request_id = None
        self._pending_deadline_ns = 0
        self._pending_miss_recorded = False
        self._consumed_miss_recorded = False
        self._discard_pending_response = False

    def warmup(self) -> None:
        """Load and verify the configured checkpoint before its first live action."""
        frame = self.frame_provider()
        if frame is None:
            raise RuntimeError("cannot warm learned policy without a captured frame")
        self._ensure_started(len(frame.bgra))

    def _ensure_started(self, required_size: int) -> None:
        if (
            self._process is not None
            and self._process.poll() is None
            and self._memory is not None
            and required_size <= self._memory_size
        ):
            return
        self.close()
        self._memory_size = required_size
        self._memory = shared_memory.SharedMemory(create=True, size=required_size)
        command = [
            self.config.python_path,
            "-m",
            "minecraft_ai.policy_service",
            "serve",
            "--provider",
            self.config.provider,
            "--shared-memory",
            self._memory.name,
            "--source-path",
            self.config.source_path,
            "--model-path",
            self.config.model_path,
            "--weights-path",
            self.config.weights_path,
            "--model-sha256",
            self.config.model_sha256,
            "--weights-sha256",
            self.config.weights_sha256,
            "--model-version",
            self.config.model_version,
            "--device",
            self.config.device,
            "--threads",
            str(self.config.threads),
            "--seed",
            str(self.config.seed),
            "--condition-scale",
            str(self.config.condition_scale),
            "--camera-scale",
            str(self.config.camera_scale),
        ]
        if self.config.stochastic:
            command.append("--stochastic")
        if self.config.deterministic_condition:
            command.append("--deterministic-condition")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_response(self.config.startup_timeout_s)
        if ready is None or ready.get("type") != "ready":
            raise RuntimeError(f"learned policy did not become ready: {ready}")

    def _consume_pending_response(self) -> dict[str, Any] | None:
        self._consumed_miss_recorded = False
        if self._pending_request_id is None:
            return None
        response = self._read_response(0.0)
        if response is None:
            return None
        request_id = self._pending_request_id
        self._pending_request_id = None
        self._pending_deadline_ns = 0
        self._consumed_miss_recorded = self._pending_miss_recorded
        self._pending_miss_recorded = False
        if response.get("request_id") != request_id:
            raise RuntimeError("learned policy response/request identity mismatch")
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error", "policy inference failed")))
        if response.get("type") != "prediction":
            raise RuntimeError(f"unexpected policy response: {response.get('type')}")
        return response

    def _submit(
        self,
        frame: CapturedFrame,
        intent: MotorIntent,
        blackboard: PerceptionBlackboard,
    ) -> None:
        assert self._memory is not None
        assert self._process is not None
        assert self._process.stdin is not None
        self._memory.buf[: len(frame.bgra)] = frame.bgra  # type: ignore[index]
        request_id = uuid.uuid4().hex
        deadline_ns = time.monotonic_ns() + self.config.deadline_ms * 1_000_000
        request = {
            "type": "infer",
            "request_id": request_id,
            "frame": {
                "width": frame.width,
                "height": frame.height,
                "length": len(frame.bgra),
                "captured_ns": frame.captured_ns,
            },
            "intent": self._conditioned_intent(intent, blackboard),
            "deadline_ns": deadline_ns,
        }
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        self.metrics.requests += 1
        self._pending_request_id = request_id
        self._pending_deadline_ns = deadline_ns
        self._pending_miss_recorded = False

    def _read_response(self, timeout_s: float) -> dict[str, Any] | None:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("learned policy process is not running")
        readable, _, _ = select.select([process.stdout], [], [], timeout_s)
        if not readable:
            if process.poll() is not None:
                raise RuntimeError(f"learned policy exited with code {process.returncode}")
            return None
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("learned policy closed its response stream")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError("learned policy returned a non-object response")
        return payload

    def _output_action(self, output: LearnedPolicyOutput, sequence: int) -> MotorAction:
        mouse_dx, mouse_dy = self._filter_camera(output.mouse_dx, output.mouse_dy)
        desired_keys = set(output.keys)
        desired_buttons = set(output.buttons)
        action = MotorAction(
            sequence=sequence,
            keys_down=tuple(sorted(desired_keys - self._held_keys)),
            keys_up=tuple(sorted(self._held_keys - desired_keys)),
            buttons_down=tuple(sorted(desired_buttons - self._held_buttons)),
            buttons_up=tuple(sorted(self._held_buttons - desired_buttons)),
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            duration_ms=50,
        )
        self._held_keys = desired_keys
        self._held_buttons = desired_buttons
        self._held_until_ns = time.monotonic_ns() + self.config.action_hold_ms * 1_000_000
        return action

    def _conditioned_intent(
        self,
        intent: MotorIntent,
        blackboard: PerceptionBlackboard | None = None,
    ) -> dict[str, object]:
        payload = intent.model_dump(mode="json")
        if blackboard is not None:
            latest = blackboard.latest()
            tracks = () if latest is None else latest.tracks
            candidates = [
                track
                for track in tracks
                if intent.target_label is None
                or track.label.casefold() == intent.target_label.casefold()
            ]
            if candidates:
                target = max(candidates, key=lambda track: track.confidence)
                payload["target_track"] = target.model_dump(mode="json")
        payload["interaction_id"] = _rocket_interaction_id(intent.mode)
        return payload

    def _filter_camera(self, mouse_dx: int, mouse_dy: int) -> tuple[int, int]:
        max_step = self.config.camera_max_step
        if max_step > 0:
            mouse_dx = max(-max_step, min(max_step, mouse_dx))
            mouse_dy = max(-max_step, min(max_step, mouse_dy))
        pitch_limit = self.config.camera_pitch_limit
        if pitch_limit > 0:
            proposed = self._estimated_pitch_units + mouse_dy
            bounded = max(-pitch_limit, min(pitch_limit, proposed))
            mouse_dy = bounded - self._estimated_pitch_units
            self._estimated_pitch_units = bounded
            # Saturation is sufficient: it preserves the learned controller's
            # task conditioning while preventing cumulative pitch runaway.
            # Replacing the task with a horizon-recovery prompt caused a live
            # deadlock because the policy could keep choosing the saturated
            # direction while all locomotion/interaction was suppressed.
            self._camera_recovery_active = False
        return mouse_dx, mouse_dy

    def _hold(self, sequence: int) -> MotorAction:
        """Respect the model's 50 ms action timing while inference is pending."""
        if self._held_keys or self._held_buttons:
            if time.monotonic_ns() >= self._held_until_ns:
                return self._release(sequence)
        return MotorAction(sequence=sequence, duration_ms=50)

    def _release(self, sequence: int) -> MotorAction:
        action = MotorAction(
            sequence=sequence,
            keys_up=tuple(sorted(self._held_keys)),
            buttons_up=tuple(sorted(self._held_buttons)),
        )
        self._held_keys.clear()
        self._held_buttons.clear()
        self._held_until_ns = 0
        return action


@dataclass
class GroundedPolicyRouter:
    """Route grounded interactions to ROCKET while STEVE handles open-ended goals.

    ROCKET is only eligible when the current blackboard contains a recent,
    confident localized target and the requested interaction belongs to its
    published interaction taxonomy. This keeps target masks evidence-backed and
    prevents an empty or stale VLM box from silently becoming motor ground truth.
    """

    primary: MotorPolicy
    grounded: MotorPolicy
    min_track_confidence: float = 0.65
    max_track_age_ms: int = 15_000
    policy_id: str = field(init=False)
    _active: MotorPolicy = field(init=False)
    _active_route: str = field(default="primary", init=False)
    _last_sequence: int = field(default=-1, init=False)
    _switches: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_track_age_ms <= 0:
            raise ValueError("max_track_age_ms must be positive")
        if not 0.0 <= self.min_track_confidence <= 1.0:
            raise ValueError("min_track_confidence must be in 0..1")
        self._active = self.primary
        self.policy_id = f"router:{self.primary.policy_id}+{self.grounded.policy_id}"

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        if sequence <= self._last_sequence:
            raise ValueError("motor policy sequence must increase monotonically")
        self._last_sequence = sequence
        route = "grounded" if self._has_grounded_target(blackboard, intent) else "primary"
        selected = self.grounded if route == "grounded" else self.primary
        release: MotorAction | None = None
        if selected is not self._active:
            release = self._active.reset()
            self._active = selected
            self._active_route = route
            self._switches += 1
        action = selected.act(blackboard, intent, sequence=sequence)
        return action if release is None else _merge_policy_release(action, release)

    def reset(self) -> MotorAction:
        sequence = self._last_sequence + 1
        primary_release = self.primary.reset()
        grounded_release = self.grounded.reset()
        self._last_sequence = sequence
        self._active = self.primary
        self._active_route = "primary"
        return _merge_policy_release(
            MotorAction(sequence=sequence),
            _merge_policy_release(primary_release, grounded_release),
        )

    def close(self) -> None:
        for policy in (self.primary, self.grounded):
            close = getattr(policy, "close", None)
            if callable(close):
                close()

    def warmup(self) -> None:
        """Preload both controllers so switching cannot stall live control."""
        for policy in (self.primary, self.grounded):
            warmup = getattr(policy, "warmup", None)
            if callable(warmup):
                warmup()

    def status(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "provider": "grounded-router",
            "active_route": self._active_route,
            "switches": self._switches,
            "min_track_confidence": self.min_track_confidence,
            "max_track_age_ms": self.max_track_age_ms,
            "primary": _policy_status(self.primary),
            "grounded": _policy_status(self.grounded),
        }

    def _has_grounded_target(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
    ) -> bool:
        if _rocket_interaction_id(intent.mode) < 0:
            return False
        latest = blackboard.latest()
        if latest is None:
            return False
        cutoff = time.monotonic_ns() - self.max_track_age_ms * 1_000_000
        return any(
            track.confidence >= self.min_track_confidence
            and (
                track.attributes.get("source") == "operator"
                or track.last_seen_ns >= cutoff
            )
            and (
                intent.target_label is None
                or track.label.casefold() == intent.target_label.casefold()
            )
            for track in latest.tracks
        )


def _policy_status(policy: MotorPolicy) -> dict[str, object]:
    status = getattr(policy, "status", None)
    if callable(status):
        reported = status()
        if isinstance(reported, dict):
            return reported
    return {"policy_id": policy.policy_id}


def _merge_policy_release(action: MotorAction, release: MotorAction) -> MotorAction:
    return MotorAction(
        sequence=action.sequence,
        keys_down=action.keys_down,
        keys_up=tuple(sorted(set(action.keys_up) | set(release.keys_up))),
        buttons_down=action.buttons_down,
        buttons_up=tuple(sorted(set(action.buttons_up) | set(release.buttons_up))),
        mouse_dx=action.mouse_dx,
        mouse_dy=action.mouse_dy,
        duration_ms=action.duration_ms,
    )


def _validate_policy_config(config: PolicyConfig) -> None:
    required = {
        "python_path": config.python_path,
        "source_path": config.source_path,
        "model_path": config.model_path,
        "weights_path": config.weights_path,
        "model_sha256": config.model_sha256,
        "weights_sha256": config.weights_sha256,
        "model_version": config.model_version,
        "source_commit": config.source_commit,
        "license": config.license,
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise ValueError(f"learned policy configuration is incomplete: {', '.join(missing)}")
    for key in ("python_path", "source_path", "model_path", "weights_path"):
        if not Path(required[key]).exists():
            raise ValueError(f"learned policy {key} does not exist: {required[key]}")
    if config.license.lower() != "mit" and not config.research_only:
        raise ValueError(f"unapproved learned policy license: {config.license}")
    if (
        config.camera_pitch_limit > 0
        and config.camera_recovery_release >= config.camera_pitch_limit
    ):
        raise ValueError("camera_recovery_release must be smaller than camera_pitch_limit")


def _learned_scene_blocked(blackboard: PerceptionBlackboard) -> bool:
    playable = blackboard.fact("scene.playable", min_confidence=0.7)
    if playable is None or bool(playable.value):
        return False
    if playable.source.startswith(("safety:", "bootstrap:")):
        return True
    if not playable.source.startswith("vlm:"):
        return False
    observed = blackboard.fact("scene.observation_dhash", min_confidence=1.0)
    current = blackboard.fact("frame.dhash", min_confidence=1.0)
    if observed is None or current is None:
        return False
    if not isinstance(observed.value, str) or not isinstance(current.value, str):
        return False
    try:
        return perceptual_hash_distance(observed.value, current.value) <= 6
    except ValueError:
        return False


@dataclass
class _VPTBackend:
    source_path: Path
    model_path: Path
    weights_path: Path
    model_sha256: str
    weights_sha256: str
    model_version: str
    device: str
    threads: int
    stochastic: bool
    camera_scale: float
    seed: int
    policy: Any = field(init=False)
    mapper: Any = field(init=False)
    transformer: Any = field(init=False)
    torch: Any = field(init=False)
    numpy: Any = field(init=False)
    cv2: Any = field(init=False)
    hidden_state: Any = field(init=False)
    first: Any = field(init=False)
    button_names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        _verify_sha256(self.model_path, self.model_sha256)
        _verify_sha256(self.weights_path, self.weights_sha256)
        self._install_minerl_import_stub()
        sys.path.insert(0, str(self.source_path))
        self.torch = importlib.import_module("torch")
        self.numpy = importlib.import_module("numpy")
        self.cv2 = importlib.import_module("cv2")
        self.torch.set_num_threads(self.threads)
        self.torch.set_num_interop_threads(1)
        self.torch.manual_seed(self.seed)
        action_mapping = importlib.import_module("lib.action_mapping")
        actions = importlib.import_module("lib.actions")
        policy_module = importlib.import_module("lib.policy")
        torch_util = importlib.import_module("lib.torch_util")
        gym3_types = importlib.import_module("gym3.types")
        torch_util.set_default_torch_device(self.torch.device(self.device))
        with self.model_path.open("rb") as handle:
            parameters = pickle.load(handle)  # noqa: S301 - hash-pinned official artifact
        policy_kwargs = parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.mapper = action_mapping.CameraHierarchicalMapping(n_camera_bins=11)
        action_space = gym3_types.DictType(**self.mapper.get_action_space_update())
        self.policy = policy_module.MinecraftAgentPolicy(
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            action_space=action_space,
        ).to(self.device)
        state = self.torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=True,
        )
        self.policy.load_state_dict(state, strict=False)
        self.policy.eval()
        self.transformer = actions.ActionTransformer(
            camera_binsize=2,
            camera_maxval=10,
            camera_mu=10,
            camera_quantization_scheme="mu_law",
        )
        self.button_names = tuple(actions.Buttons.ALL)
        self.first = self.torch.tensor([False], device=self.device)
        self.reset()

    def reset(self) -> None:
        self.hidden_state = self.policy.initial_state(1)

    def infer(self, frame: Any) -> LearnedPolicyOutput:
        started = time.perf_counter_ns()
        rgb = _center_crop_16_9(frame[:, :, [2, 1, 0]], self.numpy)
        resized = self.cv2.resize(rgb, (128, 128), interpolation=self.cv2.INTER_LINEAR)
        tensor = self.torch.from_numpy(resized[None].copy()).to(self.device)
        with self.torch.inference_mode():
            action, self.hidden_state, _ = self.policy.act(
                {"img": tensor},
                self.first,
                self.hidden_state,
                stochastic=self.stochastic,
            )
        raw = {
            "buttons": action["buttons"].cpu().numpy(),
            "camera": action["camera"].cpu().numpy(),
        }
        factored = self.mapper.to_factored(raw)
        decoded = self.transformer.policy2env(factored)
        return _decoded_policy_output(
            decoded,
            inference_ns=time.perf_counter_ns() - started,
            model_version=self.model_version,
            camera_scale=self.camera_scale,
        )

    @staticmethod
    def _install_minerl_import_stub() -> None:
        names = (
            "minerl",
            "minerl.herobraine",
            "minerl.herobraine.hero",
            "minerl.herobraine.hero.mc",
        )
        for name in names:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
        sys.modules["minerl.herobraine.hero.mc"].__dict__["MINERL_ITEM_MAP"] = {}


@dataclass
class _SteveOneBackend:
    """MineStudio STEVE-1 goal-conditioned temporal policy backend."""

    source_path: Path
    model_path: Path
    weights_path: Path
    model_sha256: str
    weights_sha256: str
    model_version: str
    device: str
    threads: int
    stochastic: bool
    deterministic_condition: bool
    condition_scale: float
    camera_scale: float
    seed: int
    policy: Any = field(init=False)
    mapper: Any = field(init=False)
    transformer: Any = field(init=False)
    torch: Any = field(init=False)
    numpy: Any = field(init=False)
    cv2: Any = field(init=False)
    hidden_state: Any = field(init=False, default=None)
    condition: Any = field(init=False, default=None)
    instruction: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        _verify_sha256(self.model_path, self.model_sha256)
        _verify_sha256(self.weights_path, self.weights_sha256)
        if self.model_path.parent != self.weights_path.parent:
            raise ValueError("STEVE-1 config and weights must share a checkpoint directory")
        sys.path.insert(0, str(self.source_path))
        self.torch = importlib.import_module("torch")
        self.numpy = importlib.import_module("numpy")
        self.cv2 = importlib.import_module("cv2")
        self.torch.set_num_threads(self.threads)
        self.torch.set_num_interop_threads(1)
        self.torch.manual_seed(self.seed)
        body = importlib.import_module("minestudio.models.steve_one.body")
        action_mapping = importlib.import_module("minestudio.utils.vpt_lib.action_mapping")
        actions = importlib.import_module("minestudio.utils.vpt_lib.actions")
        self.policy = body.SteveOnePolicy.from_pretrained(self.model_path.parent)
        self.policy = self.policy.to(self.device).eval()
        self.mapper = action_mapping.CameraHierarchicalMapping(n_camera_bins=11)
        self.transformer = actions.ActionTransformer(
            camera_binsize=2,
            camera_maxval=10,
            camera_mu=10,
            camera_quantization_scheme="mu_law",
        )
        self.reset()

    def reset(self) -> None:
        self.hidden_state = None
        self.condition = None
        self.instruction = None

    def infer(self, frame: Any, intent: dict[str, Any]) -> LearnedPolicyOutput:
        started = time.perf_counter_ns()
        instruction = _intent_instruction(intent)
        if instruction != self.instruction or self.condition is None:
            self.condition = self.policy.prepare_condition(
                {"cond_scale": self.condition_scale, "text": instruction},
                deterministic=self.deterministic_condition,
            )
            self.hidden_state = self.policy.initial_state(1, self.condition)
            self.instruction = instruction
        rgb = _center_crop_16_9(frame[:, :, [2, 1, 0]], self.numpy)
        resized = self.cv2.resize(rgb, (128, 128), interpolation=self.cv2.INTER_LINEAR)
        image = self.torch.from_numpy(resized[None, None].copy()).to(self.device)
        with self.torch.inference_mode():
            action, self.hidden_state = self.policy.get_action(
                {"image": image, "condition": self.condition},
                self.hidden_state,
                input_shape="BT*",
                deterministic=not self.stochastic,
            )
        raw = {
            "buttons": action["buttons"].cpu().numpy().reshape(1, 1),
            "camera": action["camera"].cpu().numpy().reshape(1, 1),
        }
        decoded = self.transformer.policy2env(self.mapper.to_factored(raw))
        return _decoded_policy_output(
            decoded,
            inference_ns=time.perf_counter_ns() - started,
            model_version=self.model_version,
            camera_scale=self.camera_scale,
        )


@dataclass
class _RocketTwoBackend:
    """Official ROCKET-2 cross-view grounded temporal policy backend."""

    source_path: Path
    model_path: Path
    weights_path: Path
    model_sha256: str
    weights_sha256: str
    model_version: str
    device: str
    threads: int
    stochastic: bool
    condition_scale: float
    camera_scale: float
    seed: int
    policy: Any = field(init=False)
    mapper: Any = field(init=False)
    transformer: Any = field(init=False)
    torch: Any = field(init=False)
    numpy: Any = field(init=False)
    cv2: Any = field(init=False)
    hidden_state: Any = field(init=False, default=None)
    reference_image: Any = field(init=False, default=None)
    reference_mask: Any = field(init=False, default=None)
    grounding_signature: str | None = field(init=False, default=None)
    previous_action: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        _verify_sha256(self.model_path, self.model_sha256)
        _verify_sha256(self.weights_path, self.weights_sha256)
        if self.model_path.parent != self.weights_path.parent:
            raise ValueError("ROCKET-2 config and weights must share a checkpoint directory")
        sys.path.insert(0, str(self.source_path))
        self.torch = importlib.import_module("torch")
        self.numpy = importlib.import_module("numpy")
        self.cv2 = importlib.import_module("cv2")
        self.torch.set_num_threads(self.threads)
        self.torch.set_num_interop_threads(1)
        self.torch.manual_seed(self.seed)
        rocket = importlib.import_module("model")
        cfg_wrapper = importlib.import_module("cfg_wrapper")
        action_mapping = importlib.import_module("minestudio.utils.vpt_lib.action_mapping")
        actions = importlib.import_module("minestudio.utils.vpt_lib.actions")
        # Published ROCKET-2 configs predate timm's source-prefix parser. The
        # registered architecture names are unchanged after removing "timm/".
        base = rocket.CrossViewRocket.from_pretrained(
            self.model_path.parent,
            view_backbone="vit_base_patch16_224.dino",
            mask_backbone="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        )
        base = base.to(self.device).eval()
        self.policy = (
            base
            if self.condition_scale == 0.0
            else cfg_wrapper.CFGWrapper(base, k=self.condition_scale)
        )
        self.mapper = action_mapping.CameraHierarchicalMapping(n_camera_bins=11)
        self.transformer = actions.ActionTransformer(
            camera_binsize=2,
            camera_maxval=10,
            camera_mu=10,
            camera_quantization_scheme="mu_law",
        )
        self.reset()

    def reset(self) -> None:
        self.hidden_state = None
        self.reference_image = None
        self.reference_mask = None
        self.grounding_signature = None
        self.previous_action = self._empty_previous_action()

    def infer(self, frame: Any, intent: dict[str, Any]) -> LearnedPolicyOutput:
        started = time.perf_counter_ns()
        rgb = _center_crop_16_9(frame[:, :, [2, 1, 0]], self.numpy)
        current = self.cv2.resize(rgb, (224, 224), interpolation=self.cv2.INTER_LINEAR)
        track = intent.get("target_track")
        interaction_id = int(intent.get("interaction_id", -1))
        signature = _rocket_grounding_signature(track, interaction_id)
        if signature != self.grounding_signature:
            self.hidden_state = None
            self.grounding_signature = signature
            if isinstance(track, dict):
                mask = _track_mask(frame.shape[0], frame.shape[1], track, self.numpy)
                mask = _center_crop_16_9(mask, self.numpy)
                self.reference_mask = self.cv2.resize(
                    mask,
                    (224, 224),
                    interpolation=self.cv2.INTER_NEAREST,
                ).astype(self.numpy.uint8)
                self.reference_image = current.copy()
            else:
                self.reference_mask = self.numpy.zeros((224, 224), dtype=self.numpy.uint8)
                self.reference_image = self.numpy.zeros((224, 224, 3), dtype=self.numpy.uint8)
        assert self.reference_image is not None
        assert self.reference_mask is not None
        observation = {
            "image": current,
            "env_prev_action": self.previous_action,
            "cross_view": {
                "cross_view_image": self.reference_image,
                "cross_view_obj_id": self.torch.tensor(interaction_id, dtype=self.torch.long),
                "cross_view_obj_mask": self.torch.from_numpy(self.reference_mask),
            },
        }
        with self.torch.inference_mode():
            action, self.hidden_state = self.policy.get_action(
                observation,
                self.hidden_state,
                input_shape="*",
                deterministic=not self.stochastic,
            )
        raw = {
            "buttons": action["buttons"].cpu().numpy().reshape(1, 1),
            "camera": action["camera"].cpu().numpy().reshape(1, 1),
        }
        decoded = self.transformer.policy2env(self.mapper.to_factored(raw))
        self.previous_action = self._decoded_previous_action(decoded)
        return _decoded_policy_output(
            decoded,
            inference_ns=time.perf_counter_ns() - started,
            model_version=self.model_version,
            camera_scale=self.camera_scale,
        )

    def _empty_previous_action(self) -> dict[str, Any]:
        return {
            "camera": self.numpy.zeros(2, dtype=self.numpy.float32),
            **{
                key.replace("_", "."): self.numpy.asarray(0, dtype=self.numpy.int64)
                for key in _ROCKET_BINARY_KEYS
            },
        }

    def _decoded_previous_action(self, decoded: dict[str, Any]) -> dict[str, Any]:
        return {
            "camera": self.numpy.asarray(decoded["camera"][0], dtype=self.numpy.float32),
            **{
                key.replace("_", "."): self.numpy.asarray(
                    int(bool(decoded[key.replace("_", ".")][0])),
                    dtype=self.numpy.int64,
                )
                for key in _ROCKET_BINARY_KEYS
            },
        }


_ROCKET_BINARY_KEYS = (
    "forward",
    "back",
    "left",
    "right",
    "inventory",
    "sprint",
    "sneak",
    "jump",
    "attack",
    "use",
    "hotbar_1",
    "hotbar_2",
    "hotbar_3",
    "hotbar_4",
    "hotbar_5",
    "hotbar_6",
    "hotbar_7",
    "hotbar_8",
    "hotbar_9",
)


def _rocket_interaction_id(mode: str) -> int:
    normalized = mode.casefold()
    if normalized in {"attack", "hunt", "combat"}:
        return 0
    if normalized in {"mine", "gather_wood", "gather", "break"}:
        return 2
    if normalized in {"use", "interact"}:
        return 3
    if normalized.startswith("craft") or normalized == "gui":
        return 4
    if normalized in {"switch", "hotbar"}:
        return 5
    if normalized in {"approach", "navigate"}:
        return 6
    return -1


def _rocket_grounding_signature(track: object, interaction_id: int) -> str:
    if not isinstance(track, dict):
        return f"ungrounded:{interaction_id}"
    return f"{track.get('track_id', 'unknown')}:{interaction_id}"


def _track_mask(height: int, width: int, track: dict[str, Any], numpy_module: Any) -> Any:
    region = track.get("region")
    if not isinstance(region, dict):
        raise ValueError("ROCKET-2 target track is missing a screen region")
    x0 = max(0, min(width - 1, int(float(region["x"]) * width)))
    y0 = max(0, min(height - 1, int(float(region["y"]) * height)))
    x1 = max(x0 + 1, min(width, int(float(region["x"] + region["width"]) * width)))
    y1 = max(y0 + 1, min(height, int(float(region["y"] + region["height"]) * height)))
    mask = numpy_module.zeros((height, width), dtype=numpy_module.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _intent_instruction(intent: dict[str, Any]) -> str:
    instruction = intent.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    skill_id = str(intent.get("skill_id") or "explore safely")
    return skill_id.replace("_", " ")


def _decoded_policy_output(
    decoded: dict[str, Any],
    *,
    inference_ns: int,
    model_version: str,
    camera_scale: float = 1.0,
) -> LearnedPolicyOutput:
    keys: set[str] = set()
    buttons: set[str] = set()
    key_map = {
        "forward": "w",
        "back": "s",
        "left": "a",
        "right": "d",
        "jump": "space",
        "sneak": "shift",
        "sprint": "ctrl",
        "inventory": "e",
        "drop": "q",
    }
    for source, target in key_map.items():
        if bool(decoded[source][0]):
            keys.add(target)
    for slot in range(1, 10):
        if bool(decoded[f"hotbar.{slot}"][0]):
            keys.add(str(slot))
    if bool(decoded["attack"][0]):
        buttons.add("left")
    if bool(decoded["use"][0]):
        buttons.add("right")
    pitch, yaw = decoded["camera"][0]
    return LearnedPolicyOutput(
        keys=tuple(sorted(keys)),
        buttons=tuple(sorted(buttons)),
        mouse_dx=int(round(float(yaw) * camera_scale)),
        mouse_dy=int(round(float(pitch) * camera_scale)),
        inference_ns=inference_ns,
        model_version=model_version,
    )


def _center_crop_16_9(frame: Any, numpy_module: Any) -> Any:
    height, width = frame.shape[:2]
    target_ratio = 16.0 / 9.0
    if width / height > target_ratio:
        crop_width = int(round(height * target_ratio))
        x0 = max(0, (width - crop_width) // 2)
        return numpy_module.ascontiguousarray(frame[:, x0 : x0 + crop_width])
    crop_height = int(round(width / target_ratio))
    y0 = max(0, (height - crop_height) // 2)
    return numpy_module.ascontiguousarray(frame[y0 : y0 + crop_height, :])


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _write_response(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _serve(args: argparse.Namespace) -> int:
    memory = shared_memory.SharedMemory(name=args.shared_memory)
    # The parent owns unlinking this segment. Python 3.12 otherwise registers
    # the attached child as a second owner and emits false leak warnings.
    resource_tracker.unregister(memory._name, "shared_memory")  # type: ignore[attr-defined]
    try:
        common = {
            "source_path": Path(args.source_path),
            "model_path": Path(args.model_path),
            "weights_path": Path(args.weights_path),
            "model_sha256": args.model_sha256,
            "weights_sha256": args.weights_sha256,
            "model_version": args.model_version,
            "device": args.device,
            "threads": args.threads,
            "stochastic": args.stochastic,
            "camera_scale": args.camera_scale,
            "seed": args.seed,
        }
        if args.provider == "openai-vpt":
            backend: _VPTBackend | _SteveOneBackend | _RocketTwoBackend = _VPTBackend(**common)
        elif args.provider == "minestudio-steve1":
            # Third-party model loaders may print status text. Keep stdout a
            # strict line-delimited JSON protocol for the realtime parent.
            with contextlib.redirect_stdout(sys.stderr):
                backend = _SteveOneBackend(
                    **common,
                    deterministic_condition=args.deterministic_condition,
                    condition_scale=args.condition_scale,
                )
        elif args.provider == "minestudio-rocket2":
            with contextlib.redirect_stdout(sys.stderr):
                backend = _RocketTwoBackend(
                    **common,
                    condition_scale=args.condition_scale,
                )
        else:
            raise ValueError(f"unsupported learned policy provider: {args.provider}")
        _write_response({"type": "ready", "model_version": args.model_version})
        for line in sys.stdin:
            request: Any = {}
            try:
                request = json.loads(line)
                request_type = request.get("type")
                if request_type == "stop":
                    return 0
                if request_type == "reset":
                    backend.reset()
                    continue
                if request_type != "infer":
                    raise ValueError(f"unknown request type: {request_type}")
                frame_spec = request["frame"]
                width = int(frame_spec["width"])
                height = int(frame_spec["height"])
                length = int(frame_spec["length"])
                if length != width * height * 4 or length > memory.size:
                    raise ValueError("invalid shared frame dimensions")
                frame = backend.numpy.ndarray(
                    (height, width, 4),
                    dtype=backend.numpy.uint8,
                    buffer=memory.buf,
                )
                if isinstance(backend, (_SteveOneBackend, _RocketTwoBackend)):
                    output = backend.infer(frame, request["intent"])
                else:
                    output = backend.infer(frame)
                _write_response(
                    {
                        "type": "prediction",
                        "request_id": request["request_id"],
                        "output": output.model_dump(mode="json"),
                    }
                )
            except Exception as exc:
                _write_response(
                    {
                        "type": "error",
                        "request_id": request.get("request_id")
                        if isinstance(request, dict)
                        else None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return 0
    finally:
        memory.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minecraft AI learned policy service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument(
        "--provider",
        choices=("openai-vpt", "minestudio-steve1", "minestudio-rocket2"),
        required=True,
    )
    serve.add_argument("--shared-memory", required=True)
    serve.add_argument("--source-path", required=True)
    serve.add_argument("--model-path", required=True)
    serve.add_argument("--weights-path", required=True)
    serve.add_argument("--model-sha256", required=True)
    serve.add_argument("--weights-sha256", required=True)
    serve.add_argument("--model-version", required=True)
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--threads", type=int, default=4)
    serve.add_argument("--seed", type=int, default=1928)
    serve.add_argument("--stochastic", action="store_true")
    serve.add_argument("--deterministic-condition", action="store_true")
    serve.add_argument("--condition-scale", type=float, default=4.0)
    serve.add_argument("--camera-scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
