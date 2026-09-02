from __future__ import annotations

import argparse
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
from .motor import MotorIntent
from .perception import PerceptionBlackboard
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
    _last_sequence: int = field(default=-1, init=False)
    _pending_request_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config.provider != "openai-vpt":
            raise ValueError(f"unsupported learned policy provider: {self.config.provider}")
        _validate_policy_config(self.config)
        self.policy_id = f"learned:{self.config.provider}:{self.config.model_version}"

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        del blackboard
        if sequence <= self._last_sequence:
            raise ValueError("motor policy sequence must increase monotonically")
        self._last_sequence = sequence
        frame = self.frame_provider()
        if frame is None:
            return self._release(sequence)
        try:
            self._ensure_started(len(frame.bgra))
            if not self._drain_pending_response():
                self.metrics.deadline_misses += 1
                return self._release(sequence)
            assert self._memory is not None
            assert self._process is not None
            assert self._process.stdin is not None
            memory = self._memory
            memory.buf[: len(frame.bgra)] = frame.bgra  # type: ignore[index]
            request_id = uuid.uuid4().hex
            request = {
                "type": "infer",
                "request_id": request_id,
                "frame": {
                    "width": frame.width,
                    "height": frame.height,
                    "length": len(frame.bgra),
                    "captured_ns": frame.captured_ns,
                },
                "intent": intent.model_dump(mode="json"),
                "deadline_ns": time.monotonic_ns() + self.config.deadline_ms * 1_000_000,
            }
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            self.metrics.requests += 1
            self._pending_request_id = request_id
            response = self._read_response(self.config.deadline_ms / 1000.0)
            if response is None:
                self.metrics.deadline_misses += 1
                return self._release(sequence)
            self._pending_request_id = None
            return self._response_action(response, sequence)
        except Exception as exc:
            self.metrics.failures += 1
            self.metrics.last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            return self._release(sequence)

    def reset(self) -> MotorAction:
        sequence = self._last_sequence + 1
        process = self._process
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"type":"reset"}\n')
                process.stdin.flush()
            except OSError:
                pass
        self._pending_request_id = None
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
            "goal_conditioned": False,
            "temporal_memory": True,
            "process_alive": bool(process is not None and process.poll() is None),
            "requests": self.metrics.requests,
            "responses": self.metrics.responses,
            "deadline_misses": self.metrics.deadline_misses,
            "failures": self.metrics.failures,
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
        ]
        if self.config.stochastic:
            command.append("--stochastic")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read_response(15.0)
        if ready is None or ready.get("type") != "ready":
            raise RuntimeError(f"learned policy did not become ready: {ready}")

    def _drain_pending_response(self) -> bool:
        if self._pending_request_id is None:
            return True
        response = self._read_response(0.0)
        if response is None:
            return False
        self._pending_request_id = None
        if response.get("type") == "error":
            self.metrics.failures += 1
            self.metrics.last_error = str(response.get("error", "policy inference failed"))
        return True

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

    def _response_action(self, response: dict[str, Any], sequence: int) -> MotorAction:
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error", "policy inference failed")))
        if response.get("type") != "prediction":
            raise RuntimeError(f"unexpected policy response: {response.get('type')}")
        output = LearnedPolicyOutput.model_validate(response["output"])
        self.metrics.responses += 1
        self.metrics.last_inference_ms = output.inference_ns / 1_000_000.0
        self.metrics.last_error = None
        desired_keys = set(output.keys)
        desired_buttons = set(output.buttons)
        action = MotorAction(
            sequence=sequence,
            keys_down=tuple(sorted(desired_keys - self._held_keys)),
            keys_up=tuple(sorted(self._held_keys - desired_keys)),
            buttons_down=tuple(sorted(desired_buttons - self._held_buttons)),
            buttons_up=tuple(sorted(self._held_buttons - desired_buttons)),
            mouse_dx=output.mouse_dx,
            mouse_dy=output.mouse_dy,
            duration_ms=50,
        )
        self._held_keys = desired_keys
        self._held_buttons = desired_buttons
        return action

    def _release(self, sequence: int) -> MotorAction:
        action = MotorAction(
            sequence=sequence,
            keys_up=tuple(sorted(self._held_keys)),
            buttons_up=tuple(sorted(self._held_buttons)),
        )
        self._held_keys.clear()
        self._held_buttons.clear()
        return action


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
    if config.license.lower() != "mit":
        raise ValueError(f"unapproved learned policy license: {config.license}")


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
            mouse_dx=int(round(float(yaw))),
            mouse_dy=int(round(float(pitch))),
            inference_ns=time.perf_counter_ns() - started,
            model_version=self.model_version,
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
        backend = _VPTBackend(
            source_path=Path(args.source_path),
            model_path=Path(args.model_path),
            weights_path=Path(args.weights_path),
            model_sha256=args.model_sha256,
            weights_sha256=args.weights_sha256,
            model_version=args.model_version,
            device=args.device,
            threads=args.threads,
            stochastic=args.stochastic,
            seed=args.seed,
        )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
