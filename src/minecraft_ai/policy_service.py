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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import PolicyConfig
from .motor import MotorIntent, MotorPolicy
from .perception import PerceptionBlackboard, PerceptionFact, ScreenRegion
from .perception_service import perceptual_hash_distance
from .platforms.bedrock_x11 import CapturedFrame
from .safety import MotorAction


class LearnedPolicyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keys: tuple[str, ...] = ()
    buttons: tuple[str, ...] = ()
    mouse_dx: int = 0
    mouse_dy: int = 0
    camera_semantics: Literal["world", "cursor"] = "world"
    inference_ns: int = Field(ge=0)
    model_version: str
    target_exists_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    target_point_yx: tuple[float, float] | None = None
    target_bbox_xyxy: tuple[float, float, float, float] | None = None
    scene_mode: Literal["world", "inventory", "chat", "unknown"] | None = None
    scene_playable: bool | None = None
    scene_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    scene_class_probabilities: dict[str, float] = Field(default_factory=dict)
    scene_model_version: str | None = None
    suppressed_actions: tuple[str, ...] = ()


@dataclass
class PolicyServiceMetrics:
    requests: int = 0
    responses: int = 0
    deadline_misses: int = 0
    failures: int = 0
    scene_blocks: int = 0
    camera_feedback_waits: int = 0
    last_inference_ms: float = 0.0
    last_error: str | None = None


@dataclass
class WorldCameraState:
    """Actuator-domain pitch shared by all policies driving one game view.

    Learned controllers retain independent temporal model states, but STEVE
    and ROCKET move the same physical Minecraft camera. A per-client pitch
    accumulator lets route switches spend the physical envelope repeatedly.
    """

    estimated_pitch_units: int = 0


@dataclass(frozen=True)
class GroundedTargetObservation:
    """One ROCKET auxiliary localization result tied to its captured frame."""

    observed_ns: int
    probability: float
    point_yx: tuple[float, float] | None
    bbox_xyxy: tuple[float, float, float, float] | None
    model_version: str


@dataclass(frozen=True)
class LearnedSceneObservation:
    """One learned scene belief tied to the exact consumed Bedrock frame."""

    observed_ns: int
    mode: Literal["world", "inventory", "chat", "unknown"]
    playable: bool | None
    confidence: float
    class_probabilities: dict[str, float]
    model_version: str


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
    _pending_frame_captured_ns: int = field(default=0, init=False)
    _consumed_frame_captured_ns: int = field(default=0, init=False)
    _discard_pending_response: bool = field(default=False, init=False)
    _world_camera_state: WorldCameraState = field(
        default_factory=WorldCameraState,
        init=False,
    )
    _camera_recovery_active: bool = field(default=False, init=False)
    _pending_camera: tuple[int, int] = field(default=(0, 0), init=False)
    _pending_camera_semantics: Literal["world", "cursor"] = field(
        default="world",
        init=False,
    )
    _last_prediction: LearnedPolicyOutput | None = field(default=None, init=False)
    _last_emitted_camera: tuple[int, int] = field(default=(0, 0), init=False)
    _predicted_camera_total: tuple[int, int] = field(default=(0, 0), init=False)
    _emitted_camera_total: tuple[int, int] = field(default=(0, 0), init=False)
    _accepted_predictions: int = field(default=0, init=False)
    _learned_action_counts: dict[str, int] = field(default_factory=dict, init=False)
    _last_target_observation: GroundedTargetObservation | None = field(
        default=None,
        init=False,
    )
    _last_scene_observation: LearnedSceneObservation | None = field(
        default=None,
        init=False,
    )
    _last_scene_feedback_ns: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _validate_policy_config(self.config)
        self.policy_id = f"learned:{self.config.provider}:{self.config.model_version}"

    def bind_world_camera_state(self, state: WorldCameraState) -> None:
        """Bind this controller to the physical camera shared with its peers."""
        self._world_camera_state = state

    def restore_world_camera_state(self, *, estimated_pitch_units: int) -> None:
        """Restore actuator pitch owned by the persistent supervisor."""
        self._world_camera_state.estimated_pitch_units = estimated_pitch_units

    def target_observation(self) -> GroundedTargetObservation | None:
        return self._last_target_observation

    def scene_observation(self) -> LearnedSceneObservation | None:
        return self._last_scene_observation

    def merge_perception(self, blackboard: PerceptionBlackboard) -> bool:
        observation = self._last_scene_observation
        if (
            observation is None
            or observation.observed_ns <= self._last_scene_feedback_ns
        ):
            return False
        updated = _merge_learned_scene_observation(
            blackboard,
            observation,
            policy_id=self.policy_id,
        )
        if updated:
            self._last_scene_feedback_ns = observation.observed_ns
        return updated

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
        if _learned_scene_blocked(blackboard, intent):
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
                if self._pending_camera != (0, 0):
                    # Do not ask the recurrent controller to react to a frame
                    # captured before its complete previous camera action.
                    # Drain the learned delta through the bounded actuator,
                    # then submit the first post-action frame on a later tick.
                    self.metrics.camera_feedback_waits += 1
                    return self._hold(sequence)
                self._submit(frame, intent, blackboard)
                return self._hold(sequence)
            output = LearnedPolicyOutput.model_validate(response["output"])
            self.metrics.responses += 1
            self.metrics.last_inference_ms = output.inference_ns / 1_000_000.0
            self.metrics.last_error = None
            if output.inference_ns > self.config.deadline_ms * 1_000_000:
                if not self._consumed_miss_recorded:
                    self.metrics.deadline_misses += 1
                return self._release(sequence)
            action = self._output_action(output, sequence)
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
            "scene_probe_interval": self.config.scene_probe_interval,
            "scene_model_version": self.config.scene_model_version or None,
            "scene_min_confidence": self.config.scene_min_confidence,
            "camera_scale": self.config.camera_scale,
            "gui_camera_scale": self.config.gui_camera_scale,
            "camera_max_step": self.config.camera_max_step,
            "camera_pitch_limit": self.config.camera_pitch_limit,
            "action_hold_ms": self.config.action_hold_ms,
            "estimated_pitch_units": self._world_camera_state.estimated_pitch_units,
            "camera_recovery_active": self._camera_recovery_active,
            "button_zero_order_hold": True,
            "last_prediction": (
                None
                if self._last_prediction is None
                else {
                    "keys": self._last_prediction.keys,
                    "buttons": self._last_prediction.buttons,
                    "mouse_dx": self._last_prediction.mouse_dx,
                    "mouse_dy": self._last_prediction.mouse_dy,
                    "camera_semantics": self._last_prediction.camera_semantics,
                    "target_exists_probability": (
                        self._last_prediction.target_exists_probability
                    ),
                    "target_point_yx": self._last_prediction.target_point_yx,
                    "target_bbox_xyxy": self._last_prediction.target_bbox_xyxy,
                    "scene_mode": self._last_prediction.scene_mode,
                    "scene_playable": self._last_prediction.scene_playable,
                    "scene_confidence": self._last_prediction.scene_confidence,
                    "scene_class_probabilities": (
                        self._last_prediction.scene_class_probabilities
                    ),
                    "scene_model_version": (
                        self._last_prediction.scene_model_version
                    ),
                    "suppressed_actions": self._last_prediction.suppressed_actions,
                }
            ),
            "last_emitted_camera": {
                "mouse_dx": self._last_emitted_camera[0],
                "mouse_dy": self._last_emitted_camera[1],
            },
            "pending_camera": {
                "mouse_dx": self._pending_camera[0],
                "mouse_dy": self._pending_camera[1],
            },
            "predicted_camera_total": {
                "mouse_dx": self._predicted_camera_total[0],
                "mouse_dy": self._predicted_camera_total[1],
            },
            "emitted_camera_total": {
                "mouse_dx": self._emitted_camera_total[0],
                "mouse_dy": self._emitted_camera_total[1],
            },
            "accepted_predictions": self._accepted_predictions,
            "target_observation_ns": (
                None
                if self._last_target_observation is None
                else self._last_target_observation.observed_ns
            ),
            "scene_observation": (
                None
                if self._last_scene_observation is None
                else {
                    "observed_ns": self._last_scene_observation.observed_ns,
                    "mode": self._last_scene_observation.mode,
                    "playable": self._last_scene_observation.playable,
                    "confidence": self._last_scene_observation.confidence,
                    "class_probabilities": (
                        self._last_scene_observation.class_probabilities
                    ),
                    "model_version": self._last_scene_observation.model_version,
                }
            ),
            "learned_action_counts": dict(sorted(self._learned_action_counts.items())),
            "process_alive": bool(process is not None and process.poll() is None),
            "requests": self.metrics.requests,
            "responses": self.metrics.responses,
            "deadline_misses": self.metrics.deadline_misses,
            "failures": self.metrics.failures,
            "scene_blocks": self.metrics.scene_blocks,
            "camera_feedback_waits": self.metrics.camera_feedback_waits,
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
        self._pending_frame_captured_ns = 0
        self._consumed_frame_captured_ns = 0
        self._discard_pending_response = False
        self._last_target_observation = None
        self._last_scene_observation = None
        self._last_scene_feedback_ns = 0
        self._pending_camera = (0, 0)
        self._pending_camera_semantics = "world"

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
            "--scene-probe-interval",
            str(self.config.scene_probe_interval),
            "--scene-model-path",
            self.config.scene_model_path,
            "--scene-model-sha256",
            self.config.scene_model_sha256,
            "--scene-model-version",
            self.config.scene_model_version,
            "--scene-min-confidence",
            str(self.config.scene_min_confidence),
            "--camera-scale",
            str(self.config.camera_scale),
            "--gui-camera-scale",
            str(self.config.gui_camera_scale),
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
        self._consumed_frame_captured_ns = self._pending_frame_captured_ns
        self._pending_request_id = None
        self._pending_deadline_ns = 0
        self._pending_frame_captured_ns = 0
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
        self._pending_frame_captured_ns = frame.captured_ns
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
        self._record_learned_action(output)
        self._last_prediction = output
        if (
            output.target_exists_probability is not None
            and self._consumed_frame_captured_ns > 0
        ):
            self._last_target_observation = GroundedTargetObservation(
                observed_ns=self._consumed_frame_captured_ns,
                probability=output.target_exists_probability,
                point_yx=output.target_point_yx,
                bbox_xyxy=output.target_bbox_xyxy,
                model_version=output.model_version,
            )
        if (
            output.scene_mode is not None
            and output.scene_confidence is not None
            and self._consumed_frame_captured_ns > 0
        ):
            self._last_scene_observation = LearnedSceneObservation(
                observed_ns=self._consumed_frame_captured_ns,
                mode=output.scene_mode,
                playable=output.scene_playable,
                confidence=output.scene_confidence,
                class_probabilities=dict(output.scene_class_probabilities),
                model_version=output.scene_model_version
                or f"{output.model_version}/mineclip",
            )
        self._predicted_camera_total = (
            self._predicted_camera_total[0] + output.mouse_dx,
            self._predicted_camera_total[1] + output.mouse_dy,
        )
        if (
            self._pending_camera != (0, 0)
            and output.camera_semantics != self._pending_camera_semantics
        ):
            # World motion is measured in calibrated relative counts while GUI
            # motion is measured in pointer pixels. Never add or replay one unit
            # after the interaction mode has switched to the other.
            self._pending_camera = (0, 0)
        self._pending_camera = (
            self._pending_camera[0] + output.mouse_dx,
            self._pending_camera[1] + output.mouse_dy,
        )
        self._pending_camera_semantics = output.camera_semantics
        camera_semantics = self._pending_camera_semantics
        mouse_dx, mouse_dy = self._drain_camera()
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
            camera_semantics=camera_semantics,
            duration_ms=50,
        )
        self._held_keys = desired_keys
        self._held_buttons = desired_buttons
        self._held_until_ns = time.monotonic_ns() + self.config.action_hold_ms * 1_000_000
        return action

    def _record_learned_action(self, output: LearnedPolicyOutput) -> None:
        """Expose what the checkpoint selected before actuator filtering.

        These counters distinguish learned decisions from 20 Hz state-hold and
        release actions. They make live camera/jump evaluation evidence-based
        without altering, boosting, or replacing the checkpoint's outputs.
        """
        self._accepted_predictions += 1
        keys = set(output.keys)
        buttons = set(output.buttons)
        labels = {
            "camera": bool(output.mouse_dx or output.mouse_dy),
            "jump": "space" in keys,
            "sprint_jump": {"ctrl", "space", "w"}.issubset(keys),
            "forward": "w" in keys,
            "inventory": "e" in keys,
            "attack": "left" in buttons,
            "use": "right" in buttons,
        }
        labels.update(
            {
                f"constraint_suppressed.{action}": True
                for action in output.suppressed_actions
            }
        )
        for label, selected in labels.items():
            if selected:
                self._learned_action_counts[label] = (
                    self._learned_action_counts.get(label, 0) + 1
                )

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

    def _drain_camera(self) -> tuple[int, int]:
        """Emit one smooth actuator slice without discarding learned motion.

        Policy camera outputs represent an angular delta for one source step.
        CPU inference is slower than the 20 Hz actuator, so clipping that delta
        once permanently loses part of the learned action. Queue the converted
        relative motion and drain it over live motor ticks instead.
        """

        pending_dx, pending_dy = self._pending_camera
        max_step = self.config.camera_max_step
        if max_step > 0:
            mouse_dx = max(-max_step, min(max_step, pending_dx))
            mouse_dy = max(-max_step, min(max_step, pending_dy))
        else:
            mouse_dx, mouse_dy = pending_dx, pending_dy
        remaining_dx = pending_dx - mouse_dx
        remaining_dy = pending_dy - mouse_dy
        pitch_limit = self.config.camera_pitch_limit
        if pitch_limit > 0 and self._pending_camera_semantics == "world":
            current_pitch = self._world_camera_state.estimated_pitch_units
            proposed = current_pitch + mouse_dy
            bounded = max(-pitch_limit, min(pitch_limit, proposed))
            bounded_dy = bounded - current_pitch
            if bounded_dy != mouse_dy:
                # The envelope rejected motion farther toward a pitch pole.
                # Discard queued motion in that same direction so it cannot
                # reappear after a later, valid correction moves away.
                remaining_dy = 0
            mouse_dy = bounded_dy
            self._world_camera_state.estimated_pitch_units = bounded
            # Saturation is sufficient: it preserves the learned controller's
            # task conditioning while preventing cumulative pitch runaway.
            # Replacing the task with a horizon-recovery prompt caused a live
            # deadlock because the policy could keep choosing the saturated
            # direction while all locomotion/interaction was suppressed.
            self._camera_recovery_active = False
        self._pending_camera = (remaining_dx, remaining_dy)
        if remaining_dx == 0 and remaining_dy == 0:
            self._pending_camera_semantics = "world"
        self._last_emitted_camera = (mouse_dx, mouse_dy)
        self._emitted_camera_total = (
            self._emitted_camera_total[0] + mouse_dx,
            self._emitted_camera_total[1] + mouse_dy,
        )
        return mouse_dx, mouse_dy

    def _hold(self, sequence: int) -> MotorAction:
        """Bound locomotion while preserving continuous interaction semantics.

        ROCKET and STEVE emit state, not clicks.  When inference is slower than
        the 20 Hz training clock, releasing an attack button at the locomotion
        hold boundary repeatedly cancels Minecraft's block-breaking progress.
        Keep buttons latched until the next prediction (or the request deadline)
        while releasing movement keys at the bounded sample-and-hold boundary.
        """
        keys_up: tuple[str, ...] = ()
        if self._held_keys and time.monotonic_ns() >= self._held_until_ns:
            keys_up = tuple(sorted(self._held_keys))
            self._held_keys.clear()
            self._held_until_ns = 0
        camera_semantics = self._pending_camera_semantics
        mouse_dx, mouse_dy = self._drain_camera()
        return MotorAction(
            sequence=sequence,
            keys_up=keys_up,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            camera_semantics=camera_semantics,
            duration_ms=50,
        )

    def _release(self, sequence: int) -> MotorAction:
        action = MotorAction(
            sequence=sequence,
            keys_up=tuple(sorted(self._held_keys)),
            buttons_up=tuple(sorted(self._held_buttons)),
        )
        self._held_keys.clear()
        self._held_buttons.clear()
        self._held_until_ns = 0
        self._pending_camera = (0, 0)
        self._pending_camera_semantics = "world"
        self._last_emitted_camera = (0, 0)
        return action


@dataclass
class GroundedPolicyRouter:
    """Route open-world, grounded, and blocking-GUI actions to learned experts.

    ROCKET is only eligible when the current blackboard contains a recent,
    confident localized target and the requested interaction belongs to its
    published interaction taxonomy. This keeps target masks evidence-backed and
    prevents an empty or stale VLM box from silently becoming motor ground truth.
    """

    primary: MotorPolicy
    grounded: MotorPolicy
    gui: MotorPolicy | None = None
    min_track_confidence: float = 0.65
    max_track_age_ms: int = 15_000
    target_confidence_alpha: float = 0.2
    target_near_min_screen_fraction: float = 0.10
    target_near_max_center_error: float = 0.35
    policy_id: str = field(init=False)
    _active: MotorPolicy = field(init=False)
    _active_route: str = field(default="primary", init=False)
    _last_sequence: int = field(default=-1, init=False)
    _switches: int = field(default=0, init=False)
    _grounded_track_id: str | None = field(default=None, init=False)
    _grounded_interaction_id: int | None = field(default=None, init=False)
    _grounded_confidence: float = field(default=0.0, init=False)
    _last_grounded_feedback_ns: int = field(default=0, init=False)
    _world_camera_state: WorldCameraState = field(
        default_factory=WorldCameraState,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.max_track_age_ms <= 0:
            raise ValueError("max_track_age_ms must be positive")
        if not 0.0 <= self.min_track_confidence <= 1.0:
            raise ValueError("min_track_confidence must be in 0..1")
        if not 0.0 < self.target_confidence_alpha <= 1.0:
            raise ValueError("target_confidence_alpha must be in (0, 1]")
        if not 0.0 < self.target_near_min_screen_fraction <= 1.0:
            raise ValueError("target_near_min_screen_fraction must be in (0, 1]")
        if not 0.0 <= self.target_near_max_center_error <= 2**0.5:
            raise ValueError("target_near_max_center_error is outside the normalized frame")
        self._active = self.primary
        policy_ids = [self.primary.policy_id, self.grounded.policy_id]
        if self.gui is not None:
            policy_ids.append(self.gui.policy_id)
        self.policy_id = "router:" + "+".join(policy_ids)
        for policy in self._policies():
            bind = getattr(policy, "bind_world_camera_state", None)
            if callable(bind):
                bind(self._world_camera_state)

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
        if self.gui is not None and intent.mode.casefold() == "death_gui":
            route = "gui"
            selected = self.gui
        elif self._has_grounded_target(blackboard, intent):
            route = "grounded"
            selected = self.grounded
        else:
            route = "primary"
            selected = self.primary
        release: MotorAction | None = None
        if selected is not self._active:
            release = self._active.reset()
            self._active = selected
            self._active_route = route
            self._switches += 1
        if route != "grounded":
            self._grounded_track_id = None
            self._grounded_interaction_id = None
            self._grounded_confidence = 0.0
            self._last_grounded_feedback_ns = 0
        action = selected.act(blackboard, intent, sequence=sequence)
        return action if release is None else _merge_policy_release(action, release)

    def reset(self) -> MotorAction:
        sequence = self._last_sequence + 1
        release = MotorAction(sequence=sequence)
        for policy in self._policies():
            release = _merge_policy_release(release, policy.reset())
        self._last_sequence = sequence
        self._active = self.primary
        self._active_route = "primary"
        self._grounded_track_id = None
        self._grounded_interaction_id = None
        self._grounded_confidence = 0.0
        self._last_grounded_feedback_ns = 0
        return release

    def close(self) -> None:
        for policy in self._policies():
            close = getattr(policy, "close", None)
            if callable(close):
                close()

    def warmup(self) -> None:
        """Preload every learned expert so an event-time switch cannot stall control."""
        for policy in self._policies():
            warmup = getattr(policy, "warmup", None)
            if callable(warmup):
                warmup()

    def status(self) -> dict[str, object]:
        status = {
            "policy_id": self.policy_id,
            "provider": "grounded-router",
            "active_route": self._active_route,
            "switches": self._switches,
            "min_track_confidence": self.min_track_confidence,
            "max_track_age_ms": self.max_track_age_ms,
            "target_confidence_alpha": self.target_confidence_alpha,
            "target_near_min_screen_fraction": self.target_near_min_screen_fraction,
            "target_near_max_center_error": self.target_near_max_center_error,
            "world_camera": {
                "estimated_pitch_units": self._world_camera_state.estimated_pitch_units,
            },
            "primary": _policy_status(self.primary),
            "grounded": _policy_status(self.grounded),
        }
        if self.gui is not None:
            status["gui"] = _policy_status(self.gui)
        return status

    def restore_world_camera_state(self, *, estimated_pitch_units: int) -> None:
        """Restore the one physical pitch accumulator shared by every route."""
        self._world_camera_state.estimated_pitch_units = estimated_pitch_units

    def _policies(self) -> tuple[MotorPolicy, ...]:
        return (
            (self.primary, self.grounded)
            if self.gui is None
            else (self.primary, self.grounded, self.gui)
        )

    def merge_perception(self, blackboard: PerceptionBlackboard) -> bool:
        """Merge STEVE scene belief and ROCKET target belief independently."""
        primary_merge = getattr(self.primary, "merge_perception", None)
        scene_updated = bool(callable(primary_merge) and primary_merge(blackboard))
        return self._merge_grounded_target_perception(blackboard) or scene_updated

    def _merge_grounded_target_perception(
        self,
        blackboard: PerceptionBlackboard,
    ) -> bool:
        """Publish ROCKET's learned current-view localization as online belief.

        The auxiliary head is an inference signal, not evaluator ground truth or
        a training label. An exponential temporal belief prevents one noisy
        frame from destroying an otherwise stable recurrent target lease.
        """
        if self._active_route != "grounded" or self._grounded_track_id is None:
            return False
        observe = getattr(self.grounded, "target_observation", None)
        if not callable(observe):
            return False
        observation = observe()
        if not isinstance(observation, GroundedTargetObservation):
            return False
        if observation.observed_ns <= self._last_grounded_feedback_ns:
            return False
        now_ns = time.monotonic_ns()
        if now_ns - observation.observed_ns > self.max_track_age_ms * 1_000_000:
            return False
        latest = blackboard.latest()
        if latest is None:
            return False
        target = next(
            (
                track
                for track in latest.tracks
                if track.track_id == self._grounded_track_id
            ),
            None,
        )
        if target is None:
            return False
        raw_confidence = max(0.0, min(1.0, observation.probability))
        alpha = self.target_confidence_alpha
        self._grounded_confidence = (
            raw_confidence
            if self._grounded_confidence <= 0.0
            else alpha * raw_confidence + (1.0 - alpha) * self._grounded_confidence
        )
        region = target.region
        point_yx = _crop_point_to_full(
            latest.width,
            latest.height,
            observation.point_yx,
        )
        bbox_xyxy = _crop_bbox_to_full(
            latest.width,
            latest.height,
            observation.bbox_xyxy,
        )
        if raw_confidence >= self.min_track_confidence and bbox_xyxy is not None:
            x0, y0, x1, y1 = bbox_xyxy
            if x1 > x0 and y1 > y0:
                region = ScreenRegion(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        source = (
            f"learned:{self.grounded.policy_id}:aux-localization:not-training-label"
        )
        attributes = dict(target.attributes)
        attributes.update(
            {
                "tracking_source": source,
                "tracking_model_version": observation.model_version,
                "target_exists_probability": round(raw_confidence, 6),
            }
        )
        updated_track = target.model_copy(
            update={
                "confidence": self._grounded_confidence,
                "region": region,
                "last_seen_ns": observation.observed_ns,
                "attributes": attributes,
            }
        )
        visible = self._grounded_confidence >= self.min_track_confidence
        fact_values: list[tuple[str, str | int | float | bool, float]] = [
            ("target.exists_probability", raw_confidence, 1.0),
            ("target.tracking_confidence", self._grounded_confidence, 1.0),
            (
                "target.visible",
                visible,
                max(self._grounded_confidence, 1.0 - self._grounded_confidence),
            ),
            ("target.kind", target.label, self._grounded_confidence),
        ]
        if point_yx is not None and raw_confidence >= self.min_track_confidence:
            point_y, point_x = point_yx
            fact_values.extend(
                (
                    ("target.dx", max(-1.0, min(1.0, 2.0 * point_x - 1.0)), raw_confidence),
                    ("target.dy", max(-1.0, min(1.0, 2.0 * point_y - 1.0)), raw_confidence),
                )
            )
        if bbox_xyxy is not None and raw_confidence >= self.min_track_confidence:
            x0, y0, x1, y1 = bbox_xyxy
            screen_fraction = max(0.0, (x1 - x0) * (y1 - y0))
            center_x = (x0 + x1) / 2.0
            center_y = (y0 + y1) / 2.0
            center_error = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
            proximity = min(1.0, screen_fraction / self.target_near_min_screen_fraction)
            near = (
                screen_fraction >= self.target_near_min_screen_fraction
                and center_error <= self.target_near_max_center_error
            )
            fact_values.extend(
                (
                    ("target.screen_fraction", screen_fraction, raw_confidence),
                    ("target.center_error", center_error, raw_confidence),
                    ("target.proximity", proximity, raw_confidence),
                    ("target.near", near, raw_confidence),
                )
            )
        facts = tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=confidence,
                observed_ns=observation.observed_ns,
                source=source,
                expires_after_ms=min(1500, self.max_track_age_ms),
            )
            for key, value, confidence in fact_values
        )
        self._last_grounded_feedback_ns = observation.observed_ns
        track_updated = blackboard.upsert_semantic_track(
            instance_id=latest.instance_id,
            track=updated_track,
        )
        facts_updated = blackboard.merge_semantics(
            instance_id=latest.instance_id,
            facts=facts,
        )
        return track_updated and facts_updated

    def _has_grounded_target(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
    ) -> bool:
        interaction_id = _rocket_interaction_id(intent.mode)
        if interaction_id < 0:
            return False
        latest = blackboard.latest()
        if latest is None:
            return False
        candidates = tuple(
            track
            for track in latest.tracks
            if track.confidence >= self.min_track_confidence
            and (
                intent.target_label is None
                or track.label.casefold() == intent.target_label.casefold()
            )
        )
        # Once ROCKET has captured a valid reference image, preserve that
        # recurrent option while the same target and interaction remain active.
        # A stale rectangle alone cannot re-arm a process; a verified persisted
        # cross-view image can, because it contains the actual visual reference.
        if (
            self._active_route == "grounded"
            and interaction_id == self._grounded_interaction_id
            and any(track.track_id == self._grounded_track_id for track in candidates)
        ):
            return True
        cutoff = time.monotonic_ns() - self.max_track_age_ms * 1_000_000
        current_hash = blackboard.fact("frame.dhash", min_confidence=1.0)
        selected = next(
            (
                track
                for track in candidates
                if track.last_seen_ns >= cutoff
                or _operator_reference_matches(track, current_hash)
                or _operator_reference_artifact_available(track)
            ),
            None,
        )
        if selected is None:
            return False
        self._grounded_track_id = selected.track_id
        self._grounded_interaction_id = interaction_id
        self._grounded_confidence = selected.confidence
        self._last_grounded_feedback_ns = 0
        return True


def _operator_reference_matches(track: object, current_hash: object) -> bool:
    if not hasattr(track, "attributes") or not hasattr(current_hash, "value"):
        return False
    attributes = track.attributes
    if attributes.get("source") != "operator":
        return False
    reference = attributes.get("reference_dhash")
    observed = current_hash.value
    if not isinstance(reference, str) or not isinstance(observed, str):
        return False
    try:
        return perceptual_hash_distance(reference, observed) <= 6
    except ValueError:
        return False


def _operator_reference_artifact_available(track: object) -> bool:
    if not hasattr(track, "attributes"):
        return False
    attributes = track.attributes
    if attributes.get("source") != "operator":
        return False
    path = attributes.get("reference_image_path")
    digest = attributes.get("reference_image_sha256")
    return (
        isinstance(path, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and Path(path).is_file()
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
        camera_semantics=action.camera_semantics,
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
    scene_model = {
        "scene_model_path": config.scene_model_path,
        "scene_model_sha256": config.scene_model_sha256,
        "scene_model_version": config.scene_model_version,
    }
    if any(scene_model.values()) and not all(scene_model.values()):
        missing_scene = sorted(key for key, value in scene_model.items() if not value)
        raise ValueError(
            "learned scene model configuration is incomplete: "
            + ", ".join(missing_scene)
        )
    if all(scene_model.values()):
        if config.provider != "minestudio-steve1":
            raise ValueError("learned scene model is currently supported by STEVE-1 only")
        if not Path(config.scene_model_path).exists():
            raise ValueError(
                f"learned scene model does not exist: {config.scene_model_path}"
            )
    if config.license.lower() != "mit" and not config.research_only:
        raise ValueError(f"unapproved learned policy license: {config.license}")
    if (
        config.camera_pitch_limit > 0
        and config.camera_recovery_release >= config.camera_pitch_limit
    ):
        raise ValueError("camera_recovery_release must be smaller than camera_pitch_limit")


def _learned_scene_blocked(
    blackboard: PerceptionBlackboard,
    intent: MotorIntent | None = None,
) -> bool:
    if intent is not None and intent.mode.casefold() in {
        "gui",
        "death_gui",
        "close_inventory",
    }:
        return False
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


_MINECLIP_SCENE_PROMPTS = {
    "world": "playing Minecraft in the world",
    "inventory": "open Minecraft inventory and crafting menu",
    "chat": "open Minecraft chat screen",
    "wall": "looking directly at a wall",
}


def _fast_scene_belief(
    probabilities: dict[str, float],
    *,
    min_confidence: float,
) -> tuple[
    Literal["world", "inventory", "chat", "unknown"],
    bool | None,
    float,
    dict[str, float],
]:
    """Convert the evidence-gated Bedrock scene head into a safe verdict."""
    required = {"world", "inventory"}
    if set(probabilities) != required:
        missing = sorted(required - set(probabilities))
        extra = sorted(set(probabilities) - required)
        raise ValueError(
            f"invalid fast scene classes: missing={missing}, extra={extra}"
        )
    normalized = {
        key: max(0.0, min(1.0, float(probabilities[key])))
        for key in sorted(required)
    }
    mode = max(normalized, key=normalized.__getitem__)
    confidence = normalized[mode]
    if confidence < min_confidence:
        return "unknown", None, confidence, normalized
    if mode == "inventory":
        return "inventory", False, confidence, normalized
    return "world", True, confidence, normalized


def _mineclip_scene_belief(
    probabilities: dict[str, float],
) -> tuple[
    Literal["world", "inventory", "chat", "unknown"],
    bool | None,
    float,
    dict[str, float],
]:
    """Calibrate MineCLIP prompt scores into a conservative online belief.

    The world class intentionally groups ordinary world views with close wall
    views. This distinction is precisely where the old top-band interlock and
    generic VLM narration were unreliable. Ambiguous prompt scores publish no
    playable verdict and therefore cannot silently authorize motor control.
    """
    required = set(_MINECLIP_SCENE_PROMPTS)
    if set(probabilities) != required:
        missing = sorted(required - set(probabilities))
        extra = sorted(set(probabilities) - required)
        raise ValueError(f"invalid MineCLIP scene classes: missing={missing}, extra={extra}")
    normalized = {
        key: max(0.0, min(1.0, float(probabilities[key]))) for key in sorted(required)
    }
    inventory = normalized["inventory"]
    chat = normalized["chat"]
    world = normalized["world"] + normalized["wall"]
    if inventory >= 0.65 and inventory > chat and inventory > world:
        return "inventory", False, inventory, normalized
    if chat >= 0.45 and chat > inventory and chat > world:
        return "chat", False, chat, normalized
    if world >= 0.50 and world > inventory and world > chat:
        return "world", True, min(1.0, world), normalized
    return "unknown", None, max(inventory, chat, min(1.0, world)), normalized


def _merge_learned_scene_observation(
    blackboard: PerceptionBlackboard,
    observation: LearnedSceneObservation,
    *,
    policy_id: str,
) -> bool:
    latest = blackboard.raw_latest()
    if latest is None:
        return False
    now_ns = time.monotonic_ns()
    if now_ns - observation.observed_ns > 10_000_000_000:
        return False
    source = f"learned:{policy_id}:scene:{observation.model_version}"
    values: list[tuple[str, str | int | float | bool, float]] = [
        ("perception.scene_confidence", observation.confidence, 1.0),
        ("perception.scene_model", observation.model_version, 1.0),
    ]
    values.extend(
        (
            f"perception.scene_probability.{label}",
            probability,
            1.0,
        )
        for label, probability in sorted(observation.class_probabilities.items())
    )
    if observation.playable is not None:
        values.extend(
            (
                ("scene.mode", observation.mode, observation.confidence),
                ("scene.playable", observation.playable, observation.confidence),
                ("scene.ui_overlay", not observation.playable, observation.confidence),
            )
        )
    facts = tuple(
        PerceptionFact(
            key=key,
            value=value,
            confidence=confidence,
            observed_ns=observation.observed_ns,
            source=source,
            expires_after_ms=5_000,
        )
        for key, value, confidence in values
    )
    return blackboard.merge_semantics(instance_id=latest.instance_id, facts=facts)


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
    gui_camera_scale: float
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

    def infer(
        self,
        frame: Any,
        intent: dict[str, Any] | None = None,
    ) -> LearnedPolicyOutput:
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
            camera_scale=_intent_camera_scale(
                intent or {},
                world_scale=self.camera_scale,
                gui_scale=self.gui_camera_scale,
            ),
            camera_semantics=_intent_camera_semantics(intent or {}),
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
    scene_probe_interval: int
    scene_model_path: Path | None
    scene_model_sha256: str
    scene_model_version: str
    scene_min_confidence: float
    camera_scale: float
    gui_camera_scale: float
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
    inference_count: int = field(init=False, default=0)
    scene_model: Any = field(init=False, default=None)
    discrete_actions_emitted: set[str] = field(init=False, default_factory=set)

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
        if self.scene_model_path is not None:
            _verify_sha256(self.scene_model_path, self.scene_model_sha256)
            self.scene_model = self.torch.jit.load(
                self.scene_model_path,
                map_location=self.device,
            ).eval()
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
        self.inference_count = 0
        self.discrete_actions_emitted.clear()

    def infer(self, frame: Any, intent: dict[str, Any]) -> LearnedPolicyOutput:
        started = time.perf_counter_ns()
        self.inference_count += 1
        instruction = _intent_instruction(intent)
        if instruction != self.instruction or self.condition is None:
            self.condition = self.policy.prepare_condition(
                {"cond_scale": self.condition_scale, "text": instruction},
                deterministic=self.deterministic_condition,
            )
            self.hidden_state = self.policy.initial_state(1, self.condition)
            self.instruction = instruction
            self.discrete_actions_emitted.clear()
        rgb = _center_crop_16_9(frame[:, :, [2, 1, 0]], self.numpy)
        resized = self.cv2.resize(rgb, (128, 128), interpolation=self.cv2.INTER_LINEAR)
        image = self.torch.from_numpy(resized[None, None].copy()).to(self.device)
        scene_belief: tuple[
            Literal["world", "inventory", "chat", "unknown"],
            bool | None,
            float,
            dict[str, float],
            str,
        ] | None = None
        with self.torch.inference_mode():
            if self._should_probe_scene(intent):
                scene_belief = self._infer_scene(frame)
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
        decoded, scene_suppressed = _apply_observed_scene_action_contract(
            decoded,
            intent,
            scene_belief,
        )
        decoded, event_suppressed = _apply_discrete_action_contract(
            decoded,
            intent,
            self.discrete_actions_emitted,
        )
        decoded, constraint_suppressed = _apply_action_constraints(decoded, intent)
        suppressed = (*scene_suppressed, *event_suppressed, *constraint_suppressed)
        scene_mode = None if scene_belief is None else scene_belief[0]
        scene_playable = None if scene_belief is None else scene_belief[1]
        scene_confidence = None if scene_belief is None else scene_belief[2]
        scene_probabilities = {} if scene_belief is None else scene_belief[3]
        scene_model_version = None if scene_belief is None else scene_belief[4]
        return _decoded_policy_output(
            decoded,
            inference_ns=time.perf_counter_ns() - started,
            model_version=self.model_version,
            camera_scale=_intent_camera_scale(
                intent,
                world_scale=self.camera_scale,
                gui_scale=self.gui_camera_scale,
            ),
            camera_semantics=_intent_camera_semantics(intent),
            scene_mode=scene_mode,
            scene_playable=scene_playable,
            scene_confidence=scene_confidence,
            scene_class_probabilities=scene_probabilities,
            scene_model_version=scene_model_version,
            suppressed_actions=suppressed,
        )

    def _should_probe_scene(self, intent: dict[str, Any]) -> bool:
        mode = str(intent.get("mode") or "").casefold()
        if mode in {"gui", "close_inventory"} or mode.startswith("craft"):
            return True
        interval = self.scene_probe_interval
        return interval > 0 and (
            self.inference_count == 1 or self.inference_count % interval == 0
        )

    def _infer_scene(
        self,
        frame: Any,
    ) -> tuple[
        Literal["world", "inventory", "chat", "unknown"],
        bool | None,
        float,
        dict[str, float],
        str,
    ]:
        if self.scene_model is not None:
            rgb = self.numpy.ascontiguousarray(frame[:, :, [2, 1, 0]])
            resized = self.cv2.resize(
                rgb,
                (160, 96),
                interpolation=self.cv2.INTER_AREA,
            )
            image = (
                self.torch.from_numpy(
                    resized.transpose(2, 0, 1)[None].copy()
                )
                .float()
                .div_(255.0)
                .to(self.device)
            )
            logits = self.scene_model(image)
            probabilities = self.torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
            belief = _fast_scene_belief(
                dict(zip(("world", "inventory"), probabilities, strict=True)),
                min_confidence=self.scene_min_confidence,
            )
            return (*belief, self.scene_model_version)
        # MineCLIP is already resident inside STEVE-1. Its published visual
        # resolution is 160x256, so this event-time head adds no new checkpoint
        # or process and avoids routing every frame through a general VLM.
        rgb = self.numpy.ascontiguousarray(frame[:, :, [2, 1, 0]])
        resized = self.cv2.resize(rgb, (256, 160), interpolation=self.cv2.INTER_LINEAR)
        video = (
            self.torch.from_numpy(resized.transpose(2, 0, 1)[None, None].copy())
            .to(self.device)
        )
        features = self.policy.mineclip.encode_video(video)
        logits, _ = self.policy.mineclip.forward_reward_head(
            features,
            text_tokens=list(_MINECLIP_SCENE_PROMPTS.values()),
        )
        probabilities = self.torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
        belief = _mineclip_scene_belief(
            dict(zip(_MINECLIP_SCENE_PROMPTS, probabilities, strict=True))
        )
        return (*belief, f"{self.model_version}/mineclip")


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
    gui_camera_scale: float
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
                reference = _rocket_reference_frame(track, frame, self.cv2)
                reference_rgb = reference[:, :, [2, 1, 0]]
                mask = _track_mask(
                    reference.shape[0],
                    reference.shape[1],
                    track,
                    self.numpy,
                )
                mask = _center_crop_16_9(mask, self.numpy)
                self.reference_mask = self.cv2.resize(
                    mask,
                    (224, 224),
                    interpolation=self.cv2.INTER_NEAREST,
                ).astype(self.numpy.uint8)
                reference_rgb = _center_crop_16_9(reference_rgb, self.numpy)
                self.reference_image = self.cv2.resize(
                    reference_rgb,
                    (224, 224),
                    interpolation=self.cv2.INTER_LINEAR,
                )
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
        decoded = _rocket_action_contract(decoded)
        decoded, suppressed = _apply_action_constraints(decoded, intent)
        self.previous_action = self._decoded_previous_action(decoded)
        exists_probability, point_yx, bbox_xyxy = _rocket_target_estimate(self.policy)
        return _decoded_policy_output(
            decoded,
            inference_ns=time.perf_counter_ns() - started,
            model_version=self.model_version,
            camera_scale=_intent_camera_scale(
                intent,
                world_scale=self.camera_scale,
                gui_scale=self.gui_camera_scale,
            ),
            camera_semantics=_intent_camera_semantics(intent),
            target_exists_probability=exists_probability,
            target_point_yx=point_yx,
            target_bbox_xyxy=bbox_xyxy,
            suppressed_actions=suppressed,
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


def _rocket_action_contract(decoded: dict[str, Any]) -> dict[str, Any]:
    """Mask actions outside ROCKET-2's published interaction controller.

    The released ROCKET-2 recurrent previous-action contract omits ``drop``.
    Some generic MineStudio action mappers still decode that bit from the
    shared VPT button vocabulary. Never let an unsupported bit discard the
    operator's inventory during a grounded interaction.
    """

    drop = decoded.get("drop")
    if drop is None:
        return decoded
    safe_drop = drop.copy()
    safe_drop[...] = 0
    return {**decoded, "drop": safe_drop}


def _apply_action_constraints(
    decoded: dict[str, Any],
    intent: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Mask explicit strategic prohibitions without replacing learned behavior."""
    parameters = intent.get("parameters")
    if not isinstance(parameters, dict):
        return decoded, ()
    constrained = decoded
    suppressed: list[str] = []
    action_permissions: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("allow_attack", ("attack",)),
        ("allow_use", ("use",)),
        ("allow_jump", ("jump",)),
        ("allow_drop", ("drop",)),
        ("allow_inventory", ("inventory",)),
        ("allow_hotbar", tuple(f"hotbar.{slot}" for slot in range(1, 10))),
    )
    for parameter, actions in action_permissions:
        if parameters.get(parameter) is not False:
            continue
        parameter_suppressed = False
        for action in actions:
            if action not in decoded:
                continue
            value = decoded[action]
            try:
                selected = bool(value[0])
                safe_value = value.copy()
                safe_value[...] = 0
            except (IndexError, TypeError, AttributeError):
                continue
            if constrained is decoded:
                constrained = dict(decoded)
            constrained[action] = safe_value
            parameter_suppressed = parameter_suppressed or selected
        if parameter_suppressed:
            suppressed.append(parameter.removeprefix("allow_"))
    return constrained, tuple(suppressed)


def _apply_discrete_action_contract(
    decoded: dict[str, Any],
    intent: dict[str, Any],
    emitted: set[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Make edge-triggered learned actions single-shot within one option run.

    STEVE-1 predicts an action at every temporal step. Repeating a toggle while
    Bedrock animates the first transition can close and immediately reopen the
    same menu. The checkpoint still decides whether and when to emit the event;
    this contract converts the selected event into one edge and then waits for
    the skill executor's visual success/failure observation.
    """
    actions_by_mode = {"close_inventory": ("inventory",)}
    actions = actions_by_mode.get(str(intent.get("mode") or "").casefold(), ())
    if not actions:
        return decoded, ()
    constrained = decoded
    suppressed: list[str] = []
    for action in actions:
        value = decoded.get(action)
        if value is None:
            continue
        try:
            selected = bool(value[0])
        except (IndexError, TypeError, AttributeError):
            continue
        if not selected:
            continue
        if action not in emitted:
            emitted.add(action)
            continue
        safe_value = value.copy()
        safe_value[...] = 0
        if constrained is decoded:
            constrained = dict(decoded)
        constrained[action] = safe_value
        suppressed.append(f"{action}:repeat")
    return constrained, tuple(suppressed)


def _apply_observed_scene_action_contract(
    decoded: dict[str, Any],
    intent: dict[str, Any],
    scene_belief: tuple[
        Literal["world", "inventory", "chat", "unknown"],
        bool | None,
        float,
        dict[str, float],
        str,
    ]
    | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Prevent a learned toggle from undoing its verified transition.

    The fast scene head and STEVE action are inferred from the same consumed
    frame. If that frame is already a confidently playable world, applying a
    newly predicted inventory toggle would reopen the menu before the executor
    can consume the visual success event on its next tick.
    """
    if (
        str(intent.get("mode") or "").casefold() != "close_inventory"
        or scene_belief is None
        or scene_belief[1] is not True
    ):
        return decoded, ()
    inventory = decoded.get("inventory")
    if inventory is None:
        return decoded, ()
    try:
        selected = bool(inventory[0])
        safe_inventory = inventory.copy()
        safe_inventory[...] = 0
    except (IndexError, TypeError, AttributeError):
        return decoded, ()
    if not selected:
        return decoded, ()
    return {**decoded, "inventory": safe_inventory}, ("inventory:scene-playable",)


def _rocket_interaction_id(mode: str) -> int:
    normalized = mode.casefold()
    if normalized in {"attack", "hunt", "combat"}:
        return 0
    if normalized in {"mine", "gather_wood", "gather", "break"}:
        return 2
    if normalized in {"use", "interact", "gui"}:
        return 3
    if normalized.startswith("craft"):
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


def _rocket_reference_frame(track: dict[str, Any], current_frame: Any, cv2_module: Any) -> Any:
    """Load and verify a persisted cross-view image, or use the current view."""
    attributes = track.get("attributes")
    if not isinstance(attributes, dict):
        return current_frame
    path_value = attributes.get("reference_image_path")
    digest = attributes.get("reference_image_sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        return current_frame
    path = Path(path_value)
    _verify_sha256(path, digest)
    reference = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if reference is None or reference.ndim != 3 or reference.shape[2] != 3:
        raise RuntimeError(f"unable to decode ROCKET-2 reference image {path}")
    return reference


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


def _intent_camera_semantics(intent: dict[str, Any]) -> Literal["world", "cursor"]:
    mode = str(intent.get("mode") or "").casefold()
    return (
        "cursor"
        if mode in {"gui", "death_gui", "close_inventory"} or mode.startswith("craft")
        else "world"
    )


def _intent_camera_scale(
    intent: dict[str, Any],
    *,
    world_scale: float,
    gui_scale: float,
) -> float:
    return gui_scale if _intent_camera_semantics(intent) == "cursor" else world_scale


def _decoded_policy_output(
    decoded: dict[str, Any],
    *,
    inference_ns: int,
    model_version: str,
    camera_scale: float = 1.0,
    camera_semantics: Literal["world", "cursor"] = "world",
    target_exists_probability: float | None = None,
    target_point_yx: tuple[float, float] | None = None,
    target_bbox_xyxy: tuple[float, float, float, float] | None = None,
    scene_mode: Literal["world", "inventory", "chat", "unknown"] | None = None,
    scene_playable: bool | None = None,
    scene_confidence: float | None = None,
    scene_class_probabilities: dict[str, float] | None = None,
    scene_model_version: str | None = None,
    suppressed_actions: tuple[str, ...] = (),
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
        camera_semantics=camera_semantics,
        inference_ns=inference_ns,
        model_version=model_version,
        target_exists_probability=target_exists_probability,
        target_point_yx=target_point_yx,
        target_bbox_xyxy=target_bbox_xyxy,
        scene_mode=scene_mode,
        scene_playable=scene_playable,
        scene_confidence=scene_confidence,
        scene_class_probabilities=scene_class_probabilities or {},
        scene_model_version=scene_model_version,
        suppressed_actions=suppressed_actions,
    )


def _rocket_target_estimate(
    policy: Any,
) -> tuple[
    float | None,
    tuple[float, float] | None,
    tuple[float, float, float, float] | None,
]:
    """Read ROCKET-2's published auxiliary current-view localization heads."""
    cached = getattr(policy, "cache_latents", None)
    if not isinstance(cached, dict):
        return None, None, None
    exist = cached.get("exist")
    point = cached.get("point")
    bbox = cached.get("bbox")
    if exist is None or point is None or bbox is None:
        return None, None, None
    probability = float(exist.detach().sigmoid().cpu().reshape(-1)[0].item())
    point_values = [
        float(value) for value in point.detach().cpu().reshape(-1).tolist()[:2]
    ]
    bbox_values = [
        float(value) for value in bbox.detach().cpu().reshape(-1).tolist()[:4]
    ]
    if len(point_values) != 2 or len(bbox_values) != 4:
        return probability, None, None
    point_yx = tuple(max(0.0, min(1.0, value)) for value in point_values)
    x0, y0, x1, y1 = (max(0.0, min(1.0, value)) for value in bbox_values)
    bbox_xyxy = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return probability, (point_yx[0], point_yx[1]), bbox_xyxy


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


def _crop_point_to_full(
    width: int,
    height: int,
    point_yx: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Map a normalized point from the policy's 16:9 crop to the full frame."""
    if point_yx is None or width <= 0 or height <= 0:
        return None
    point_y, point_x = point_yx
    target_ratio = 16.0 / 9.0
    if width / height > target_ratio:
        crop_width = int(round(height * target_ratio))
        left = max(0, (width - crop_width) // 2)
        point_x = (left + point_x * crop_width) / width
    elif width / height < target_ratio:
        crop_height = int(round(width / target_ratio))
        top = max(0, (height - crop_height) // 2)
        point_y = (top + point_y * crop_height) / height
    return (
        max(0.0, min(1.0, point_y)),
        max(0.0, min(1.0, point_x)),
    )


def _crop_bbox_to_full(
    width: int,
    height: int,
    bbox_xyxy: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Map a normalized policy crop box back into full-frame coordinates."""
    if bbox_xyxy is None:
        return None
    x0, y0, x1, y1 = bbox_xyxy
    top_left = _crop_point_to_full(width, height, (y0, x0))
    bottom_right = _crop_point_to_full(width, height, (y1, x1))
    if top_left is None or bottom_right is None:
        return None
    full_y0, full_x0 = top_left
    full_y1, full_x1 = bottom_right
    return (
        min(full_x0, full_x1),
        min(full_y0, full_y1),
        max(full_x0, full_x1),
        max(full_y0, full_y1),
    )


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
            "gui_camera_scale": args.gui_camera_scale,
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
                    scene_probe_interval=args.scene_probe_interval,
                    scene_model_path=(
                        Path(args.scene_model_path) if args.scene_model_path else None
                    ),
                    scene_model_sha256=args.scene_model_sha256,
                    scene_model_version=args.scene_model_version,
                    scene_min_confidence=args.scene_min_confidence,
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
                output = backend.infer(frame, request["intent"])
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
    serve.add_argument("--scene-probe-interval", type=int, default=0)
    serve.add_argument("--scene-model-path", default="")
    serve.add_argument("--scene-model-sha256", default="")
    serve.add_argument("--scene-model-version", default="")
    serve.add_argument("--scene-min-confidence", type=float, default=0.80)
    serve.add_argument("--camera-scale", type=float, default=1.0)
    serve.add_argument("--gui-camera-scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
