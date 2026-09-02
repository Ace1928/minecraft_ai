from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..trajectory import ReplayTrajectorySample


class TraceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    values: dict[str, float | int | bool | str] = Field(default_factory=dict)

    def merged(self, extra: dict[str, float | int | bool | str]) -> TraceMetrics:
        values = dict(self.values)
        values.update(extra)
        return TraceMetrics(values=values)


def trace_metrics(samples: Iterable[ReplayTrajectorySample]) -> TraceMetrics:
    accumulator = TraceMetricAccumulator()
    for sample in samples:
        accumulator.add(sample)
    return accumulator.finish()


class TraceMetricAccumulator:
    """Incrementally summarize a trajectory without retaining decoded frames."""

    def __init__(self) -> None:
        self.steps = 0
        self.keyboard_updates = 0
        self.button_updates = 0
        self.camera_updates = 0
        self.forward_presses = 0
        self.jump_presses = 0
        self.sprint_jump_presses = 0
        self.attack_presses = 0
        self.use_presses = 0
        self.inventory_presses = 0
        self.hotbar_presses = 0
        self.mouse_distance = 0
        self.sequence_violations = 0
        self.latencies: list[float] = []
        self.longest_static_moving = 0
        self.static_moving = 0
        self.previous_hash: str | None = None
        self.first_captured_ns: int | None = None
        self.last_captured_ns: int | None = None

    def add(self, sample: ReplayTrajectorySample) -> None:
        action = sample.step.action
        keys_down = set(action.keys_down)
        self.steps += 1
        self.keyboard_updates += bool(action.keys_down or action.keys_up)
        self.button_updates += bool(action.buttons_down or action.buttons_up)
        self.camera_updates += bool(action.mouse_dx or action.mouse_dy)
        self.forward_presses += "w" in keys_down
        self.jump_presses += "space" in keys_down
        self.sprint_jump_presses += {"ctrl", "space", "w"}.issubset(keys_down)
        self.attack_presses += "left" in action.buttons_down
        self.use_presses += "right" in action.buttons_down
        self.inventory_presses += "e" in keys_down
        self.hotbar_presses += bool(keys_down.intersection(set("123456789")))
        self.mouse_distance += abs(action.mouse_dx) + abs(action.mouse_dy)
        self.sequence_violations += action.sequence != sample.step.step_index
        if sample.step.accepted_ns is not None:
            self.latencies.append(
                (sample.step.accepted_ns - sample.step.captured_ns) / 1_000_000.0
            )
        moving = bool(keys_down.intersection({"w", "a", "s", "d"}))
        if moving and sample.step.frame_hash == self.previous_hash:
            self.static_moving += 1
            self.longest_static_moving = max(
                self.longest_static_moving, self.static_moving
            )
        else:
            self.static_moving = 0
        self.previous_hash = sample.step.frame_hash
        if self.first_captured_ns is None:
            self.first_captured_ns = sample.step.captured_ns
        self.last_captured_ns = sample.step.captured_ns

    def finish(self) -> TraceMetrics:
        values: dict[str, float | int | bool | str] = {
            "trace.steps": self.steps,
            "action.keyboard_updates": self.keyboard_updates,
            "action.button_updates": self.button_updates,
            "action.camera_updates": self.camera_updates,
            "action.forward_presses": self.forward_presses,
            "action.jump_presses": self.jump_presses,
            "action.sprint_jump_presses": self.sprint_jump_presses,
            "action.attack_presses": self.attack_presses,
            "action.use_presses": self.use_presses,
            "action.inventory_presses": self.inventory_presses,
            "action.hotbar_presses": self.hotbar_presses,
            "action.mouse_distance": self.mouse_distance,
            "safety.sequence_violations": self.sequence_violations,
            "trace.stuck_max_identical_moving_steps": self.longest_static_moving,
        }
        if self.first_captured_ns is not None and self.last_captured_ns is not None:
            values["trace.duration_s"] = max(
                0.0, (self.last_captured_ns - self.first_captured_ns) / 1e9
            )
        if self.latencies:
            values["latency.frame_to_accept_p50_ms"] = percentile(self.latencies, 0.50)
            values["latency.frame_to_accept_p95_ms"] = percentile(self.latencies, 0.95)
            values["latency.frame_to_accept_p99_ms"] = percentile(self.latencies, 0.99)
        return TraceMetrics(values=values)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wilson_interval(
    successes: int,
    samples: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if samples <= 0:
        return 0.0, 1.0
    proportion = successes / samples
    denominator = 1.0 + z * z / samples
    center = (proportion + z * z / (2 * samples)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples + z * z / (4 * samples * samples)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def compare_reports(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_summary = _summary(baseline)
    candidate_summary = _summary(candidate)
    baseline_passed = _int_value(baseline_summary.get("passed", 0))
    baseline_scored = _int_value(baseline_summary.get("scored", 0))
    candidate_passed = _int_value(candidate_summary.get("passed", 0))
    candidate_scored = _int_value(candidate_summary.get("scored", 0))
    baseline_rate = baseline_passed / baseline_scored if baseline_scored else None
    candidate_rate = candidate_passed / candidate_scored if candidate_scored else None
    same_suite = baseline.get("suite_id") == candidate.get("suite_id")
    baseline_complete = baseline_summary.get("promotion_eligible") is True
    candidate_complete = candidate_summary.get("promotion_eligible") is True
    return {
        "baseline_run_id": baseline.get("benchmark_run_id"),
        "candidate_run_id": candidate.get("benchmark_run_id"),
        "same_suite": same_suite,
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "success_rate_delta": (
            candidate_rate - baseline_rate
            if candidate_rate is not None and baseline_rate is not None
            else None
        ),
        "baseline_wilson_95": wilson_interval(baseline_passed, baseline_scored),
        "candidate_wilson_95": wilson_interval(candidate_passed, candidate_scored),
        "promotion_evidence_sufficient": bool(
            same_suite and baseline_complete and candidate_complete
        ),
    }


def _summary(report: dict[str, object]) -> dict[str, object]:
    summary = report.get("summary")
    return summary if isinstance(summary, dict) else {}


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
