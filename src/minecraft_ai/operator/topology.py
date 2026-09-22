"""Read-only system topology for the operator dashboard.

Combines the same allowlisted status and observation payloads the dashboard
already reads into a compact part graph with live activity, plus the observed
model populations for the topology viewer. No capture, inference or control.
"""
from __future__ import annotations

import time
from typing import Any

SCHEMA = "minecraft.topology.v1"

# (id, label, kind, upstream ids)
PARTS: tuple[tuple[str, str, str], ...] = (
    ("capture", "Frame capture", "sense"),
    ("perception", "ROCKET-2 perception", "sense"),
    ("policy", "Motor policy", "act"),
    ("skills", "Skill execution", "act"),
    ("bedrock", "Bedrock game", "world"),
    ("supervisor", "Supervisor", "govern"),
    ("agent", "Cognition agent", "think"),
    ("native_policy", "Native policy model", "model"),
    ("association_brain", "Association brain", "model"),
    ("memory", "Memory & trajectory", "remember"),
)

EDGES: tuple[tuple[str, str], ...] = (
    ("capture", "perception"),
    ("perception", "policy"),
    ("perception", "native_policy"),
    ("policy", "skills"),
    ("skills", "bedrock"),
    ("bedrock", "capture"),
    ("supervisor", "skills"),
    ("agent", "policy"),
    ("agent", "memory"),
    ("association_brain", "agent"),
    ("native_policy", "agent"),
    ("policy", "memory"),
)


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) not in (int, float):
        return default
    return float(value)


def _integer(value: Any, default: int = 0) -> int:
    if type(value) is not int:
        return default
    return value


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _state(activity: float, available: bool) -> str:
    if not available:
        return "offline"
    return "live" if activity >= 0.5 else "idle"


def _model_activity(model: dict[str, Any]) -> float:
    if not model.get("available"):
        return 0.0
    kcs, active = _integer(model.get("source_kc_count")), _integer(model.get("source_active_kcs"))
    if kcs:
        return _clamp(active / kcs * 8)  # one-in-eight busy reads as full activity
    return 1.0 if _integer(model.get("calls")) else 0.0


def _observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    online = observation.get("online") is True
    action = observation.get("action")
    receptive = observation.get("receptive_fields") or {}
    state = observation.get("state")
    if not isinstance(state, str):
        state = None
    return {
        "online": online,
        "source_id": observation.get("source_id") if online else None,
        "stream_id": observation.get("stream_id") if online else None,
        "sequence": _integer(observation.get("sequence")) if online else None,
        "frame_age_ms": _number(observation.get("frame_age_ms")) if online else None,
        # An expired sample is a normal gap, not a fake outage: carry the true
        # last-seen age and the last explicit producer phase.
        "state": state,
        "last_seen_age_ms": (
            None if online else _number(observation.get("last_seen_age_ms"))
        ),
        "reason": None if online else observation.get("reason"),
        "action": None if not isinstance(action, dict) else {
            "kind": action.get("kind"), "accepted": action.get("accepted") is True,
            "outcome": action.get("outcome"), "buttons": action.get("buttons") or [],
        },
        "receptive_scope": receptive.get("scope"),
        "receptive_samples": len(receptive.get("samples") or []),
    }


def build_topology(
    status: dict[str, Any], observation: dict[str, Any], *, now_ns: int | None = None,
) -> dict[str, Any]:
    """Derive the live part graph; every value is copied, never inferred pixels."""
    telemetry = status.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    bedrock = status.get("bedrock")
    bedrock = bedrock if isinstance(bedrock, dict) else {}
    supervisor = status.get("supervisor")
    supervisor = supervisor if isinstance(supervisor, dict) else {}
    agent = status.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    models = observation.get("models") if observation.get("online") is True else None
    models = models if isinstance(models, dict) else {}

    capture_ms = telemetry.get("last_capture_ms")
    capture_activity = 1.0 if type(capture_ms) in (int, float) and 0 <= capture_ms < 250 else 0.0
    facts = (telemetry.get("perception") or {}).get("fresh_facts") if isinstance(
        telemetry.get("perception"), dict) else None
    perception_activity = _clamp(len(facts) / 8) if isinstance(facts, dict) else 0.0
    policy = telemetry.get("policy")
    policy = policy if isinstance(policy, dict) else {}
    route = policy.get("primary") if isinstance(policy.get("primary"), dict) else {}
    inference_ms = route.get("last_inference_ms")
    policy_activity = 1.0 if type(inference_ms) in (int, float) and inference_ms < 250 else 0.0
    active_skill = telemetry.get("active_skill")
    skills_activity = 1.0 if active_skill else 0.0
    recording = telemetry.get("trajectory_recording")
    recording = recording if isinstance(recording, dict) else {}
    memory_activity = 1.0 if recording.get("enabled") is True else 0.0

    activities = {
        "capture": capture_activity,
        "perception": perception_activity,
        "policy": policy_activity,
        "skills": skills_activity,
        "bedrock": 1.0 if bedrock.get("instances") else 0.0,
        "supervisor": 1.0 if status.get("supervisor_reachable") and supervisor.get("state") not in
                              {"FAILSAFE", "STOPPED", None} else 0.0,
        "agent": 1.0 if agent.get("alive") else 0.0,
        "native_policy": _model_activity(models.get("native_policy") or {}),
        "association_brain": _model_activity(models.get("association_brain") or {}),
        "memory": memory_activity,
    }
    details = {
        "capture": None if capture_ms is None else f"{capture_ms} ms",
        "perception": None if not isinstance(facts, dict) else f"{len(facts)} facts",
        "policy": route.get("model_version") or policy.get("policy_id"),
        "skills": active_skill,
        "bedrock": (bedrock.get("version") or "unknown") + (
            f" · {len(bedrock.get('instances') or [])} instance(s)"),
        "supervisor": supervisor.get("state"),
        "agent": "running" if agent.get("alive") else "disarmed",
        "native_policy": _model_detail(models.get("native_policy")),
        "association_brain": _model_detail(models.get("association_brain")),
        "memory": f"{_integer(recording.get('written_steps'))} saved · "
                  f"{_integer(recording.get('queued_samples'))} queued"
                  if recording.get("enabled") is True else (recording.get("disabled_reason") or "paused"),
    }
    available = {
        "capture": capture_ms is not None,
        "perception": isinstance(facts, dict),
        "policy": bool(route),
        "skills": active_skill is not None,
        "bedrock": bool(bedrock.get("instances")),
        "supervisor": bool(supervisor),
        "agent": bool(agent),
        "native_policy": bool((models.get("native_policy") or {}).get("available")),
        "association_brain": bool((models.get("association_brain") or {}).get("available")),
        "memory": bool(recording),
    }
    parts = [
        {
            "id": part_id, "label": label, "kind": kind,
            "state": _state(activities[part_id], available[part_id]),
            "activity": round(activities[part_id], 4),
            "detail": details[part_id],
        }
        for part_id, label, kind in PARTS
    ]
    edges = [
        {"source": source, "target": target,
         "flow": round(min(activities.get(source, 0.0), activities.get(target, 0.0)), 4)}
        for source, target in EDGES
    ]
    populations = []
    for key in ("native_policy", "association_brain"):
        model = models.get(key)
        if not isinstance(model, dict) or not model.get("available"):
            continue
        population = model.get("population")
        if not isinstance(population, dict):
            continue
        populations.append({"model": key, **population,
                            "calls": _integer(model.get("calls")),
                            "source_unit_count": _integer(model.get("source_unit_count")),
                            "activity_basis": model.get("activity_basis", "observed-activation"),
                            "activity_reused": model.get("activity_reused") is True})
    return {
        "schema": SCHEMA,
        "generated_ns": time.time_ns() if now_ns is None else now_ns,
        "observation": _observation_summary(observation),
        "parts": parts,
        "edges": edges,
        "populations": populations,
        "scope": "allowlisted status and observation fields only; no inferred pixels or activations",
    }


def _model_detail(model: dict[str, Any]) -> str | None:
    if not isinstance(model, dict) or not model.get("available"):
        return "unavailable"
    return (f"{_integer(model.get('calls'))} calls · "
            f"{_integer(model.get('source_unit_count'))} units · "
            f"{_integer(model.get('source_active_kcs'))}/{_integer(model.get('source_kc_count'))} KC busy")
