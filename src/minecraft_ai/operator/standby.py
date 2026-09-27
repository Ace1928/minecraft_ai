"""Bounded local reasoning standby; never pauses capture or revokes the game lease."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from platformdirs import user_runtime_dir

from minecraft_ai.models import local_model_inference_available
from minecraft_ai.skills import SkillOutcome

STANDBY_FILE = Path(user_runtime_dir("minecraft-ai")) / "reasoning-standby.json"
MAX_STANDBY_NS = 120_000_000_000


def standby_remaining_ns(*, path=None):
    path = STANDBY_FILE if path is None else path
    try:
        data = json.loads(path.read_text())
        remaining = data["expires_monotonic_ns"] - time.monotonic_ns()
        return remaining if type(remaining) is int and 0 < remaining <= MAX_STANDBY_NS else 0
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def set_reasoning_standby(enabled, *, path=None):
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    path = STANDBY_FILE if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    staged.write_text(
        json.dumps(
            {
                "expires_monotonic_ns": time.monotonic_ns() + MAX_STANDBY_NS if enabled else 0,
            }
        )
    )
    staged.replace(path)
    return {"requested": enabled, "maximum_seconds": 120}


def standby_status(runtime):
    remaining = standby_remaining_ns()
    if not remaining:
        return {"state": "off", "remaining_seconds": 0}
    pool = getattr(runtime, "_pool", None)
    pool_idle = pool is None or (callable(getattr(pool, "is_idle", None)) and pool.is_idle())
    active_vlm = getattr(getattr(runtime, "perception", None), "active_vlm", None)
    vision = active_vlm.status() if active_vlm is not None else {}
    # available() holds the worker's admission lock and covers the handoff
    # between dequeuing a job and setting its visible busy flag.
    vision_idle = active_vlm is None or (
        callable(getattr(active_vlm, "available", None)) and active_vlm.available()
    )
    draining = (
        not getattr(runtime, "_reasoning_standby_engaged", False)
        or not pool_idle
        or not local_model_inference_available()
        or not vision_idle
        or bool(vision.get("busy"))
        or bool(vision.get("pending_requests"))
    )
    return {
        "state": "draining" if draining else "standby",
        "remaining_seconds": round(remaining / 1e9, 1),
    }


def apply_reasoning_standby(runtime):
    if not standby_remaining_ns():
        if getattr(runtime, "_reasoning_standby_engaged", False):
            runtime._reasoning_standby_engaged = False
            runtime._cognition_requested = True
        return False
    future = getattr(runtime, "_pending_decision", None)
    if future is not None:
        runtime._reject_bound_cognition(future, "operator_reasoning_standby")
        future.cancel()  # Running work drains; its result cannot enter execution.
        runtime._pending_decision = None
        runtime._pending_operator_message_ids = ()
        runtime._pending_operator_message_kinds = {}
    run = runtime.executor.run
    if run is not None and run.parameters.get("direction_request_id"):
        runtime._reasoning_standby_engaged = True
        return True  # The typed receipt retains its own action authority.
    if not getattr(runtime, "_reasoning_standby_engaged", False):
        if run is not None and run.outcome == SkillOutcome.RUNNING:
            cancelled = runtime.executor.cancel()
            runtime._record_terminal_run(cancelled.run, advance_plan=False)
        runtime._execution_revision += 1
        runtime._headroom_recovery = None
        runtime._cognition_perception_probe = None
        runtime._reasoning_standby_engaged = runtime._release_and_reconcile_inputs()
    return True
