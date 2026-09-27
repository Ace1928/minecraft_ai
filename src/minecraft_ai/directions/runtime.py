"""Bind the inventory catalogue transaction to the existing skill executor.

No separate motor controller: the two native GUI skills retain their own
observation checks, supervisor lease and bounded inventory pulses.
"""

from __future__ import annotations

import time
import uuid

from minecraft_ai.cognition.prompts import _urgent_safety_required
from minecraft_ai.directions.gateway import (
    ControlUnavailableError,
    DirectionsGateway,
    DirectionsError,
)
from minecraft_ai.perception_service import BEDROCK_HUD_SAFETY_SOURCE
from minecraft_ai.runtime_support.types import SkillStartSource
from minecraft_ai.skills import SkillOutcome


def _release(runtime):
    run = runtime.executor.run
    if run is not None and run.outcome == SkillOutcome.RUNNING:
        cancelled = runtime.executor.cancel()
        runtime._record_terminal_run(cancelled.run, advance_plan=False)
    runtime._release_and_reconcile_inputs()
    runtime._active_direction = None
    runtime._execution_revision += 1
    runtime._cognition_requested = True


def _start(runtime, gateway, receipt, skill_id):
    run_id = uuid.uuid4().hex
    parameters = gateway.start_step(receipt["request_id"], receipt["attempt_id"], skill_id, run_id)
    runtime._start_skill(
        runtime.skills.get(skill_id),
        source=SkillStartSource.CONTINUATION,
        run_id=run_id,
        context_key=f"operator:{receipt['message_id']}",
        parameters=parameters,
    )


def tick_inventory_direction(runtime) -> bool:
    """Return true only when this typed transaction owns the current tick."""
    database = runtime.state_db
    if database is None:
        return False
    gateway = getattr(runtime, "_direction_gateway", None)
    if gateway is None:
        if (
            database.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paid_directions'"
            ).fetchone()
            is None
        ):
            return False
        gateway = runtime._direction_gateway = DirectionsGateway(database.path)
    receipt = getattr(runtime, "_active_direction", None)
    if receipt is None:
        pending = gateway.next_pending()
        if pending is None:
            return False
        # Restarted in-flight attempts are uncertain, never automatically replayed.
        if pending["state"] == "running":
            gateway.claim(pending["request_id"])
            return True
        current = runtime.executor.run
        if current is not None and current.outcome == SkillOutcome.RUNNING:
            if current.context_key != "explore-keepalive":
                gateway.status(pending["request_id"])
                return False  # The existing atomic owner must finish first.
        world = runtime.blackboard.fact("scene.playable", min_confidence=0.99)
        if (
            _urgent_safety_required(runtime.blackboard)
            or world is None
            or world.value is not True
            or world.source != BEDROCK_HUD_SAFETY_SOURCE
        ):
            gateway.status(pending["request_id"])
            return False  # Existing scene/death recovery keeps priority.
        receipt = gateway.claim(pending["request_id"])
        if receipt is None:
            return True
        runtime._active_direction = receipt
        if current is not None and current.outcome == SkillOutcome.RUNNING:
            cancelled = runtime.executor.cancel()
            runtime._record_terminal_run(cancelled.run, advance_plan=False)
        if not runtime._release_and_reconcile_inputs():
            gateway.cancel(receipt["request_id"], reason="input_release_unconfirmed")
            _release(runtime)
            return True
        future = getattr(runtime, "_pending_decision", None)
        if future is not None:
            runtime._reject_bound_cognition(future, "typed_direction_owns_execution")
            future.cancel()
            runtime._pending_decision = None
            runtime._pending_operator_message_ids = ()
            runtime._pending_operator_message_kinds = {}
        runtime._execution_revision += 1
        try:
            _start(runtime, gateway, receipt, "open_inventory")
        except DirectionsError:
            gateway.cancel(receipt["request_id"], reason="direction_start_revoked")
            _release(runtime)
        return True  # First input uses a fresh post-release capture next tick.

    try:
        status = gateway.status(receipt["request_id"])
        if status.state.value != "running" or _urgent_safety_required(runtime.blackboard):
            gateway.cancel(receipt["request_id"], reason="safety_preempted")
            _release(runtime)
            return True
        result = runtime.executor.tick(
            runtime.blackboard,
            sequence=runtime._sequence,
            now_ns=time.monotonic_ns(),
            capture=runtime.perception.last_capture,
        )
        if result.action is not None:
            runtime._send_motor(result.action, execution=result)
        if result.run.outcome != SkillOutcome.RUNNING:
            runtime._record_terminal_run(result.run, advance_plan=False)
            complete = gateway.finish_step(
                receipt["request_id"],
                receipt["attempt_id"],
                result.run,
                runtime.blackboard,
                runtime.perception.last_capture,
            )
            if complete and result.run.skill_id == "open_inventory":
                _start(runtime, gateway, receipt, "close_open_inventory")
            else:
                _release(runtime)
    except (ControlUnavailableError, DirectionsError):
        gateway.cancel(receipt["request_id"], reason="direction_authority_revoked")
        _release(runtime)
    return True
