from __future__ import annotations

import argparse
import json
from typing import Callable


def _start_permitted() -> tuple[bool, dict[str, object]]:
    from .emergency import emergency_reason, emergency_stop_latched
    from .supervisor import operator_pause_latched

    latched = emergency_stop_latched()
    paused = operator_pause_latched()
    return not latched and not paused, {
        "emergency_stop_latched": latched,
        "operator_pause_latched": paused,
        "reason": emergency_reason(),
    }


def _bedrock_health() -> tuple[bool, dict[str, object]]:
    from .platforms.bedrock_session import (
        BEDROCK_SESSION_FILE,
        BedrockSession,
        bedrock_session_alive,
    )

    try:
        session = BedrockSession.load(BEDROCK_SESSION_FILE)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}
    alive = session.mode in {"weston", "xephyr"} and bedrock_session_alive(session)
    return alive, {
        "alive": alive,
        "display": session.display,
        "mode": session.mode,
        "created_ns": session.created_ns,
    }


def _playable_bedrock_health() -> tuple[bool, dict[str, object]]:
    from .perception_service import bedrock_survival_hud_present
    from .platforms.bedrock_session import BEDROCK_SESSION_FILE, BedrockSession
    from .platforms.bedrock_x11 import IsolatedX11Capture

    healthy, detail = _bedrock_health()
    if not healthy:
        return False, detail
    try:
        session = BedrockSession.load(BEDROCK_SESSION_FILE)
        window_id = session.find_window()
        if window_id is None:
            return False, {**detail, "window": None, "survival_hud": False}
        capture = IsolatedX11Capture(session.display, window_id, allow_host=False)
        try:
            frame = capture.capture()
        finally:
            capture.close()
        survival_hud = bedrock_survival_hud_present(frame)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return False, {**detail, "error": f"{type(exc).__name__}: {exc}"}
    return survival_hud, {
        **detail,
        "window": window_id,
        "survival_hud": survival_hud,
        "capture": [frame.width, frame.height],
    }


def _supervisor_health() -> tuple[bool, dict[str, object]]:
    from .supervisor import send_command, supervisor_alive

    if not supervisor_alive():
        return False, {"alive": False}
    try:
        status = send_command("status")
    except Exception as exc:
        return False, {"alive": False, "error": f"{type(exc).__name__}: {exc}"}
    state = str(status.get("state", ""))
    healthy = state in {"SAFE_IDLE", "RUNNING"}
    return healthy, {"alive": True, "state": state, "role": status.get("role")}


def _live_agent_health() -> tuple[bool, dict[str, object]]:
    from .agent_lifecycle import AgentProcess, agent_alive
    from .supervisor import send_command, supervisor_alive

    try:
        process = AgentProcess.load()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, {"alive": False, "error": f"{type(exc).__name__}: {exc}"}
    if not agent_alive(process) or not supervisor_alive():
        return False, {"alive": False, "pid": process.pid}
    try:
        status = send_command("status")
    except Exception as exc:
        return False, {"alive": False, "error": f"{type(exc).__name__}: {exc}"}
    healthy = (
        status.get("state") == "RUNNING"
        and bool(status.get("live_capable"))
        and bool(status.get("motor_lease_active"))
    )
    return healthy, {
        "alive": True,
        "pid": process.pid,
        "state": status.get("state"),
        "live_capable": status.get("live_capable"),
        "motor_lease_active": status.get("motor_lease_active"),
    }


def _stop_agent() -> tuple[bool, dict[str, object]]:
    from .agent_lifecycle import AGENT_FILE, stop_agent_process
    from .supervisor import send_command, supervisor_alive

    stopped = stop_agent_process()
    state: object = None
    if supervisor_alive():
        try:
            current = send_command("status")
            if current.get("state") in {"ARMED", "RUNNING"}:
                current = send_command("disarm")
            state = current.get("state")
        except Exception as exc:
            return False, {"agent_stop_attempted": stopped, "error": f"{type(exc).__name__}: {exc}"}
    contained = not AGENT_FILE.exists()
    return contained, {
        "agent_stop_attempted": stopped,
        "agent_containment_confirmed": contained,
        "supervisor_state": state,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minecraft AI stack health and rollback probes")
    parser.add_argument(
        "probe",
        choices=(
            "start-permitted",
            "bedrock",
            "playable-bedrock",
            "supervisor",
            "live-agent",
            "stop-agent",
        ),
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probes: dict[str, Callable[[], tuple[bool, dict[str, object]]]] = {
        "start-permitted": _start_permitted,
        "bedrock": _bedrock_health,
        "playable-bedrock": _playable_bedrock_health,
        "supervisor": _supervisor_health,
        "live-agent": _live_agent_health,
        "stop-agent": _stop_agent,
    }
    healthy, detail = probes[args.probe]()
    if args.json_output:
        print(json.dumps({"healthy": healthy, **detail}, sort_keys=True))
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
