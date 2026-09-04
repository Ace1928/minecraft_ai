from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import typer
from platformdirs import user_data_dir, user_runtime_dir
from rich import print

from .agent_lifecycle import (
    AGENT_FILE,
    AGENT_LOG,
    AgentProcess,
    agent_alive,
    launch_agent_process,
    stop_agent_process,
)
from .camera_calibration import (
    load_camera_calibration,
    read_bedrock_mouse_sensitivity,
)
from .bedrock_menu import (
    DEFAULT_BEDROCK_CONNECT_SERVERS,
    BedrockMenuNavigator,
    MenuNavigationError,
    NestedXTestMenuInput,
    TesseractMenuTextReader,
    load_configured_local_server,
)
from .config import app_paths, ensure_default_config, load_config
from .emergency import (
    clear_emergency_stop,
    emergency_reason,
    emergency_stop_latched,
    engage_emergency_stop,
    terminate_registered_supervisor,
)
from .knowledge import Edition, GameVersion, KnowledgeGraph
from .knowledge.importers import import_java_datapack, import_minecraft_data
from .human_recording import HumanRecordingRequest, record_human_session
from .eval import (
    BenchmarkRunner,
    TraceMetricAccumulator,
    bedrock_baseline_suite,
    compare_reports,
    load_evidence,
)
from .operator_server import serve_operator_dashboard
from .perception_service import bedrock_in_world_hud_present
from .platforms import (
    IsolatedX11Capture,
    IsolationError,
    create_bedrock_capture,
    discover_bedrock_linux_install,
    find_bedrock_linux_instances,
)
from .platforms.bedrock_session import (
    BEDROCK_SESSION_FILE,
    DEFAULT_BEDROCK_HEIGHT,
    DEFAULT_BEDROCK_WIDTH,
    BedrockSession,
    bedrock_lifecycle_lock,
    bedrock_session_alive,
    bind_direct_session_to_monitor,
    launch_isolated_bedrock_session,
    stop_bedrock_session,
    wait_for_minecraft_window,
)
from .roles import BUILTIN_ROLES
from .service_control import (
    persistent_agent_service_load_state,
    persistent_agent_service_state,
    start_persistent_agent_service,
    stop_persistent_agent_service,
)
from .storage import StateDatabase
from .supervisor import (
    CONTROL_FILE,
    STATUS_FILE,
    ControlEndpoint,
    clear_operator_pause,
    control_endpoint_process_state,
    current_control_owner_state,
    latch_operator_pause,
    operator_intent_lock,
    operator_pause_latched,
    send_command,
    supervisor_alive,
)
from .trajectory import TrajectoryReader
from .wiki import WikiService

app = typer.Typer(help="Minecraft AI lifecycle and tooling CLI.")
knowledge_app = typer.Typer(help="Versioned game knowledge commands.")
bedrock_app = typer.Typer(help="BedrockOnLinux isolated-session commands.")
roles_app = typer.Typer(help="Agent role/archetype commands.")
config_app = typer.Typer(help="Local runtime/model configuration commands.")
dataset_app = typer.Typer(help="Inspect and verify frame/action trajectory datasets.")
eval_app = typer.Typer(help="Run and compare evidence-gated Bedrock evaluations.")
benchmark_app = typer.Typer(help="Build frozen-suite benchmark reports.")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(bedrock_app, name="bedrock")
app.add_typer(roles_app, name="roles")
app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")
app.add_typer(eval_app, name="eval")
app.add_typer(benchmark_app, name="benchmark")

APP_NAME = "minecraft-ai"
DATA_DIR = Path(user_data_dir(APP_NAME))
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
LOG_FILE = DATA_DIR / "logs" / "supervisor.log"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
WIKI_CACHE = DATA_DIR / "wiki-cache"
DEFAULT_EDITION = Edition.BEDROCK


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_CACHE.mkdir(parents=True, exist_ok=True)


def _read_status_file() -> dict[str, object]:
    if not STATUS_FILE.exists():
        return {"state": "STOPPED", "live_capable": False}
    try:
        payload = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"state": "UNKNOWN"}
    except Exception:
        return {"state": "UNKNOWN", "live_capable": False}


def _command(
    command: str,
    *,
    timeout_s: float = 1.5,
    **payload: object,
) -> dict[str, object]:
    try:
        return send_command(command, timeout_s=timeout_s, **payload)
    except Exception as exc:
        raise typer.BadParameter(f"supervisor unavailable: {exc}") from exc


def _graph_path(version: GameVersion) -> Path:
    safe_version = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in version.version_id)
    return KNOWLEDGE_DIR / f"{version.edition.value}-{safe_version}.json"


def _load_graph(path: Path) -> KnowledgeGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"invalid graph file: {path}")
    return KnowledgeGraph.from_dict(payload)


def _bedrock_version_or_error(explicit: str | None) -> str:
    if explicit:
        return explicit
    install = discover_bedrock_linux_install()
    if install is not None and install.selected_build is not None:
        return install.selected_build.version
    raise typer.BadParameter(
        "No exact Bedrock version was supplied or detected from BedrockOnLinux. "
        "Pass --version explicitly."
    )


def _session_payload(session: BedrockSession) -> dict[str, object]:
    alive = bedrock_session_alive(session)
    host_monitor = session.host_monitor_binding()
    return {
        "display": session.display,
        "host_display": session.host_display,
        "xserver_pid": session.xserver_pid,
        "launcher_pid": session.launcher_pid,
        "width": session.width,
        "height": session.height,
        "mode": session.mode,
        "compositor_fullscreen": session.compositor_fullscreen,
        "wayland_socket": session.wayland_socket,
        "compositor_log": session.compositor_log,
        "launcher_log": session.launcher_log,
        "host_monitor": None if host_monitor is None else host_monitor.payload(),
        "alive": alive,
        "minecraft_window": session.find_window() if alive else None,
    }


def _agent_payload() -> dict[str, object] | None:
    try:
        process = AgentProcess.load()
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return {
        "pid": process.pid,
        "alive": agent_alive(process),
        "role": process.role,
        "display": process.display,
        "window_id": process.window_id,
        "instance_id": process.instance_id,
    }


@app.command()
def demo(
    ticks: int = typer.Option(12, min=1, max=1000, help="Number of ticks to run."),
    role: str = typer.Option("generalist", help="Role profile."),
) -> None:
    """Print an offline bootstrap smoke sequence without claiming live capability."""
    _ensure_dirs()
    from .builtin_skills import build_bootstrap_skill_library
    from .roles import get_role

    role_profile = get_role(role)
    library = build_bootstrap_skill_library()
    print(f"[bold green]Bootstrap smoke sequence[/bold green] ticks={ticks} role={role}")
    print(f"Role: {role_profile.role_id} - {role_profile.description or 'Inhabitant'}")
    print(f"Standing goals: {list(role_profile.standing_goals)}")
    print(f"Loaded skills: {len(library.specs)}")

    print("\n[bold yellow]--- Synthetic tick sequence ---[/bold yellow]")
    for i in range(ticks):
        subgoal = "approach_target" if i < 5 else "mine_block"
        verb = "look" if i == 0 else ("walk" if i < 5 else "mine")
        print(f"t={i:02d} subgoal={subgoal:<18} critic=continue verb={verb:<6}")
        time.sleep(0.05)

    print("[green]Offline smoke sequence completed.[/green]")
    print("No live-control, intelligence, or performance claim is implied.")


@app.command()
def install() -> None:
    """Prepare local state/config and report runtime dependencies/capabilities."""
    _ensure_dirs()
    config_path = ensure_default_config()
    print(f"[green]Minecraft AI bootstrap prepared.[/green] config={config_path}")
    doctor()


@app.command()
def doctor() -> None:
    """Report platform capabilities without enabling live control."""
    _ensure_dirs()
    linux = platform.system() == "Linux"
    bol = discover_bedrock_linux_install() if linux else None
    bedrock_instances = find_bedrock_linux_instances() if linux else []
    selected_build = bol.selected_build if bol is not None else None
    try:
        session = BedrockSession.load() if BEDROCK_SESSION_FILE.exists() else None
    except (OSError, ValueError, TypeError, KeyError):
        session = None
    profile = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "default_edition": DEFAULT_EDITION.value,
        "reference_runtime": "bedrock-on-linux/winegdk",
        "data_dir": str(DATA_DIR),
        "runtime_dir": str(RUNTIME_DIR),
        "supervisor": "running" if supervisor_alive() else "stopped",
        "agent": _agent_payload(),
        "emergency_stop": {
            "latched": emergency_stop_latched(),
            "reason": emergency_reason(),
        },
        "bedrock_on_linux": {
            "detected": bol is not None,
            "data_dir": str(bol.data_dir) if bol is not None else None,
            "wine_prefix": str(bol.wine_prefix) if bol is not None else None,
            "launcher": bol.launcher_command if bol is not None else None,
            "selected_version": selected_build.version if selected_build is not None else None,
            "selected_channel": selected_build.edition_id if selected_build is not None else None,
            "running_instances": [instance.instance_id for instance in bedrock_instances],
        },
        "isolation": {
            "weston": shutil.which("weston"),
            "xephyr": shutil.which("Xephyr"),
            "preferred_backend": "weston" if shutil.which("weston") else "xephyr",
            "host_display": os.environ.get("DISPLAY"),
            "managed_session": _session_payload(session) if session is not None else None,
        },
        "python_optional_modules": {
            "xlib": _module_available("Xlib"),
            "mss": _module_available("mss"),
            "httpx": _module_available("httpx"),
            "pillow": _module_available("PIL"),
            "numpy": _module_available("numpy"),
        },
    }
    print(json.dumps(profile, indent=2, sort_keys=True))


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _persistent_agent_service_state() -> str:
    return persistent_agent_service_state()


def _stop_persistent_agent_service() -> bool:
    return stop_persistent_agent_service()


def _try_latch_operator_pause() -> str | None:
    try:
        latch_operator_pause()
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _report_pause_persistence_error(error: str | None) -> None:
    if error is None:
        return
    service_stopped = _stop_persistent_agent_service()
    detail = (
        "persistent recovery service is confirmed stopped"
        if service_stopped
        else "persistent recovery service shutdown could not be confirmed"
    )
    raise typer.BadParameter(
        "control was revoked, but the durable operator-pause marker could not be written "
        f"({error}); {detail}"
    )


def _prepare_human_recording_takeover() -> None:
    """Atomically revoke autonomous ownership before accepting human input."""

    persistence_error: str | None = None
    revocation_error: Exception | None = None
    disarm_confirmed = False
    service_stopped = False
    agent_contained = False
    pause_confirmed = False
    owner_state = "unreadable"

    with operator_intent_lock():
        persistence_error = _try_latch_operator_pause()
        try:
            if supervisor_alive():
                result = _command("disarm", timeout_s=5.0)
                if result.get("motor_lease_active") is not False:
                    raise typer.BadParameter(
                        "Supervisor did not confirm motor revocation for human takeover."
                    )
                disarm_confirmed = True
        except Exception as exc:
            revocation_error = exc
            terminate_registered_supervisor()
        finally:
            stop_agent_process()

        service_state = _persistent_agent_service_state()
        service_stopped = service_state == "inactive" or _stop_persistent_agent_service()

        # Stopping the recovery owner can race its final child cleanup. Repeat
        # containment while the durable operator intent is still serialized.
        if revocation_error is not None:
            terminate_registered_supervisor()
        stop_agent_process()
        owner_state = current_control_owner_state()
        agent_contained = not AGENT_FILE.exists()
        pause_confirmed = operator_pause_latched()

    _report_pause_persistence_error(persistence_error)
    if not pause_confirmed:
        raise typer.BadParameter(
            "Durable operator pause could not be verified; human recording refused."
        )
    if not service_stopped:
        raise typer.BadParameter(
            "Persistent recovery service shutdown could not be confirmed; recording refused."
        )
    if not agent_contained:
        raise typer.BadParameter(
            "Realtime agent containment could not be confirmed; human recording refused."
        )
    if not disarm_confirmed and owner_state in {
        "verified-live",
        "unverifiable",
        "unreadable",
    }:
        detail = "" if revocation_error is None else f": {revocation_error}"
        raise typer.BadParameter(
            f"Supervisor revocation could not be confirmed ({owner_state}){detail}"
        )


def _require_autonomous_isolated_session(session: BedrockSession) -> None:
    if session.mode not in {"xephyr", "weston"}:
        raise typer.BadParameter(
            "Autonomous live control requires a nested Weston/Xephyr Bedrock session. "
            "Direct and host-monitor sessions share the operator display and are manual-debug "
            "or capture-only modes; stop this session and run `minecraft-ai bedrock launch`."
        )


def _launch_realtime_agent_transaction(
    *,
    target: str,
    display: str,
    window_id: int,
    role: str,
    allow_host_capture: bool,
    capture_source: str,
) -> AgentProcess:
    """Serialize the final arm/activate/spawn boundary with operator intent."""

    with operator_intent_lock():
        if emergency_stop_latched():
            raise typer.BadParameter("Emergency stop was requested before agent launch.")
        if operator_pause_latched():
            raise typer.BadParameter("Operator pause was requested before agent launch.")
        if agent_alive():
            raise typer.BadParameter("Realtime agent process is already running.")

        armed = _command("arm", target_instance=target)
        lease = armed.get("lease")
        if not isinstance(lease, dict) or not isinstance(lease.get("lease_id"), str):
            raise typer.BadParameter("supervisor did not return a valid motor lease")
        lease_id = str(lease["lease_id"])
        try:
            _command("activate")
            # A failed/out-of-process operator client may have persisted its
            # marker while SAFE_IDLE was transitioning; check once more before
            # publishing a child descriptor.
            if operator_pause_latched():
                raise typer.BadParameter("Operator pause was requested before agent launch.")
            return launch_agent_process(
                lease_id=lease_id,
                display=display,
                window_id=window_id,
                instance_id=target,
                role=role,
                allow_host_capture=allow_host_capture,
                capture_source=capture_source,
            )
        except Exception as exc:
            try:
                _command("disarm")
            except Exception:
                # Emergency stop deliberately bypasses the ordinary intent
                # lock, so its supervisor kill can make this IPC unavailable.
                pass
            if isinstance(exc, typer.BadParameter):
                raise
            raise typer.BadParameter(f"realtime agent failed to start: {exc}") from exc


@app.command()
def run(
    role: str = typer.Option("generalist", help="Role/archetype profile."),
    edition: Edition = typer.Option(DEFAULT_EDITION, "--edition"),
    live: bool = typer.Option(False, help="Run the realtime isolated Bedrock agent."),
    capture_source: str = typer.Option(
        "pipewire",
        "--capture-source",
        help="Capture source: pipewire (default) or x11 (window-targeted XGetImage).",
    ),
) -> None:
    """Start supervisor; with --live attach Bedrock and start the player loop."""
    _ensure_dirs()
    ensure_default_config()
    if role not in BUILTIN_ROLES:
        raise typer.BadParameter(f"unknown role {role!r}; see `minecraft-ai roles list`")
    if emergency_stop_latched():
        raise typer.BadParameter(
            "Emergency stop is latched. Run `minecraft-ai reset-emergency-stop` "
            "explicitly before starting again."
        )
    if live and operator_pause_latched():
        raise typer.BadParameter(
            "Operator pause is latched. Run `minecraft-ai resume` explicitly before starting."
        )
    if live and edition != Edition.BEDROCK:
        raise typer.BadParameter("The production live backend currently targets Bedrock on Linux.")
    if not supervisor_alive():
        _start_supervisor(role)
    current = send_command("status")
    print(
        f"[green]Supervisor ready[/green] role={role} "
        f"edition={edition.value} state={current['state']}"
    )
    if not live:
        print("Motor control remains SAFE_IDLE/unarmed.")
        return
    if agent_alive():
        print("[yellow]Realtime agent process is already running.[/yellow]")
        return

    try:
        session = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise typer.BadParameter(
            "No managed isolated Bedrock session exists. Run `minecraft-ai bedrock launch` first."
        ) from exc
    if not bedrock_session_alive(session):
        raise typer.BadParameter("The managed Bedrock session is not alive.")
    _require_autonomous_isolated_session(session)
    window_id = wait_for_minecraft_window(session, timeout_s=30.0)
    host_binding = None
    allow_host = False
    install = discover_bedrock_linux_install()
    build = install.selected_build if install is not None else None
    if install is None or build is None:
        raise typer.BadParameter(
            "The exact active BedrockOnLinux build and Wine prefix are required "
            "before live camera control can be calibrated."
        )
    version = build.version

    import time as _time

    launch_frame = None
    hud_verified = False
    last_capture_error: Exception | None = None
    for _attempt in range(12):
        capture = None
        try:
            capture = create_bedrock_capture(
                session.display,
                window_id,
                allow_host=allow_host,
                host_monitor_binding=host_binding,
                source=capture_source,
            )
            for _ in range(2):
                launch_frame = capture.capture()
                if bedrock_in_world_hud_present(launch_frame):
                    hud_verified = True
                    break
        except Exception as exc:
            last_capture_error = exc
        finally:
            if capture is not None:
                capture.close()
        if hud_verified:
            break
        _time.sleep(2.0)
    if not hud_verified or launch_frame is None:
        raise typer.BadParameter(
            "Live control requires a complete in-world survival HUD before arming. "
            f"last capture error: {last_capture_error}"
        )
    print(
        "[green]Complete survival HUD verified[/green] "
        f"capture={launch_frame.width}x{launch_frame.height}"
    )

    attached = _command(
        "attach-bedrock-x11",
        display=session.display,
        window_id=window_id,
        allow_host=allow_host,
        host_monitor_binding=None if host_binding is None else host_binding.payload(),
    )
    config = load_config()
    try:
        profile = load_camera_calibration(
            app_paths().data_dir,
            game_version=version,
        )
        sensitivity = read_bedrock_mouse_sensitivity(install.wine_prefix)
        profile.require_compatible(
            game_version=version,
            mouse_sensitivity=sensitivity,
            configured_yaw_counts_per_degree=(
                config.policy.camera_scale if config.policy.enabled else None
            ),
            configured_pitch_counts_per_degree=(
                config.policy.effective_camera_pitch_scale if config.policy.enabled else None
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "Live camera origin cannot be established from a compatible measured "
            f"Bedrock profile: {exc}"
        ) from exc
    camera_state = attached.get("world_camera")
    reported_pitch_scale = (
        camera_state.get("pitch_counts_per_degree") if isinstance(camera_state, dict) else None
    )
    camera_calibrated = (
        isinstance(camera_state, dict)
        and bool(camera_state.get("origin_calibrated"))
        and camera_state.get("calibration_id") == profile.profile_id
        and isinstance(reported_pitch_scale, (int, float))
        and not isinstance(reported_pitch_scale, bool)
        and abs(float(reported_pitch_scale) - profile.pitch_counts_per_degree) <= 1e-6
    )
    if not camera_calibrated:
        _command(
            "calibrate-world-camera",
            timeout_s=10.0,
            pitch_counts_per_degree=profile.pitch_counts_per_degree,
            calibration_id=profile.profile_id,
        )
        print(
            "[green]Physical camera horizon calibrated[/green] "
            f"Bedrock={version} pitch_counts_per_degree="
            f"{profile.pitch_counts_per_degree:.6f}"
        )
    else:
        print("[green]Physical camera horizon calibration preserved[/green]")

    target = f"bedrock:{version}:x11:{window_id}"
    process = _launch_realtime_agent_transaction(
        target=target,
        display=session.display,
        window_id=window_id,
        role=role,
        allow_host_capture=allow_host,
        capture_source=capture_source,
    )
    print(
        "[bold green]LIVE BEDROCK AGENT STARTED[/bold green] "
        f"pid={process.pid} display={session.display} window={window_id}"
    )
    if host_binding is None:
        print("Host-global input is not enabled; emergency stop remains independent.")
    else:
        print(
            "Dedicated host-monitor guard active: "
            f"{host_binding.output_name} {host_binding.monitor.width}x"
            f"{host_binding.monitor.height}+{host_binding.monitor.x}+"
            f"{host_binding.monitor.y}. Focus loss or geometry drift releases input."
        )


def _start_supervisor(role: str) -> None:
    try:
        existing = ControlEndpoint.load(CONTROL_FILE)
    except FileNotFoundError:
        existing = None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise typer.BadParameter(
            f"supervisor control descriptor is unreadable; refusing unsafe replacement: {exc}"
        ) from exc
    if existing is not None:
        state = control_endpoint_process_state(existing)
        if state in {"verified-live", "unverifiable"}:
            raise typer.BadParameter(
                "existing supervisor owner is alive but its control socket is unavailable; "
                "refusing to discard its identity"
            )
        # Do not unlink a stale pathname here. The candidate supervisor must
        # first acquire the kernel lifecycle lock, then atomically publishes
        # its own endpoint. That closes the stale-check/replacement race.
    with LOG_FILE.open("ab", buffering=0) as log:
        subprocess.Popen(
            [sys.executable, "-m", "minecraft_ai.supervisor", "--role", role],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=(os.name != "nt"),
        )
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if supervisor_alive():
            return
        time.sleep(0.05)
    raise typer.BadParameter(f"supervisor failed to start; inspect {LOG_FILE}")


@app.command()
def status() -> None:
    """Show supervisor, realtime agent and emergency-stop status."""
    if supervisor_alive():
        payload = send_command("status")
    else:
        payload = _read_status_file()
        payload["supervisor_reachable"] = False
        payload["emergency_stop_latched"] = emergency_stop_latched()
        payload["emergency_stop_reason"] = emergency_reason()
    payload["operator_pause_latched"] = operator_pause_latched()
    payload["agent"] = _agent_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def pause() -> None:
    """Pause the agent and revoke motor capability immediately."""
    persistence_error: str | None = None
    command_error: Exception | None = None
    fallback_owner_state: str | None = None
    if supervisor_alive():
        try:
            payload = _command("pause", timeout_s=5.0)
            if payload.get("operator_pause_persisted") is False:
                persistence_error = "supervisor could not persist operator pause"
            if payload.get("agent_containment_confirmed") is not True:
                raise typer.BadParameter(
                    "supervisor could not confirm realtime agent containment after pause"
                )
        except Exception as exc:
            command_error = exc
            with operator_intent_lock():
                persistence_error = _try_latch_operator_pause()
                terminate_registered_supervisor()
                stop_agent_process()
                fallback_owner_state = current_control_owner_state()
            payload = {"state": "STOPPED"}
    else:
        with operator_intent_lock():
            persistence_error = _try_latch_operator_pause()
            terminate_registered_supervisor()
            stop_agent_process()
            fallback_owner_state = current_control_owner_state()
        payload = {"state": "STOPPED"}
    _report_pause_persistence_error(persistence_error)
    if AGENT_FILE.exists():
        raise typer.BadParameter("Agent process containment could not be confirmed after pause.")
    if fallback_owner_state in {"verified-live", "unverifiable", "unreadable"}:
        raise typer.BadParameter(
            f"Supervisor ownership is {fallback_owner_state}; revocation is unconfirmed."
        )
    if command_error is not None:
        owner_state = current_control_owner_state()
        if owner_state in {"verified-live", "unverifiable", "unreadable"}:
            raise typer.BadParameter(
                "Supervisor pause IPC failed and OS-level revocation could not be confirmed: "
                f"{command_error}"
            ) from command_error
        raise typer.BadParameter(f"Supervisor pause IPC failed: {command_error}") from command_error
    print(f"[yellow]{payload['state']}[/yellow] — motor capability revoked.")


@app.command()
def resume() -> None:
    """Return a paused supervisor to SAFE_IDLE; run --live to re-arm gameplay."""
    _resume_operator_intent()


def _resume_operator_intent() -> None:
    """Execute the shared fail-closed durable-resume transaction."""

    if emergency_stop_latched():
        raise typer.BadParameter("Emergency stop is latched; reset it explicitly first.")
    if supervisor_alive():
        payload = _command("resume", timeout_s=5.0)
        print(f"[green]{payload['state']}[/green] — persistent recovery is permitted.")
    else:
        late_supervisor = False
        with operator_intent_lock():
            late_supervisor = supervisor_alive()
            if not late_supervisor:
                owner_state = current_control_owner_state()
                if owner_state in {"verified-live", "unverifiable", "unreadable"}:
                    raise typer.BadParameter(
                        "Supervisor ownership is live or ambiguous; refusing to clear pause."
                    )
                load_state = persistent_agent_service_load_state()
                if load_state == "loaded" and not start_persistent_agent_service():
                    raise typer.BadParameter(
                        "Persistent recovery service did not start; operator pause remains latched."
                    )
                if load_state == "unknown":
                    raise typer.BadParameter(
                        "Persistent recovery service availability is unknown; operator pause "
                        "remains latched."
                    )
                clear_operator_pause()
        if late_supervisor:
            payload = _command("resume", timeout_s=5.0)
            print(f"[green]{payload['state']}[/green] — persistent recovery is permitted.")
            return
        suffix = " and service started" if load_state == "loaded" else " for manual startup"
        print(f"[green]RESUME REQUESTED[/green] — recovery is permitted{suffix}.")


@app.command()
def stop(
    transient: bool = typer.Option(
        False,
        "--transient",
        hidden=True,
        help="Internal service cleanup that does not change persistent operator intent.",
    ),
) -> None:
    """Durably stop autonomous control until an explicit resume."""
    persistence_error: str | None = None
    command_error: Exception | None = None
    if supervisor_alive():
        try:
            result = send_command(
                "stop", timeout_s=5.0, persistent_intent=not transient
            )
            if not transient and result.get("operator_pause_persisted") is False:
                persistence_error = "supervisor could not persist operator pause"
            if result.get("agent_containment_confirmed") is not True:
                raise typer.BadParameter(
                    "supervisor could not confirm realtime agent containment after stop"
                )
        except Exception as exc:
            command_error = exc
            print(f"[yellow]Control socket stop failed ({exc}); using OS fallback.[/yellow]")
            with operator_intent_lock():
                if not transient:
                    persistence_error = _try_latch_operator_pause()
                terminate_registered_supervisor()
                stop_agent_process()
    else:
        with operator_intent_lock():
            if not transient:
                persistence_error = _try_latch_operator_pause()
            terminate_registered_supervisor()
            stop_agent_process()
    _report_pause_persistence_error(persistence_error)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        owner_state = current_control_owner_state()
        if owner_state in {"absent", "dead", "mismatch"} and not AGENT_FILE.exists():
            print("[bold red]STOPPED[/bold red] — agent and supervisor are no longer reachable.")
            return
        time.sleep(0.05)
    detail = "" if command_error is None else f" after IPC failure: {command_error}"
    raise typer.BadParameter(
        f"supervisor process shutdown could not be confirmed{detail}; "
        "use `minecraft-ai emergency-stop`"
    )


@app.command("emergency-stop")
def emergency_stop(reason: str = typer.Option("operator-emergency-stop", "--reason")) -> None:
    """Latch stop and terminate agent/supervisor without depending on normal IPC."""
    pause_error = _try_latch_operator_pause()
    emergency_error: str | None = None
    try:
        engage_emergency_stop(reason)
    except OSError as exc:
        emergency_error = f"{type(exc).__name__}: {exc}"
    # Emergency revocation deliberately bypasses the ordinary intent lock. It
    # must remain available while a slow launch/control transaction is stuck;
    # the independent emergency marker prevents every subsequent re-arm.
    supervisor_terminated = terminate_registered_supervisor()
    agent_terminated = stop_agent_process(timeout_s=1.0)
    service_stopped = _stop_persistent_agent_service()
    # Repeat after stopping the recovery owner to close a process respawn race.
    supervisor_terminated = terminate_registered_supervisor() or supervisor_terminated
    agent_terminated = stop_agent_process(timeout_s=1.0) or agent_terminated
    print("[bold red]EMERGENCY STOP LATCHED[/bold red]")
    print(f"Persistent service stopped: {service_stopped}")
    print(f"Agent termination attempted: {agent_terminated}")
    print(f"Supervisor termination attempted: {supervisor_terminated}")
    if pause_error is not None:
        print(
            "[bold red]WARNING:[/bold red] the operator-pause marker could not be "
            f"persisted ({pause_error})."
        )
    if emergency_error is not None:
        print(
            "[bold red]WARNING:[/bold red] immediate stop was attempted, but the emergency "
            f"latch could not be persisted ({emergency_error}). Keep the service stopped."
        )
    if pause_error is not None and emergency_error is not None:
        raise typer.BadParameter(
            "neither durable stop marker could be written; immediate shutdown was attempted"
        )
    print(
        "The agent cannot restart until both `minecraft-ai reset-emergency-stop` "
        "and `minecraft-ai resume` are run."
    )


@app.command("reset-emergency-stop")
def reset_emergency_stop() -> None:
    """Clear persistent stop latch. This does not start or arm the agent."""
    service_load_state = persistent_agent_service_load_state()
    service_state = (
        _persistent_agent_service_state()
        if service_load_state == "loaded"
        else service_load_state
    )
    if service_state not in {"inactive", "not-found"}:
        raise typer.BadParameter(
            "The persistent minecraft-ai-agent-live service must be confirmed inactive before "
            f"resetting the latch (observed {service_state})."
        )
    if CONTROL_FILE.exists():
        try:
            endpoint = ControlEndpoint.load(CONTROL_FILE)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise typer.BadParameter(
                f"Supervisor control descriptor is unreadable; refusing reset: {exc}"
            ) from exc
        endpoint_state = control_endpoint_process_state(endpoint)
        if endpoint_state in {"verified-live", "unverifiable"}:
            raise typer.BadParameter(
                "A supervisor owner may still be active; refusing to clear the emergency latch."
            )
    if supervisor_alive() or agent_alive():
        raise typer.BadParameter("Stop the agent and supervisor before resetting the latch.")
    try:
        latch_operator_pause()
    except OSError as exc:
        raise typer.BadParameter(
            f"Could not establish the required operator-pause marker: {exc}"
        ) from exc
    if not operator_pause_latched():
        raise typer.BadParameter(
            "Could not verify the required operator-pause marker; emergency latch retained."
        )
    clear_emergency_stop()
    print(
        "[green]Emergency stop latch cleared.[/green] "
        "Agent remains stopped until `minecraft-ai resume`."
    )


@app.command()
def logs(lines: int = typer.Option(80, min=1, max=2000)) -> None:
    """Show the tail of supervisor and realtime-agent logs."""
    for label, path in (("supervisor", LOG_FILE), ("agent", AGENT_LOG)):
        print(f"[bold]{label}[/bold] {path}")
        if not path.exists():
            print("No log yet.")
            continue
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(content[-lines:]))


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Numeric loopback address."),
    port: int = typer.Option(8765, min=1, max=65535),
) -> None:
    """Serve the local telemetry and high-level agent interaction surface."""
    _ensure_dirs()
    print(f"[green]Operator dashboard[/green] http://{host}:{port}/")
    serve_operator_dashboard(host=host, port=port)


@app.command("record-human")
def record_human(
    duration_s: float = typer.Option(300.0, "--duration", min=1.0, max=86_400.0),
    capture_hz: float = typer.Option(20.0, "--capture-hz", min=5.0, max=60.0),
    label: str = typer.Option("human-demonstration", "--label"),
    task_id: str | None = typer.Option(None, "--task"),
    fov: float | None = typer.Option(None, "--fov", min=30.0, max=180.0),
    mouse_sensitivity: float | None = typer.Option(
        None,
        "--mouse-sensitivity",
        min=0.0,
        max=1.0,
    ),
    takeover: bool = typer.Option(
        False,
        "--takeover",
        help="Stop autonomous control and pause its supervisor before recording.",
    ),
    resume_live: bool = typer.Option(
        False,
        "--resume-live",
        help="Explicitly restart autonomous play after recording finishes.",
    ),
) -> None:
    """Record Bedrock pixels and raw human input through the isolated display."""
    _ensure_dirs()
    if task_id is not None:
        try:
            bedrock_baseline_suite().task(task_id)
        except KeyError as exc:
            raise typer.BadParameter(f"unknown benchmark task: {task_id}") from exc
    config = load_config()
    try:
        recorded_agent = AgentProcess.load()
    except FileNotFoundError:
        recorded_agent = None
        was_alive = False
        agent_ownership_unconfirmed = False
    except (OSError, ValueError, TypeError, KeyError):
        recorded_agent = None
        was_alive = False
        agent_ownership_unconfirmed = True
    else:
        was_alive = agent_alive(recorded_agent)
        agent_ownership_unconfirmed = not was_alive and AGENT_FILE.exists()
    supervisor_was_alive = supervisor_alive()
    control_owner_state = current_control_owner_state()
    supervisor_ownership_unconfirmed = control_owner_state in {
        "verified-live",
        "unverifiable",
        "unreadable",
    }
    persistent_service_state = _persistent_agent_service_state()
    persistent_service_may_run = persistent_service_state != "inactive"
    if (
        was_alive
        or agent_ownership_unconfirmed
        or supervisor_was_alive
        or supervisor_ownership_unconfirmed
        or persistent_service_may_run
    ) and not takeover:
        raise typer.BadParameter(
            "Autonomous ownership is active or unconfirmed. Pass --takeover to prevent "
            "mixed human/model labels."
        )
    if takeover:
        _prepare_human_recording_takeover()
    try:
        session = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise typer.BadParameter(
            "No managed isolated Bedrock session exists. Run `minecraft-ai bedrock launch` first."
        ) from exc
    if not bedrock_session_alive(session) or session.mode not in {"xephyr", "weston"}:
        raise typer.BadParameter("Human recording requires a live managed isolated session.")
    window_id = wait_for_minecraft_window(session, timeout_s=10.0)
    install = discover_bedrock_linux_install()
    build = install.selected_build if install is not None else None
    version = build.version if build is not None else "unknown"
    paths = app_paths()
    print(
        "[bold green]HUMAN BEDROCK RECORDING[/bold green] "
        f"duration={duration_s:.1f}s rate={capture_hz:.1f}Hz display={session.display}"
    )
    print("Focus the Bedrock window and play normally; Ctrl-C also seals the trajectory.")
    manifest = record_human_session(
        HumanRecordingRequest(
            display=session.display,
            window_id=window_id,
            instance_id=f"bedrock:{version}:x11:{window_id}",
            role=config.role,
            game_version=version,
            artifact_root=paths.data_dir / "trajectories",
            state_db_path=paths.state_db,
            duration_s=duration_s,
            capture_hz=capture_hz,
            label=label,
            task_id=task_id,
            fov=fov,
            mouse_sensitivity=mouse_sensitivity,
            shard_steps=config.trajectory.shard_steps,
            queue_size=config.trajectory.queue_size,
        )
    )
    destination = paths.data_dir / "trajectories" / manifest.trajectory_id
    print(
        "[green]Sealed human trajectory[/green] "
        f"steps={manifest.accepted_steps} dropped={manifest.dropped_steps} -> {destination}"
    )
    if resume_live:
        _resume_operator_intent()
        run(role=config.role, edition=Edition.BEDROCK, live=True)


@dataset_app.command("inspect")
def dataset_inspect(
    trajectory: Path = typer.Argument(..., exists=True),
) -> None:
    """Inspect one portable trajectory and summarize accepted motor behavior."""
    reader = TrajectoryReader(trajectory)
    accumulator = TraceMetricAccumulator()
    validation = reader.validate(on_sample=accumulator.add)
    metrics: dict[str, float | int | bool | str] = {}
    if validation.valid:
        metrics = accumulator.finish().values
    payload = {
        "manifest": reader.manifest.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


@dataset_app.command("validate")
def dataset_validate(
    trajectory: Path = typer.Argument(..., exists=True),
) -> None:
    """Replay and integrity-check every sealed sample in one trajectory."""
    report = TrajectoryReader(trajectory).validate()
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if not report.valid:
        raise typer.Exit(code=1)


@eval_app.command("tasks")
def eval_tasks() -> None:
    """List the immutable Bedrock M1 benchmark task contracts."""
    suite = bedrock_baseline_suite()
    print(json.dumps(suite.model_dump(mode="json"), indent=2, sort_keys=True))


@eval_app.command("run")
def eval_run(
    trajectory: Path = typer.Option(..., "--trajectory", exists=True),
    task_ids: list[str] = typer.Option(..., "--task", help="Repeat for multiple tasks."),
    evidence: Path | None = typer.Option(
        None,
        "--evidence",
        exists=True,
        dir_okay=False,
        help="Controlled evaluator evidence JSON; never exposed to the agent.",
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Evaluate accepted actions plus independently observed task outcomes."""
    suite = bedrock_baseline_suite()
    for task_id in task_ids:
        try:
            suite.task(task_id)
        except KeyError as exc:
            raise typer.BadParameter(f"unknown benchmark task: {task_id}") from exc
    runner = BenchmarkRunner(suite)
    report = runner.evaluate_trajectory(
        trajectory,
        task_ids=tuple(task_ids),
        evidence=None if evidence is None else load_evidence(evidence),
        git_commit=_git_commit(),
    )
    destination = output or _benchmark_output(report.benchmark_run_id)
    report.write(destination)
    with StateDatabase(app_paths().state_db) as database:
        database.save_benchmark_report(report)
    print(report.model_dump_json(indent=2))
    print(f"[green]Benchmark report:[/green] {destination}")


@eval_app.command("compare")
def eval_compare(
    baseline: Path = typer.Argument(..., exists=True, dir_okay=False),
    candidate: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Compare two reports without treating small samples as promotion evidence."""
    comparison = compare_reports(_load_json_object(baseline), _load_json_object(candidate))
    if output is not None:
        _write_json_atomic(output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


@benchmark_app.command("report")
def benchmark_report(
    trajectory_root: Path | None = typer.Option(
        None,
        "--trajectory-root",
        exists=True,
        file_okay=False,
        help="Root containing one directory per trajectory.",
    ),
    evidence_dir: Path | None = typer.Option(
        None,
        "--evidence-dir",
        exists=True,
        file_okay=False,
        help="Optional evaluator evidence files named <trajectory-id>.json.",
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Aggregate task-tagged trajectories into the frozen baseline report."""
    root = trajectory_root or app_paths().data_dir / "trajectories"
    if not root.is_dir():
        raise typer.BadParameter(f"trajectory root does not exist: {root}")
    trajectories = tuple(sorted(path.parent for path in root.glob("*/manifest.json")))
    evidence_by_trajectory = {}
    if evidence_dir is not None:
        for evidence_path in sorted(evidence_dir.glob("*.json")):
            evidence_by_trajectory[evidence_path.stem] = load_evidence(evidence_path)
    runner = BenchmarkRunner(bedrock_baseline_suite())
    report = runner.evaluate_many(
        trajectories,
        evidence_by_trajectory=evidence_by_trajectory,
        git_commit=_git_commit(),
    )
    destination = output or _benchmark_output(report.benchmark_run_id)
    report.write(destination)
    with StateDatabase(app_paths().state_db) as database:
        database.save_benchmark_report(report)
    print(report.model_dump_json(indent=2))
    print(f"[green]Benchmark report:[/green] {destination}")


def _benchmark_output(benchmark_run_id: str) -> Path:
    return app_paths().data_dir / "benchmarks" / f"{benchmark_run_id}.json"


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staged.replace(path)


@config_app.command("show")
def config_show() -> None:
    config = load_config()
    print(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))


@config_app.command("init")
def config_init() -> None:
    path = ensure_default_config()
    print(f"[green]Configuration ready:[/green] {path}")


@bedrock_app.command("status")
def bedrock_status() -> None:
    """Show BedrockOnLinux install/build/process and managed isolation status."""
    install = discover_bedrock_linux_install()
    instances = find_bedrock_linux_instances()
    try:
        session = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError):
        session = None
    payload = {
        "install": None
        if install is None
        else {
            "data_dir": str(install.data_dir),
            "wine_prefix": str(install.wine_prefix),
            "launcher": install.launcher_command,
            "selected_build": None
            if install.selected_build is None
            else {
                "edition": install.selected_build.edition_id,
                "version": install.selected_build.version,
                "root": str(install.selected_build.game_root),
            },
        },
        "processes": [instance.instance_id for instance in instances],
        "managed_session": _session_payload(session) if session is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


@bedrock_app.command("launch")
def bedrock_launch(
    width: int = typer.Option(DEFAULT_BEDROCK_WIDTH, min=320, max=7680),
    height: int = typer.Option(DEFAULT_BEDROCK_HEIGHT, min=240, max=4320),
    fullscreen: bool = typer.Option(
        True,
        "--fullscreen/--windowed",
        help="Present isolated Weston fullscreen so Bedrock's complete HUD remains visible.",
    ),
    direct: bool = typer.Option(
        False,
        "--direct-debug",
        help="Launch on the host display for manual debugging; autonomous input stays disabled.",
    ),
) -> None:
    """Launch BedrockOnLinux in an accelerated isolated compositor."""
    # Acquire locks in the same order as durable Bedrock stop, but retain only
    # the lifecycle lock for the potentially slow launcher. A later stop can
    # publish its pause immediately, then waits and reaps this exact launch.
    with ExitStack() as launch_stack:
        with operator_intent_lock():
            if emergency_stop_latched():
                raise typer.BadParameter("Emergency stop is latched; Bedrock launch is disabled.")
            if operator_pause_latched():
                raise typer.BadParameter("Operator pause is latched; Bedrock launch is disabled.")
            launch_stack.enter_context(bedrock_lifecycle_lock())
        _bedrock_launch_locked(
            width=width,
            height=height,
            fullscreen=fullscreen,
            direct=direct,
        )


def _bedrock_launch_locked(
    *,
    width: int,
    height: int,
    fullscreen: bool,
    direct: bool,
) -> None:
    try:
        existing_session = BedrockSession.load()
    except FileNotFoundError:
        existing_session = None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise typer.BadParameter(
            "managed Bedrock descriptor is unreadable; refusing unsafe replacement: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if existing_session is not None:
        if bedrock_session_alive(existing_session):
            print("[yellow]Managed Bedrock session is already running.[/yellow]")
            print(json.dumps(_session_payload(existing_session), indent=2, sort_keys=True))
            return
        # The launcher can exit while its nested compositor remains healthy.
        # Reap that exact persisted session before replacing the descriptor so
        # an orphaned fullscreen surface cannot occupy the Bedrock monitor.
        stop_bedrock_session(existing_session)
    if direct:
        from .platforms.bedrock_session import launch_direct_bedrock_session

        session = launch_direct_bedrock_session(width=width, height=height)
        print(
            "[yellow]Direct debug session: capture inspection is allowed, but autonomous "
            "motor control cannot be armed.[/yellow]"
        )
    else:
        session = launch_isolated_bedrock_session(
            width=width,
            height=height,
            fullscreen=fullscreen,
        )
    print("[green]Bedrock session launched.[/green]")
    print(json.dumps(_session_payload(session), indent=2, sort_keys=True))


@bedrock_app.command("navigate")
def bedrock_navigate(
    lan_name: str = typer.Option(
        "BedrockConnect",
        "--lan-name",
        help="Exact discovered LAN world label to select.",
    ),
    server_name: str | None = typer.Option(
        None,
        "--server-name",
        help="Exact local custom-server label; required when more than one exists.",
    ),
    custom_servers: Path = typer.Option(
        DEFAULT_BEDROCK_CONNECT_SERVERS,
        "--custom-servers",
        help="BedrockConnect custom_servers.json path.",
    ),
    timeout_s: float = typer.Option(120.0, "--timeout-s", min=10.0, max=600.0),
    retries: int = typer.Option(2, "--retries", min=1, max=5),
) -> None:
    """Navigate a managed nested Bedrock client into its configured local server."""
    if emergency_stop_latched():
        raise typer.BadParameter("Emergency stop is latched; menu input remains disabled.")
    if operator_pause_latched():
        raise typer.BadParameter("Operator pause is latched; menu input remains disabled.")
    try:
        session = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise typer.BadParameter("No managed Bedrock session exists.") from exc
    if not bedrock_session_alive(session):
        raise typer.BadParameter("The managed Bedrock session is not alive.")
    _require_autonomous_isolated_session(session)
    server = load_configured_local_server(custom_servers, requested_name=server_name)
    window_id = wait_for_minecraft_window(session, timeout_s=30.0)

    input_backend: NestedXTestMenuInput | None = None
    capture: IsolatedX11Capture | None = None
    try:
        input_backend = NestedXTestMenuInput(
            session.display,
            window_id,
            host_display=session.host_display,
            input_permitted=lambda: (
                not emergency_stop_latched() and not operator_pause_latched()
            ),
        )
        capture = IsolatedX11Capture(
            session.display,
            input_backend.input_window_id,
            host_display=session.host_display,
            allow_host=False,
        )
        navigator = BedrockMenuNavigator(
            capture=capture,
            text_reader=TesseractMenuTextReader(),
            click_backend=input_backend,
            lan_name=lan_name,
            server=server,
            timeout_s=timeout_s,
            max_retries=retries,
            input_permitted=lambda: (
                not emergency_stop_latched() and not operator_pause_latched()
            ),
        )
        result = navigator.run()
    except (IsolationError, MenuNavigationError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        if capture is not None:
            capture.close()
        if input_backend is not None:
            input_backend.close()
    print(json.dumps(result.payload(), indent=2, sort_keys=True))


@bedrock_app.command("bind-monitor")
def bedrock_bind_monitor(
    output: str = typer.Argument(
        ...,
        help="Exact active RandR output exclusively occupied by Minecraft (for example DP-2).",
    ),
) -> None:
    """Bind a direct Bedrock window to one dedicated monitor without moving it."""

    with bedrock_lifecycle_lock():
        try:
            session = BedrockSession.load()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise typer.BadParameter("No managed Bedrock session exists.") from exc
        try:
            bound = bind_direct_session_to_monitor(session, output_name=output)
        except (OSError, RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    binding = bound.host_monitor_binding()
    if binding is None:
        raise typer.BadParameter("Host-monitor binding was not persisted.")
    print(
        "[bold green]DEDICATED MINECRAFT MONITOR VERIFIED[/bold green] "
        f"{binding.output_name}={binding.monitor.width}x{binding.monitor.height}+"
        f"{binding.monitor.x}+{binding.monitor.y} window={binding.window_id}"
    )
    print(
        "Autonomous input may now be armed only while this exact window remains "
        "fullscreen on this output and Minecraft retains input focus."
    )


@bedrock_app.command("stop")
def bedrock_stop(
    transient: bool = typer.Option(
        False,
        "--transient",
        hidden=True,
        help="Internal service cleanup that does not change persistent operator intent.",
    ),
) -> None:
    """Durably stop control, then stop the managed Bedrock session."""
    persistence_error: str | None = None

    def revoke_and_stop_session() -> None:
        nonlocal persistence_error
        if not transient:
            persistence_error = _try_latch_operator_pause()
        if supervisor_alive():
            try:
                current = send_command("status")
                if bool(current.get("live_capable")):
                    # The durable marker is already authoritative. Disarm does
                    # not acquire the intent lock, so a concurrent resume waits
                    # until the entire Bedrock shutdown transaction completes.
                    send_command("disarm", timeout_s=5.0)
            except Exception as exc:
                # A compositor crash also closes the supervisor's X11 backend.
                # Shutdown must still reap the launcher/compositor descriptor.
                print(f"[yellow]Supervisor revocation failed during cleanup: {exc}[/yellow]")
                terminate_registered_supervisor()
        stop_agent_process()
        # A launch already owns this lock without holding operator intent. Wait
        # for it to publish its exact descriptor, then reap that same session;
        # failing immediately would let a completed launch survive a durable
        # stop request that had already latched and disarmed control.
        with bedrock_lifecycle_lock(wait_timeout_s=30.0):
            stop_bedrock_session()

    if transient:
        revoke_and_stop_session()
    else:
        with operator_intent_lock():
            revoke_and_stop_session()
            if persistence_error is not None:
                stop_persistent_agent_service()
    _report_pause_persistence_error(persistence_error)
    if AGENT_FILE.exists():
        raise typer.BadParameter("Agent process containment could not be confirmed.")
    print("[bold red]Managed Bedrock session stopped.[/bold red]")


@roles_app.command("list")
def roles_list() -> None:
    payload = {
        role_id: {
            "description": role.description,
            "standing_goals": list(role.standing_goals),
            "risk_tolerance": role.risk_tolerance,
            "utility_weights": role.utility_weights,
        }
        for role_id, role in sorted(BUILTIN_ROLES.items())
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def wiki(
    query: str,
    version: str | None = typer.Option(None, "--version"),
    edition: Edition = typer.Option(DEFAULT_EDITION, "--edition"),
) -> None:
    """Search version-locked Minecraft Wiki evidence for player/game questions."""
    resolved = _bedrock_version_or_error(version) if edition == Edition.BEDROCK else version
    if not resolved:
        raise typer.BadParameter("--version is required for Java wiki lookup")
    service = WikiService(WIKI_CACHE)
    evidence = service.search(query, GameVersion(edition=edition, version_id=resolved))
    payload = [item.model_dump(mode="json") for item in evidence]
    print(json.dumps(payload, indent=2, sort_keys=True))


@knowledge_app.command("sync")
def knowledge_sync(
    version: str | None = typer.Option(None, "--version", help="Exact Minecraft version."),
    data_root: Path = typer.Option(..., "--data-root", exists=True, file_okay=False),
    edition: Edition = typer.Option(DEFAULT_EDITION, "--edition"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Compile exact-version machine-readable game data into a provenance graph."""
    _ensure_dirs()
    resolved_version = _bedrock_version_or_error(version) if edition == Edition.BEDROCK else version
    if not resolved_version:
        raise typer.BadParameter("--version is required for Java knowledge sync")
    game_version = GameVersion(edition=edition, version_id=resolved_version)
    if edition == Edition.JAVA:
        graph = import_java_datapack(data_root, game_version)
    else:
        graph = import_minecraft_data(data_root, game_version)
    errors = graph.validate()
    if errors:
        raise typer.BadParameter("compiled graph failed validation: " + "; ".join(errors[:10]))
    destination = output or _graph_path(game_version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[green]Compiled[/green] {game_version.key}: "
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges -> {destination}"
    )


@knowledge_app.command("summary")
def knowledge_summary(graph_file: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Inspect a compiled graph without loading any model."""
    graph = _load_graph(graph_file)
    counts: dict[str, int] = {}
    for node in graph.nodes.values():
        counts[node.kind.value] = counts.get(node.kind.value, 0) + 1
    payload = {
        "version": graph.version.model_dump(mode="json"),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "node_kinds": dict(sorted(counts.items())),
        "validation_errors": graph.validate(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


@knowledge_app.command("requirements")
def knowledge_requirements(
    graph_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    node_id: str = typer.Argument(...),
    recursive: bool = typer.Option(False, "--recursive"),
) -> None:
    """Show direct or transitive prerequisites for a graph node."""
    graph = _load_graph(graph_file)
    if node_id not in graph.nodes:
        raise typer.BadParameter(f"node not found: {node_id}")
    if recursive:
        result: object = sorted(graph.prerequisite_closure(node_id))
    else:
        result = [edge.model_dump(mode="json") for edge in graph.requirements(node_id)]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
