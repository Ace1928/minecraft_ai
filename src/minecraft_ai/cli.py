from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import typer
from platformdirs import user_data_dir, user_runtime_dir
from rich import print

from .agent_lifecycle import (
    AGENT_LOG,
    AgentProcess,
    agent_alive,
    launch_agent_process,
    stop_agent_process,
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
from .platforms import discover_bedrock_linux_install, find_bedrock_linux_instances
from .platforms.bedrock_session import (
    BEDROCK_SESSION_FILE,
    DEFAULT_BEDROCK_HEIGHT,
    DEFAULT_BEDROCK_WIDTH,
    BedrockSession,
    bedrock_session_alive,
    launch_isolated_bedrock_session,
    stop_bedrock_session,
    wait_for_minecraft_window,
)
from .roles import BUILTIN_ROLES
from .storage import StateDatabase
from .supervisor import CONTROL_FILE, STATUS_FILE, send_command, supervisor_alive
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


def _command(command: str, **payload: object) -> dict[str, object]:
    try:
        return send_command(command, **payload)
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


@app.command()
def run(
    role: str = typer.Option("generalist", help="Role/archetype profile."),
    edition: Edition = typer.Option(DEFAULT_EDITION, "--edition"),
    live: bool = typer.Option(False, help="Run the realtime isolated Bedrock agent."),
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
    if session.mode not in {"xephyr", "weston"}:
        raise typer.BadParameter(
            "Direct host-display sessions are observation/debug only and cannot be armed. "
            "Stop it and launch the default isolated session first."
        )
    window_id = wait_for_minecraft_window(session, timeout_s=30.0)
    _command(
        "attach-bedrock-x11",
        display=session.display,
        window_id=window_id,
        allow_host=False,
    )
    install = discover_bedrock_linux_install()
    build = install.selected_build if install is not None else None
    version = build.version if build is not None else "unknown"
    target = f"bedrock:{version}:x11:{window_id}"
    armed = _command("arm", target_instance=target)
    lease = armed.get("lease")
    if not isinstance(lease, dict) or not isinstance(lease.get("lease_id"), str):
        raise typer.BadParameter("supervisor did not return a valid motor lease")
    lease_id = str(lease["lease_id"])
    _command("activate")
    try:
        process = launch_agent_process(
            lease_id=lease_id,
            display=session.display,
            window_id=window_id,
            instance_id=target,
            role=role,
        )
    except Exception as exc:
        _command("disarm")
        raise typer.BadParameter(f"realtime agent failed to start: {exc}") from exc
    print(
        "[bold green]LIVE BEDROCK AGENT STARTED[/bold green] "
        f"pid={process.pid} display={session.display} window={window_id}"
    )
    print("Host-global input is not enabled; emergency stop remains independent.")


def _start_supervisor(role: str) -> None:
    try:
        CONTROL_FILE.unlink()
    except FileNotFoundError:
        pass
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
    payload["agent"] = _agent_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def pause() -> None:
    """Pause the agent and revoke motor capability immediately."""
    stop_agent_process()
    payload = _command("pause")
    print(f"[yellow]{payload['state']}[/yellow] — motor capability revoked.")


@app.command()
def resume() -> None:
    """Return a paused supervisor to SAFE_IDLE; run --live to re-arm gameplay."""
    if emergency_stop_latched():
        raise typer.BadParameter("Emergency stop is latched; reset it explicitly first.")
    payload = _command("resume")
    print(f"[green]{payload['state']}[/green] — run --live to attach gameplay again.")


@app.command()
def stop() -> None:
    """Stop realtime agent first, then stop the independent supervisor."""
    stop_agent_process()
    if not CONTROL_FILE.exists():
        print("[bold red]STOPPED[/bold red] — no supervisor endpoint exists.")
        return
    try:
        send_command("stop")
    except Exception as exc:
        print(f"[yellow]Control socket stop failed ({exc}); using OS fallback.[/yellow]")
        terminate_registered_supervisor()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not supervisor_alive():
            print("[bold red]STOPPED[/bold red] — agent and supervisor are no longer reachable.")
            return
        time.sleep(0.05)
    raise typer.BadParameter("supervisor did not stop; use `minecraft-ai emergency-stop`")


@app.command("emergency-stop")
def emergency_stop(reason: str = typer.Option("operator-emergency-stop", "--reason")) -> None:
    """Latch stop and terminate agent/supervisor without depending on normal IPC."""
    engage_emergency_stop(reason)
    agent_terminated = stop_agent_process(timeout_s=1.0)
    supervisor_terminated = terminate_registered_supervisor()
    print("[bold red]EMERGENCY STOP LATCHED[/bold red]")
    print(f"Agent termination attempted: {agent_terminated}")
    print(f"Supervisor termination attempted: {supervisor_terminated}")
    print("The agent cannot restart until `minecraft-ai reset-emergency-stop` is run.")


@app.command("reset-emergency-stop")
def reset_emergency_stop() -> None:
    """Clear persistent stop latch. This does not start or arm the agent."""
    if supervisor_alive() or agent_alive():
        raise typer.BadParameter("Stop the agent and supervisor before resetting the latch.")
    clear_emergency_stop()
    print("[green]Emergency stop latch cleared.[/green] Agent remains stopped.")


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
    was_alive = agent_alive()
    if was_alive and not takeover:
        raise typer.BadParameter(
            "The autonomous agent is running. Pass --takeover to prevent mixed human/model labels."
        )
    if was_alive:
        stop_agent_process()
        if supervisor_alive():
            _command("pause")
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
        if supervisor_alive():
            state = str(send_command("status").get("state", ""))
            if state == "PAUSED":
                _command("resume")
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
    try:
        existing_session = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError):
        existing_session = None
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


@bedrock_app.command("stop")
def bedrock_stop() -> None:
    """Stop realtime control then the managed Bedrock nested session."""
    stop_agent_process()
    if supervisor_alive():
        try:
            current = send_command("status")
            if bool(current.get("live_capable")):
                send_command("pause")
        except Exception as exc:
            # A compositor crash also closes the supervisor's X11 backend.
            # Shutdown must still reap the launcher/compositor descriptor.
            print(f"[yellow]Supervisor pause failed during cleanup: {exc}[/yellow]")
    stop_bedrock_session()
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
