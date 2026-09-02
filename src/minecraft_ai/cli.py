from __future__ import annotations

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

from .emergency import (
    clear_emergency_stop,
    emergency_reason,
    emergency_stop_latched,
    engage_emergency_stop,
    terminate_registered_supervisor,
)
from .knowledge import Edition, GameVersion, KnowledgeGraph
from .knowledge.importers import import_java_datapack, import_minecraft_data
from .platforms import discover_bedrock_linux_install, find_bedrock_linux_instances
from .platforms.bedrock_session import (
    BEDROCK_SESSION_FILE,
    BedrockSession,
    bedrock_session_alive,
    launch_xephyr_bedrock_session,
    stop_bedrock_session,
    wait_for_minecraft_window,
)
from .roles import BUILTIN_ROLES
from .supervisor import CONTROL_FILE, STATUS_FILE, send_command, supervisor_alive

app = typer.Typer(help="Minecraft AI lifecycle and tooling CLI.")
knowledge_app = typer.Typer(help="Versioned game knowledge commands.")
bedrock_app = typer.Typer(help="BedrockOnLinux isolated-session commands.")
roles_app = typer.Typer(help="Agent role/archetype commands.")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(bedrock_app, name="bedrock")
app.add_typer(roles_app, name="roles")

APP_NAME = "minecraft-ai"
DATA_DIR = Path(user_data_dir(APP_NAME))
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
LOG_FILE = DATA_DIR / "logs" / "supervisor.log"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
DEFAULT_EDITION = Edition.BEDROCK


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


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
    return {
        "display": session.display,
        "host_display": session.host_display,
        "xserver_pid": session.xserver_pid,
        "launcher_pid": session.launcher_pid,
        "width": session.width,
        "height": session.height,
        "mode": session.mode,
        "alive": bedrock_session_alive(session),
        "minecraft_window": session.find_window() if bedrock_session_alive(session) else None,
    }


@app.command()
def install() -> None:
    """Prepare local state and report all runtime dependencies/capabilities."""
    _ensure_dirs()
    print("[green]Minecraft AI bootstrap prepared.[/green]")
    doctor()


@app.command()
def doctor() -> None:
    """Report platform capabilities without enabling live control."""
    _ensure_dirs()
    linux = platform.system() == "Linux"
    bol = discover_bedrock_linux_install() if linux else None
    bedrock_instances = find_bedrock_linux_instances() if linux else []
    selected_build = bol.selected_build if bol is not None else None
    session: BedrockSession | None
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
            "xephyr": shutil.which("Xephyr"),
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
    import importlib.util

    return importlib.util.find_spec(name) is not None


@app.command()
def run(
    role: str = typer.Option("generalist", help="Role/archetype profile."),
    edition: Edition = typer.Option(DEFAULT_EDITION, "--edition"),
    live: bool = typer.Option(False, help="Attach the isolated Bedrock motor backend."),
) -> None:
    """Start the independent supervisor; optionally attach isolated Bedrock input."""
    _ensure_dirs()
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

    try:
        session = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise typer.BadParameter(
            "No managed isolated Bedrock session exists. Run `minecraft-ai bedrock launch` first."
        ) from exc
    if not bedrock_session_alive(session):
        raise typer.BadParameter("The managed Bedrock session is not alive.")
    window_id = wait_for_minecraft_window(session, timeout_s=5.0)
    _command("attach-bedrock-x11", display=session.display, window_id=window_id)
    install = discover_bedrock_linux_install()
    build = install.selected_build if install is not None else None
    target = f"bedrock:{build.version if build is not None else 'unknown'}:x11:{window_id}"
    armed = _command("arm", target_instance=target)
    lease = armed.get("lease")
    if not isinstance(lease, dict) or not isinstance(lease.get("lease_id"), str):
        raise typer.BadParameter("supervisor did not return a valid motor lease")
    _command("activate")
    print(
        "[bold green]LIVE BEDROCK BACKEND ARMED[/bold green] "
        f"display={session.display} window={window_id}"
    )
    print("No host-global input backend is enabled. Emergency stop remains independently latched.")


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
    """Show live supervisor state, or the last persisted safe state."""
    if supervisor_alive():
        payload = send_command("status")
    else:
        payload = _read_status_file()
        payload["supervisor_reachable"] = False
        payload["emergency_stop_latched"] = emergency_stop_latched()
        payload["emergency_stop_reason"] = emergency_reason()
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def pause() -> None:
    """Pause the agent and revoke motor capability immediately."""
    payload = _command("pause")
    print(f"[yellow]{payload['state']}[/yellow] — motor capability revoked.")


@app.command()
def resume() -> None:
    """Return a paused supervisor to SAFE_IDLE; this never arms live input."""
    if emergency_stop_latched():
        raise typer.BadParameter("Emergency stop is latched; reset it explicitly first.")
    payload = _command("resume")
    print(f"[green]{payload['state']}[/green] — live input remains unarmed.")


@app.command()
def stop() -> None:
    """Normal stop: control socket first, OS-level PID termination as fallback."""
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
            print("[bold red]STOPPED[/bold red] — supervisor is no longer reachable.")
            return
        time.sleep(0.05)
    raise typer.BadParameter("supervisor did not stop; use `minecraft-ai emergency-stop`")


@app.command("emergency-stop")
def emergency_stop(reason: str = typer.Option("operator-emergency-stop", "--reason")) -> None:
    """Latch a stop and terminate the registered supervisor without using IPC."""
    engage_emergency_stop(reason)
    terminated = terminate_registered_supervisor()
    print("[bold red]EMERGENCY STOP LATCHED[/bold red]")
    print(f"OS-level supervisor termination attempted: {terminated}")
    print("The agent cannot be restarted until `minecraft-ai reset-emergency-stop` is run.")


@app.command("reset-emergency-stop")
def reset_emergency_stop() -> None:
    """Clear the persistent stop latch. This does not start or arm the agent."""
    if supervisor_alive():
        raise typer.BadParameter("Stop the supervisor before resetting the emergency latch.")
    clear_emergency_stop()
    print("[green]Emergency stop latch cleared.[/green] Agent remains stopped.")


@app.command()
def logs(lines: int = typer.Option(80, min=1, max=2000)) -> None:
    """Show the tail of the supervisor log."""
    if not LOG_FILE.exists():
        print("No supervisor log yet.")
        return
    content = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


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
    width: int = typer.Option(1280, min=320, max=7680),
    height: int = typer.Option(720, min=240, max=4320),
) -> None:
    """Launch BedrockOnLinux inside a dedicated nested X input namespace."""
    if bedrock_session_alive():
        session = BedrockSession.load()
        print("[yellow]Managed Bedrock session is already running.[/yellow]")
        print(json.dumps(_session_payload(session), indent=2, sort_keys=True))
        return
    session = launch_xephyr_bedrock_session(width=width, height=height)
    print("[green]Isolated Bedrock session launched.[/green]")
    print(json.dumps(_session_payload(session), indent=2, sort_keys=True))


@bedrock_app.command("stop")
def bedrock_stop() -> None:
    """Stop the managed Bedrock nested session and its launcher process group."""
    if supervisor_alive():
        current = send_command("status")
        if bool(current.get("live_capable")):
            send_command("pause")
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
def wiki(query: str) -> None:
    """Placeholder for exact-version sourced game Q&A."""
    print(f"Wiki service not implemented yet. Query preserved: {query!r}")


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
