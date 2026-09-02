from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import typer
from platformdirs import user_data_dir, user_runtime_dir
from rich import print

from .supervisor import CONTROL_FILE, STATUS_FILE, ControlEndpoint, send_command, supervisor_alive

app = typer.Typer(help="Minecraft AI lifecycle and tooling CLI.")
knowledge_app = typer.Typer(help="Versioned game knowledge commands.")
app.add_typer(knowledge_app, name="knowledge")

APP_NAME = "minecraft-ai"
DATA_DIR = Path(user_data_dir(APP_NAME))
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
LOG_FILE = DATA_DIR / "logs" / "supervisor.log"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


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


@app.command()
def install() -> None:
    """Prepare local directories and report the platform profile.

    Model/runtime downloads and Minecraft input bridge installation remain
    gated until their license, compatibility and safety checks land.
    """
    _ensure_dirs()
    print("[green]Minecraft AI bootstrap prepared.[/green]")
    doctor()


@app.command()
def doctor() -> None:
    """Report platform capabilities without enabling live control."""
    _ensure_dirs()
    profile = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "data_dir": str(DATA_DIR),
        "runtime_dir": str(RUNTIME_DIR),
        "supervisor": "running" if supervisor_alive() else "stopped",
        "scoped_input": "not-installed",
        "live_input": False,
    }
    print(json.dumps(profile, indent=2, sort_keys=True))


@app.command()
def run(
    role: str = typer.Option("generalist", help="Role/archetype profile."),
    live: bool = typer.Option(False, help="Request live input when a gated backend exists."),
) -> None:
    """Start the independent supervisor in SAFE_IDLE.

    Phase 0 intentionally exposes only the fake motor backend. `--live` fails
    closed until a scoped Minecraft backend has passed the safety gates.
    """
    _ensure_dirs()
    if live:
        raise typer.BadParameter(
            "Live input is not enabled yet. The independent supervisor and "
            "scoped-input backend must pass docs/SAFETY.md first."
        )
    if supervisor_alive():
        current = send_command("status")
        print("[yellow]Supervisor is already running.[/yellow]")
        print(json.dumps(current, indent=2, sort_keys=True))
        return

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
            current = send_command("status")
            print(f"[green]Supervisor started[/green] role={role} state={current['state']}")
            print("Live motor control remains disabled by design.")
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
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def pause() -> None:
    """Pause the agent and revoke motor capability immediately."""
    payload = _command("pause")
    print(f"[yellow]{payload['state']}[/yellow] — motor capability revoked.")


@app.command()
def resume() -> None:
    """Return a paused supervisor to SAFE_IDLE; this never arms live input."""
    payload = _command("resume")
    print(f"[green]{payload['state']}[/green] — live input remains unarmed.")


@app.command()
def stop() -> None:
    """Stop through the control socket, with a process-level fallback.

    The fallback exists specifically so a broken cognition/control client cannot
    make the operator dependent on that same client to stop the supervisor.
    """
    if not CONTROL_FILE.exists():
        print("[bold red]STOPPED[/bold red] — no supervisor endpoint exists.")
        return

    endpoint: ControlEndpoint | None = None
    try:
        endpoint = ControlEndpoint.load()
        send_command("stop")
    except Exception as exc:
        print(f"[yellow]Control socket stop failed ({exc}); using process fallback.[/yellow]")
        if endpoint is not None:
            try:
                os.kill(endpoint.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not supervisor_alive():
            print("[bold red]STOPPED[/bold red] — supervisor is no longer reachable.")
            return
        time.sleep(0.05)
    raise typer.BadParameter("supervisor did not stop; use the independent OS/process stop path")


@app.command()
def logs(lines: int = typer.Option(80, min=1, max=2000)) -> None:
    """Show the tail of the supervisor log."""
    if not LOG_FILE.exists():
        print("No supervisor log yet.")
        return
    content = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


@app.command(hidden=True)
def arm_fake(target_instance: str = "fake-instance") -> None:
    """Phase-0 test hook. Never controls Minecraft or the desktop."""
    result = _command("arm-fake", target_instance=target_instance)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def wiki(query: str) -> None:
    """Placeholder for exact-version sourced game Q&A."""
    print(f"Wiki service not implemented yet. Query preserved: {query!r}")


@knowledge_app.command("sync")
def knowledge_sync() -> None:
    """Placeholder for exact-version game knowledge compilation."""
    print("Knowledge compiler not implemented yet; see docs/ROADMAP.md Phase 2.")


if __name__ == "__main__":
    app()
