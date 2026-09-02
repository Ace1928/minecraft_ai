from __future__ import annotations

import json
import platform
from pathlib import Path

import typer
from platformdirs import user_data_dir, user_runtime_dir
from rich import print

app = typer.Typer(help="Minecraft AI lifecycle and tooling CLI.")
knowledge_app = typer.Typer(help="Versioned game knowledge commands.")
app.add_typer(knowledge_app, name="knowledge")

APP_NAME = "minecraft-ai"
DATA_DIR = Path(user_data_dir(APP_NAME))
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
STATE_FILE = RUNTIME_DIR / "supervisor-state.json"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _read_state() -> dict[str, object]:
    if not STATE_FILE.exists():
        return {"state": "STOPPED", "live_input": False}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "UNKNOWN", "live_input": False}


def _write_state(state: str, **extra: object) -> None:
    _ensure_dirs()
    payload: dict[str, object] = {"state": state, "live_input": False, **extra}
    STATE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@app.command()
def install() -> None:
    """Prepare local directories and report the platform profile.

    Model/runtime downloads and input bridge installation are deliberately not
    implemented until their license and safety gates land.
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
        "scoped_input": "not-installed",
        "live_input": False,
    }
    print_json = json.dumps(profile, indent=2)
    print(print_json)


@app.command()
def run(
    role: str = typer.Option("generalist", help="Role/archetype profile."),
    live: bool = typer.Option(False, help="Request live input when a gated backend exists."),
) -> None:
    """Start the safe supervisor scaffold.

    Current bootstrap intentionally refuses live motor control.
    """
    if live:
        raise typer.BadParameter(
            "Live input is not enabled in the bootstrap release. Implement and pass "
            "the independent supervisor + scoped-input safety gates first."
        )
    _write_state("SAFE_IDLE", role=role)
    print(f"[green]Supervisor scaffold started[/green] role={role} state=SAFE_IDLE")
    print("Live motor control remains disabled by design.")


@app.command()
def status() -> None:
    """Show supervisor state."""
    print(json.dumps(_read_state(), indent=2, sort_keys=True))


@app.command()
def pause() -> None:
    """Pause the agent and revoke motor capability."""
    state = _read_state()
    _write_state("PAUSED", previous_state=state.get("state"))
    print("[yellow]PAUSED[/yellow] — motor capability revoked.")


@app.command()
def resume() -> None:
    """Return to safe idle; arming live control is a separate gated transition."""
    _write_state("SAFE_IDLE")
    print("[green]SAFE_IDLE[/green] — live input remains disabled.")


@app.command()
def stop() -> None:
    """Stop the supervisor scaffold and revoke all motor capability."""
    _write_state("STOPPED")
    print("[bold red]STOPPED[/bold red] — motor capability revoked.")


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
