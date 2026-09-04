from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="persistent launcher is a POSIX bash service",
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _launcher_harness(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "runtime"
    bin_dir = root / "bin"
    venv_bin = root / ".venv" / "bin"
    state = root / "state"
    config_home = root / "config-home"
    bin_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    state.mkdir()
    (config_home / "minecraft-ai").mkdir(parents=True)
    shutil.copy2(Path(__file__).parents[1] / "start.sh", root / "start.sh")
    _write_executable(
        venv_bin / "python",
        f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} \"$@\"\n",
    )

    _write_executable(
        venv_bin / "minecraft-ai",
        """#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >> "$TEST_STATE/cli-calls"
case "${1:-}" in
    install|stop)
        exit 0
        ;;
    config)
        test "${2:-}" = show
        role="$(sed -n 's/^role: *//p' "$TEST_CONFIG")"
        printf '{"role":"%s"}\n' "$role"
        ;;
    status)
        generation="$(cat "$TEST_STATE/generation")"
        alive=true
        if [ -e "$TEST_STATE/agent-dead" ]; then
            alive=false
        fi
        printf '{"state":"RUNNING",'
        printf '"live_capable":true,"motor_lease_active":true,'
        printf '"emergency_stop_latched":false,"operator_pause_latched":false,'
        printf '"agent":{"alive":%s,"pid":%s}}\n' "$alive" "$generation"
        ;;
    bedrock)
        if [ "${2:-}" = status ]; then
            printf '{"alive": true}\n'
        fi
        exit 0
        ;;
    run)
        role=""
        previous=""
        for argument in "$@"; do
            if [ "$previous" = --role ]; then
                role="$argument"
                break
            fi
            previous="$argument"
        done
        printf '%s\n' "$role" >> "$TEST_STATE/run-roles"
        runs="$(wc -l < "$TEST_STATE/run-roles")"
        if [ "${CHANGE_ROLE_AFTER_FIRST:-0}" = 1 ] && [ "$runs" -eq 1 ]; then
            printf 'role: generalist\n' > "$TEST_CONFIG"
            : > "$TEST_STATE/agent-dead"
        fi
        if [ -n "${STOP_AFTER_RUNS:-}" ] && [ "$runs" -ge "$STOP_AFTER_RUNS" ]; then
            kill -TERM "$PPID"
        fi
        exit 0
        ;;
esac
exit 0
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -u
url="${!#}"
case "$url" in
    */livez)
        count=0
        if [ -e "$TEST_STATE/livez-count" ]; then
            count="$(cat "$TEST_STATE/livez-count")"
        fi
        count=$((count + 1))
        printf '%s\n' "$count" > "$TEST_STATE/livez-count"
        if [ "$count" -le "${LIVEZ_FAILURES:-0}" ]; then
            exit 22
        fi
        if [ "${STOP_AFTER_LIVEZ_SUCCESS:-0}" = 1 ]; then
            kill -TERM "$PPID"
        fi
        exit 0
        ;;
    *)
        exit 0
        ;;
esac
""",
    )
    _write_executable(
        bin_dir / "sleep",
        """#!/usr/bin/env bash
set -u
count=0
if [ -e "$TEST_STATE/sleep-count" ]; then
    count="$(cat "$TEST_STATE/sleep-count")"
fi
count=$((count + 1))
printf '%s\n' "$count" > "$TEST_STATE/sleep-count"
if [ "${TRANSITION_ON_FIRST_SLEEP:-0}" = 1 ] && [ "$count" -eq 1 ]; then
    printf '222\n' > "$TEST_STATE/generation"
fi
exit 0
""",
    )
    for command in ("systemctl", "bedrock-on-linux"):
        _write_executable(bin_dir / command, "#!/usr/bin/env bash\nexit 0\n")

    config = config_home / "minecraft-ai" / "config.yaml"
    config.write_text("role: generalist\n", encoding="utf-8")
    (state / "generation").write_text("111\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_STATE": str(state),
        "TEST_CONFIG": str(config),
        "XDG_CONFIG_HOME": str(config_home),
        "MINECRAFT_AI_CHECK_INTERVAL_S": "0",
        "MINECRAFT_AI_FAILURES_BEFORE_RECOVERY": "3",
        "MINECRAFT_AI_AGENT_TRANSITION_GRACE_S": "60",
    }
    return root, env, state


def _run_launcher(
    root: Path,
    env: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "start.sh"), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_recovery_reloads_current_configured_role(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    Path(env["TEST_CONFIG"]).write_text("role: creative_builder\n", encoding="utf-8")
    env.update(
        {
            "CHANGE_ROLE_AFTER_FIRST": "1",
            "LIVEZ_FAILURES": "3",
            "STOP_AFTER_RUNS": "2",
        }
    )

    result = _run_launcher(root, env, "creative_builder")

    assert result.returncode == 0, result.stderr
    assert (state / "run-roles").read_text(encoding="utf-8").splitlines() == [
        "creative_builder",
        "generalist",
    ]
    cli_calls = (state / "cli-calls").read_text(encoding="utf-8").splitlines()
    run_calls = [call for call in cli_calls if call.startswith("run ")]
    assert all(not call.endswith(" creative_builder") for call in run_calls)
    assert all(not call.startswith("config show") for call in cli_calls)
    assert "Recovering the isolated Bedrock session and agent." in result.stdout


def test_new_healthy_agent_generation_gets_bounded_warmup_grace(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    env.update(
        {
            "TRANSITION_ON_FIRST_SLEEP": "1",
            "LIVEZ_FAILURES": "3",
            "STOP_AFTER_LIVEZ_SUCCESS": "1",
        }
    )

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    assert (state / "run-roles").read_text(encoding="utf-8").splitlines() == ["generalist"]
    assert "New agent generation 222 is warming" in result.stderr
    assert "Recovering the isolated Bedrock session and agent." not in result.stdout


def test_new_agent_generation_is_recovered_after_grace_expires(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    env.update(
        {
            "TRANSITION_ON_FIRST_SLEEP": "1",
            "LIVEZ_FAILURES": "3",
            "STOP_AFTER_RUNS": "2",
            "MINECRAFT_AI_AGENT_TRANSITION_GRACE_S": "0",
        }
    )

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    assert (state / "run-roles").read_text(encoding="utf-8").splitlines() == [
        "generalist",
        "generalist",
    ]
    assert "Agent generation 222 exceeded its warm-up grace." in result.stderr
    assert "Recovering the isolated Bedrock session and agent." in result.stdout


def test_same_generation_failure_recovers_at_three_strikes(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    env.update(
        {
            "LIVEZ_FAILURES": "3",
            "STOP_AFTER_RUNS": "2",
        }
    )

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    assert (state / "livez-count").read_text(encoding="utf-8").strip() == "3"
    assert (state / "run-roles").read_text(encoding="utf-8").splitlines() == [
        "generalist",
        "generalist",
    ]
    assert "New agent generation" not in result.stderr
    assert "Runtime health check failed (3/3)." in result.stderr
    assert "Recovering the isolated Bedrock session and agent." in result.stdout


def test_option_first_invocation_is_forwarded_without_being_consumed(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    env["STOP_AFTER_RUNS"] = "1"

    result = _run_launcher(root, env, "--edition", "bedrock")

    assert result.returncode == 0, result.stderr
    run_call = next(
        call
        for call in (state / "cli-calls").read_text(encoding="utf-8").splitlines()
        if call.startswith("run ")
    )
    assert run_call.endswith("--edition bedrock")


def test_persistent_service_uses_dynamic_configured_role() -> None:
    unit = (Path(__file__).parents[1] / "systemd" / "minecraft-ai-agent-live.service").read_text(
        encoding="utf-8"
    )

    assert "ExecStart=%h/minecraft_ai/start.sh\n" in unit
