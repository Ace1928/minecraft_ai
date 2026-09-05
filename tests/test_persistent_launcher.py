from __future__ import annotations

import os
import json
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
        if [ -e "$TEST_STATE/status-override" ]; then
            cat "$TEST_STATE/status-override"
            exit 0
        fi
        generation="$(cat "$TEST_STATE/generation")"
        alive=true
        if [ -e "$TEST_STATE/agent-dead" ]; then
            alive=false
        fi
        printf '{"state":"RUNNING","session_id":"%s","fault_code":null,' "$generation"
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
        if [ -n "${RUN_FAILURE_KIND:-}" ]; then
            if [ "$RUN_FAILURE_KIND" = route ]; then
                printf '{"state":"FAILSAFE","session_id":"111",' > "$TEST_STATE/status-override"
                printf '"fault_code":"input-route-unverified",' >> "$TEST_STATE/status-override"
                printf '"emergency_stop_latched":false,"operator_pause_latched":false}\n' \
                    >> "$TEST_STATE/status-override"
            fi
            exit 2
        fi
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
seconds=0
if [ -e "$TEST_STATE/sleep-seconds" ]; then
    seconds="$(cat "$TEST_STATE/sleep-seconds")"
fi
printf '%s\n' "$((seconds + ${1:-0} * ${TEST_CLOCK_MULTIPLIER:-1}))" \
    > "$TEST_STATE/sleep-seconds"
if [ "$count" -eq "${REPLACE_STATUS_ON_SLEEP:-0}" ]; then
    cp "$TEST_STATE/replacement-status" "$TEST_STATE/status-override"
fi
if [ "$count" -eq "${STOP_AFTER_SLEEPS:-0}" ]; then
    cp "$TEST_STATE/cli-calls" "$TEST_STATE/calls-before-stop"
    kill -TERM "$PPID"
fi
if [ "${TRANSITION_ON_FIRST_SLEEP:-0}" = 1 ] && [ "$count" -eq 1 ]; then
    printf '222\n' > "$TEST_STATE/generation"
fi
exit 0
""",
    )
    for command in ("systemctl", "bedrock-on-linux"):
        _write_executable(
            bin_dir / command,
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> \"$TEST_STATE/service-calls\"\n"
            'if [ "${TEST_SERVICES_FAIL:-0}" = 1 ]; then exit 1; fi\n'
            "exit 0\n",
        )
    _write_executable(
        bin_dir / "date",
        """#!/usr/bin/env bash
seconds=0
if [ -e "$TEST_STATE/sleep-seconds" ]; then
    seconds="$(cat "$TEST_STATE/sleep-seconds")"
fi
printf '%s\n' "$((1000 + seconds))"
""",
    )

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
        timeout=20,
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


def _fault_status(**changes: object) -> str:
    return json.dumps(
        {
            "state": "FAILSAFE",
            "session_id": "111",
            "fault_code": "input-route-unverified",
            "emergency_stop_latched": False,
            "operator_pause_latched": False,
            **changes,
        }
    )


@pytest.mark.parametrize("retired", [False, True])
@pytest.mark.parametrize("host_isolation_failed", [False, True])
def test_existing_route_fault_holds_without_starting_or_stopping_game(
    tmp_path: Path, retired: bool, host_isolation_failed: bool
) -> None:
    root, env, state = _launcher_harness(tmp_path)
    (state / "status-override").write_text(
        _fault_status(
            state="STOPPED" if retired else "FAILSAFE", supervisor_reachable=not retired,
            fault_code=None if host_isolation_failed else "input-route-unverified",
            input_isolation={"verified": not host_isolation_failed},
        ),
        encoding="utf-8",
    )
    env.update(
        {
            "STOP_AFTER_SLEEPS": "50", "MINECRAFT_AI_CHECK_INTERVAL_S": "10",
            "TEST_SERVICES_FAIL": "1",
        }
    )

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    calls = (state / "calls-before-stop").read_text(encoding="utf-8").splitlines()
    assert set(calls) == {"status"}
    assert not (state / "service-calls").exists()
    assert int((state / "sleep-seconds").read_text()) > 420
    assert result.stderr.count("holding automatic recovery") == 1


@pytest.mark.parametrize("replacement", ["unknown", "same-generation", "unready-new-generation"])
def test_startup_route_fault_does_not_retry_after_status_changes(
    tmp_path: Path, replacement: str
) -> None:
    root, env, state = _launcher_harness(tmp_path)
    env.update(
        {
            "RUN_FAILURE_KIND": "route",
            "STOP_AFTER_SLEEPS": "50",
            "REPLACE_STATUS_ON_SLEEP": "2",
            "MINECRAFT_AI_CHECK_INTERVAL_S": "10",
        }
    )
    updated = "not-json" if replacement == "unknown" else _fault_status(
        fault_code=None,
        state="RUNNING" if replacement == "same-generation" else "SAFE_IDLE",
        session_id="111" if replacement == "same-generation" else "222",
        live_capable=True,
        motor_lease_active=True,
        agent={"alive": True, "pid": 222},
    )
    (state / "replacement-status").write_text(updated, encoding="utf-8")

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    calls = (state / "calls-before-stop").read_text(encoding="utf-8").splitlines()
    assert calls.count("stop --transient") == 1  # Normal initial attach, before the fault.
    assert "bedrock stop --transient" not in calls
    assert sum(call.startswith("run ") for call in calls) == 1
    assert sum(call.startswith("bedrock navigate") for call in calls) == 1
    assert int((state / "sleep-seconds").read_text()) > 420
    assert "failed startup attempts" not in result.stderr


def test_route_hold_adopts_only_ready_healthy_new_generation(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    (state / "status-override").write_text(_fault_status(), encoding="utf-8")
    (state / "replacement-status").write_text(
        _fault_status(
            state="RUNNING", session_id="222", fault_code=None,
            live_capable=True, motor_lease_active=True, agent={"alive": True, "pid": 222},
        ),
        encoding="utf-8",
    )
    env.update({"REPLACE_STATUS_ON_SLEEP": "2", "STOP_AFTER_SLEEPS": "5"})

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    calls = (state / "calls-before-stop").read_text(encoding="utf-8").splitlines()
    assert set(calls) == {"status"}
    assert "adopting the recovered agent" in result.stdout


@pytest.mark.parametrize("grace_limit", [False, True])
def test_unrelated_startup_failure_still_replaces_bedrock_at_existing_limit(
    tmp_path: Path, grace_limit: bool
) -> None:
    root, env, state = _launcher_harness(tmp_path)
    attempts = 5 if grace_limit else 22
    env.update({"RUN_FAILURE_KIND": "other", "STOP_AFTER_SLEEPS": str(attempts)})
    if grace_limit:
        env["MINECRAFT_AI_START_FAILURES_BEFORE_BEDROCK_RELAUNCH"] = "1000"
        env["TEST_CLOCK_MULTIPLIER"] = "14"  # Two retries reach the unchanged 420s grace.

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    calls = (state / "calls-before-stop").read_text(encoding="utf-8").splitlines()
    assert calls.count("bedrock stop --transient") == 1
    assert sum(call.startswith("run ") for call in calls) == attempts
    limit = 3 if grace_limit else 20
    assert (
        f"Replacing unhealthy Bedrock session after {limit} failed startup attempts"
        in result.stderr
    )


def test_live_agent_route_failure_holds_before_recovery_destroys_supervisor(tmp_path: Path) -> None:
    root, env, state = _launcher_harness(tmp_path)
    (state / "replacement-status").write_text(_fault_status(), encoding="utf-8")
    env.update(
        {
            "REPLACE_STATUS_ON_SLEEP": "1", "STOP_AFTER_SLEEPS": "50",
            "LIVEZ_FAILURES": "1000", "MINECRAFT_AI_CHECK_INTERVAL_S": "10",
        }
    )

    result = _run_launcher(root, env)

    assert result.returncode == 0, result.stderr
    calls = (state / "calls-before-stop").read_text(encoding="utf-8").splitlines()
    assert calls.count("stop --transient") == 1  # Only the normal initial startup.
    assert "bedrock stop --transient" not in calls
    assert sum(call.startswith("run ") for call in calls) == 1
    assert "Recovering the isolated Bedrock session and agent" not in result.stdout


@pytest.mark.parametrize("interlock", ["operator_pause_latched", "emergency_stop_latched"])
def test_route_hold_never_overrides_operator_interlocks(tmp_path: Path, interlock: str) -> None:
    root, env, state = _launcher_harness(tmp_path)
    (state / "status-override").write_text(_fault_status(**{interlock: True}), encoding="utf-8")
    env["STOP_AFTER_SLEEPS"] = "3"

    result = _run_launcher(root, env)

    assert result.returncode == (64 if interlock == "emergency_stop_latched" else 0), result.stderr
    calls_path = state / (
        "cli-calls" if interlock == "emergency_stop_latched" else "calls-before-stop"
    )
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert not any(call.startswith(("run ", "bedrock ")) for call in calls)
    assert "holding automatic recovery" not in result.stderr
