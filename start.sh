#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$SCRIPT_DIR/.venv/bin/minecraft-ai"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
CAPTURE_SOURCE="${MINECRAFT_AI_CAPTURE_SOURCE:-x11}"
CHECK_INTERVAL_S="${MINECRAFT_AI_CHECK_INTERVAL_S:-10}"
FAILURES_BEFORE_RECOVERY="${MINECRAFT_AI_FAILURES_BEFORE_RECOVERY:-3}"
NONPLAYABLE_FAILURES_BEFORE_RECOVERY="${MINECRAFT_AI_NONPLAYABLE_FAILURES_BEFORE_RECOVERY:-12}"
AGENT_TRANSITION_GRACE_S="${MINECRAFT_AI_AGENT_TRANSITION_GRACE_S:-60}"
START_FAILURES_BEFORE_BEDROCK_RELAUNCH="${MINECRAFT_AI_START_FAILURES_BEFORE_BEDROCK_RELAUNCH:-20}"
START_FAILURE_GRACE_S="${MINECRAFT_AI_START_FAILURE_GRACE_S:-420}"
bedrock_start_failures=0
bedrock_failure_started_s=0
input_route_hold_generation=""
input_route_recovery_adopted=false

if [ ! -x "$CLI" ] || [ ! -x "$PYTHON" ]; then
    echo "Missing repo environment. Run: python3 -m venv .venv && .venv/bin/pip install -e '.[full,dev]'" >&2
    exit 2
fi

# Older installed units supplied the role as argv[1]. Consume only a valid
# role token so option-first manual invocations remain intact, while persisted
# config stays authoritative for every launch/recovery generation.
if [ "$#" -gt 0 ] && [[ "$1" != -* ]] \
    && "$PYTHON" -c '
import sys
from minecraft_ai.roles import get_role
get_role(sys.argv[1])
' "$1" >/dev/null 2>&1
then
    shift
fi
RUN_ARGS=("$@")

stop_runtime() {
    "$CLI" stop --transient >/dev/null 2>&1 || true
}

stop_runtime_required() {
    if "$CLI" stop --transient >/dev/null; then
        return 0
    fi
    echo "Existing supervisor/agent containment could not be confirmed; startup is blocked." >&2
    return 1
}

stop_bedrock() {
    "$CLI" bedrock stop --transient >/dev/null 2>&1 || true
}

operator_paused() {
    "$CLI" status 2>/dev/null | grep -q '"operator_pause_latched": true'
}

emergency_latched() {
    "$CLI" status 2>/dev/null | grep -q '"emergency_stop_latched": true'
}

configured_runtime_role() {
    "$PYTHON" -c '
from minecraft_ai.config import load_config
from minecraft_ai.roles import get_role

role = load_config().role
get_role(role)
print(role)
'
}

runtime_role() {
    configured_runtime_role
}

healthy_agent_generation() {
    local payload
    payload="$("$CLI" status 2>/dev/null)" || return 1
    printf '%s' "$payload" | "$PYTHON" -c '
import json
import sys

payload = json.load(sys.stdin)
agent = payload.get("agent") if isinstance(payload, dict) else None
healthy = (
    isinstance(agent, dict)
    and agent.get("alive") is True
    and payload.get("state") == "RUNNING"
    and payload.get("live_capable") is True
    and payload.get("motor_lease_active") is True
)
pid = agent.get("pid") if isinstance(agent, dict) else None
if not healthy or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
    raise SystemExit(1)
print(pid)
'
}

reset_start_failures() {
    bedrock_start_failures=0
    bedrock_failure_started_s=0
}

input_route_held() {
    local payload observation
    input_route_recovery_adopted=false
    payload="$("$CLI" status 2>/dev/null)" || payload=""
    observation="$(printf '%s' "$payload" | "$PYTHON" -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except (ValueError, TypeError):
    raise SystemExit(1)
if not isinstance(payload, dict):
    raise SystemExit(1)
generation = payload.get("session_id")
if payload.get("fault_code") == "input-route-unverified":
    print("hold:" + (generation if isinstance(generation, str) and generation else "unknown"))
elif (
    isinstance(generation, str) and generation
    and payload.get("fault_code", "unknown") is None
    and payload.get("supervisor_reachable") is not False
    and payload.get("state") == "RUNNING"
    and payload.get("live_capable") is True
    and payload.get("motor_lease_active") is True
    and payload.get("emergency_stop_latched") is False
    and payload.get("operator_pause_latched") is False
    and isinstance(payload.get("agent"), dict)
    and payload["agent"].get("alive") is True
):
    print("healthy:" + generation)
' 2>/dev/null)" || observation=""
    if [[ "$observation" == hold:* ]]; then
        if [ -z "$input_route_hold_generation" ]; then
            echo "Input routing is unverified; holding automatic recovery and preserving Bedrock." \
                "A deliberately recovered healthy supervisor generation is required." >&2
        fi
        input_route_hold_generation="${observation#hold:}"
    fi
    if [ -z "$input_route_hold_generation" ]; then
        return 1
    fi
    if [[ "$observation" == healthy:* ]] \
        && [ "${observation#healthy:}" != "$input_route_hold_generation" ] \
        && readiness_ok
    then
        echo "A healthy replacement supervisor is ready; adopting the recovered agent."
        input_route_hold_generation=""
        input_route_recovery_adopted=true
        reset_start_failures
        return 1
    fi
    # No stop, navigation, calibration, pointer warp, or retry is allowed here.
    # Remain alive: exiting also invokes the service unit's Bedrock ExecStop.
    return 0
}

abort_failed_start() {
    local result="$1"
    local force_relaunch="${2:-false}"
    local now_s elapsed_s
    if [ "$result" -eq 64 ]; then
        stop_runtime
        return "$result"
    fi
    if emergency_latched; then
        stop_runtime
        return 64
    fi
    if operator_paused; then
        stop_runtime
        return 65
    fi
    if input_route_held; then
        return 69
    fi
    if [ "$input_route_recovery_adopted" = true ]; then
        return 0
    fi
    stop_runtime
    now_s="$(date +%s)"
    if [ "$bedrock_failure_started_s" -eq 0 ]; then
        bedrock_failure_started_s="$now_s"
    fi
    bedrock_start_failures=$((bedrock_start_failures + 1))
    elapsed_s=$((now_s - bedrock_failure_started_s))
    # A cold Bedrock client can render unreadable transitional frames for
    # several minutes. Preserve that warm-up window, but replace a session
    # that stays unhealthy beyond a bounded number of retries or grace time.
    if [ "$force_relaunch" = true ] \
        || [ "$bedrock_start_failures" -ge "$START_FAILURES_BEFORE_BEDROCK_RELAUNCH" ] \
        || { [ "$bedrock_start_failures" -ge 3 ] \
            && [ "$elapsed_s" -ge "$START_FAILURE_GRACE_S" ]; }
    then
        echo "Replacing unhealthy Bedrock session after $bedrock_start_failures failed startup attempts." >&2
        stop_bedrock
        reset_start_failures
    else
        echo "Preserving the warming Bedrock session ($bedrock_start_failures failed startup attempts)." >&2
    fi
    return "$result"
}

cleanup() {
    local result=$?
    trap - EXIT INT TERM
    stop_runtime
    stop_bedrock
    exit "$result"
}

terminate() {
    trap - EXIT INT TERM
    stop_runtime
    stop_bedrock
    exit 0
}
trap cleanup EXIT
trap terminate INT TERM

readiness_ok() {
    curl --fail --silent --show-error --max-time 5 \
        http://127.0.0.1:8765/readyz >/dev/null \
        && curl --fail --silent --show-error --max-time 5 \
            http://127.0.0.1:8081/v1/models >/dev/null
}

runtime_health_ok() {
    curl --fail --silent --show-error --max-time 5 \
        http://127.0.0.1:8765/livez >/dev/null \
        && curl --fail --silent --show-error --max-time 5 \
            http://127.0.0.1:8081/v1/models >/dev/null
}

start_support_services() {
    local service
    for service in \
        minecraft-ai-dashboard-live.service \
        minecraft-ai-vlm-gemma4-vulkan-all.service
    do
        if ! systemctl --user cat "$service" >/dev/null 2>&1; then
            echo "Required user service is missing: $service" >&2
            return 1
        fi
        systemctl --user start "$service" || return $?
    done
}

wait_for_bedrock_session() {
    local attempt
    for attempt in $(seq 1 45); do
        if "$CLI" bedrock status 2>/dev/null | grep -q '"alive": true'; then
            return 0
        fi
        sleep 1
    done
    echo "Managed Bedrock process did not become alive within 45 seconds." >&2
    return 1
}

bedrock_session_alive() {
    "$CLI" bedrock status 2>/dev/null | grep -q '"alive": true'
}

wait_for_model() {
    local attempt
    for attempt in $(seq 1 60); do
        if curl --fail --silent --max-time 5 \
            http://127.0.0.1:8081/v1/models >/dev/null 2>&1
        then
            return 0
        fi
        sleep 2
    done
    echo "Local planner/VLM service did not become ready within 120 seconds." >&2
    return 1
}

start_live_runtime() {
    local result selected_role
    if emergency_latched; then
        echo "Emergency stop is latched; persistent startup is suspended." >&2
        return 64
    fi
    if operator_paused; then
        echo "Explicit operator pause is active; persistent startup remains suspended." >&2
        return 65
    fi
    if input_route_held; then
        return 69
    fi
    if [ "$input_route_recovery_adopted" = true ]; then
        return 0
    fi

    selected_role="$(runtime_role)"
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "Configured agent role is unreadable or invalid; containing runtime." >&2
        if ! stop_runtime_required; then
            return 66
        fi
        return 68
    fi
    # A live launcher intentionally owns the GPU marker, so Doctor reports it
    # as busy. Run the host preflight only when a fresh GPU launch is needed.
    if ! bedrock_session_alive && ! bedrock-on-linux doctor; then
        # A supervised stop can be interrupted after Wine dies but before the
        # launch marker is removed. BOL's dedicated recovery action only
        # clears an exact dead same-boot marker under its global launch lock;
        # it cannot acknowledge previous-boot driver incidents.
        if ! bedrock-on-linux doctor --recover-interrupted-launch; then
            echo "BedrockOnLinux preflight is blocked; automatic launch retries are suspended." >&2
            return 67
        fi
    fi

    if ! stop_runtime_required; then
        return 66
    fi
    if emergency_latched; then
        echo "Emergency stop was latched during cleanup; startup is suspended." >&2
        return 64
    fi
    if operator_paused; then
        echo "Operator pause was requested during cleanup; startup is suspended." >&2
        return 65
    fi
    "$CLI" bedrock launch
    result=$?
    if [ "$result" -ne 0 ]; then
        abort_failed_start "$result" true
        return $?
    fi
    if operator_paused; then
        return 65
    fi
    wait_for_bedrock_session
    result=$?
    if [ "$result" -ne 0 ]; then
        abort_failed_start "$result" true
        return $?
    fi
    if operator_paused; then
        return 65
    fi
    "$CLI" bedrock navigate --timeout-s 180 --retries 3
    result=$?
    if [ "$result" -ne 0 ]; then
        abort_failed_start "$result"
        return $?
    fi
    if operator_paused; then
        return 65
    fi
    "$CLI" run \
        --role "$selected_role" \
        --live \
        --capture-source "$CAPTURE_SOURCE" \
        ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
    result=$?
    if [ "$result" -ne 0 ]; then
        abort_failed_start "$result"
        return $?
    fi

    local attempt
    for attempt in $(seq 1 45); do
        if readiness_ok; then
            reset_start_failures
            echo "Minecraft AI is ready: isolated Bedrock, supervisor, and agent are healthy" \
                "(role=$selected_role)."
            return 0
        fi
        if emergency_latched; then
            abort_failed_start 64
            return $?
        fi
        if operator_paused; then
            stop_runtime
            return 65
        fi
        sleep 2
    done
    echo "Agent did not become ready within 90 seconds." >&2
    abort_failed_start 1
}

echo "=================================================="
echo " Minecraft AI persistent Bedrock runtime"
echo " Role: configured | capture: $CAPTURE_SOURCE"
echo "=================================================="

# A retired route fault also owns startup: fallible model/service bootstrap
# must not exit through cleanup and destroy an otherwise healthy game.
while true; do
    if emergency_latched; then
        trap - EXIT INT TERM
        exit 64
    fi
    if operator_paused; then
        sleep "$CHECK_INTERVAL_S"
        continue
    fi
    if ! input_route_held; then
        break
    fi
    sleep "$CHECK_INTERVAL_S"
done

if [ "$input_route_recovery_adopted" != true ]; then
    if ! "$CLI" install; then
        trap - EXIT INT TERM
        exit 2
    fi
    start_support_services || exit $?
    wait_for_model || exit $?
fi

while [ "$input_route_recovery_adopted" != true ]; do
    start_live_runtime
    result=$?
    if [ "$result" -eq 0 ]; then
        break
    fi
    if [ "$result" -eq 64 ]; then
        trap - EXIT INT TERM
        exit 64
    fi
    if [ "$result" -eq 67 ]; then
        exit 67
    fi
    if [ "$result" -eq 65 ] || [ "$result" -eq 69 ]; then
        sleep "$CHECK_INTERVAL_S"
        continue
    fi
    echo "Startup failed; retrying safely in 15 seconds." >&2
    sleep 15
done

failures=0
nonplayable_failures=0
healthy_generation="$(healthy_agent_generation 2>/dev/null || true)"
transition_generation=""
transition_started_s=0
while true; do
    sleep "$CHECK_INTERVAL_S"
    if runtime_health_ok; then
        failures=0
        observed_generation="$(healthy_agent_generation 2>/dev/null || true)"
        if [ -n "$observed_generation" ]; then
            healthy_generation="$observed_generation"
        fi
        transition_generation=""
        transition_started_s=0
        if readiness_ok; then
            nonplayable_failures=0
            continue
        fi
        if operator_paused; then
            nonplayable_failures=0
            continue
        fi
        nonplayable_failures=$((nonplayable_failures + 1))
        echo "In-world HUD check is pending ($nonplayable_failures/$NONPLAYABLE_FAILURES_BEFORE_RECOVERY)." >&2
        if [ "$nonplayable_failures" -lt "$NONPLAYABLE_FAILURES_BEFORE_RECOVERY" ]; then
            continue
        fi
    else
        nonplayable_failures=0
    fi
    if emergency_latched; then
        echo "Emergency stop was latched; persistent startup is suspended." >&2
        trap - EXIT INT TERM
        stop_runtime
        exit 64
    fi
    if operator_paused; then
        failures=0
        continue
    fi
    if input_route_held; then
        failures=0
        nonplayable_failures=0
        continue
    fi
    if [ "$input_route_recovery_adopted" = true ]; then
        failures=0
        nonplayable_failures=0
        healthy_generation="$(healthy_agent_generation 2>/dev/null || true)"
        continue
    fi
    if [ "$nonplayable_failures" -eq 0 ]; then
        failures=$((failures + 1))
        observed_generation="$(healthy_agent_generation 2>/dev/null || true)"
        if [ -n "$observed_generation" ] \
            && [ "$observed_generation" != "$healthy_generation" ]
        then
            now_s="$(date +%s)"
            if [ "$transition_started_s" -eq 0 ]; then
                transition_started_s="$now_s"
            fi
            transition_generation="$observed_generation"
            transition_elapsed_s=$((now_s - transition_started_s))
            if [ "$transition_elapsed_s" -lt "$AGENT_TRANSITION_GRACE_S" ]; then
                echo "New agent generation $transition_generation is warming" \
                    "($transition_elapsed_s/$AGENT_TRANSITION_GRACE_S seconds); preserving it." >&2
                continue
            fi
            echo "Agent generation $transition_generation exceeded its warm-up grace." >&2
        else
            transition_generation=""
            transition_started_s=0
        fi
        echo "Runtime health check failed ($failures/$FAILURES_BEFORE_RECOVERY)." >&2
        if [ "$failures" -lt "$FAILURES_BEFORE_RECOVERY" ]; then
            continue
        fi
    fi

    echo "Recovering the isolated Bedrock session and agent."
    failures=0
    nonplayable_failures=0
    while true; do
        start_live_runtime
        result=$?
        if [ "$result" -eq 0 ]; then
            healthy_generation="$(healthy_agent_generation 2>/dev/null || true)"
            transition_generation=""
            transition_started_s=0
            break
        fi
        if [ "$result" -eq 64 ]; then
            trap - EXIT INT TERM
            exit 64
        fi
        if [ "$result" -eq 67 ]; then
            exit 67
        fi
        if [ "$result" -eq 65 ] || [ "$result" -eq 69 ]; then
            sleep "$CHECK_INTERVAL_S"
            continue
        fi
        echo "Recovery failed; retrying safely in 15 seconds." >&2
        sleep 15
    done
done
