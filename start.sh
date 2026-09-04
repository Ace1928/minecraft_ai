#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$SCRIPT_DIR/.venv/bin/minecraft-ai"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
ROLE="${1:-generalist}"
CAPTURE_SOURCE="${MINECRAFT_AI_CAPTURE_SOURCE:-x11}"
CHECK_INTERVAL_S="${MINECRAFT_AI_CHECK_INTERVAL_S:-10}"
FAILURES_BEFORE_RECOVERY="${MINECRAFT_AI_FAILURES_BEFORE_RECOVERY:-3}"
NONPLAYABLE_FAILURES_BEFORE_RECOVERY="${MINECRAFT_AI_NONPLAYABLE_FAILURES_BEFORE_RECOVERY:-12}"
START_FAILURES_BEFORE_BEDROCK_RELAUNCH="${MINECRAFT_AI_START_FAILURES_BEFORE_BEDROCK_RELAUNCH:-20}"
START_FAILURE_GRACE_S="${MINECRAFT_AI_START_FAILURE_GRACE_S:-420}"
bedrock_start_failures=0
bedrock_failure_started_s=0

if [ "$#" -gt 0 ]; then
    shift
fi
RUN_ARGS=("$@")

if [ ! -x "$CLI" ] || [ ! -x "$PYTHON" ]; then
    echo "Missing repo environment. Run: python3 -m venv .venv && .venv/bin/pip install -e '.[full,dev]'" >&2
    exit 2
fi

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

reset_start_failures() {
    bedrock_start_failures=0
    bedrock_failure_started_s=0
}

abort_failed_start() {
    local result="$1"
    local force_relaunch="${2:-false}"
    local now_s elapsed_s
    stop_runtime
    if [ "$result" -eq 64 ]; then
        return "$result"
    fi
    if emergency_latched; then
        return 64
    fi
    if operator_paused; then
        return 65
    fi
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
    local result
    if emergency_latched; then
        echo "Emergency stop is latched; persistent startup is suspended." >&2
        return 64
    fi
    if operator_paused; then
        echo "Explicit operator pause is active; persistent startup remains suspended." >&2
        return 65
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
        --role "$ROLE" \
        --live \
        --capture-source "$CAPTURE_SOURCE" \
        "${RUN_ARGS[@]}"
    result=$?
    if [ "$result" -ne 0 ]; then
        abort_failed_start "$result"
        return $?
    fi

    local attempt
    for attempt in $(seq 1 45); do
        if readiness_ok; then
            reset_start_failures
            echo "Minecraft AI is ready: isolated Bedrock, supervisor, and agent are healthy."
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
echo " Role: $ROLE | capture: $CAPTURE_SOURCE"
echo "=================================================="

if ! "$CLI" install; then
    trap - EXIT INT TERM
    exit 2
fi
start_support_services || exit $?
wait_for_model || exit $?

while true; do
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
    if [ "$result" -eq 65 ]; then
        sleep "$CHECK_INTERVAL_S"
        continue
    fi
    echo "Startup failed; retrying safely in 15 seconds." >&2
    sleep 15
done

failures=0
nonplayable_failures=0
while true; do
    sleep "$CHECK_INTERVAL_S"
    if runtime_health_ok; then
        failures=0
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
    if [ "$nonplayable_failures" -eq 0 ]; then
        failures=$((failures + 1))
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
            break
        fi
        if [ "$result" -eq 64 ]; then
            trap - EXIT INT TERM
            exit 64
        fi
        if [ "$result" -eq 67 ]; then
            exit 67
        fi
        if [ "$result" -eq 65 ]; then
            sleep "$CHECK_INTERVAL_S"
            continue
        fi
        echo "Recovery failed; retrying safely in 15 seconds." >&2
        sleep 15
    done
done
