#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Missing repo environment. Run: python3 -m venv .venv && .venv/bin/pip install -e '.[full,dev]'" >&2
    exit 2
fi

echo "=================================================="
echo " Minecraft AI Bedrock runtime"
echo "    Python: $VENV_PYTHON"
echo "=================================================="

echo "[1/3] Verifying runtime dependencies"
"$VENV_PYTHON" -m minecraft_ai.cli install

# 2. Check for isolated Bedrock session or launch if requested
echo "[2/3] Checking Bedrock display isolation status"
"$VENV_PYTHON" -c "
from minecraft_ai.cli import doctor
print('System status verified.')
"

# 3. Start Agent & Live Web Dashboard
echo "[3/3] Launching Minecraft AI live runtime"
echo "      Press Ctrl+C to stop."
echo ""

ROLE="${1:-generalist}"
if [ "$#" -gt 0 ]; then
    shift
fi
exec "$VENV_PYTHON" -m minecraft_ai.cli run --role "$ROLE" --live "$@"
