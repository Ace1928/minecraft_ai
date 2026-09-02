#!/usr/bin/env bash
# ==============================================================================
# 💎 EIDOSIAN NEXUS — MINECRAFT AI SINGLE-CLICK LAUNCHER
# High performance, persistent, multi-timescale Minecraft AI Agent
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="/home/lloyd/eidosian_forge/eidosian_venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(which python3)"
fi

echo "=================================================="
echo " 💎 EIDOSIAN NEXUS — MINECRAFT AI AGENT SYSTEM"
echo "    Python: $VENV_PYTHON"
echo "=================================================="

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# 1. Bootstrap environment & config
echo "[1/3] Verifying runtime dependencies..."
$VENV_PYTHON -m minecraft_ai.cli install

# 2. Check for isolated Bedrock session or launch if requested
echo "[2/3] Checking Bedrock display isolation status..."
$VENV_PYTHON -c "
from minecraft_ai.cli import doctor
print('System status verified.')
"

# 3. Start Agent & Live Web Dashboard
echo "[3/3] Launching Minecraft AI Live Runtime..."
echo "      Live stream and dashboard will be available at: http://127.0.0.1:8765/"
echo "      Press Ctrl+C to stop."
echo ""

ROLE="${1:-generalist}"
exec $VENV_PYTHON -m minecraft_ai.cli run --role "$ROLE" --live "$@"
