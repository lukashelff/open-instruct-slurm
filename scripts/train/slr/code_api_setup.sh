#!/bin/bash
# Code API setup for Slurm (SLR) runs.
# Launches a single uvicorn process with N workers to serve the code execution API.
# Adapted from configs/beaker_configs/code_api_setup.sh.
#
# Usage: source this script from the judge-node block in your sbatch script.
#   Required env vars:
#     CODE_API_PORT  - the port to serve on
#     BASE_DIR       - path to the repo root (for uvicorn module resolution)
#   Optional env vars:
#     CODE_SERVER_CPUS   - number of uvicorn workers (default: 24)

set -e

# --- Configuration ---
BASE_DIR="${BASE_DIR:?BASE_DIR must be set}"
CODE_API_PORT="${CODE_API_PORT:?CODE_API_PORT must be set}"
CODE_SERVER_CPUS="${CODE_SERVER_CPUS:-48}"

echo "=========================================="
echo "Code API setup (SLR)"
echo "  Workers: $CODE_SERVER_CPUS"
echo "  Port:    $CODE_API_PORT"
echo "  Repo:    $BASE_DIR"
echo "=========================================="

cd "$BASE_DIR"
mkdir -p "$BASE_DIR/logs/code_api"

LOG_FILE="$BASE_DIR/logs/code_api/${JOB_NAME}_${SLURM_JOB_ID}.log"

echo "[code_api] Starting uvicorn with $CODE_SERVER_CPUS workers on port $CODE_API_PORT ..."
nohup uv run uvicorn open_instruct.code_utils.api:app \
    --host 0.0.0.0 \
    --port "$CODE_API_PORT" \
    --workers "$CODE_SERVER_CPUS" \
    > "$LOG_FILE" 2>&1 &
CODE_API_PID=$!
echo "[code_api] PID: $CODE_API_PID"

# Wait for health
echo "[code_api] Waiting for health check on port $CODE_API_PORT ..."
for attempt in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${CODE_API_PORT}/health" >/dev/null 2>&1; then
        echo "[code_api] ✓ Code API healthy after ${attempt}s"
        break
    fi
    if ! kill -0 "$CODE_API_PID" 2>/dev/null; then
        echo "[code_api] ✗ Code API process died. Logs:"
        tail -20 "$LOG_FILE"
        break
    fi
    /usr/bin/sleep 1
done

if ! curl -sf "http://127.0.0.1:${CODE_API_PORT}/health" >/dev/null 2>&1; then
    echo "[code_api] ⚠ Code API NOT healthy on port $CODE_API_PORT after 30s"
fi

echo "[code_api] Setup complete."
