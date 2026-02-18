#!/bin/bash
set -euo pipefail

# Simple node tester: start one uvicorn instance and verify /health
# Usage: ./node_tester.sh [PORT]

PORT=${1:-1234}
REPO_PATH=${REPO_PATH:-$(pwd)}
LOG_DIR="$REPO_PATH/logs/node_tester"
mkdir -p "$LOG_DIR"

# Activate virtualenv if present
if [ -f "$REPO_PATH/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_PATH/.venv/bin/activate"
fi

echo "Node tester: REPO_PATH=$REPO_PATH  PORT=$PORT"
cd "$REPO_PATH"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn not found in PATH; install with: pip install 'uvicorn[standard]'" >&2
  exit 2
fi

nohup uvicorn open_instruct.code_utils.api:app --host 0.0.0.0 --port "$PORT" --workers 1 > "$LOG_DIR/node_tester_$PORT.log" 2>&1 &
PID=$!
echo "Started uvicorn (PID=$PID). Logs: $LOG_DIR/node_tester_$PORT.log"

trap 'echo "Stopping uvicorn (PID=$PID)"; kill "$PID" 2>/dev/null || true; exit' INT TERM EXIT

echo "Waiting for /health to respond..."
for i in {1..15}; do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "✓ Health endpoint responding on port $PORT"
    break
  fi
  sleep 1
done

if ! curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "✗ Health endpoint did not respond. See log: $LOG_DIR/node_tester_$PORT.log" >&2
  exit 3
fi

echo "Node tester succeeded. Press Ctrl-C to stop the server and exit." 

# Wait on the server process so the script keeps running until user interrupts
wait "$PID"
