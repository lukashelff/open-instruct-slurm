#!/bin/bash
# Code API setup for Slurm (SLR) runs.
# Launches N individual uvicorn instances behind nginx, mirroring the Beaker setup.
# Adapted from configs/beaker_configs/code_api_setup.sh.
#
# Usage: source this script from the judge-node block in your sbatch script.
#   Required env vars:
#     CODE_API_PORT  - the public port clients connect to (nginx)
#     BASE_DIR       - path to the repo root (for uvicorn module resolution)
#   Optional env vars:
#     CODE_SERVER_CPUS   - number of uvicorn instances to spawn (default: 96)

set -e

# --- Configuration ---
BASE_DIR="${BASE_DIR:?BASE_DIR must be set}"
CODE_API_PORT="${CODE_API_PORT:?CODE_API_PORT must be set}"
CODE_SERVER_CPUS="${CODE_SERVER_CPUS:-96}"
API_BASE_PORT=9100  # internal ports: 9100, 9101, ..., 9100+N-1
LOG_FILE="$BASE_DIR/logs/${JOB_NAME}_${SLURM_JOB_ID}/code_api.log"

echo "=========================================="
echo "Code API setup (SLR)"
echo "  Instances: $CODE_SERVER_CPUS"
echo "  Public port (nginx): $CODE_API_PORT"
echo "  Internal ports: ${API_BASE_PORT}–$((API_BASE_PORT + CODE_SERVER_CPUS - 1))"
echo "  Repo: $BASE_DIR"
echo "=========================================="

cd "$BASE_DIR"
mkdir -p "$BASE_DIR/logs/code_api"

# --- Register cleanup trap so Apptainer can exit cleanly ---
# Without this, background uvicorn/nginx processes prevent fuse-overlayfs unmount.
code_api_cleanup() {
    echo "[code_api] Cleaning up background processes ..."
    # Kill nginx
    if [ -n "${NGINX_RUN_DIR:-}" ] && [ -f "$NGINX_RUN_DIR/nginx.pid" ]; then
        "${NGINX_BIN:-nginx}" -s stop -c "$NGINX_RUN_DIR/nginx.conf" 2>/dev/null || true
    fi
    # Kill all uvicorn children we spawned
    pkill -f "uvicorn open_instruct.code_utils.api:app" 2>/dev/null || true
    /usr/bin/sleep 1
    pkill -9 -f "uvicorn open_instruct.code_utils.api:app" 2>/dev/null || true
    echo "[code_api] Cleanup done."
}
trap code_api_cleanup EXIT

# --- Start individual uvicorn instances ---
echo "[code_api] Starting $CODE_SERVER_CPUS uvicorn instances ..."
for ((i=0; i<CODE_SERVER_CPUS; i++)); do
    PORT=$((API_BASE_PORT + i))
    nohup uv run uvicorn open_instruct.code_utils.api:app \
        --host 127.0.0.1 \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 &

    # Brief progress
    if (( i == 0 )); then
        echo "[code_api] First instance PID: $!"
    fi
done
echo "[code_api] All $CODE_SERVER_CPUS instances launched."

# Wait for the first instance to respond before configuring nginx
echo "[code_api] Waiting for first backend (port $API_BASE_PORT) to become healthy ..."
for attempt in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${API_BASE_PORT}/health" >/dev/null 2>&1; then
        echo "[code_api] ✓ First backend healthy after ${attempt}s"
        break
    fi
    /usr/bin/sleep 1
done

# --- Configure nginx as a load-balancer (standalone config, no system deps) ---
NGINX_BIN=$(command -v nginx 2>/dev/null || true)
HAS_NGINX=false

if [ -n "$NGINX_BIN" ]; then
    echo "[code_api] Found nginx at $NGINX_BIN, configuring standalone load-balancer ..."

    # Create temp dirs nginx needs (works even in Apptainer with --writable-tmpfs)
    NGINX_RUN_DIR="/tmp/nginx_code_api"
    mkdir -p "$NGINX_RUN_DIR"/{logs,tmp}

    # Find the mime.types file (needed for a standalone config)
    MIME_TYPES=""
    for candidate in /etc/nginx/mime.types /usr/local/nginx/conf/mime.types /usr/share/nginx/mime.types; do
        if [ -f "$candidate" ]; then
            MIME_TYPES="$candidate"
            break
        fi
    done

    # Build upstream block
    UPSTREAM=""
    for ((i=0; i<CODE_SERVER_CPUS; i++)); do
        UPSTREAM+="    server 127.0.0.1:$((API_BASE_PORT + i));
"
    done

    # Write a fully self-contained nginx.conf (no include of /etc/nginx/*)
    NGINX_CONF="$NGINX_RUN_DIR/nginx.conf"
    cat > "$NGINX_CONF" << EOF
# Standalone nginx config for code API load-balancing
worker_processes auto;
pid        $NGINX_RUN_DIR/nginx.pid;
error_log  $NGINX_RUN_DIR/logs/error.log warn;

events {
    # Use ulimit minus a small margin; cap at 8192 to stay within container limits
    worker_connections $(( $(ulimit -n 2>/dev/null || echo 1024) < 8192 ? $(ulimit -n 2>/dev/null || echo 1024) - 64 : 8192 ));
}

http {
    ${MIME_TYPES:+include $MIME_TYPES;}
    default_type  application/octet-stream;
    access_log    off;
    sendfile      on;

    upstream code_api_servers {
        least_conn;
${UPSTREAM}
    }

    server {
        listen ${CODE_API_PORT};
        client_max_body_size 0;

        location / {
            proxy_pass http://code_api_servers;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_connect_timeout 10s;
            proxy_send_timeout 120s;
            proxy_read_timeout 120s;
        }
    }
}
EOF

    echo "[code_api] Testing nginx config ..."
    if "$NGINX_BIN" -t -c "$NGINX_CONF" 2>&1; then
        # Kill any stale nginx for this config
        if [ -f "$NGINX_RUN_DIR/nginx.pid" ] && [ -s "$NGINX_RUN_DIR/nginx.pid" ]; then
            "$NGINX_BIN" -s stop -c "$NGINX_CONF" 2>/dev/null || true
            /usr/bin/sleep 1
        fi
        "$NGINX_BIN" -c "$NGINX_CONF"
        HAS_NGINX=true
        echo "[code_api] ✓ nginx started on port $CODE_API_PORT (pid=$(cat "$NGINX_RUN_DIR/nginx.pid" 2>/dev/null))"
    else
        echo "[code_api] ⚠ nginx config test FAILED (see above), falling back"
    fi
fi

if ! $HAS_NGINX; then
    # Fallback: run one uvicorn directly on CODE_API_PORT with --workers
    echo "[code_api] nginx not available — falling back to uvicorn --workers on port $CODE_API_PORT"
    nohup uv run uvicorn open_instruct.code_utils.api:app \
        --host 0.0.0.0 \
        --port "$CODE_API_PORT" \
        --workers "$CODE_SERVER_CPUS" \
        > "$LOG_FILE" 2>&1 &
    echo "[code_api] Fallback PID: $!"
fi

# --- Final health check on the public port ---
echo "[code_api] Waiting for public endpoint (port $CODE_API_PORT) ..."
for attempt in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${CODE_API_PORT}/health" >/dev/null 2>&1; then
        echo "[code_api] ✓ Code API healthy on port $CODE_API_PORT after ${attempt}s"
        break
    fi
    /usr/bin/sleep 1
done

if ! curl -sf "http://127.0.0.1:${CODE_API_PORT}/health" >/dev/null 2>&1; then
    echo "[code_api] ⚠ Code API NOT healthy on port $CODE_API_PORT after 60s"
    echo "[code_api] Checking backend processes ..."
    pgrep -a uvicorn | head -5
    if $HAS_NGINX; then
        "$NGINX_BIN" -t -c "$NGINX_CONF" 2>&1 || true
        echo "[code_api] nginx error log:"
        tail -10 "$NGINX_RUN_DIR/logs/error.log" 2>/dev/null || true
    fi
fi

echo "[code_api] Setup complete."
