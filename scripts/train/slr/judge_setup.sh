#!/bin/bash
# LLM Judge (vLLM) setup for Slurm (SLR) runs.
# Starts a vLLM server for the judge model with TP across local GPUs.
# Inspired by configs/judge_configs/general_verifier_judge.yaml.
#
# Usage: source this script from the judge-node block in your sbatch script.
#   Required env vars:
#     LLM_JUDGE_MODEL       - HuggingFace model name (e.g. Qwen/Qwen3-32B)
#     LLM_JUDGE_PORT        - port to serve on (e.g. 8000)
#     LLM_JUDGE_NUM_ENGINES - tensor parallel size / number of GPUs
#   Optional env vars:
#     LLM_JUDGE_MAX_MODEL_LEN  - max context length (default: 32768)
#     LLM_JUDGE_EXTRA_ARGS     - additional vllm serve arguments (default: "")
#     LLM_JUDGE_LOG_FILE       - log file path (default: $BASE_DIR/logs/judge/judge.log)

set -e

# --- Configuration ---
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:?LLM_JUDGE_MODEL must be set}"
LLM_JUDGE_PORT="${LLM_JUDGE_PORT:?LLM_JUDGE_PORT must be set}"
LLM_JUDGE_NUM_ENGINES="${LLM_JUDGE_NUM_ENGINES:?LLM_JUDGE_NUM_ENGINES must be set}"
LLM_JUDGE_MAX_MODEL_LEN="${LLM_JUDGE_MAX_MODEL_LEN:-32768}"
LLM_JUDGE_EXTRA_ARGS="${LLM_JUDGE_EXTRA_ARGS:-}"
BASE_DIR="${BASE_DIR:-.}"
LLM_JUDGE_LOG_FILE="${LLM_JUDGE_LOG_FILE:-${BASE_DIR}/logs/${JOB_NAME}_${SLURM_JOB_ID}/judge.log}"

echo "=========================================="
echo "LLM Judge setup (SLR)"
echo "  Model:          $LLM_JUDGE_MODEL"
echo "  Port:           $LLM_JUDGE_PORT"
echo "  TP size:        $LLM_JUDGE_NUM_ENGINES"
echo "  Max model len:  $LLM_JUDGE_MAX_MODEL_LEN"
echo "  Log file:       $LLM_JUDGE_LOG_FILE"
echo "  Extra args:     ${LLM_JUDGE_EXTRA_ARGS:-<none>}"
echo "=========================================="

mkdir -p "$(dirname "$LLM_JUDGE_LOG_FILE")"

# Select GPUs for the judge (first N GPUs)
JUDGE_GPUS=$(seq -s, 0 $((LLM_JUDGE_NUM_ENGINES - 1)))

echo "[judge] Starting vLLM serve on GPUs: $JUDGE_GPUS"

# Launch vLLM in the background so we can wait for it
CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
  uv run vllm serve "$LLM_JUDGE_MODEL" \
    --host 0.0.0.0 \
    --port "$LLM_JUDGE_PORT" \
    --tensor-parallel-size "$LLM_JUDGE_NUM_ENGINES" \
    --max-model-len "$LLM_JUDGE_MAX_MODEL_LEN" \
    --trust-remote-code \
    $LLM_JUDGE_EXTRA_ARGS \
    > "$LLM_JUDGE_LOG_FILE" 2>&1 &
JUDGE_PID=$!

echo "[judge] vLLM PID: $JUDGE_PID"

# Wait for the server to become healthy
echo "[judge] Waiting for vLLM to become healthy on port $LLM_JUDGE_PORT ..."
MAX_WAIT=600  # 10 minutes for large models
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "[judge] ✗ vLLM process died. Last 20 lines of log:"
        tail -20 "$LLM_JUDGE_LOG_FILE"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${LLM_JUDGE_PORT}/health" >/dev/null 2>&1; then
        echo "[judge] ✓ vLLM is healthy after ${WAITED}s (port $LLM_JUDGE_PORT)"
        break
    fi
    /usr/bin/sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[judge] ✗ vLLM did not become healthy within ${MAX_WAIT}s. Last 20 lines:"
    tail -20 "$LLM_JUDGE_LOG_FILE"
    exit 1
fi

echo "[judge] Setup complete. Judge serving $LLM_JUDGE_MODEL on port $LLM_JUDGE_PORT"

# Keep the script alive (foreground wait) so that the srun task doesn't exit.
# The caller can use `exec source ...` or just `source ...` at the end of their block.
wait "$JUDGE_PID"
