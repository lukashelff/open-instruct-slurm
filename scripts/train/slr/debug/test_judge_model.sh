#!/bin/bash
# ===========================================================================
# Judge-model smoke test  (runs on a SINGLE node, 8 GPUs)
#
# Always uses the official vLLM container (vllm/vllm-openai:qwen3_5).
# Starts a vLLM server for the candidate judge model, sends a batch of
# realistic judge prompts (quality_ref style), and reports:
#   1. Throughput (tok/s, req/s)
#   2. Parse-success rate  (valid JSON with REASONING + SCORE)
#   3. Latency distribution (p50, p90, p99)
#   4. Sample outputs for manual inspection
#
# Usage:
#   sbatch scripts/train/slr/test_judge_model.sh
#
# Override the model:
#   LLM_JUDGE_MODEL=Qwen/Qwen3.5-32B-A17B sbatch scripts/train/slr/test_judge_model.sh
#
# Compatible models:
#   Qwen/Qwen3.5-397B-A17B-FP8  - MoE 17B active, FP8 (default)
#   Qwen/Qwen3.5-397B-A17B      - MoE 17B active, BF16
#   Qwen/Qwen3.5-32B-A17B       - MoE 17B active, smaller
# ===========================================================================
#SBATCH --job-name=judge-test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/judge-test_%j.out
#SBATCH --error=logs/judge-test_%j.err
#SBATCH --qos=normal

set -euo pipefail

# --- Configuration (override via env before sbatch) ---
# LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen3-Next-80B-A3B-Thinking}"
LLM_JUDGE_PORT="${LLM_JUDGE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_REQUESTS="${NUM_REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-16}"

BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
VLLM_CONTAINER_IMAGE="docker://vllm/vllm-openai:qwen3_5"

echo "=========================================="
echo "Judge Model Smoke Test"
echo "  Model:       $LLM_JUDGE_MODEL"
echo "  TP size:     $TP_SIZE"
echo "  Max len:     $MAX_MODEL_LEN"
echo "  Max tokens:  $MAX_COMPLETION_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Requests:    $NUM_REQUESTS"
echo "  Concurrency: $CONCURRENCY"
echo "=========================================="

mkdir -p "$BASE_DIR/logs"

# --- Load secrets ---
if [ -f "$BASE_DIR/secrets.env" ]; then
  source "$BASE_DIR/secrets.env"
fi

# --- Container setup ---
SIF_CACHE="$BASE_DIR/.cache/apptainer"
mkdir -p "$SIF_CACHE"
SIF_FILE="$SIF_CACHE/vllm_openai_qwen3_5.sif"

if [ ! -f "$SIF_FILE" ]; then
    echo "[setup] Pulling container to $SIF_FILE ..."
    apptainer pull --force "$SIF_FILE" "$VLLM_CONTAINER_IMAGE"
    echo "[setup] Pull complete."
fi

APPTAINER_ENV=(
  --bind "$BASE_DIR:/stage"
  --bind "/dev/shm:/dev/shm"
  --env "TMPDIR=/tmp"
  --env "HF_HOME=/stage/.cache/huggingface"
  --env "HF_TOKEN=$HF_TOKEN"
  --env "HF_HUB_OFFLINE=False"
  --env "HF_HUB_DISABLE_PROGRESS_BARS=0"
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  --env "VLLM_LOGGING_LEVEL=INFO"
  --env "VLLM_USE_DEEP_GEMM=0"
  --env "NCCL_CUMEM_ENABLE=0"
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
  --env "LLM_JUDGE_MODEL=$LLM_JUDGE_MODEL"
  --env "LLM_JUDGE_PORT=$LLM_JUDGE_PORT"
  --env "TP_SIZE=$TP_SIZE"
  --env "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  --env "MAX_COMPLETION_TOKENS=$MAX_COMPLETION_TOKENS"
  --env "TEMPERATURE=$TEMPERATURE"
  --env "NUM_REQUESTS=$NUM_REQUESTS"
  --env "CONCURRENCY=$CONCURRENCY"
  --env "VLLM_LOG=/stage/logs/vllm_judge_test_${SLURM_JOB_ID}.log"
)

srun --nodes=1 --ntasks=1 apptainer exec --nv --writable-tmpfs "${APPTAINER_ENV[@]}" "$SIF_FILE" \
  bash -c '
set -euo pipefail
cd /stage

echo "[1/3] Starting vLLM server: $LLM_JUDGE_MODEL (TP=$TP_SIZE) ..."
echo "[1/3] vLLM log: $VLLM_LOG"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE - 1))) \
  vllm serve "$LLM_JUDGE_MODEL" \
    --host 0.0.0.0 \
    --port "$LLM_JUDGE_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --language-model-only \
    --reasoning-parser deepseek_r1 \
    --enable-prefix-caching \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Stream log live to show download progress and startup
tail -f "$VLLM_LOG" &
TAIL_PID=$!

# Wait for health
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        kill $TAIL_PID 2>/dev/null || true
        echo "[FAIL] vLLM process died."
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${LLM_JUDGE_PORT}/health" >/dev/null 2>&1; then
        kill $TAIL_PID 2>/dev/null || true
        echo "[OK] vLLM healthy after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ $WAITED -ge $MAX_WAIT ]; then
    kill $TAIL_PID 2>/dev/null || true
    echo "[FAIL] vLLM did not start in ${MAX_WAIT}s."
    exit 1
fi

echo "[2/3] Running judge benchmark ($NUM_REQUESTS requests, concurrency=$CONCURRENCY) ..."
pip install openai -q 2>/dev/null || true
python3 /stage/scripts/train/slr/debug/judge_benchmark.py

echo ""
echo "[3/3] Cleaning up ..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
'
