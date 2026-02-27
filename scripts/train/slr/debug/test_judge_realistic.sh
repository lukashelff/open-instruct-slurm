#!/bin/bash
# ===========================================================================
# Realistic judge-model speed test  (SINGLE node, 8 GPUs)
#
# Uses REAL OLMo rollout data as judge inputs to get accurate throughput
# numbers for long-context judge prompts. Compares reasoning-ON vs
# reasoning-OFF to measure the overhead of chain-of-thought.
#
# Default model: Qwen/Qwen3.5-35B-A3B  (MoE, 3B active params)
#
# Usage:
#   sbatch scripts/train/slr/debug/test_judge_realistic.sh
#
# Override model:
#   LLM_JUDGE_MODEL=Qwen/Qwen3.5-35B-A3B sbatch scripts/train/slr/debug/test_judge_realistic.sh
#
# Override mode (both|no_reasoning|reasoning):
#   BENCHMARK_MODE=no_reasoning sbatch scripts/train/slr/debug/test_judge_realistic.sh
#
# Compatible models:
#   Qwen/Qwen3.5-35B-A3B         - MoE 3B active (default, fastest)
# ===========================================================================
#SBATCH --job-name=judge-realistic
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=logs/judge-realistic_%j.out
#SBATCH --error=logs/judge-realistic_%j.err
#SBATCH --qos=normal
#SBATCH --exclude=cn13,cn06,cn05

set -euo pipefail

# --- Configuration (override via env before sbatch) ---
# LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen3-32B}"
LLM_JUDGE_PORT="${LLM_JUDGE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-45536}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_REQUESTS="${NUM_REQUESTS:-30}"
CONCURRENCY="${CONCURRENCY:-16}"
# "both" runs no-reasoning then reasoning; "no_reasoning" or "reasoning" for single
BENCHMARK_MODE="${BENCHMARK_MODE:-both}"
# Rollout data config
ROLLOUT_FILE="${ROLLOUT_FILE:-/stage/output/RLVR-soofi-Olmo-IsomorphicRL/rollouts/RLVR-soofi-Olmo-IsomorphicRL__1__1771357951_rollouts_000000.jsonl}"
ROLLOUT_TOKENIZER="${ROLLOUT_TOKENIZER:-allenai/Olmo-3-7B-Think-DPO}"
ROLLOUT_SAMPLE_POOL="${ROLLOUT_SAMPLE_POOL:-500}"

BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
VLLM_CONTAINER_IMAGE="docker://vllm/vllm-openai:qwen3_5"

echo "=========================================="
echo "Realistic Judge Model Speed Test"
echo "  Model:          $LLM_JUDGE_MODEL"
echo "  TP size:        $TP_SIZE"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  Max tokens:     $MAX_COMPLETION_TOKENS"
echo "  Temperature:    $TEMPERATURE"
echo "  Requests:       $NUM_REQUESTS"
echo "  Concurrency:    $CONCURRENCY"
echo "  Benchmark mode: $BENCHMARK_MODE"
echo "  Rollout file:   $ROLLOUT_FILE"
echo "  Tokenizer:      $ROLLOUT_TOKENIZER"
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
  --env "BENCHMARK_MODE=$BENCHMARK_MODE"
  --env "ROLLOUT_FILE=$ROLLOUT_FILE"
  --env "ROLLOUT_TOKENIZER=$ROLLOUT_TOKENIZER"
  --env "ROLLOUT_SAMPLE_POOL=$ROLLOUT_SAMPLE_POOL"
)

srun --nodes=1 --ntasks=1 apptainer exec --nv --writable-tmpfs "${APPTAINER_ENV[@]}" "$SIF_FILE" \
  bash -c '
set -euo pipefail
cd /stage

echo "[1/3] Starting vLLM server: $LLM_JUDGE_MODEL (TP=$TP_SIZE) ..."

# vLLM outputs directly to stdout/stderr (captured by SLURM .out/.err)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE - 1))) \
  vllm serve "$LLM_JUDGE_MODEL" \
    --host 0.0.0.0 \
    --port "$LLM_JUDGE_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    &
VLLM_PID=$!

# Wait for health
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[FAIL] vLLM process died."
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${LLM_JUDGE_PORT}/health" >/dev/null 2>&1; then
        echo "[OK] vLLM healthy after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[FAIL] vLLM did not start in ${MAX_WAIT}s."
    exit 1
fi

echo "[2/3] Running realistic judge benchmark ($NUM_REQUESTS requests, concurrency=$CONCURRENCY, mode=$BENCHMARK_MODE) ..."
pip install openai transformers -q 2>/dev/null || true
python3 /stage/scripts/train/slr/debug/judge_benchmark_realistic.py

echo ""
echo "[3/3] Cleaning up ..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
'
