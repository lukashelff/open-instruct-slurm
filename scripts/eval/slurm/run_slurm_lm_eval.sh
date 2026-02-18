#!/bin/bash
# Slurm eval script for standard benchmarks (MMLU, GSM8K, BBH, etc.) without Beaker.
# Runs lm-eval (EleutherAI) on a saved checkpoint. Compatible with OLMo-Think and other HF models.
#
# Prerequisites:
#   uv sync --extra lm-eval   # install lm-eval optional dependency
#
# Usage:
#   # Eval final model
#   sbatch scripts/eval/slurm/run_slurm_lm_eval.sh \
#     /path/to/output/RLVR-soofi-Basev2-Isomorphic-RLv2 \
#     output/eval/RLVR-soofi-step-final
#
#   # Eval checkpoint at step 100
#   sbatch scripts/eval/slurm/run_slurm_lm_eval.sh \
#     /path/to/output/RLVR-soofi-Basev2-Isomorphic-RLv2/checkpoints/step_100 \
#     output/eval/RLVR-soofi-step-100
#
#   # With custom tasks (comma-separated)
#   LMEVAL_TASKS="mmlu,gsm8k,truthfulqa" sbatch scripts/eval/slurm/run_slurm_lm_eval.sh MODEL_PATH OUTPUT_DIR
#
#SBATCH --job-name=lm-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lm_eval_%j.out
#SBATCH --error=logs/lm_eval_%j.err
#SBATCH --qos=normal

set -e

# --- Arguments ---
MODEL_PATH="${1:?Usage: $0 MODEL_PATH OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: $0 MODEL_PATH OUTPUT_DIR}"

# Optional: comma-separated lm-eval task names (default: mmlu, gsm8k, bbh, truthfulqa)
# See: lm_eval tasks list  (e.g. mmlu, gsm8k, gsm8k_cot, bbh, hendrycksTest-*, truthfulqa_mc2, humaneval, mbpp)
LMEVAL_TASKS="${LMEVAL_TASKS:-mmlu,gsm8k,bbh,truthfulqa_mc2}"

# --- Paths ---
BASE_DIR="${BASE_DIR:-/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://helffml/open_instruct_dev:slr}"

mkdir -p "$BASE_DIR/logs" "$OUTPUT_DIR"

# Resolve model path to absolute (for use inside container)
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$(cd "$BASE_DIR" && readlink -f "$MODEL_PATH")"
fi

echo "=========================================="
echo "LM Eval Job: $SLURM_JOB_ID"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Tasks: $LMEVAL_TASKS"
echo "=========================================="

# --- Environment ---
export HOME="$BASE_DIR"
export TOKENIZERS_PARALLELISM=false
export HF_TRUST_REMOTE_CODE=true
export HF_HOME="${HF_HOME:-$BASE_DIR/.cache/huggingface}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

APPTAINER_ENV=(
  --bind "$BASE_DIR:/stage"
  --env "UV_CACHE_DIR=/stage/.cache/uv"
  --env "HF_HOME=/stage/.cache/huggingface"
  --env "HF_TRUST_REMOTE_CODE=true"
  --env "TOKENIZERS_PARALLELISM=false"
  --env "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
)

# Model path inside container (stage = BASE_DIR)
MODEL_PATH_IN=""
if [[ "$MODEL_PATH" == "$BASE_DIR"/* ]]; then
  MODEL_PATH_IN="/stage/${MODEL_PATH#$BASE_DIR/}"
else
  MODEL_PATH_IN="$MODEL_PATH"
fi

OUTPUT_DIR_IN=""
if [[ "$OUTPUT_DIR" == "$BASE_DIR"/* ]]; then
  OUTPUT_DIR_IN="/stage/${OUTPUT_DIR#$BASE_DIR/}"
else
  OUTPUT_DIR_IN="$OUTPUT_DIR"
fi

# --- Run lm-eval ---
# Uses HuggingFace backend. Batch size 8 for 7B on 1x A100. --apply_chat_template for instruction-tuned models (OLMo-Think, etc.).
apptainer exec --nv "${APPTAINER_ENV[@]}" "$CONTAINER_IMAGE" \
  bash -c "
    cd /stage
    uv sync --extra lm-eval
    uv run lm_eval run \
      --model hf \
      --model_args pretrained=$MODEL_PATH_IN \
      --tasks $LMEVAL_TASKS \
      --batch_size 8 \
      --output_path $OUTPUT_DIR_IN \
      --log_samples \
      --trust_remote_code \
      --apply_chat_template
  "

echo "=========================================="
echo "Eval complete. Results in $OUTPUT_DIR"
echo "=========================================="
