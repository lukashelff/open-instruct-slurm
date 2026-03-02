#!/bin/bash
# Slurm eval script for OLMo-paper benchmarks (MMLU, BBH, GPQA, AGIEval, AIME24/25, IFEval).
# Uses vLLM backend for efficient batched inference on 8 GPUs.
# Compatible with OLMo-Think and other HuggingFace instruction-tuned models.
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
#   LMEVAL_TASKS="mmlu,bbh,gpqa_diamond" sbatch scripts/eval/slurm/run_slurm_lm_eval.sh MODEL_PATH OUTPUT_DIR
#
#   # Sweep all checkpoints sequentially:
#   bash scripts/eval/slurm/run_slurm_lm_eval_sweep.sh MODEL_BASE_DIR OUTPUT_BASE_DIR
#
#SBATCH --job-name=lm-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lm_eval/run_%j.out
#SBATCH --error=logs/lm_eval/run_%j.err
#SBATCH --qos=normal

set -e

# --- Arguments ---
MODEL_PATH="${1:?Usage: $0 MODEL_PATH OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: $0 MODEL_PATH OUTPUT_DIR}"

# Comma-separated lm-eval task names matching the OLMo paper eval suite.
# Zebra Grid, CHE, and LiveCodeBench are not available in lm-eval harness and are omitted.
# See: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
# Paper metrics: MMLU  BBH  GPQA  AGI  AIME25  AIME24  IFEval
LMEVAL_TASKS="${LMEVAL_TASKS:-mmlu,bbh,gpqa,agieval,aime24,aime25,ifeval}"
LMEVAL_TASKS2="${LMEVAL_TASKS2:-logiqa,logiqa2,ai2_arc,hellaswag}"

# Number of GPUs for tensor-parallel vLLM inference (must match --gpus-per-node above).
TP_SIZE="${TP_SIZE:-8}"

# Max sequence length for vLLM. Defaults to the model config value.
# For long-chain-of-thought models (OLMo-Think) keep this large; set lower (e.g. 4096) only
# if you hit OOM and don't need full-length CoT for a given task.
MAX_MODEL_LEN="32768"

# --- Paths ---
BASE_DIR="${BASE_DIR:-/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://helffml/open_instruct_dev:slr}"

mkdir -p "$BASE_DIR/logs" "$OUTPUT_DIR"

if [ -f "$BASE_DIR/secrets.env" ]; then
  source "$BASE_DIR/secrets.env"
fi

# Resolve model path to absolute (for use inside container)
if [[ "$MODEL_PATH" != /* ]]; then
  if [[ -e "$BASE_DIR/$MODEL_PATH" ]]; then
    MODEL_PATH="$(cd "$BASE_DIR" && readlink -f "$MODEL_PATH")"
  elif [[ -e "$MODEL_PATH" ]]; then
    MODEL_PATH="$(readlink -f "$MODEL_PATH")"
  else
    echo "Treating MODEL_PATH as Hugging Face repo ID: $MODEL_PATH"
  fi
fi

echo "=========================================="
echo "LM Eval Job: $SLURM_JOB_ID"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Tasks: $LMEVAL_TASKS","$LMEVAL_TASKS2"
echo "TP size: $TP_SIZE"
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


apptainer exec --nv "${APPTAINER_ENV[@]}" "$CONTAINER_IMAGE" \
  bash -c "
    cd /stage
    uv sync --extra lm-eval
    uv run lm_eval \
      --model vllm \
      --model_args pretrained=$MODEL_PATH_IN,tensor_parallel_size=$TP_SIZE,dtype=auto,gpu_memory_utilization=0.7,max_model_len=$MAX_MODEL_LEN \
      --tasks $LMEVAL_TASKS \
      --batch_size auto \
      --output_path $OUTPUT_DIR_IN \
      --log_samples \
      --seed 42 \
      --fewshot_as_multiturn \
      --trust_remote_code \
      --apply_chat_template \
      --gen_kwargs "temperature=0.6,top_p=0.95"
  "

echo "=========================================="
echo "Eval complete. Results in $OUTPUT_DIR"
echo "=========================================="
