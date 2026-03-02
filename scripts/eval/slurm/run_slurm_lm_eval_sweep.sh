#!/bin/bash
# Submit lm-eval jobs sequentially for every checkpoint under MODEL_BASE_DIR.
# Each job is chained with --dependency=afterok so they run one after another,
# avoiding OOM from multiple vLLM instances and keeping GPU utilisation clean.
#
# Checkpoint discovery (sorted numerically by step number):
#   MODEL_BASE_DIR/checkpoints/step_100/
#   MODEL_BASE_DIR/checkpoints/step_200/
#   MODEL_BASE_DIR/checkpoints/step_300/
#   ...
# Optionally also evaluates the final (non-checkpoint) model at MODEL_BASE_DIR.
#
# Usage:
#   # Eval all checkpoints + final model
#   bash scripts/eval/slurm/run_slurm_lm_eval_sweep.sh \
#     /path/to/output/MY-RUN \
#     output/eval/MY-RUN
#
#   # Eval only checkpoints (skip final model)
#   EVAL_FINAL=0 bash scripts/eval/slurm/run_slurm_lm_eval_sweep.sh \
#     /path/to/output/MY-RUN \
#     output/eval/MY-RUN
#
#   # Override task list for the whole sweep
#   LMEVAL_TASKS="mmlu,bbh" bash scripts/eval/slurm/run_slurm_lm_eval_sweep.sh \
#     /path/to/output/MY-RUN output/eval/MY-RUN
#
#   # Limit to specific steps (glob pattern)
#   CKPT_GLOB="checkpoints/step_{100,200,500}" bash ... MODEL_BASE_DIR OUTPUT_BASE_DIR

set -e

# --- Arguments ---
MODEL_BASE_DIR="${1:?Usage: $0 MODEL_BASE_DIR OUTPUT_BASE_DIR}"
OUTPUT_BASE_DIR="${2:?Usage: $0 MODEL_BASE_DIR OUTPUT_BASE_DIR}"

# Set to 0 to skip evaluating the final (non-checkpoint) model.
EVAL_FINAL="${EVAL_FINAL:-1}"

# Glob pattern relative to MODEL_BASE_DIR for checkpoint discovery.
CKPT_GLOB="${CKPT_GLOB:-checkpoints/step_*}"

BASE_DIR="${BASE_DIR:-/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$BASE_DIR/scripts/eval/slurm/run_slurm_lm_eval.sh}"

# Resolve model base dir to absolute path.
if [[ "$MODEL_BASE_DIR" != /* ]]; then
  MODEL_BASE_DIR="$(readlink -f "$MODEL_BASE_DIR")"
fi

echo "=========================================="
echo "LM Eval Sweep"
echo "Model base: $MODEL_BASE_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo "Checkpoint glob: $CKPT_GLOB"
echo "Tasks: ${LMEVAL_TASKS:-<default>}"
echo "=========================================="

# --- Collect and sort checkpoints ---
# sort -V: natural/version sort so step_10 < step_100 < step_200.
mapfile -t CKPTS < <(
  ls -d "$MODEL_BASE_DIR"/$CKPT_GLOB 2>/dev/null | sort -V
)

if [[ ${#CKPTS[@]} -eq 0 && "$EVAL_FINAL" -ne 1 ]]; then
  echo "ERROR: no checkpoints found matching $MODEL_BASE_DIR/$CKPT_GLOB and EVAL_FINAL=0."
  exit 1
fi

echo "Found ${#CKPTS[@]} checkpoint(s) to evaluate."

# --- Submit jobs ---
PREV_JID=""

submit_job() {
  local ckpt_path="$1"
  local out_dir="$2"
  local label="$3"

  local dep_flag=""
  if [[ -n "$PREV_JID" ]]; then
    dep_flag="--dependency=afterok:$PREV_JID"
  fi

  local jid
  jid=$(sbatch $dep_flag "$EVAL_SCRIPT" "$ckpt_path" "$out_dir" | awk '{print $NF}')

  echo "  Submitted [$label] -> job $jid  (depends on: ${PREV_JID:-none})  output: $out_dir"
  PREV_JID="$jid"
}

for CKPT in "${CKPTS[@]}"; do
  STEP=$(basename "$CKPT")    # e.g. step_100
  OUT_DIR="$OUTPUT_BASE_DIR/$STEP"
  submit_job "$CKPT" "$OUT_DIR" "$STEP"
done

if [[ "$EVAL_FINAL" -eq 1 ]]; then
  submit_job "$MODEL_BASE_DIR" "$OUTPUT_BASE_DIR/final" "final"
fi

echo "=========================================="
echo "All jobs submitted. Chain head: job ${PREV_JID}."
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
