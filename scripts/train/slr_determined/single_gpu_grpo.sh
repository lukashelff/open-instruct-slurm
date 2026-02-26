#!/bin/bash
# Single-GPU GRPO test on Determined AI.
# Runs a minimal GRPO training with 1 GPU to verify the full pipeline.
# Uses a small model and few episodes to keep it fast.
set -e

echo "=== Single GPU GRPO Test ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "GPUs:"
nvidia-smi -L 2>&1 || echo "No nvidia-smi"

# --- Load secrets ---
SECRETS_FILE="${BASE_DIR}/secrets.env"
if [ -f "$SECRETS_FILE" ]; then
    echo "[setup] Loading secrets from $SECRETS_FILE"
    source "$SECRETS_FILE"
fi

cd /stage

# --- Dirs ---
mkdir -p "${BASE_DIR}/logs" \
         "${BASE_DIR}/.cache/nltk_data" \
         "${BASE_DIR}/.cache/open_instruct_dataset_cache" \
         "${OUTPUT_DIR}" \
         "${OUTPUT_DIR}/rollouts"

# --- NLTK data ---
uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" || true

# --- Start Ray head (single node) ---
ray stop --force 2>/dev/null || true
ray start --head --port="${RAY_PORT:-6379}" --dashboard-host=0.0.0.0
sleep 5

# --- Run GRPO with minimal config ---
# Single GPU: 1 learner, 1 vLLM engine, small batch, few episodes
uv run python open_instruct/grpo_fast.py \
  --exp_name "${JOB_NAME}" \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 2 \
  --num_unique_prompts_rollout 4 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --output_dir "${OUTPUT_DIR}" \
  --rollouts_save_path "${OUTPUT_DIR}/rollouts" \
  --dataset_local_cache_dir "${BASE_DIR}/.cache/open_instruct_dataset_cache" \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 4 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 512 \
  --response_length 512 \
  --pack_length 1024 \
  --model_name_or_path allenai/OLMo-2-0425-1B \
  --non_stop_penalty False \
  --mask_truncated_completions False \
  --temperature 1.0 \
  --ground_truths_key ground_truth \
  --sft_messages_key prompt \
  --total_episodes 16 \
  --deepspeed_stage 3 \
  --num_learners_per_node 1 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.5 \
  --vllm_sync_backend gloo \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --seed 42 \
  --local_eval_every -1 \
  --save_freq 999 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --gradient_checkpointing \
  --push_to_hub false \
  || true

ray stop --force 2>/dev/null || true
echo "=== Single GPU GRPO Test Complete ==="
