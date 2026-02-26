#!/bin/bash
# Determined AI entrypoint for GRPO training.
# Runs inside each Determined container. Uses environment variables
# from Determined to figure out the node role:
#
# Role assignment (for multi-node runs):
#   - Chief (rank 0, first node):  Ray head + GRPO training
#   - Other ranks:                  Ray workers
#
# For single-node runs (slots_per_trial <= 8):
#   - Everything runs on the single node.
#
# Required environment variables (set by Determined or the YAML):
#   BASE_DIR      - path to the shared project directory
#   OUTPUT_DIR    - path to write outputs
#   JOB_NAME      - experiment name for W&B and output dirs
#
# These are automatically set by Determined:
#   DET_SLOT_IDS  - comma-separated GPU IDs for this container
#   DET_CHIEF_IP  - IP of the chief (rank 0) container
#   DET_NUM_NODES - total number of nodes (derived from slots)

set -e

# --- Source secrets if available on shared fs ---
SECRETS_FILE="${BASE_DIR}/secrets.env"
if [ -f "$SECRETS_FILE" ]; then
    echo "[entrypoint] Loading secrets from $SECRETS_FILE"
    source "$SECRETS_FILE"
fi

# --- Determine node role ---
# Determined sets RANK or DET_UNIQUE_PORT_OFFSET for distributed; fallback to 0 for single
# For HPC/Slurm-based Determined, SLURM_PROCID may be available
if [ -n "${SLURM_PROCID:-}" ]; then
    NODE_RANK="$SLURM_PROCID"
elif [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
    NODE_RANK="$OMPI_COMM_WORLD_RANK"
elif [ -n "${DET_UNIQUE_PORT_OFFSET:-}" ]; then
    # DET_UNIQUE_PORT_OFFSET is 0 for chief, >0 for workers
    NODE_RANK="$DET_UNIQUE_PORT_OFFSET"
else
    NODE_RANK=0
fi

echo "=========================================="
echo "Determined GRPO Entrypoint"
echo "  JOB_NAME:   ${JOB_NAME:-unset}"
echo "  NODE_RANK:  $NODE_RANK"
echo "  DET_CHIEF_IP: ${DET_CHIEF_IP:-localhost}"
echo "  DET_SLOT_IDS: ${DET_SLOT_IDS:-none}"
echo "  Hostname:   $(hostname)"
echo "  IP:         $(hostname -I 2>/dev/null | awk '{print $1}')"
echo "=========================================="

# --- Common setup ---
cd /stage

# Ensure cache dirs exist on shared fs
mkdir -p "${BASE_DIR}/logs" \
         "${BASE_DIR}/.cache/nltk_data" \
         "${BASE_DIR}/.cache/open_instruct_dataset_cache" \
         "${OUTPUT_DIR}" \
         "${OUTPUT_DIR}/rollouts"

export NLTK_DATA="${BASE_DIR}/.cache/nltk_data"
export DATASET_CACHE="${BASE_DIR}/.cache/open_instruct_dataset_cache"

# --- Ray configuration ---
RAY_PORT="${RAY_PORT:-6379}"
HEAD_IP="${DET_CHIEF_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

# --- Judge URLs (judge runs on a separate node, rank 1 in multi-node) ---
# For single-node, judge is disabled. For multi-node, judge node = rank 1.
JUDGE_IP="${JUDGE_IP:-$HEAD_IP}"
CODE_API_URL="${CODE_API_URL:-http://${JUDGE_IP}:${CODE_API_PORT:-1234}/test_program}"
HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-http://${JUDGE_IP}:${LLM_JUDGE_PORT:-8000}/v1}"

# --- GRPO arguments (set via GRPO_ARGS env var or use defaults below) ---
if [ -z "${GRPO_ARGS:-}" ]; then
    GRPO_ARGS="--exp_name ${JOB_NAME} \
  --queue_dashboard_port 8765 \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 8 \
  --num_unique_prompts_rollout 64 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --output_dir ${OUTPUT_DIR} \
  --rollouts_save_path ${OUTPUT_DIR}/rollouts \
  --save_traces \
  --dataset_local_cache_dir ${BASE_DIR}/.cache/open_instruct_dataset_cache \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 AIML-TUDA/SLR-Bench:v1-All 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 AIML-TUDA/SLR-Bench:v1-All 4 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 5000 \
  --response_length 30000 \
  --pack_length 35840 \
  --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
  --chat_template_name olmo_thinker \
  --non_stop_penalty False \
  --mask_truncated_completions False \
  --temperature 1.0 \
  --ground_truths_key ground_truth \
  --sft_messages_key prompt \
  --total_episodes 10000000 \
  --deepspeed_stage 3 \
  --num_learners_per_node 8 \
  --vllm_num_engines 48 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.85 \
  --vllm_sync_backend nccl \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --llm_judge_model hosted_vllm/${LLM_JUDGE_MODEL} \
  --llm_judge_timeout 1200 \
  --llm_judge_max_tokens 2048 \
  --llm_judge_max_context_length 32768 \
  --clip_higher 0.272 \
  --code_api_url ${CODE_API_URL} \
  --code_pass_rate_reward_threshold 0.99 \
  --code_max_execution_time 6 \
  --seed 1 \
  --local_eval_every -1 \
  --eval_receive_timeout 600 \
  --save_freq 50 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --gradient_checkpointing \
  --with_tracking \
  --checkpoint_state_freq 100 \
  --checkpoint_state_dir ${OUTPUT_DIR}/checkpoints \
  --backend_timeout 1200 \
  --inflight_updates true \
  --async_steps 8 \
  --advantage_normalization_type centered \
  --truncated_importance_sampling_ratio_cap 2.0 \
  --push_to_hub false"
fi

if [ "$NODE_RANK" = "0" ]; then
    echo "[entrypoint] === CHIEF NODE (rank 0) ==="

    # --- Start code API if configured ---
    if [ -n "${CODE_API_PORT:-}" ]; then
        echo "[entrypoint] Starting Code API on port $CODE_API_PORT..."
        export BASE_DIR
        # Set a dummy SLURM_JOB_ID/JOB_NAME for the code_api_setup script
        export SLURM_JOB_ID="${DET_EXPERIMENT_ID:-0}"
        source scripts/train/slr/code_api_setup.sh &
        CODE_API_PID=$!
        sleep 5
    fi

    # --- Start Ray head ---
    echo "[entrypoint] Starting Ray head on port $RAY_PORT..."
    ray stop --force 2>/dev/null || true
    ray start --head --port="$RAY_PORT" --dashboard-host=0.0.0.0
    sleep 5

    # Wait for workers to join (if multi-node)
    TOTAL_NODES="${DET_NUM_NODES:-1}"
    if [ "$TOTAL_NODES" -gt 1 ]; then
        echo "[entrypoint] Waiting 30s for $((TOTAL_NODES - 1)) workers to join..."
        sleep 30
    fi

    # --- Download NLTK data ---
    uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" || true

    # --- Run GRPO training ---
    echo "[entrypoint] Starting GRPO training..."
    uv run python open_instruct/grpo_fast.py $GRPO_ARGS || true

    # Cleanup
    ray stop --force 2>/dev/null || true
    if [ -n "${CODE_API_PID:-}" ]; then
        kill "$CODE_API_PID" 2>/dev/null || true
    fi

else
    echo "[entrypoint] === WORKER NODE (rank $NODE_RANK) ==="

    # --- Start LLM judge on rank 1 if configured ---
    if [ "$NODE_RANK" = "1" ] && [ -n "${LLM_JUDGE_MODEL:-}" ] && [ -n "${LLM_JUDGE_PORT:-}" ]; then
        echo "[entrypoint] Starting LLM judge on this worker..."
        export SLURM_JOB_ID="${DET_EXPERIMENT_ID:-0}"
        source scripts/train/slr/judge_setup.sh &
        JUDGE_PID=$!
    fi

    # --- Start Ray worker ---
    echo "[entrypoint] Starting Ray worker, connecting to head at $RAY_ADDRESS..."
    ray stop --force 2>/dev/null || true

    # Stagger worker joins
    DELAY=$((15 + NODE_RANK * 2))
    echo "[entrypoint] Worker $NODE_RANK: waiting ${DELAY}s before joining cluster"
    sleep "$DELAY"

    ray start --address="$RAY_ADDRESS" --dashboard-host=0.0.0.0
    echo "[entrypoint] Worker $NODE_RANK: joined cluster"

    # Poll head availability. Exit 0 when head is gone.
    while ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; do
        sleep 5
    done
    echo "[entrypoint] Worker $NODE_RANK: head unreachable, exiting."
    ray stop --force 2>/dev/null || true

    if [ -n "${JUDGE_PID:-}" ]; then
        kill "$JUDGE_PID" 2>/dev/null || true
    fi
fi

echo "[entrypoint] Done."
