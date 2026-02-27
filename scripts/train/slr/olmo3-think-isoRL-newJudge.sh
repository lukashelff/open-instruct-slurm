#!/bin/bash
# OLMo-3 7B Think RL (GRPO) on Slurm (8 nodes):
#   Task 0 = judge (code API + LLM judge vLLM in official vLLM container)
#   Task 1 = Ray head (gradient updates via grpo_fast.py)
#   Tasks 2–7 = Ray workers (48 vLLM inference engines)
#
# Dual-container mode:
#   - Judge node runs vLLM in the official vLLM container (latest model support)
#   - All other nodes (+ code API on judge) use the open_instruct container
#
#SBATCH --job-name=RLVR-Isomorphic
#SBATCH --partition=all
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=1T
#SBATCH --time=100:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal
#SBATCH --open-mode=append
#SBATCH --exclude=cn13,cn06,cn05

# --- 1. Configuration ---
JOB_NAME="RLVR-soofi-Olmo-IsomorphicRL"
BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
CONTAINER_IMAGE="docker://helffml/open_instruct_dev:slr"
VLLM_CONTAINER_IMAGE="docker://vllm/vllm-openai:qwen3_5"  # Official vLLM image (latest model support)
OUTPUT_DIR="$BASE_DIR/output/$JOB_NAME"
RAY_PORT=6379
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
LLM_JUDGE_PORT=8000
LLM_JUDGE_NUM_ENGINES=8
LLM_JUDGE_MAX_MODEL_LEN="${LLM_JUDGE_MAX_MODEL_LEN:-45536}"
CODE_API_PORT=1234

export HOME="$BASE_DIR"
export JOB_NAME="$JOB_NAME"

# --- 2. Resolve node IPs ---
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '2p')
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JUDGE_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
JUDGE_IP=$(srun --nodes=1 --ntasks=1 -w "$JUDGE_NODE" hostname --ip-address)
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
HOSTED_VLLM_API_BASE="http://${JUDGE_IP}:${LLM_JUDGE_PORT}/v1"
CODE_API_URL="http://${JUDGE_IP}:${CODE_API_PORT}/test_program"


echo "=========================================="
echo "Job: $JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Nodes: $SLURM_NODELIST"
echo "Head: $HEAD_NODE ($HEAD_IP:$RAY_PORT)"
echo "Judge: $JUDGE_NODE ($JUDGE_IP) — LLM: $LLM_JUDGE_PORT, Code: $CODE_API_PORT"
echo "Judge model: $LLM_JUDGE_MODEL"
echo "Ray dashboard: ssh -L 8265:$HEAD_IP:8265 <login-node>"
echo "=========================================="

# --- 3. Directories ---
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/nltk_data" "$BASE_DIR/.cache/open_instruct_dataset_cache" "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

# --- 3a. Load secrets (API keys, tokens) from file if it exists. This file is not in the repo and should be created by each user with their own keys.
if [ -f "$BASE_DIR/secrets.env" ]; then
source $BASE_DIR/secrets.env
fi

# --- 3b. Pre-pull container image(s) (avoid 8 nodes racing to build SIF) ---
SIF_CACHE="$BASE_DIR/.cache/apptainer"
mkdir -p "$SIF_CACHE"

# Training container (open_instruct)
SIF_FILE="$SIF_CACHE/open_instruct_dev_slr.sif"
if [ ! -f "$SIF_FILE" ]; then
    echo "[setup] Pulling training container to $SIF_FILE ..."
    apptainer pull --force "$SIF_FILE" "$CONTAINER_IMAGE"
    echo "[setup] Pull complete."
else
    echo "[setup] Using cached training SIF: $SIF_FILE"
fi

# Judge container (vLLM official)
VLLM_SIF_FILE="$SIF_CACHE/vllm_openai_qwen3_5.sif"
if [ ! -f "$VLLM_SIF_FILE" ]; then
    echo "[setup] Pulling vLLM container to $VLLM_SIF_FILE ..."
    apptainer pull --force "$VLLM_SIF_FILE" "$VLLM_CONTAINER_IMAGE"
    echo "[setup] Pull complete."
else
    echo "[setup] Using cached vLLM SIF: $VLLM_SIF_FILE"
fi

# --- 4. Container environment ---
APPTAINER_ENV=(
  --bind "$BASE_DIR:/stage"
  --env "TMPDIR=/tmp"
  --env "BASE_DIR=/stage"
  --env "UV_CACHE_DIR=/stage/.cache/uv"
  --env "HF_HOME=/stage/.cache/huggingface"
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
  --env "HF_TOKEN=$HF_TOKEN"
  --env "HF_HUB_OFFLINE=True"
  --env "NLTK_DATA=/stage/.cache/nltk_data"
  --env "TOKENIZERS_PARALLELISM=FALSE"
  --env "WANDB_ENTITY=helff"
  --env "WANDB_PROJECT=Reward-Shortcut"
  --env "WANDB_API_KEY=$WANDB_API_KEY"
  --env "RAY_ADDRESS=$RAY_ADDRESS"
  --env "RAY_PORT=$RAY_PORT"
  --env "RAY_DEDUP_LOGS=0"
  --env "HOSTED_VLLM_API_BASE=$HOSTED_VLLM_API_BASE"
  --env "CODE_API_URL=$CODE_API_URL"
  --env "CODE_API_PORT=$CODE_API_PORT"
  --env "LLM_JUDGE_MODEL=$LLM_JUDGE_MODEL"
  --env "LLM_JUDGE_PORT=$LLM_JUDGE_PORT"
  --env "LLM_JUDGE_NUM_ENGINES=$LLM_JUDGE_NUM_ENGINES"
  --env "NCCL_CUMEM_ENABLE=0"
  --env "NCCL_DEBUG=ERROR"
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  --env "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
  --env "VLLM_LOGGING_LEVEL=WARNING"
)

# --- 5. GRPO training arguments ---
GRPO_ARGS="--exp_name $JOB_NAME \
  --queue_dashboard_port 8765 \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 8 \
  --num_unique_prompts_rollout 64 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --output_dir $OUTPUT_DIR \
  --rollouts_save_path $OUTPUT_DIR/rollouts \
  --save_traces \
  --dataset_local_cache_dir /stage/.cache/open_instruct_dataset_cache \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 AIML-TUDA/SLR-Bench:v1-All 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 AIML-TUDA/SLR-Bench:v1-All 4 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 5000 \
  --response_length 32768 \
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
  --llm_judge_model hosted_vllm/$LLM_JUDGE_MODEL \
  --llm_judge_timeout 1200 \
  --llm_judge_max_tokens 8048 \
  --llm_judge_max_context_length $LLM_JUDGE_MAX_MODEL_LEN \
  --llm_judge_temperature 0.7 \
  --clip_higher 0.272 \
  --code_api_url $CODE_API_URL \
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
  --checkpoint_state_dir $OUTPUT_DIR/checkpoints \
  --backend_timeout 1200 \
  --inflight_updates true \
  --async_steps 8 \
  --advantage_normalization_type centered \
  --truncated_importance_sampling_ratio_cap 2.0 \
  --push_to_hub false"

# --- 5b. Judge-specific container env (for vLLM container, minimal) ---
JUDGE_APPTAINER_ENV=(
  --bind "$BASE_DIR:/stage"
  --bind "/dev/shm:/dev/shm"
  --env "TMPDIR=/tmp"
  --env "HF_HOME=/stage/.cache/huggingface"
  --env "HF_TOKEN=$HF_TOKEN"
  --env "HF_HUB_OFFLINE=False"
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
  --env "LLM_JUDGE_MODEL=$LLM_JUDGE_MODEL"
  --env "LLM_JUDGE_PORT=$LLM_JUDGE_PORT"
  --env "LLM_JUDGE_NUM_ENGINES=$LLM_JUDGE_NUM_ENGINES"
  --env "LLM_JUDGE_MAX_MODEL_LEN=$LLM_JUDGE_MAX_MODEL_LEN"
  --env "NCCL_CUMEM_ENABLE=0"
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  --env "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
  --env "VLLM_LOGGING_LEVEL=WARNING"
  --env "VLLM_WORKER_MULTIPROC_METHOD=spawn"
)

# --- 6. Launch ---
# ===========================================================================
# Dual-container mode:
#   srun #1: vLLM container on judge node (LLM judge server)
#   srun #2: open_instruct container on ALL 8 nodes
#            - judge node (PROCID=0): code API + wait
#            - head node  (PROCID=1): Ray head + training
#            - others     (PROCID=2-7): Ray workers
#
# Both srun steps share the judge node via --overlap.
# The vLLM container only runs the judge server (GPU-heavy).
# The open_instruct container on the judge node only runs the code API (CPU).
# ===========================================================================

# Step 1: Launch vLLM judge server (vLLM container, judge node, background)
LLM_JUDGE_LOG_FILE="$BASE_DIR/logs/judge/${JOB_NAME}_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$LLM_JUDGE_LOG_FILE")"

srun --overlap --nodes=1 --ntasks=1 -w "$JUDGE_NODE" \
  apptainer exec --nv --writable-tmpfs "${JUDGE_APPTAINER_ENV[@]}" "$VLLM_SIF_FILE" \
  bash -c '
    set -euo pipefail
    JUDGE_GPUS=$(seq -s, 0 $((LLM_JUDGE_NUM_ENGINES - 1)))
    echo "[judge-vllm] Starting vLLM serve: $LLM_JUDGE_MODEL (TP=$LLM_JUDGE_NUM_ENGINES, GPUs=$JUDGE_GPUS)"
    CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
      vllm serve "$LLM_JUDGE_MODEL" \
        --host 0.0.0.0 \
        --port "$LLM_JUDGE_PORT" \
        --tensor-parallel-size "$LLM_JUDGE_NUM_ENGINES" \
        --max-model-len "$LLM_JUDGE_MAX_MODEL_LEN" \
        --trust-remote-code \
        --language-model-only \
        --reasoning-parser qwen3 \
        --enable-prefix-caching \
  ' > "$LLM_JUDGE_LOG_FILE" 2>&1 &
JUDGE_SRUN_PID=$!
echo "[launch] Judge vLLM srun PID: $JUDGE_SRUN_PID (log: $LLM_JUDGE_LOG_FILE)"

 # Step 2: Main training (open_instruct container, all 8 nodes)
# On the judge node (PROCID=0), only the code API runs (CPU-only, no GPU conflict).
srun --overlap --nodes=8 --ntasks=8 apptainer exec --nv --writable-tmpfs "${APPTAINER_ENV[@]}" "$SIF_FILE" \
  bash -c '
    cd /stage

    if [ "${SLURM_PROCID:-0}" = "1" ]; then
      # --- Ray head + training ---
      source scripts/train/slr/ray_setup.sh
      /usr/bin/sleep 20  # extra wait for workers to join
      uv run python -c "import nltk; nltk.download(\"punkt_tab\", quiet=True); nltk.download(\"punkt\", quiet=True)"
      uv run python open_instruct/grpo_fast.py '"$GRPO_ARGS"' || true
      ray stop --force 2>/dev/null || true

    elif [ "${SLURM_PROCID:-0}" = "0" ]; then
      # --- Judge node: code API only (vLLM runs in separate container) ---
      source scripts/train/slr/code_api_setup.sh

      # Wait for vLLM judge to become healthy (started in the vLLM container)
      echo "[judge-wait] Waiting for vLLM server on port $LLM_JUDGE_PORT ..."
      MAX_WAIT=600
      WAITED=0
      while [ $WAITED -lt $MAX_WAIT ]; do
          if curl -sf "http://127.0.0.1:${LLM_JUDGE_PORT}/health" >/dev/null 2>&1; then
              echo "[judge-wait] vLLM healthy after ${WAITED}s"
              break
          fi
          /usr/bin/sleep 5
          WAITED=$((WAITED + 5))
      done
      if [ $WAITED -ge $MAX_WAIT ]; then
          echo "[judge-wait] WARNING: vLLM not healthy after ${MAX_WAIT}s"
      fi

      # Block until training finishes (keeps the srun task alive)
      echo "[judge-wait] Code API running, sleeping until job ends ..."
      while true; do /usr/bin/sleep 60; done

    else
      # --- Ray worker (blocks until head exits) ---
      source scripts/train/slr/ray_setup.sh
    fi
  '

# Training srun finished — clean up judge
echo "[launch] Training complete, stopping judge vLLM ..."
kill $JUDGE_SRUN_PID 2>/dev/null || true
wait $JUDGE_SRUN_PID 2>/dev/null || true