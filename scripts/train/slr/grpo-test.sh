#!/bin/bash
#SBATCH --job-name=grpo-rlvr-test
#SBATCH --partition=all          # Match your cluster: all, gpu, etc.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1        # 1 GPU for a short test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        # Adjust if your cluster expects different
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal             # Use normal QoS (match example)
#SBATCH --open-mode=append      # Keep logs continuous across restarts

# --- 1. Variables & Paths ---
BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
# Use your Docker image via Apptainer/Singularity (no .sif needed; pulls from Docker Hub)
# container name https://hub.docker.com/r/helffml/open_instruct_dev
CONTAINER_IMAGE="docker://helffml/open_instruct_dev"

# Optional: if you have a pre-pulled .sif, set it here and use $SIF_IMAGE in the exec below
# SIF_IMAGE="$BASE_DIR/helffml_open_instruct_dev.sif"

# --- 2. Slurm env (for logging) ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $BASE_DIR"

# --- 3. Environment variables (inside container) ---
# Use writable paths under bind-mounted /stage (read-only /root fails in Apptainer)
# TMPDIR must be a short path inside container so Ray's AF_UNIX socket path stays under 107 bytes
export HOME="$BASE_DIR" # Sets home inside container to your workspace
export TOKENIZERS_PARALLELISM=FALSE
export WANDB_ENTITY="SLR-RLVR"
export HF_TRUST_REMOTE_CODE=TRUE
export HF_TOKEN="HF_TOKEN"
export WANDB_API_KEY="wandb_TOKEN"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TRITON_CACHE_DIR="$BASE_DIR/.cache/triton"

# --- 4. Run GRPO + RLVR inside container ---
mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/tmp"
mkdir -p "$BASE_DIR/.cache/triton"

echo "UV_CACHE_DIR: $UV_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo "Starting container..."
# Use apptainer (or singularity on older clusters) with --nv for GPU access.
# If `apptainer` is not found, try: singularity exec --nv ...
apptainer exec --nv \
  --bind "$BASE_DIR:/stage" \
  --env "UV_CACHE_DIR=/stage/.cache/uv" \
  --env "HF_HOME=/stage/.cache/huggingface" \
  --env "TMPDIR=/tmp/ray_run" \
  "$CONTAINER_IMAGE" \
  bash -c "mkdir -p /tmp/ray_run && cd /stage && uv run python open_instruct/grpo_fast.py \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 512 \
  --response_length 512 \
  --pack_length 1024 \
  --per_device_train_batch_size 1 \
  --num_unique_prompts_rollout 8 \
  --num_samples_per_prompt_rollout 4 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --stop_strings '</answer>' \
  --apply_verifiable_reward true \
  --temperature 0.7 \
  --ground_truths_key ground_truth \
  --chat_template_name r1_simple_chat_postpend_think \
  --learning_rate 3e-7 \
  --total_episodes 10 \
  --deepspeed_stage 2 \
  --num_epochs 1 \
  --num_learners_per_node 1 \
  --vllm_tensor_parallel_size 1 \
  --beta 0.01 \
  --seed 3 \
  --local_eval_every 1 \
  --vllm_sync_backend gloo \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_enforce_eager \
  --gradient_checkpointing \
  --single_gpu_mode \
  --push_to_hub false"
