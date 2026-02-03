#!/bin/bash
# OLMo-3 7B Think RL (GRPO) - Slurm version of scripts/train/olmo3/7b_think_rl.sh
# Single-node 8-GPU run. For multi-node, add srun + Ray head/workers and adjust SBATCH.
#SBATCH --job-name=olmo3-7b-think-rl
#SBATCH --partition=all          # Match your cluster: all, gpu, etc.
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal
#SBATCH --open-mode=append
#SBATCH --exclusive

# --- 1. Variables & Paths ---
BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
CONTAINER_IMAGE="docker://helffml/open_instruct_dev"
OUTPUT_DIR="$BASE_DIR/output/olmo3_7b_think_rl"

# --- 2. Slurm env (for logging) ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $BASE_DIR"
echo "Output dir: $OUTPUT_DIR"

# --- 3. Environment variables (inside container) ---
export HOME="$BASE_DIR"
export TOKENIZERS_PARALLELISM=FALSE
export WANDB_ENTITY="SLR-RLVR"
export HF_TRUST_REMOTE_CODE=TRUE
export HF_TOKEN="HF_TOKEN"
export WANDB_API_KEY="wandb_TOKEN"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRITON_CACHE_DIR="$BASE_DIR/.cache/triton"

# --- 4. Run OLMo-3 7B Think RL inside container ---
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/triton" "$OUTPUT_DIR"

apptainer exec --nv \
  --bind "$BASE_DIR:/stage" \
  --env "UV_CACHE_DIR=/stage/.cache/uv" \
  --env "HF_HOME=/stage/.cache/huggingface" \
  --env "TMPDIR=/tmp/ray_run" \
  "$CONTAINER_IMAGE" \
  bash -c "mkdir -p /tmp/ray_run && cd /stage && uv run python open_instruct/grpo_fast.py \
  --exp_name pipelinerl_7b_olmo3_thinker \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 8 \
  --num_unique_prompts_rollout 64 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --output_dir /stage/output/olmo3_7b_think_rl \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 \
  --dataset_mixer_eval_list_splits train \
  # --max_token_length 10240 \
  --max_prompt_token_length 2048 \
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
  --num_learners_per_node 8 8 \
  --vllm_num_engines 16 \
  --vllm_tensor_parallel_size 1 \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --seed 1 \
  --local_eval_every 50 \
  --save_freq 25 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --gradient_checkpointing \
  --with_tracking \
  --clip_higher 0.272 \
  --checkpoint_state_freq 100 \
  --backend_timeout 1200 \
  --inflight_updates true \
  --async_steps 8 \
  --advantage_normalization_type centered \
  --truncated_importance_sampling_ratio_cap 2.0 \
  --push_to_hub false"
