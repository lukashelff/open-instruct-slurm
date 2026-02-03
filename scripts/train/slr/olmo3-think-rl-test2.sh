#!/bin/bash
# OLMo-3 7B Think RL (GRPO) - Slurm version, DEBUG: 1 node, 2 GPUs (1 learner + 1 vLLM), small batch/lengths.
# No single-GPU sharing; no multi-node Ray setup needed.
#SBATCH --job-name=olmo3-7b-think-rl-test
#SBATCH --partition=all          # Match your cluster: all, gpu, etc.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal
#SBATCH --open-mode=append

# --- 1. Variables & Paths ---
BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
CONTAINER_IMAGE="docker://helffml/open_instruct_dev"
OUTPUT_DIR="$BASE_DIR/output/olmo3_7b_think_rl"


# --- 2. Slurm env (for logging) ---
echo "=========================================="
echo "SLR Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment Dir: $BASE_DIR"
echo "Start Time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="


# --- 3. Environment variables (inside container) ---
export HOME="$BASE_DIR"
export TOKENIZERS_PARALLELISM=FALSE
# Use your wandb username/team; "SLR-RLVR" must exist in your wandb account or use your username
export WANDB_ENTITY="helff"
export WANDB_PROJECT="Reward-Shortcut"
export HF_TRUST_REMOTE_CODE=TRUE
export HF_TOKEN="HF_TOKEN"
export WANDB_API_KEY="wandb_TOKEN"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRITON_CACHE_DIR="$BASE_DIR/.cache/triton"
# From Tillicum-style GRPO runs: quieter logs, avoid cache issues in containers
export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_DEBUG=ERROR
export VLLM_LOGGING_LEVEL=WARNING

# --- 4. Run OLMo-3 7B Think RL inside container ---
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/triton" "$BASE_DIR/.cache/open_instruct_dataset_cache" "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

apptainer exec --nv \
  --bind "$BASE_DIR:/stage" \
  --env "UV_CACHE_DIR=/stage/.cache/uv" \
  --env "HF_HOME=/stage/.cache/huggingface" \
  --env "TMPDIR=/tmp/ray_run" \
  --env "VLLM_DISABLE_COMPILE_CACHE=1" \
  --env "NCCL_DEBUG=ERROR" \
  --env "VLLM_LOGGING_LEVEL=WARNING" \
  "$CONTAINER_IMAGE" \
  bash -c "mkdir -p /tmp/ray_run && cd /stage && uv run python open_instruct/grpo_fast.py \
  --exp_name pipelinerl_7b_olmo3_thinker_debug \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 2 \
  --num_unique_prompts_rollout 2 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --output_dir /stage/output/olmo3_7b_think_rl \
  --rollouts_save_path /stage/output/olmo3_7b_think_rl/rollouts \
  --dataset_local_cache_dir /stage/.cache/open_instruct_dataset_cache \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 4 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 512 \
  --response_length 2048 \
  --pack_length 2560 \
  --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
  --chat_template_name olmo_thinker \
  --non_stop_penalty False \
  --mask_truncated_completions False \
  --temperature 1.0 \
  --ground_truths_key ground_truth \
  --sft_messages_key prompt \
  --total_episodes 32 \
  --deepspeed_stage 3 \
  --num_learners_per_node 4 \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_enforce_eager \
  --vllm_sync_backend nccl \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --seed 1 \
  --local_eval_every 1 \
  --save_freq 5 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --gradient_checkpointing \
  --with_tracking \
  --clip_higher 0.272 \
  --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
  --code_pass_rate_reward_threshold 0.99 \
  --checkpoint_state_freq 0 \
  --backend_timeout 1200 \
  --inflight_updates true \
  --async_steps 8 \
  --advantage_normalization_type centered \
  --truncated_importance_sampling_ratio_cap 2.0 \
  --push_to_hub false"
