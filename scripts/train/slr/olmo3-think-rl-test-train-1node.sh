#!/bin/bash
# OLMo-3 7B Think RL (GRPO) - Slurm test2: scaled batch/rollouts, same fixes as test (NLTK, LLM judge, code API).
# 1 node, 8 GPUs. For 2+ nodes use olmo3-think-rl-test-train-2node.sh (starts Ray cluster so placement group does not hang).
#SBATCH --job-name=olmo-GRPO-test2
#SBATCH --partition=all          # Match your cluster: all, gpu, etc.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=4:00:00
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
echo "SLR Job Started (test2: scaled rollouts)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment Dir: $BASE_DIR"
echo "Start Time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="
# Dashboards: Ray = 8265; ActorManager queue dashboard = 8765 (fixed via --queue_dashboard_port).
# Forward from your machine: ssh -L 8265:NODE:8265 -L 8765:NODE:8765 USER@LOGIN
# Then open http://localhost:8265 (Ray) and http://localhost:8765 (ActorManager queue status).

# --- 3. Environment variables (inside container) ---
export HOME="$BASE_DIR"
export TOKENIZERS_PARALLELISM=FALSE
export WANDB_ENTITY="helff"
export WANDB_PROJECT="Reward-Shortcut"
export HF_TRUST_REMOTE_CODE=TRUEf
export HF_TOKEN="HF_TOKEN"
export WANDB_API_KEY="wandb_TOKEN"
export OPENAI_API_KEY="openAI_TOKEN"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=ERROR
export TRITON_CACHE_DIR="$BASE_DIR/.cache/triton"
export HOSTED_VLLM_API_KEY="${HOSTED_VLLM_API_KEY:-EMPTY}"

# --- 4. Run OLMo-3 7B Think RL inside container ---
# Verifiers: Math (rule-based), Code (--code_api_url), Instruction (IFEval), Chat (--llm_judge_model hosted_vllm/...).
# For hosted_vllm judge: HOSTED_VLLM_API_BASE and HOSTED_VLLM_API_KEY=EMPTY avoid "fake-api-key" rejection.
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/triton" "$BASE_DIR/.cache/nltk_data" "$BASE_DIR/.cache/open_instruct_dataset_cache" "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

apptainer exec --nv \
  --bind "$BASE_DIR:/stage" \
  --env "UV_CACHE_DIR=/stage/.cache/uv" \
  --env "HF_HOME=/stage/.cache/huggingface" \
  --env "TMPDIR=/tmp/ray_run" \
  --env "NLTK_DATA=/stage/.cache/nltk_data" \
  --env "HOSTED_VLLM_API_BASE=http://ceres-cs-aus-447.reviz.ai2.in:8001/v1" \
  --env "HOSTED_VLLM_API_KEY=$HOSTED_VLLM_API_KEY" \
  --env "VLLM_DISABLE_COMPILE_CACHE=1" \
  --env "NCCL_DEBUG=ERROR" \
  --env "VLLM_LOGGING_LEVEL=WARNING" \
  "$CONTAINER_IMAGE" \
  bash -c "mkdir -p /tmp/ray_run && cd /stage && uv run python -c \"import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)\" && uv run python open_instruct/grpo_fast.py \
  --exp_name pipelinerl_7b_olmo3_thinker_test2 \
  --queue_dashboard_port 8765 \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 4 \
  --num_unique_prompts_rollout 16 \
  --num_mini_batches 2 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 2 \
  --output_dir /stage/output/olmo3_7b_think_rl \
  --rollouts_save_path /stage/output/olmo3_7b_think_rl/rollouts \
  --dataset_local_cache_dir /stage/.cache/open_instruct_dataset_cache \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 1024 \
  --response_length 4096 \
  --pack_length 5120 \
  --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
  --chat_template_name olmo_thinker \
  --non_stop_penalty False \
  --mask_truncated_completions False \
  --temperature 1.0 \
  --ground_truths_key ground_truth \
  --sft_messages_key prompt \
  --total_episodes 128 \
  --deepspeed_stage 3 \
  --num_learners_per_node 4 \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_enforce_eager \
  --vllm_sync_backend nccl \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --llm_judge_model hosted_vllm/Qwen/Qwen3-4B-Instruct-2507 \
  --llm_judge_timeout 600 \
  --llm_judge_max_tokens 2048 \
  --llm_judge_max_context_length 8192 \
  --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
  --code_pass_rate_reward_threshold 0.99 \
  --seed 1 \
  --local_eval_every 2 \
  --save_freq 10 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --gradient_checkpointing \
  --with_tracking \
  --clip_higher 0.272 \
  --checkpoint_state_freq 0 \
  --backend_timeout 1200 \
  --inflight_updates true \
  --async_steps 8 \
  --advantage_normalization_type centered \
  --truncated_importance_sampling_ratio_cap 2.0 \
  --push_to_hub false"
