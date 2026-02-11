#!/bin/bash
# OLMo-3 7B Think RL (GRPO) - Slurm 9-node run: 2 nodes for training (learners), 7 nodes for inference (vLLM).
# grpo_fast.py uses a STRICT_SPREAD placement group: --num_learners_per_node 8 8 → 2 nodes (16 learners); --vllm_num_engines 56 → 7 nodes (56 vLLM engines).
# Task 0 = head (Ray head + grpo_fast.py); tasks 1–8 = Ray workers. Ray places learners on first 2 nodes, vLLM on remaining 7.
#SBATCH --job-name=SOOFI-RLVR
# (If you change RUN_NAME below, submit with: sbatch -J "$RUN_NAME" this_script.sh so the job name matches.)
#SBATCH --partition=all
#SBATCH --nodes=9
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=200:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal
#SBATCH --open-mode=append
JOB_NAME="RLVR-soofi-slr2"


# --- 1. Variables & Paths (define run name once; used for job name, exp_name, output and checkpoint dirs) ---
BASE_DIR="/mnt/vast/home/lh22zyta/shortcut-RL/open-instruct"
CONTAINER_IMAGE="docker://helffml/open_instruct_dev:slr"
OUTPUT_DIR="$BASE_DIR/output/$JOB_NAME"
RAY_PORT=6379

# --- 2. Slurm env (for logging) ---
echo "=========================================="
echo "RLVR Post-Training Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# --- 3. Head node and IP (for Ray cluster) ---
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" apptainer exec --nv "$CONTAINER_IMAGE" hostname --ip-address)
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "Ray head: $HEAD_NODE ($HEAD_IP:$RAY_PORT)"
echo "Port forwarding: ssh -L 8265:$HEAD_IP:8265 -L 8765:$HEAD_IP:8765 43_cluster  # then open http://localhost:8265 (Ray) and http://localhost:8765 (ActorManager)"

# --- 4. Environment variables (inside container) ---
export HOME="$BASE_DIR"
export TOKENIZERS_PARALLELISM=FALSE
export WANDB_ENTITY="helff"
export WANDB_PROJECT="Reward-Shortcut"
export WANDB_API_KEY="wandb_TOKEN"
export HF_TRUST_REMOTE_CODE=TRUE
export HF_TOKEN="HF_TOKEN"
export OPENAI_API_KEY="openAI_TOKEN"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_DEBUG=ERROR
export TRITON_CACHE_DIR="/tmp/triton"
export SLR_VERIFIER_DISABLE_TQDM=1

mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/triton" "$BASE_DIR/.cache/nltk_data" "$BASE_DIR/.cache/open_instruct_dataset_cache" "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

APPTAINER_ENV=(
  --bind "$BASE_DIR:/stage"
  --env "UV_CACHE_DIR=/stage/.cache/uv"
  --env "HF_HOME=/stage/.cache/huggingface"
  --env "TMPDIR=/tmp"
  --env "NLTK_DATA=/stage/.cache/nltk_data"
  --env "HOSTED_VLLM_API_BASE=http://ceres-cs-aus-447.reviz.ai2.in:8001/v1"
  --env "HOSTED_VLLM_API_KEY=${HOSTED_VLLM_API_KEY:-EMPTY}"
  --env "NCCL_DEBUG=ERROR"
  --env "VLLM_LOGGING_LEVEL=WARNING"
  --env "HEAD_NODE=$HEAD_NODE"
  --env "HEAD_IP=$HEAD_IP"
  --env "RAY_ADDRESS=$RAY_ADDRESS"
  --env "RAY_PORT=$RAY_PORT"
  --env "RAY_DEDUP_LOGS=0"
  --env "LITELLM_DEBUG=1"
)


# --- 5. One srun, N tasks: task 0 = head (Ray + grpo_fast.py), tasks 1–8 = workers. 2 nodes for learners, 7 for vLLM (grpo_fast placement). ---
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
  --dataset_local_cache_dir /stage/.cache/open_instruct_dataset_cache \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 \
  --dataset_mixer_eval_list_splits train \
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
  --vllm_num_engines 56 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.8 \
  --vllm_sync_backend nccl \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
  --llm_judge_timeout 1200 \
  --llm_judge_max_tokens 1048 \
  --llm_judge_max_context_length 32768 \
  --clip_higher 0.272 \
  --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
  --code_pass_rate_reward_threshold 0.99 \
  --seed 1 \
  --local_eval_every 50 \
  --save_freq 25 \
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
  --push_to_hub false \
  --oe_eval_tasks mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek"

# Do not pass SLURM_PROCID=... (script's value is unset; each srun task has its own in the environment). Container inherits it.
srun --nodes=9 --ntasks=9 apptainer exec --nv "${APPTAINER_ENV[@]}" "$CONTAINER_IMAGE" \
  bash -c '
    cd /stage
    mkdir -p /tmp/triton
    # Use SLURM_PROCID to pick head (task 0 = head); hostname can differ inside container
    if [ "${SLURM_PROCID:-0}" = "0" ]; then
      echo "Head: starting Ray head then grpo_fast.py"
      uv run ray start --head --port=$RAY_PORT --node-ip-address=$HEAD_IP --dashboard-host=0.0.0.0
      sleep 35
      uv run python -c "import nltk; nltk.download(\"punkt_tab\", quiet=True); nltk.download(\"punkt\", quiet=True)"
      uv run python open_instruct/grpo_fast.py '"$GRPO_ARGS"' || true
      uv run ray stop --force 2>/dev/null || true
    else
      echo "Worker: starting Ray worker (inference node)"
      sleep $((40 + SLURM_PROCID * 3))
      uv run ray start --address=$RAY_ADDRESS --dashboard-host=0.0.0.0
      while uv run ray status --address=$RAY_ADDRESS >/dev/null 2>&1; do sleep 5; done
      uv run ray stop --force 2>/dev/null || true
    fi
  '