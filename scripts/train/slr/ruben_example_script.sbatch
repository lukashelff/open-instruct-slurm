#!/bin/bash
#SBATCH --job-name=smollm2_train_${LANG_CODE}
#SBATCH --partition=all    # Partition from docs 
#SBATCH --nodes=8                 # CHANGE HERE ONLY: 8 nodes * 8 GPUs = 64 slots
#SBATCH --gpus-per-node=8         # H100 nodes have 8 GPUs 
#SBATCH --ntasks-per-node=1       # One task per node to launch the container
#SBATCH --cpus-per-task=32        # Adjust based on CPU availability
#SBATCH --mem=250G              # memory
#SBATCH --time=17:00:00           # Walltime limit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive               # Full node access for DDP
#SBATCH --qos=normal              # Use normal QoS
#SBATCH --signal=B:USR1@120  # Send SIGUSR1 120 seconds before time limit
#SBATCH --open-mode=append   # Keep logs continuous across restarts

# --- 1. Variables & Paths ---
# Use the base dir you provided
BASE_DIR="/mnt/vast/workspaces/jackal_ai/clustering/evaluation"
SIF_IMAGE="/mnt/vast/workspaces/jackal_ai/uv42.sif" # Ensure you pulled the image here!

# Auto-detect node count from Slurm (Prevents mismatch!)
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
# Fallback if SLURM_GPUS_ON_NODE is undefined (rare but possible)
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=8
fi

echo "Master Node: $SLURM_JOB_NODELIST"
echo "Nodes: $NNODES | GPUs per node: $GPUS_PER_NODE"

# --- 2. Distributed Setup (Master Address) ---
# Get the hostname of the first node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export RDZV_ID=$RANDOM

# --- 3. Environment Variables ---
export TMPDIR="$BASE_DIR/tmp"
export HOME="$BASE_DIR" # Sets home inside container to your workspace
export HF_TOKEN="HF_TOKEN"
export WANDB_API_KEY="wandb_TOKEN"
export WANDB_ENTITY="occiglot"
export HF_TRUST_REMOTE_CODE=TRUE
export TOKENIZERS_PARALLELISM=FALSE
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export SSL_CERT_FILE='/mnt/vast/workspaces/jackal_ai/clustering/evaluation/.cache/cacert.pem'

# --- 4. Execution ---
# We bind /mnt/vast so the container can see your code and data.
# We use $NNODES and $GPUS_PER_NODE dynamically in the torchrun command.

# --- 5. Dynamic Config Generation ---
# We save the generated config in the tmp folder
mkdir -p "$BASE_DIR/tmp"
ACTIVE_CONFIG="$BASE_DIR/tmp/active_${RUN_NAME}_${SLURM_JOB_ID}.yaml"

# Multi-variable replacement
sed -e "s/{{LANG}}/${LANG_CODE}/g" \
    -e "s/{{STRATEGY}}/${STRATEGY}/g" \
    -e "s/{{VERSION}}/${VERSION}/g" \
    -e "s/{{RUN_NAME}}/${RUN_NAME}/g" \
    -e "s/{{SHUFFLE}}/${SHUFFLE}/g" \
    -e "s/{{DOC_MASKING}}/${DOC_MASKING}/g" \
    -e "s/{{THRESHOLD}}/${THRESHOLD}/g" \
    "$BASE_DIR/configs/config_smollm2_2b_X.yaml" > "$ACTIVE_CONFIG"

srun bash -c "apptainer exec --nv \
--bind /mnt/vast:/mnt/vast \
$SIF_IMAGE \
bash -c '\
cd $BASE_DIR && \
source ./nanotron_v2/bin/activate && \
torchrun \
--nnodes=$NNODES \
--nproc_per_node=$GPUS_PER_NODE \
--rdzv_id=$RDZV_ID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
$BASE_DIR/run_train_parquet.py \
--config-file $ACTIVE_CONFIG'"
