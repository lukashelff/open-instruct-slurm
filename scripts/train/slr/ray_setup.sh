#!/bin/bash
# Ray node setup for Slurm (SLR) runs.
# Adapted from configs/beaker_configs/ray_node_setup.sh for Slurm/Apptainer.
#
# Dispatches based on SLURM_PROCID:
#   SLURM_PROCID == $RAY_HEAD_PROCID (default 1):  Ray head node
#   Otherwise:                                       Ray worker node
#
# Usage: source this script from within the srun bash block.
#   Required env vars:
#     RAY_ADDRESS  - head address (e.g. 10.0.0.1:6379), set for workers
#     RAY_PORT     - port for the Ray head (e.g. 6379)
#   Optional env vars:
#     RAY_HEAD_PROCID  - SLURM_PROCID of the head node (default: 1)
#     RAY_WORKER_DELAY - base delay before workers join (default: 15)

set -e
RAY_PORT="${RAY_PORT:?RAY_PORT must be set}"
RAY_HEAD_PROCID="${RAY_HEAD_PROCID:-1}"
RAY_WORKER_DELAY="${RAY_WORKER_DELAY:-15}"
PROC_ID="${SLURM_PROCID:-0}"


# --- Common setup ---
uv run ray stop --force 2>/dev/null || true

if [ "$PROC_ID" = "$RAY_HEAD_PROCID" ]; then
    # --- Head node ---
    echo "[ray_setup] Starting Ray head (SLURM_PROCID=$PROC_ID, port=$RAY_PORT)"
    uv run ray start --head --port="$RAY_PORT" --dashboard-host=0.0.0.0
    /usr/bin/sleep 5
    echo "[ray_setup] Ray head started."

    # The caller is responsible for running the training script after sourcing this.
    # Example:
    #   source scripts/train/slr/ray_setup.sh
    #   uv run python open_instruct/grpo_fast.py $GRPO_ARGS || true
    #   uv run ray stop --force 2>/dev/null || true

else
    # --- Worker node ---
    echo "[ray_setup] Starting Ray worker (SLURM_PROCID=$PROC_ID, head=$RAY_ADDRESS)"

    # Trap signals so workers exit 0 when head shuts down (prevents Slurm FAILED status)
    cleanup() {
        echo "[ray_setup] Worker $PROC_ID: cleanup — stopping Ray"
        uv run ray stop --force 2>/dev/null || true
        trap - TERM INT HUP EXIT
        exit 0
    }
    trap cleanup TERM INT HUP EXIT

    # Stagger worker joins to avoid thundering herd on head
    DELAY=$((RAY_WORKER_DELAY + PROC_ID * 2))
    echo "[ray_setup] Worker $PROC_ID: waiting ${DELAY}s before joining cluster"
    /usr/bin/sleep "$DELAY"

    uv run ray start --address="$RAY_ADDRESS" --dashboard-host=0.0.0.0
    echo "[ray_setup] Worker $PROC_ID: joined cluster, monitoring head at $RAY_ADDRESS"

    # Poll head availability. Exit 0 when head is gone.
    while uv run ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; do
        /usr/bin/sleep 5
    done
    echo "[ray_setup] Worker $PROC_ID: head unreachable, exiting."
    cleanup
fi
