# Determined AI Cluster Configuration for Open-Instruct GRPO Training
# ====================================================================
#
# This directory contains experiment configs and scripts for running
# GRPO training on the TU Darmstadt Determined AI cluster.
#
# ## Quick Start
#
# 1. **Set up credentials** (one-time):
#    ```bash
#    export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080
#    source secrets.env  # or export DET_USER/DET_PASS manually
#    det user login $DET_USER
#    ```
#
# 2. **Copy secrets to shared filesystem** (one-time):
#    The secrets.env must also exist at the BASE_DIR on the cluster's
#    shared filesystem so containers can load API keys:
#    ```
#    /pfss/mlde/workspaces/mlde_wsp_P_HessianEuropeLLM/soofi_ai/open-instruct/secrets.env
#    ```
#
# 3. **Run experiments** (from repo root):
#    ```bash
#    # No-GPU smoke test (verify container + imports)
#    det experiment create scripts/train/slr_determined/smoke_test.yml .
#
#    # Single-GPU GRPO test
#    det experiment create scripts/train/slr_determined/single_gpu_grpo.yml .
#
#    # Full multi-node (8 nodes, 64 GPUs)
#    det experiment create scripts/train/slr_determined/multi_node_grpo.yml .
#    ```
#
# ## Architecture
#
# The setup mirrors the Slurm configuration:
#
# - **smoke_test.yml** — 0 GPUs, verifies container boots, filesystem mounts,
#   uv works, all Python imports succeed.
#
# - **single_gpu_grpo.yml** — 1 GPU, runs minimal GRPO with a small model
#   (OLMo-2-1B) and tiny batch sizes. Fast debug cycle (~10 min).
#
# - **multi_node_grpo.yml** — 64 GPUs (8 nodes), full IsomorphicRL training
#   with OLMo-3 7B Think model, judge, and code API.
#
# ## Key Differences from Slurm
#
# | Slurm | Determined |
# |-------|-----------|
# | `SLURM_PROCID` | `SLURM_PROCID` or `OMPI_COMM_WORLD_RANK` |
# | `scontrol show hostnames` | `DET_CHIEF_IP` |
# | Apptainer container | Docker image |
# | `sbatch` | `det experiment create` |
# | Manual node binding | `bind_mounts` in YAML |
#
# ## Files
#
# - `smoke_test.yml` + `smoke_test.sh` — No-GPU smoke test
# - `single_gpu_grpo.yml` + `single_gpu_grpo.sh` — Single-GPU debug
# - `multi_node_grpo.yml` + `entrypoint.sh` — Full multi-node training
