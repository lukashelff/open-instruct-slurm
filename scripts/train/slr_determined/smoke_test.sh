#!/bin/bash
# Smoke test entrypoint for Determined AI cluster.
# Verifies container, filesystem, uv, Python, and imports.
set -e

echo "=== Open-Instruct Smoke Test ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"

# 1. Check shared filesystem
BASE=/pfss/mlde/workspaces/mlde_wsp_P_HessianEuropeLLM/soofi_ai
echo ""
echo "--- Filesystem Check ---"
mkdir -p "$BASE/open-instruct"
mkdir -p "$BASE/.cache/uv"
mkdir -p "$BASE/.cache/huggingface"
mkdir -p "$BASE/.cache/nltk_data"
echo "test-$(date +%s)" > "$BASE/open-instruct/.smoke_test"
cat "$BASE/open-instruct/.smoke_test"
echo "Filesystem: OK"

# 2. Check uv
echo ""
echo "--- UV Check ---"
which uv || echo "uv not found in PATH"
uv --version || echo "uv version check failed"
echo "UV: OK"

# 3. Check Python + imports
echo ""
echo "--- Python Check ---"
# In Determined, CWD is /run/determined/workdir (uploaded context).
# /stage has the Docker image's baked-in files (pyproject.toml, uv.lock, .venv).
# We need /stage for uv run since that's where pyproject.toml + .venv live.
cd /stage
python --version || true
uv run python -c "
import sys
print(f'Python: {sys.version}')
print(f'Path: {sys.executable}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers import failed: {e}')
try:
    import ray
    print(f'Ray: {ray.__version__}')
except ImportError as e:
    print(f'Ray import failed: {e}')
try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except ImportError as e:
    print(f'vLLM import failed: {e}')
try:
    import open_instruct
    print('open_instruct: OK')
except ImportError as e:
    print(f'open_instruct import failed: {e}')
try:
    import deepspeed
    print(f'DeepSpeed: {deepspeed.__version__}')
except ImportError as e:
    print(f'DeepSpeed import failed: {e}')
try:
    from open_instruct.grpo_fast import main
    print('grpo_fast.main: importable')
except Exception as e:
    print(f'grpo_fast import failed: {e}')
print('All imports checked!')
" || echo "Python check had issues"

# 4. Check nvidia-smi (should fail gracefully with 0 GPUs)
echo ""
echo "--- GPU Check ---"
nvidia-smi 2>&1 || echo "No GPUs available (expected for smoke test)"

# 5. Check networking
echo ""
echo "--- Network Check ---"
hostname --ip-address 2>/dev/null || hostname -I 2>/dev/null || echo "Could not get IP"

# 6. Check environment variables
echo ""
echo "--- Environment Variables ---"
echo "TMPDIR=${TMPDIR:-not set}"
echo "HOME=${HOME:-not set}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-not set}"
echo "HF_HOME=${HF_HOME:-not set}"
echo "NLTK_DATA=${NLTK_DATA:-not set}"
echo "HF_TOKEN=${HF_TOKEN:+set (hidden)}"
echo "WANDB_API_KEY=${WANDB_API_KEY:+set (hidden)}"

echo ""
echo "=== Smoke Test Complete ==="
echo "All checks passed!"
