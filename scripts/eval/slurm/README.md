# Slurm LM Eval

Run the OLMo-paper benchmark suite on Slurm without Beaker, using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) with a **vLLM backend** (8-GPU tensor-parallel inference).

## Tasks

Matches the OLMo-Think paper eval where possible. Tasks not available in lm-eval harness (`Zebra Grid`, `CHE`, `LiveCodeBench`) are omitted.

| Metric | lm-eval task name |
|---|---|
| MMLU | `mmlu` |
| BBH | `bbh` |
| GPQA | `gpqa_diamond` |
| AGI Eval | `agieval` |
| AIME 2024 | `aime24` |
| AIME 2025 | `aime25` |
| IFEval | `ifeval` |

## Prerequisites

```bash
uv sync --extra lm-eval
```

`vllm` is a core project dependency and is picked up automatically by the vLLM backend.

## Usage

### Eval a single checkpoint or final model

```bash
sbatch scripts/eval/slurm/run_slurm_lm_eval.sh \
  output/MY-RUN \
  output/eval/MY-RUN-final

# specific checkpoint
sbatch scripts/eval/slurm/run_slurm_lm_eval.sh \
  output/MY-RUN/checkpoints/step_100 \
  output/eval/MY-RUN-step-100
```

### Sweep all checkpoints sequentially

Submits one Slurm job per checkpoint, chained with `--dependency=afterok` so they
run one after another (avoids multiple vLLM instances competing for GPU memory).

```bash
bash scripts/eval/slurm/run_slurm_lm_eval_sweep.sh \
  output/MY-RUN \
  output/eval/MY-RUN
```

By default this also evaluates the final (non-checkpoint) model at the end of the chain.

```bash
# Checkpoints only, no final model
EVAL_FINAL=0 bash scripts/eval/slurm/run_slurm_lm_eval_sweep.sh MODEL_BASE_DIR OUTPUT_BASE_DIR
```

### Custom task list

```bash
LMEVAL_TASKS="mmlu,bbh,gpqa_diamond" \
  sbatch scripts/eval/slurm/run_slurm_lm_eval.sh MODEL_PATH OUTPUT_DIR
```

### Available tasks

List all lm-eval tasks: `uv run lm_eval tasks list`

## Container image

Uses `docker://helffml/open_instruct_dev:slr` (same image as training). It has
vLLM, CUDA drivers and `uv` pre-installed; set `CONTAINER_IMAGE` to override.

## Output

Results are written to `OUTPUT_DIR` as JSON (one file per task). Per-document
inputs/outputs are also saved via `--log_samples`.

## vs OE Eval (Beaker)

| | Slurm LM Eval | OE Eval (Beaker) |
|---|---|---|
| Benchmarks | lm-eval tasks (MMLU, BBH, GPQA, …) | oe-eval-internal tasks |
| Infrastructure | Slurm, 1 × 8-GPU node, vLLM | Beaker jobs |
| Requires | lm-eval extra, checkpoint path | oe-eval-internal, Beaker |
