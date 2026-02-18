# Slurm OE Eval

Run standard benchmarks (MMLU, GSM8K, BBH, etc.) on Slurm without Beaker, using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

## Prerequisites

```bash
uv sync --extra lm-eval
```

## Usage

### Eval final model

```bash
sbatch scripts/eval/slurm/run_slurm_lm_eval.sh \
  output/RLVR-soofi-Basev2-Isomorphic-RLv2 \
  output/eval/RLVR-soofi-final
```

### Eval checkpoint at step N

```bash
sbatch scripts/eval/slurm/run_slurm_lm_eval.sh \
  output/RLVR-soofi-Basev2-Isomorphic-RLv2/checkpoints/step_100 \
  output/eval/RLVR-soofi-step-100
```

### Custom task list

```bash
LMEVAL_TASKS="mmlu,gsm8k,bbh,truthfulqa_mc2,humaneval" \
  sbatch scripts/eval/slurm/run_slurm_lm_eval.sh MODEL_PATH OUTPUT_DIR
```

### Available tasks

Common lm-eval tasks: `mmlu`, `gsm8k`, `gsm8k_cot`, `bbh`, `truthfulqa_mc2`, `hendrycksTest-*`, `humaneval`, `mbpp`, `arc_challenge`, `hellaswag`, etc.

List all tasks: `uv run lm_eval ls tasks`

## Output

Results are saved to `OUTPUT_DIR` as JSON. Use `--log_samples` to also save per-document inputs/outputs.

## vs OE Eval (Beaker)

| | Slurm OE Eval | OE Eval (Beaker) |
|---|---|---|
| Benchmarks | lm-eval tasks (MMLU, GSM8K, BBH, …) | oe-eval-internal tasks |
| Infrastructure | Slurm, single node | Beaker jobs |
| Requires | lm-eval, checkpoint path | oe-eval-internal, Beaker |
