#!/usr/bin/env python3
"""
Plot SLR-Bench prompt token lengths per dataset level (config).

Uses the same tokenizer and chat template as the GRPO SLR training script
(Olmo-3-7B-Think-DPO + olmo_thinker) so lengths match what you get at training time.

Usage:
  uv run python scripts/train/slr/plot_slr_bench_prompt_lengths.py
  uv run python scripts/train/slr/plot_slr_bench_prompt_lengths.py --out slr_lengths.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root so we can import open_instruct
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets import load_dataset

from open_instruct.dataset_transformation import TokenizerConfig


# SLR-Bench configs (levels) on Hugging Face — skip v1-All since it's just a concat of the others
SLR_CONFIGS = ["v1-Basic", "v1-Easy", "v1-Medium", "v1-Hard"]
DATASET_NAME = "AIML-TUDA/SLR-Bench"
SPLIT = "train"


def get_prompt_token_lengths(tokenizer, dataset, max_samples: int | None = None):
    """Return list of prompt token lengths (after chat template + add_generation_prompt).

    When max_samples is set and smaller than the dataset, rows are randomly sampled
    (not taken from the front) to avoid bias from ordered datasets.
    """
    import random

    lengths = []
    errors = 0
    total = len(dataset)
    if max_samples is not None and max_samples < total:
        indices = sorted(random.sample(range(total), max_samples))
    else:
        indices = range(total)
    for i in indices:
        row = dataset[i]
        prompt = row.get("prompt")
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            lengths.append(0)
            continue
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        try:
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors=None
            )
            lengths.append(len(ids))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Warning: tokenization failed for row {i}: {e}")
            lengths.append(-1)
    if errors > 3:
        print(f"  ... and {errors - 3} more tokenization errors")
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Plot SLR-Bench prompt token lengths per level.")
    parser.add_argument(
        "--model",
        default="allenai/Olmo-3-7B-Think-DPO",
        help="Tokenizer model (default: same as olmo3-think-rl SLR script)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max samples per config to tokenize (default: 500). Use 0 for all.",
    )
    parser.add_argument(
        "--out",
        default="slr_bench_prompt_lengths.png",
        help="Output plot path (default: slr_bench_prompt_lengths.png)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Only print stats, do not save plot.")
    args = parser.parse_args()

    # Same tokenizer setup as grpo SLR script (Olmo-3: add_bos=False for olmo_thinker)
    tc = TokenizerConfig(
        tokenizer_name_or_path=args.model,
        chat_template_name="olmo_thinker",
        get_tokenizer_fn="get_tokenizer_tulu_v2_2",
        add_bos=False,
    )
    tokenizer = tc.tokenizer

    config_stats = {}
    for config in SLR_CONFIGS:
        try:
            ds = load_dataset(DATASET_NAME, config, split=SPLIT)
        except Exception as e:
            print(f"Warning: could not load {DATASET_NAME} config={config}: {e}")
            continue
        max_n = None if args.max_samples == 0 else args.max_samples
        print(f"Tokenizing {config} ({len(ds)} rows, using {max_n or len(ds)})...")
        lengths = get_prompt_token_lengths(tokenizer, ds, max_samples=max_n)
        lengths = [x for x in lengths if x >= 0]
        if not lengths:
            config_stats[config] = {"n": 0, "lengths": []}
            continue
        config_stats[config] = {
            "n": len(lengths),
            "lengths": lengths,
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        }
        sl = sorted(lengths)
        config_stats[config]["p50"] = sl[len(sl) // 2]
        config_stats[config]["p95"] = sl[int(len(sl) * 0.95)] if len(sl) > 1 else sl[0]
        config_stats[config]["p99"] = sl[int(len(sl) * 0.99)] if len(sl) > 1 else sl[0]

    # Print table
    print("\nSLR-Bench prompt token lengths (Olmo-3 chat template, add_generation_prompt=True)")
    print("=" * 90)
    for config in SLR_CONFIGS:
        if config not in config_stats or config_stats[config]["n"] == 0:
            continue
        s = config_stats[config]
        print(
            f"  {config:12}  n={s['n']:5}  min={s['min']:6}  max={s['max']:6}  mean={s['mean']:7.1f}  "
            f"p50={s['p50']:6}  p95={s['p95']:6}  p99={s['p99']:6}"
        )
    print("=" * 90)
    all_max = max((s["max"] for s in config_stats.values() if s["n"] > 0), default=0)
    print(f"  Overall max prompt length: {all_max} tokens. Recommend --max_prompt_token_length > {all_max}.\n")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot. Use --no-plot to avoid this.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: boxplot per level (one per tier, no v1-All which is just a concat)
    ax = axes[0]
    data = [config_stats[c]["lengths"] for c in SLR_CONFIGS if c in config_stats and config_stats[c]["n"] > 0]
    labels = [c for c in SLR_CONFIGS if c in config_stats and config_stats[c]["n"] > 0]
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True, widths=0.6)
        colors = ["#a8d5e2", "#f9d56e", "#f3a683", "#e66767"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Prompt token length")
        ax.set_title("Per difficulty level")
        ax.tick_params(axis="x", rotation=15)
        # Show some reference lines
        for threshold, label, color in [(2048, "2048", "gray"), (4096, "4096", "blue"), (8192, "8192", "green")]:
            if threshold < ax.get_ylim()[1] * 1.1:
                ax.axhline(y=threshold, color=color, linestyle="--", alpha=0.5, label=label)
        ax.legend(fontsize=8)

    # Right: overlaid histograms per level
    ax = axes[1]
    colors_hist = ["#a8d5e2", "#f9d56e", "#f3a683", "#e66767"]
    for i, config in enumerate(SLR_CONFIGS):
        if config in config_stats and config_stats[config]["n"] > 0:
            ax.hist(
                config_stats[config]["lengths"],
                bins=40,
                alpha=0.5,
                edgecolor="black",
                linewidth=0.5,
                label=f"{config} (n={config_stats[config]['n']})",
                color=colors_hist[i],
            )
    ax.set_xlabel("Prompt token length")
    ax.set_ylabel("Count")
    ax.set_title("Distribution per level")
    for threshold, color in [(2048, "red"), (4096, "blue")]:
        ax.axvline(x=threshold, color=color, linestyle="--", alpha=0.6, label=str(threshold))
    ax.legend(fontsize=8)

    plt.suptitle(f"SLR-Bench prompt lengths ({args.model})", fontsize=12, y=1.02)
    plt.tight_layout()
    outpath = Path(args.out)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {outpath}")


if __name__ == "__main__":
    main()
