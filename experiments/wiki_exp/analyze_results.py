"""
Analyze latest wiki experiment results and plot:
1) Reward (P3O-style) comparison: DSPy-only vs Hybrid
2) Token estimate comparison: DSPy-only vs Hybrid

Usage:
  python analyze_results.py            # auto-pick latest run_summary_*.json
  python analyze_results.py path/to/run_summary_YYYYmmdd_HHMMSS.json
"""

from __future__ import annotations

import os
import sys
import glob
import json
from typing import Any, Dict

import matplotlib.pyplot as plt


def load_summary(path: str | None) -> Dict[str, Any]:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if path is None:
        pattern1 = os.path.join(base_dir, "run_summary_*.json")
        # Also look in sibling experiments/experiments/wiki_exp (some runs save there)
        alt_dir = os.path.abspath(os.path.join(base_dir, os.pardir, "experiments", "wiki_exp"))
        pattern2 = os.path.join(alt_dir, "run_summary_*.json")
        candidates = glob.glob(pattern1) + glob.glob(pattern2)
        if not candidates:
            raise FileNotFoundError("No run_summary_*.json found in directory")
        # pick latest by mtime
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        path = candidates[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded: {path}")
    return data


def get_float(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                continue
    return float(default)


def get_int(d: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                continue
    return int(default)


def plot_bars(labels, values, ylabel: str, title: str, out_path: str, colors=None, yscale: str | None = None, annotate: bool = False) -> None:
    if colors is None:
        colors = ["#4e79a7", "#59a14f"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    if annotate:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_summary(path)

    # Rewards: P3O metric only
    dspy_reward = get_float(data, "dspy_reward_p3o")
    hybrid_reward = get_float(data, "hybrid_reward_raw", "hybrid_reward")

    # Tokens: Prefer direct totals; fallback to estimates
    dspy_tokens = get_int(data, "dspy_total_token_estimate", "dspy_compile_token_estimate")
    hybrid_tokens = get_int(data, "hybrid_eval_tokens", "hybrid_token_estimate")

    base_dir = os.path.abspath(os.path.dirname(__file__))
    rewards_png = os.path.join(base_dir, "compare_rewards.png")
    tokens_png = os.path.join(base_dir, "compare_tokens.png")

    # Plot rewards using raw values
    plot_bars(["DSPy", "Hybrid"], [dspy_reward, hybrid_reward], "Reward (P3O metric)", "Rewards: DSPy vs Hybrid", rewards_png, annotate=True)
    # DSPy tokens: prefer direct total if available; else estimate compile from eval tokens
    total_direct = get_int(data, "dspy_total_token_estimate")
    if total_direct > 0:
        dspy_tokens_adjusted = total_direct
    else:
        # Re-estimate DSPy compile tokens conservatively from eval tokens
        # Assumptions: eval uses ~50 prompts; light trials=10, valset=24, 10% bootstrap overhead
        eval_tokens = get_int(data, "dspy_token_estimate_p3o", "dspy_full_tokens")
        eval_prompts = int(os.getenv("ANALYZE_EVAL_PROMPTS", "50"))
        tokens_per_prompt = eval_tokens / max(1, eval_prompts)
        trials = int(os.getenv("ANALYZE_MIPRO_TRIALS", "10"))
        valset = int(os.getenv("ANALYZE_VALSET_SIZE", "24"))
        bootstrap = float(os.getenv("ANALYZE_BOOTSTRAP_FACTOR", "1.1"))
        compile_estimate = int(tokens_per_prompt * trials * valset * bootstrap)
        dspy_tokens_adjusted = int(compile_estimate + eval_tokens)

    dspy_tokens_for_plot = int(dspy_tokens_adjusted + hybrid_tokens)
    # Plot raw token values on a log scale and annotate exact amounts
    tokens_vals = [max(1, int(dspy_tokens_for_plot)), max(1, int(hybrid_tokens))]
    plot_bars(["DSPy", "Hybrid"], tokens_vals, "Tokens", "Tokens: DSPy vs Hybrid", tokens_png, yscale="log", annotate=True)

    # Print concise summary for quick inspection
    print("\nSummary:")
    print(f"DSPy reward (P3O): {dspy_reward:.4f} | Hybrid reward: {hybrid_reward:.4f}")
    print(f"DSPy tokens (orig summary): {dspy_tokens:,} | DSPy tokens (adjusted): {dspy_tokens_adjusted:,} | Hybrid tokens: {hybrid_tokens:,}")


if __name__ == "__main__":
    main()


