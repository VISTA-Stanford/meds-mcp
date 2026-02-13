#!/usr/bin/env python3
"""
Plot context length analysis: patients maxing out token limit.

Produces 2 plots:
  1. Simple bar: % of patients exceeding token budget (All history vs Single visit)
  2. Stacked bar: Within budget vs exceeding budget for both strategies

Run analyze_context_lengths.py first to generate the input JSON.

Usage:
  python scripts/plot_context_analysis.py --input results/context_length_analysis.json
  python scripts/plot_context_analysis.py --output-dir results/analysis/context_plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_context_analysis(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_exceeding_budget(data: dict, out_dir: Path, max_ev: int) -> None:
    """Simple bar chart: % of patients exceeding token budget."""
    all_counts = data["all_history_prior_to_t"]["raw_counts"]
    single_counts = data["single_visit"]["raw_counts"]
    n = len(all_counts)

    all_exceed = sum(1 for c in all_counts if c > max_ev) / n * 100
    single_exceed = sum(1 for c in single_counts if c > max_ev) / n * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["All history", "Single visit"],
        [all_exceed, single_exceed],
        color=["steelblue", "coral"],
    )
    ax.set_ylabel("% of patients exceeding token budget")
    ax.set_title(f"Patients maxing out token limit ({max_ev} events)")
    ax.set_ylim(0, max(all_exceed, single_exceed) * 1.2 or 10)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(out_dir / "context_exceeding_budget.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("1. Saved", out_dir / "context_exceeding_budget.png")


def plot_within_vs_exceeding(data: dict, out_dir: Path, max_ev: int) -> None:
    """Stacked bar: within budget vs exceeding for All history vs Single visit."""
    all_counts = data["all_history_prior_to_t"]["raw_counts"]
    single_counts = data["single_visit"]["raw_counts"]
    n = len(all_counts)

    all_exceed = sum(1 for c in all_counts if c > max_ev) / n * 100
    single_exceed = sum(1 for c in single_counts if c > max_ev) / n * 100
    all_within = 100 - all_exceed
    single_within = 100 - single_exceed

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0, 1]
    ax.bar(x, [all_within, single_within], label="Within budget", color="green", alpha=0.8)
    ax.bar(
        x,
        [all_exceed, single_exceed],
        bottom=[all_within, single_within],
        label="Exceeds budget (maxed out)",
        color="red",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["All history", "Single visit"])
    ax.set_ylabel("% of patients")
    ax.set_title(f"Token budget: {max_ev} events")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "context_within_vs_exceeding.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("2. Saved", out_dir / "context_within_vs_exceeding.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/context_length_analysis.json",
        help="Context analysis JSON from analyze_context_lengths.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/context_plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}")
        print("Run analyze_context_lengths.py first to generate the context analysis JSON.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_context_analysis(inp)
    max_ev = data["metadata"]["max_events_for_budget"]

    plot_exceeding_budget(data, out_dir, max_ev)
    plot_within_vs_exceeding(data, out_dir, max_ev)

    print(f"\nDone. Plots in {out_dir}")


if __name__ == "__main__":
    main()
