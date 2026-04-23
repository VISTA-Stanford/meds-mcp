#!/usr/bin/env python3
"""
Plot the word-count distribution of vignettes in patients.jsonl.

Writes a 2-panel figure: histogram + ECDF. Also prints summary stats.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]

WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text)) if text else 0


def percentile(sorted_vals: list[int], p: float) -> int:
    """Nearest-rank percentile for a sorted list (0 <= p <= 100)."""
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--patients",
        type=Path,
        default=_REPO_ROOT
        / "experiments/fewshot_with_labels/outputs/patients.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT
        / "experiments/fewshot_with_labels/outputs/vignette_word_dist.png",
    )
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    if not args.patients.exists():
        sys.exit(f"Missing patients file: {args.patients}")

    counts: list[int] = []
    n_empty = 0
    with open(args.patients, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            v = (obj.get("vignette") or "").strip()
            if not v:
                n_empty += 1
                continue
            counts.append(count_words(v))

    if not counts:
        sys.exit("No non-empty vignettes found.")

    counts_sorted = sorted(counts)
    n = len(counts)
    mean_v = sum(counts) / n
    stats = {
        "n_with_vignette": n,
        "n_empty": n_empty,
        "min": counts_sorted[0],
        "p10": percentile(counts_sorted, 10),
        "p25": percentile(counts_sorted, 25),
        "median": percentile(counts_sorted, 50),
        "p75": percentile(counts_sorted, 75),
        "p90": percentile(counts_sorted, 90),
        "p99": percentile(counts_sorted, 99),
        "max": counts_sorted[-1],
        "mean": round(mean_v, 1),
    }

    print("Word-count distribution for vignettes in", args.patients)
    w = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f"  {k:<{w}}  {v}")

    # ----- plot -----
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

    ax_h.hist(counts, bins=args.bins, edgecolor="white", linewidth=0.4)
    ax_h.set_title(f"Vignette word count (n={n})")
    ax_h.set_xlabel("Words per vignette")
    ax_h.set_ylabel("Number of vignettes")
    ax_h.grid(axis="y", linestyle=":", alpha=0.5)
    for p, style in [("median", "solid"), ("p90", "dashed")]:
        ax_h.axvline(stats[p], color="crimson", linestyle=style, linewidth=1,
                     label=f"{p} = {stats[p]}")
    ax_h.axvline(stats["mean"], color="darkorange", linestyle="dotted", linewidth=1,
                 label=f"mean = {stats['mean']}")
    ax_h.legend(frameon=False, fontsize=9)

    # ECDF
    ys = [(i + 1) / n for i in range(n)]
    ax_c.plot(counts_sorted, ys, linewidth=1.2)
    ax_c.set_title("ECDF")
    ax_c.set_xlabel("Words per vignette")
    ax_c.set_ylabel("Cumulative fraction")
    ax_c.set_ylim(0, 1)
    ax_c.grid(linestyle=":", alpha=0.5)
    for p in (0.5, 0.9, 0.99):
        ax_c.axhline(p, color="gray", linewidth=0.6, linestyle=":")

    fig.suptitle(
        f"Vignette word-count distribution  "
        f"(n={n}, empty={n_empty}, median={stats['median']}, p90={stats['p90']}, max={stats['max']})",
        fontsize=11,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
