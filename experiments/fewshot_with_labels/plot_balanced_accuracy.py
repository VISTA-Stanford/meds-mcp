#!/usr/bin/env python3
"""
Balanced accuracy bar chart for the readmission 100-patient test run.

Reads analysis_summary.json from the ehrshot output directory and produces
a grouped bar chart comparing baseline_vignette (zero-shot) vs vignette
(few-shot) on guo_readmission.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.fewshot_with_labels import _paths  # noqa: E402

BASELINE_COLOR = "#6B9FD4"   # muted blue
FEWSHOT_COLOR  = "#E8845A"   # muted orange
CHANCE_COLOR   = "#AAAAAA"

DISPLAY_NAMES = {
    "baseline_vignette": "Zero-shot\n(baseline_vignette)",
    "vignette":          "Few-shot\n(vignette)",
}

TASK_LABELS = {
    "guo_readmission":   "Readmission",
    "chexpert":          "CheXpert",
    "guo_icu":           "ICU",
    "guo_los":           "LOS",
    "lab_anemia":        "Anemia",
    "lab_hyperkalemia":  "Hyperkalemia",
    "lab_hypoglycemia":  "Hypoglycemia",
    "lab_hyponatremia":  "Hyponatremia",
    "lab_thrombocytopenia": "Thrombocytopenia",
    "new_acutemi":       "Acute MI",
    "new_celiac":        "Celiac",
    "new_hyperlipidemia": "Hyperlipidemia",
    "new_hypertension":  "Hypertension",
    "new_lupus":         "Lupus",
    "new_pancan":        "PanCan",
}


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_readmission_only(summary: dict, out_path: Path) -> None:
    """Single bar chart for baseline_vignette vs vignette on guo_readmission."""
    variants = ["baseline_vignette", "vignette"]
    colors   = [BASELINE_COLOR, FEWSHOT_COLOR]
    labels   = [DISPLAY_NAMES[v] for v in variants]

    per_variant = summary["per_variant"]
    values = []
    ns     = []
    for v in variants:
        task_data = per_variant[v]["per_task"].get("guo_readmission", {})
        ba = task_data.get("balanced_accuracy")
        n  = task_data.get("n", 0)
        values.append(ba if ba is not None else float("nan"))
        ns.append(n)

    # Also pull the shared-set comparison
    comp = summary.get("comparisons_vs_baseline", {}).get("vignette", {})
    shared_n          = comp.get("n_shared", 0)
    ba_shared_base    = comp.get("balanced_accuracy_baseline_on_shared")
    ba_shared_variant = comp.get("balanced_accuracy_variant_on_shared")
    delta_ba          = comp.get("delta_balanced_acc_variant_minus_baseline")

    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(variants))
    bars = ax.bar(x, values, width=0.5, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # Value labels on bars
    for bar, val, n in zip(bars, values, ns):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#222222",
            )

    # Chance line
    ax.axhline(0.5, color=CHANCE_COLOR, linewidth=1.4, linestyle="--", zorder=2,
               label="Chance (0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_ylim(0.35, 0.70)
    ax.set_title(
        "Readmission — Balanced Accuracy\n"
        f"100-patient test run · guo_readmission",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Delta annotation between bars
    if delta_ba is not None:
        sign = "+" if delta_ba >= 0 else ""
        mid_y = max(v for v in values if not np.isnan(v)) + 0.035
        ax.annotate(
            "",
            xy=(x[1], mid_y), xytext=(x[0], mid_y),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.3),
        )
        ax.text(
            (x[0] + x[1]) / 2, mid_y + 0.006,
            f"Δ={sign}{delta_ba:.4f}",
            ha="center", va="bottom", fontsize=9, color="#444444",
        )

    # Shared-set footnote
    if shared_n:
        footnote = (
            f"On {shared_n} shared patients: "
            f"baseline BA={ba_shared_base:.3f}, few-shot BA={ba_shared_variant:.3f}"
        )
        ax.text(
            0.5, -0.10, footnote,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#666666",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=1)

    chance_patch = mpatches.Patch(color=CHANCE_COLOR, label="Chance (0.50)")
    ax.legend(handles=[chance_patch], fontsize=8.5, loc="upper left",
              framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_all_tasks(summary: dict, out_path: Path) -> None:
    """Grouped bar chart across all tasks for variants that have enough data."""
    # Only show variants with meaningful n (≥ 10 patients per task)
    SHOW_VARIANTS = [
        ("baseline_vignette",          BASELINE_COLOR, "Zero-shot (n=100 readmission test)"),
        ("vignette",                   FEWSHOT_COLOR,  "Few-shot  (n=100 readmission test)"),
        ("baseline_vignette_20260427_185403", "#7BC8A4", "Zero-shot full EHRSHOT"),
        ("vignette_20260427_185403",   "#D4A0CC", "Few-shot  full EHRSHOT"),
    ]

    per_task = summary["per_task_across_variants"]
    all_tasks = sorted(per_task.keys())

    # Filter to tasks that have ≥ 1 valid non-small_n data point
    def has_valid(task):
        for vname, _, _ in SHOW_VARIANTS:
            cell = per_task[task].get(vname)
            if cell and not cell.get("small_n") and cell.get("balanced_accuracy") is not None:
                return True
        return False

    tasks = [t for t in all_tasks if has_valid(t)]
    n_tasks = len(tasks)
    n_variants = len(SHOW_VARIANTS)

    width = 0.18
    offsets = np.linspace(-(n_variants - 1) / 2 * width, (n_variants - 1) / 2 * width, n_variants)
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.2), 6))

    for (vname, color, label), offset in zip(SHOW_VARIANTS, offsets):
        vals = []
        for task in tasks:
            cell = per_task[task].get(vname)
            if cell and not cell.get("small_n") and cell.get("balanced_accuracy") is not None:
                vals.append(cell["balanced_accuracy"])
            else:
                vals.append(float("nan"))
        ax.bar(x + offset, vals, width=width, color=color, label=label,
               edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(0.5, color=CHANCE_COLOR, linewidth=1.2, linestyle="--",
               label="Chance (0.50)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=30,
                       ha="right", fontsize=9)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(
        "Balanced Accuracy by Task — EHRSHOT\n"
        "100-patient readmission test vs full EHRSHOT runs",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=1)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot balanced accuracy for ehrshot experiment")
    parser.add_argument("--input-dir", type=Path, default=_paths.outputs_dir() / "ehrshot")
    parser.add_argument("--output-dir", type=Path,
                        default=_paths.outputs_dir() / "ehrshot" / "plots")
    args = parser.parse_args()

    summary_path = args.input_dir / "analysis_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run analyze_results.py first.")
        sys.exit(1)

    summary = load_summary(summary_path)

    plot_readmission_only(
        summary,
        args.output_dir / "balanced_acc_readmission_100pt.png",
    )
    plot_all_tasks(
        summary,
        args.output_dir / "balanced_acc_all_tasks.png",
    )


if __name__ == "__main__":
    main()
