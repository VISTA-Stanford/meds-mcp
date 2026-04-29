#!/usr/bin/env python3
"""
Analyze and visualize batch experiment results for EHRSHOT and VISTA.

Reads the most-recent experiment_results_{context}_{run_id}.jsonl files from
outputs/ehrshot/ and outputs/vista/, computes per-task accuracy / flip / fix /
hurt statistics, writes analysis_summary.json and per_task_accuracy.csv, and
produces four figures:

  fig1_overall_accuracy.png     -- grouped bar: baseline vs fewshot per dataset
  fig2_per_task_ehrshot.png     -- per-task bars for EHRSHOT
  fig3_per_task_vista.png       -- per-task bars for VISTA  (2 panels)
  fig4_scatter_delta.png        -- scatter: baseline acc vs fewshot acc per task
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS  = BASE_DIR / "outputs"
PLOTS    = OUTPUTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DATASETS  = ["ehrshot", "vista"]
CONTEXTS  = ["baseline_vignette", "vignette"]
LABEL_MAP = {"baseline_vignette": "Baseline", "vignette": "Few-shot (k=3)"}
DS_LABEL  = {"ehrshot": "EHRSHOT", "vista": "VISTA"}

COLORS = {
    "baseline_vignette": "#5B8DB8",   # steel blue
    "vignette":          "#E07B39",   # burnt orange
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_file(dataset: str, context: str) -> Path:
    """Return the most-recently written results file for this (dataset, context)."""
    pattern = f"experiment_results_{context}_*.jsonl"
    candidates = sorted((OUTPUTS / dataset).glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No results file matching {pattern} under {OUTPUTS / dataset}"
        )
    return candidates[-1]


def load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def accuracy_on_parsed(rows: list[dict]) -> float:
    parsed = [r for r in rows if r["pred"] is not None]
    if not parsed:
        return float("nan")
    return sum(r["correct"] for r in parsed) / len(parsed)


def per_task_metrics(rows: list[dict]) -> dict[str, dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    out = {}
    for task, recs in by_task.items():
        parsed = [r for r in recs if r["pred"] is not None]
        n_total   = len(recs)
        n_parsed  = len(parsed)
        n_correct = sum(r["correct"] for r in parsed)
        out[task] = {
            "n_total":   n_total,
            "n_parsed":  n_parsed,
            "n_correct": n_correct,
            "accuracy":  n_correct / n_parsed if n_parsed else float("nan"),
            "parse_rate": n_parsed / n_total if n_total else float("nan"),
        }
    return out


def compare_baseline_vs_fewshot(
    baseline: list[dict],
    fewshot:  list[dict],
) -> dict:
    """Row-level flip / fix / hurt by (person_id, task)."""
    key = lambda r: (str(r["person_id"]), r["task"])
    b_map = {key(r): r for r in baseline if r["pred"] is not None}
    f_map = {key(r): r for r in fewshot  if r["pred"] is not None}
    shared = sorted(set(b_map) & set(f_map))
    n = len(shared)
    if n == 0:
        return {"n_shared": 0}

    b_correct = sum(bool(b_map[k]["correct"]) for k in shared)
    f_correct = sum(bool(f_map[k]["correct"]) for k in shared)
    flips = sum(
        1 for k in shared
        if not b_map[k]["correct"] and f_map[k]["correct"]
    )
    hurts = sum(
        1 for k in shared
        if b_map[k]["correct"] and not f_map[k]["correct"]
    )
    b_wrong = sum(1 for k in shared if not b_map[k]["correct"])

    return {
        "n_shared":              n,
        "baseline_acc":          round(b_correct / n, 4),
        "fewshot_acc":           round(f_correct / n, 4),
        "delta_acc":             round((f_correct - b_correct) / n, 4),
        "flip_rate":             round(flips / n * 100, 2),
        "fix_rate_given_wrong":  round(flips / b_wrong * 100, 2) if b_wrong else None,
        "hurt_rate":             round(hurts / n * 100, 2),
    }


def per_task_comparison(
    b_metrics: dict[str, dict],
    f_metrics: dict[str, dict],
) -> dict[str, dict]:
    tasks = sorted(set(b_metrics) | set(f_metrics))
    out = {}
    for t in tasks:
        bm = b_metrics.get(t, {})
        fm = f_metrics.get(t, {})
        b_acc = bm.get("accuracy", float("nan"))
        f_acc = fm.get("accuracy", float("nan"))
        out[t] = {
            "baseline_acc":  b_acc,
            "fewshot_acc":   f_acc,
            "delta":         f_acc - b_acc if not (np.isnan(b_acc) or np.isnan(f_acc)) else float("nan"),
            "n_baseline":    bm.get("n_total", 0),
            "n_fewshot":     fm.get("n_total", 0),
        }
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis() -> dict:
    results: dict[str, dict[str, list[dict]]] = {}
    for ds in DATASETS:
        results[ds] = {}
        for ctx in CONTEXTS:
            path = latest_file(ds, ctx)
            print(f"  Loading {ds}/{ctx}: {path.name}")
            results[ds][ctx] = load_results(path)

    summary = {}
    for ds in DATASETS:
        b_rows = results[ds]["baseline_vignette"]
        f_rows = results[ds]["vignette"]
        b_task = per_task_metrics(b_rows)
        f_task = per_task_metrics(f_rows)

        summary[ds] = {
            "overall": {
                "baseline_vignette": {
                    "n_total":  len(b_rows),
                    "n_parsed": sum(r["pred"] is not None for r in b_rows),
                    "accuracy": round(accuracy_on_parsed(b_rows) * 100, 2),
                },
                "vignette": {
                    "n_total":  len(f_rows),
                    "n_parsed": sum(r["pred"] is not None for r in f_rows),
                    "accuracy": round(accuracy_on_parsed(f_rows) * 100, 2),
                },
            },
            "comparison": compare_baseline_vs_fewshot(b_rows, f_rows),
            "per_task_baseline": b_task,
            "per_task_fewshot":  f_task,
            "per_task_comparison": per_task_comparison(b_task, f_task),
        }

    return summary, results


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(summary: dict) -> None:
    path = OUTPUTS / "per_task_accuracy.csv"
    rows = []
    for ds in DATASETS:
        ptc = summary[ds]["per_task_comparison"]
        for task, m in sorted(ptc.items()):
            rows.append({
                "dataset":       ds,
                "task":          task,
                "n":             m["n_baseline"],
                "baseline_acc":  round(m["baseline_acc"] * 100, 2) if not np.isnan(m["baseline_acc"]) else "",
                "fewshot_acc":   round(m["fewshot_acc"] * 100, 2)  if not np.isnan(m["fewshot_acc"])  else "",
                "delta_acc":     round(m["delta"] * 100, 2)         if not np.isnan(m["delta"])         else "",
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset","task","n","baseline_acc","fewshot_acc","delta_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV → {path}")


# ---------------------------------------------------------------------------
# Figure 1: Overall accuracy — grouped bars
# ---------------------------------------------------------------------------

def fig_overall(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    datasets = DATASETS
    x = np.arange(len(datasets))
    w = 0.35

    b_accs = [summary[ds]["overall"]["baseline_vignette"]["accuracy"] for ds in datasets]
    f_accs = [summary[ds]["overall"]["vignette"]["accuracy"]          for ds in datasets]

    bars_b = ax.bar(x - w/2, b_accs, w, label="Baseline", color=COLORS["baseline_vignette"], zorder=3)
    bars_f = ax.bar(x + w/2, f_accs, w, label="Few-shot (k=3)", color=COLORS["vignette"], zorder=3)

    for bar in list(bars_b) + list(bars_f):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Delta annotations
    for i, ds in enumerate(datasets):
        delta = f_accs[i] - b_accs[i]
        sign  = "+" if delta >= 0 else ""
        color = "#2a9d2a" if delta >= 0 else "#d62728"
        ax.text(
            x[i], max(b_accs[i], f_accs[i]) + 3.5,
            f"Δ {sign}{delta:.1f}%",
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[ds] for ds in datasets], fontsize=12)
    ax.set_ylabel("Accuracy (%) on parsed responses", fontsize=11)
    ax.set_title("Baseline vs Few-shot Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 90)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Parse rate footnote
    notes = []
    for ds in datasets:
        b = summary[ds]["overall"]["baseline_vignette"]
        f = summary[ds]["overall"]["vignette"]
        notes.append(
            f"{DS_LABEL[ds]} — baseline parse rate: {b['n_parsed']}/{b['n_total']}; "
            f"few-shot parse rate: {f['n_parsed']}/{f['n_total']}"
        )
    fig.text(0.01, -0.04, "\n".join(notes), fontsize=7, color="grey", va="top")

    fig.tight_layout()
    out = PLOTS / "fig1_overall_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1 → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Per-task accuracy — EHRSHOT
# ---------------------------------------------------------------------------

def fig_per_task_ehrshot(summary: dict) -> None:
    ptc = summary["ehrshot"]["per_task_comparison"]
    tasks = sorted(ptc.keys())
    b_accs = [ptc[t]["baseline_acc"] * 100 for t in tasks]
    f_accs = [ptc[t]["fewshot_acc"]  * 100 for t in tasks]

    x = np.arange(len(tasks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - w/2, b_accs, w, label="Baseline",       color=COLORS["baseline_vignette"], zorder=3)
    ax.bar(x + w/2, f_accs, w, label="Few-shot (k=3)", color=COLORS["vignette"],          zorder=3)

    # Colour task labels by delta
    deltas = [ptc[t]["delta"] * 100 for t in tasks]
    tick_colors = ["#2a9d2a" if d >= 0 else "#d62728" for d in deltas]

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=8)
    for tick, col in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(col)

    ax.set_ylabel("Accuracy (%) on parsed responses", fontsize=10)
    ax.set_title("EHRSHOT — Per-task Accuracy: Baseline vs Few-shot", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Subtitle: green = fewshot helps, red = hurts
    fig.text(0.5, -0.02,
             "Task label colour: green = few-shot ≥ baseline, red = few-shot < baseline",
             ha="center", fontsize=8, color="grey")

    fig.tight_layout()
    out = PLOTS / "fig2_per_task_ehrshot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2 → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Per-task accuracy — VISTA (split into 2 panels of 20)
# ---------------------------------------------------------------------------

def fig_per_task_vista(summary: dict) -> None:
    ptc = summary["vista"]["per_task_comparison"]
    tasks = sorted(ptc.keys())
    mid   = len(tasks) // 2

    for panel_idx, task_slice in enumerate([tasks[:mid], tasks[mid:]]):
        b_accs = [ptc[t]["baseline_acc"] * 100 for t in task_slice]
        f_accs = [ptc[t]["fewshot_acc"]  * 100 for t in task_slice]
        deltas = [ptc[t]["delta"] * 100        for t in task_slice]

        x = np.arange(len(task_slice))
        w = 0.38
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.bar(x - w/2, b_accs, w, label="Baseline",       color=COLORS["baseline_vignette"], zorder=3)
        ax.bar(x + w/2, f_accs, w, label="Few-shot (k=3)", color=COLORS["vignette"],          zorder=3)

        tick_colors = ["#2a9d2a" if d >= 0 else "#d62728" for d in deltas]
        ax.set_xticks(x)
        ax.set_xticklabels(task_slice, rotation=40, ha="right", fontsize=8)
        for tick, col in zip(ax.get_xticklabels(), tick_colors):
            tick.set_color(col)

        ax.set_ylabel("Accuracy (%) on parsed responses", fontsize=10)
        ax.set_title(
            f"VISTA — Per-task Accuracy: Baseline vs Few-shot  (panel {panel_idx+1}/2)",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        fig.text(0.5, -0.02,
                 "Task label colour: green = few-shot ≥ baseline, red = few-shot < baseline",
                 ha="center", fontsize=8, color="grey")

        fig.tight_layout()
        out = PLOTS / f"fig3_per_task_vista_panel{panel_idx+1}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  fig3 panel {panel_idx+1} → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Scatter — baseline acc vs fewshot acc, coloured by dataset
# ---------------------------------------------------------------------------

def fig_scatter_delta(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    ds_colors = {"ehrshot": "#7B4FA0", "vista": "#2D7A4F"}

    for ax, ds in zip(axes, DATASETS):
        ptc = summary[ds]["per_task_comparison"]
        tasks = sorted(ptc.keys())
        b_accs = np.array([ptc[t]["baseline_acc"] * 100 for t in tasks])
        f_accs = np.array([ptc[t]["fewshot_acc"]  * 100 for t in tasks])
        deltas = f_accs - b_accs

        scatter_colors = ["#2a9d2a" if d >= 0 else "#d62728" for d in deltas]
        sc = ax.scatter(b_accs, f_accs, c=scatter_colors, s=70, alpha=0.85, zorder=3,
                        edgecolors="white", linewidths=0.5)

        # Diagonal y=x reference line
        lo = min(b_accs.min(), f_accs.min()) - 3
        hi = max(b_accs.max(), f_accs.max()) + 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x (no change)")

        # Label a few points: biggest improvements and biggest regressions
        order = np.argsort(deltas)
        highlight = list(order[:2]) + list(order[-2:])
        for i in highlight:
            ax.annotate(
                tasks[i], (b_accs[i], f_accs[i]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=6.5, alpha=0.85,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Baseline accuracy (%)", fontsize=10)
        ax.set_ylabel("Few-shot accuracy (%)", fontsize=10)
        ax.set_title(f"{DS_LABEL[ds]} — per-task accuracy scatter", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        above = int((deltas > 0).sum())
        below = int((deltas < 0).sum())
        equal = int((deltas == 0).sum())
        ax.text(0.03, 0.97,
                f"↑ few-shot better: {above} tasks\n→ tied: {equal}\n↓ baseline better: {below} tasks",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    green_patch = mpatches.Patch(color="#2a9d2a", label="Few-shot ≥ Baseline")
    red_patch   = mpatches.Patch(color="#d62728", label="Baseline > Few-shot")
    fig.legend(handles=[green_patch, red_patch], fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Per-task: Baseline vs Few-shot Accuracy", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = PLOTS / "fig4_scatter_delta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4 → {out}")


# ---------------------------------------------------------------------------
# Figure 5: Delta bar chart — sorted improvement per task, both datasets
# ---------------------------------------------------------------------------

def fig_delta_bars(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ds in zip(axes, DATASETS):
        ptc = summary[ds]["per_task_comparison"]
        tasks  = sorted(ptc.keys())
        deltas = np.array([ptc[t]["delta"] * 100 for t in tasks])
        order  = np.argsort(deltas)
        tasks_sorted  = [tasks[i]  for i in order]
        deltas_sorted = deltas[order]

        bar_colors = ["#2a9d2a" if d >= 0 else "#d62728" for d in deltas_sorted]
        y = np.arange(len(tasks_sorted))
        ax.barh(y, deltas_sorted, color=bar_colors, zorder=3, height=0.7)
        ax.axvline(0, color="black", lw=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(tasks_sorted, fontsize=8)
        ax.set_xlabel("Δ Accuracy (few-shot − baseline, %)", fontsize=10)
        ax.set_title(f"{DS_LABEL[ds]} — Per-task improvement from few-shot",
                     fontsize=11, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = PLOTS / "fig5_delta_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig5 → {out}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for ds in DATASETS:
        ov = summary[ds]["overall"]
        cmp = summary[ds]["comparison"]
        print(f"\n{DS_LABEL[ds]}")
        print(f"  Baseline : {ov['baseline_vignette']['accuracy']:.1f}%  "
              f"({ov['baseline_vignette']['n_parsed']}/{ov['baseline_vignette']['n_total']} parsed)")
        print(f"  Few-shot : {ov['vignette']['accuracy']:.1f}%  "
              f"({ov['vignette']['n_parsed']}/{ov['vignette']['n_total']} parsed)")
        if cmp.get("n_shared", 0) > 0:
            delta = (cmp['fewshot_acc'] - cmp['baseline_acc']) * 100
            sign  = "+" if delta >= 0 else ""
            print(f"  Delta    : {sign}{delta:.2f}pp  "
                  f"(on {cmp['n_shared']:,} rows with both conditions parsed)")
            print(f"  Fix rate : {cmp.get('fix_rate_given_wrong', 'N/A')}%  "
                  f"(of baseline-wrong rows fixed by few-shot)")
            print(f"  Hurt rate: {cmp['hurt_rate']}%  "
                  f"(of all shared rows hurt by few-shot)")

    print("\nPer-task deltas:")
    for ds in DATASETS:
        ptc = summary[ds]["per_task_comparison"]
        deltas = {t: m["delta"]*100 for t,m in ptc.items() if not np.isnan(m["delta"])}
        best  = max(deltas, key=deltas.get)
        worst = min(deltas, key=deltas.get)
        print(f"  {DS_LABEL[ds]}: best gain={best} ({deltas[best]:+.1f}%)  "
              f"worst={worst} ({deltas[worst]:+.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results …")
    summary, _ = run_analysis()

    print("Writing CSV …")
    write_csv(summary)

    out_path = OUTPUTS / "analysis_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON → {out_path}")

    print("Generating figures …")
    fig_overall(summary)
    fig_per_task_ehrshot(summary)
    fig_per_task_vista(summary)
    fig_scatter_delta(summary)
    fig_delta_bars(summary)

    print_summary(summary)
    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
