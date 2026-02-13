#!/usr/bin/env python3
"""
Analyze vista_bench experiment results: LLM vs LLM+tool.
Produces AUROC tables, CIs, disagreement plots, and comparison visualizations.

Requirements: polars, scikit-learn, matplotlib, seaborn.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meds_mcp.experiments.task_config import BINARY_TASKS

# Tasks to exclude from all graphs
EXCLUDED_TASKS = {"lab_hyperlipidemia", "lab_hypertension"}


def load_results(jsonl_path: Path) -> pl.DataFrame:
    """Load JSONL results, exclude ERROR rows."""
    df = pl.read_ndjson(jsonl_path)
    df = df.filter(
        pl.col("llm_only_normalized").is_not_null()
        & pl.col("llm_plus_tool_normalized").is_not_null()
        & ~pl.col("llm_only_normalized").str.starts_with("[ERROR")
        & ~pl.col("llm_plus_tool_normalized").str.starts_with("[ERROR")
    )
    return df


def to_binary_series(s: pl.Series) -> np.ndarray:
    """yes/true -> 1, no/false -> 0."""
    lower = s.str.to_lowercase().str.strip_chars()
    pos = (lower == "yes") | (lower == "true")
    return pos.cast(pl.Int64).to_numpy()


def to_cat_numeric(s) -> float:
    """severe=3, moderate=2, mild=1, normal=0."""
    d = {"severe": 3, "moderate": 2, "mild": 1, "normal": 0}
    if s is None:
        return np.nan
    return d.get(str(s).strip().lower(), np.nan)


def bootstrap_auc(y_true, y_score, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for AUROC."""
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.metrics import roc_auc_score

    np.random.seed(seed)
    idx = np.arange(len(y_true))
    aucs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for _ in range(n_bootstrap):
            bi = np.random.choice(idx, size=len(idx), replace=True)
            try:
                a = roc_auc_score(y_true[bi], y_score[bi])
                aucs.append(a)
            except ValueError:
                pass
    aucs = np.array(aucs)
    if len(aucs) == 0:
        return np.nan, np.nan, np.nan
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/vista_bench_20260211_200248.jsonl",
        help="JSONL results file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Output directory for plots and tables",
    )
    args = parser.parse_args()

    print("=== analyze_vista_bench_results ===")
    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")
    print("\nLoading results...")
    df = load_results(Path(args.input))
    print(f"  Loaded {df.height} rows")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {out_dir}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns  # noqa: F401

        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False
        print("matplotlib not available; skipping plots")

    df = df.with_columns(
        (pl.col("llm_only_normalized") == pl.col("llm_plus_tool_normalized")).alias("agree"),
        (pl.col("llm_only_normalized") == pl.col("ground_truth_normalized")).alias("llm_correct"),
        (pl.col("llm_plus_tool_normalized") == pl.col("ground_truth_normalized")).alias("tool_correct"),
    )
    df = df.filter(~pl.col("task_name").is_in(list(EXCLUDED_TASKS)))
    disagree = df.filter(~pl.col("agree"))

    # --- 0. Patients per task ---
    patients_per_task = df.group_by("task_name").len(name="n_patients").sort("task_name")
    patients_per_task.write_csv(out_dir / "patients_per_task.csv")
    print("\n0. Patients per task saved to", out_dir / "patients_per_task.csv")

    # --- 1. AUROC table with CIs (binary tasks) ---
    print("\n1. Computing AUROC with bootstrap CIs (binary tasks)...")
    binary_tasks = [t for t in BINARY_TASKS if t in df["task_name"].to_list()]
    auroc_rows = []
    for i, task in enumerate(binary_tasks):
        print(f"   AUROC task {i+1}/{len(binary_tasks)}: {task}")
        sub = df.filter(pl.col("task_name") == task).drop_nulls(
            subset=["ground_truth_normalized", "llm_only_normalized", "llm_plus_tool_normalized"]
        )
        if sub.height < 10:
            continue
        y_true = to_binary_series(sub["ground_truth_normalized"])
        y_llm = to_binary_series(sub["llm_only_normalized"])
        y_tool = to_binary_series(sub["llm_plus_tool_normalized"])

        try:
            auc_llm, lo_llm, hi_llm = bootstrap_auc(y_true, y_llm)
            auc_tool, lo_tool, hi_tool = bootstrap_auc(y_true, y_tool)
            auroc_rows.append(
                {
                    "task": task,
                    "auroc_llm": round(auc_llm, 4),
                    "ci_llm": f"[{lo_llm:.3f}, {hi_llm:.3f}]",
                    "auroc_llm_tool": round(auc_tool, 4),
                    "ci_tool": f"[{lo_tool:.3f}, {hi_tool:.3f}]",
                    "n": sub.height,
                }
            )
        except Exception as e:
            auroc_rows.append({"task": task, "error": str(e), "n": len(sub)})

    auroc_df = pl.DataFrame(auroc_rows)
    if auroc_df.height > 0:
        auroc_df.write_csv(out_dir / "auroc_by_task.csv")
        print("1. AUROC table saved to", out_dir / "auroc_by_task.csv")

    # --- 2. Bar plot: AUROC LLM vs LLM+tool ---
    print("\n2. Plotting AUROC comparison...")
    if HAS_PLOT and auroc_df.height > 0 and "auroc_llm" in auroc_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(auroc_df.height)
        w = 0.35
        ax.bar(x - w / 2, auroc_df["auroc_llm"].to_list(), w, label="LLM only", color="steelblue")
        ax.bar(x + w / 2, auroc_df["auroc_llm_tool"].to_list(), w, label="LLM+tool", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(auroc_df["task"].to_list(), rotation=45, ha="right")
        ax.set_ylabel("AUROC")
        ax.legend()
        ax.set_title("AUROC: LLM vs LLM+tool (binary tasks)")
        plt.tight_layout()
        plt.savefig(out_dir / "auroc_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("2. AUROC bar plot saved to", out_dir / "auroc_comparison.png")

    # --- 3. Disagreement vectors ---
    print("\n3. Computing disagreement vectors...")
    vec_data = []
    for task in df["task_name"].unique().to_list():
        d = disagree.filter(pl.col("task_name") == task)
        vec_data.append(
            {
                "task": task,
                "tool_helped": d.filter(pl.col("tool_correct") & ~pl.col("llm_correct")).height,
                "tool_hurt": d.filter(pl.col("llm_correct") & ~pl.col("tool_correct")).height,
                "both_wrong": d.filter(~pl.col("llm_correct") & ~pl.col("tool_correct")).height,
            }
        )
    vec_df = pl.DataFrame(vec_data)
    vec_df.write_csv(out_dir / "disagreement_vectors.csv")

    if HAS_PLOT and disagree.height > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(vec_df.height)
        w = 0.25
        ax.bar(x - w, vec_df["tool_helped"].to_list(), w, label="Tool helped", color="green", alpha=0.8)
        ax.bar(x, vec_df["tool_hurt"].to_list(), w, label="Tool hurt", color="red", alpha=0.8)
        ax.bar(x + w, vec_df["both_wrong"].to_list(), w, label="Both wrong", color="gray", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(vec_df["task"].to_list(), rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Disagreement breakdown: Tool helped vs hurt vs both wrong")
        plt.tight_layout()
        plt.savefig(out_dir / "disagreement_vectors.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("3. Disagreement vectors saved to", out_dir / "disagreement_vectors.png")

    # --- 4. Agreement vs disagreement per task ---
    print("\n4. Agreement vs disagreement per task...")
    agree_counts = (
        df.group_by("task_name")
        .agg(
            pl.col("agree").sum().alias("agree"),
            pl.col("agree").count().alias("total"),
        )
        .with_columns((pl.col("total") - pl.col("agree")).alias("disagree"))
    )
    agree_counts.write_csv(out_dir / "agreement_by_task.csv")

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(agree_counts.height)
        w = 0.4
        ax.bar(x - w / 2, agree_counts["agree"].to_list(), w, label="Agree", color="steelblue")
        ax.bar(x + w / 2, agree_counts["disagree"].to_list(), w, label="Disagree", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(agree_counts["task_name"].to_list(), rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Agreement vs Disagreement: LLM vs LLM+tool")
        plt.tight_layout()
        plt.savefig(out_dir / "agreement_disagreement.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("4. Agreement/disagreement plot saved to", out_dir / "agreement_disagreement.png")

    # --- 5a. Accuracy comparison ---
    print("\n5a. Accuracy comparison...")
    acc_rows = []
    for task in df["task_name"].unique().to_list():
        sub = df.filter(pl.col("task_name") == task)
        acc_llm = (sub["llm_only_normalized"] == sub["ground_truth_normalized"]).mean()
        acc_tool = (sub["llm_plus_tool_normalized"] == sub["ground_truth_normalized"]).mean()
        acc_llm = acc_llm if acc_llm is not None else 0.0
        acc_tool = acc_tool if acc_tool is not None else 0.0
        acc_rows.append({"task": task, "acc_llm": acc_llm, "acc_tool": acc_tool, "n": sub.height})
    acc_df = pl.DataFrame(acc_rows)
    acc_df.write_csv(out_dir / "accuracy_by_task.csv")
    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(acc_df.height)
        w = 0.35
        ax.bar(x - w / 2, acc_df["acc_llm"].to_list(), w, label="LLM only", color="steelblue")
        ax.bar(x + w / 2, acc_df["acc_tool"].to_list(), w, label="LLM+tool", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(acc_df["task"].to_list(), rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Accuracy: LLM vs LLM+tool")
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("5a. Accuracy plot saved to", out_dir / "accuracy_comparison.png")

    # --- 5b. Tool net effect ---
    print("\n5b. Tool net effect...")
    vec_df = vec_df.with_columns((pl.col("tool_helped") - pl.col("tool_hurt")).alias("net"))
    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(vec_df.height)
        net_vals = vec_df["net"].to_list()
        colors = ["green" if n > 0 else "red" if n < 0 else "gray" for n in net_vals]
        ax.bar(x, net_vals, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(vec_df["task"].to_list(), rotation=45, ha="right")
        ax.set_ylabel("Net effect (helped - hurt)")
        ax.set_title("Tool net effect per task (when LLM and tool disagreed)")
        plt.tight_layout()
        plt.savefig(out_dir / "tool_net_effect.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("5b. Tool net effect saved to", out_dir / "tool_net_effect.png")

    # --- 5c. Summary ---
    print("\n5c. Writing summary...")
    summary = {
        "total_rows": df.height,
        "agreed": int(df["agree"].sum()),
        "disagreed": int((~df["agree"]).sum()),
        "both_correct": int((df["llm_correct"] & df["tool_correct"]).sum()),
        "llm_only_correct": int((df["llm_correct"] & ~df["tool_correct"]).sum()),
        "tool_only_correct": int((df["tool_correct"] & ~df["llm_correct"]).sum()),
        "both_wrong": int((~df["llm_correct"] & ~df["tool_correct"]).sum()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved to", out_dir / "summary.json")
    print("\nDone. Outputs in", out_dir)


if __name__ == "__main__":
    main()
