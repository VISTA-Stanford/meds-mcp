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

from meds_mcp.experiments.task_config import (
    ALL_TASKS,
    BINARY_TASKS,
    get_csv_path_for_task,
    get_labels_dir,
    is_binary_task,
)

# Tasks to exclude from all graphs (new_hyperlipidemia and new_hypertension are binary and included)
EXCLUDED_TASKS: set = set()

# For class balance: population (500) vs subset (100)
POPULATION_SIZE = 500
SUBSET_SIZE = 100


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


def load_task_labels_for_balance(task_name: str):
    """Load task CSV for class balance. Tries main labels dir then labels_100 subdir."""
    csv_path = get_csv_path_for_task(task_name)
    if csv_path.exists():
        return pl.read_csv(csv_path)
    alt = get_labels_dir() / "labels_100" / csv_path.name
    if alt.exists():
        return pl.read_csv(alt)
    return None


def _is_lab_binary_labels(labels: pl.Series) -> bool:
    """True if all non-null labels are binary (true/false/0/1/yes/no)."""
    uniq = labels.drop_nulls().unique().to_list()
    binary_vals = {"true", "false", "0", "1", "yes", "no"}
    return all(str(v).strip().lower() in binary_vals for v in uniq) if uniq else False


def compute_class_balance(df_sub: pl.DataFrame, task_name: str) -> dict:
    """Return counts per class (all tasks are binary: pos/neg)."""
    if df_sub.height == 0:
        return {}
    labels = df_sub["label"].cast(pl.Utf8).str.to_lowercase().str.strip_chars()
    pos = (labels == "true") | (labels == "yes") | (labels == "1")
    neg = (labels == "false") | (labels == "no") | (labels == "0")
    return {"positive": int(pos.sum()), "negative": int(neg.sum())}


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

    # --- 0b. Class balance per task (population 500 vs subset 100) ---
    print("\n0b. Class balance per task (population vs subset)...")
    balance_table_rows = []
    for task_name in ALL_TASKS:
        if task_name in EXCLUDED_TASKS:
            continue
        task_df = load_task_labels_for_balance(task_name)
        if task_df is None:
            continue
        pop = task_df.head(POPULATION_SIZE)
        sub = task_df.head(SUBSET_SIZE)
        bal_pop = compute_class_balance(pop, task_name)
        bal_sub = compute_class_balance(sub, task_name)
        if not bal_pop:
            continue
        task_type = "binary" if ("positive" in bal_pop) else "categorical"
        row = {"task": task_name, "task_type": task_type}
        if "positive" in bal_pop:
            n_pos_p, n_neg_p = bal_pop.get("positive", 0), bal_pop.get("negative", 0)
            n_pos_s, n_neg_s = bal_sub.get("positive", 0), bal_sub.get("negative", 0)
            tot_p = n_pos_p + n_neg_p
            tot_s = n_pos_s + n_neg_s
            row["pct_positive_population"] = round(100 * n_pos_p / tot_p, 2) if tot_p else ""
            row["pct_positive_subset"] = round(100 * n_pos_s / tot_s, 2) if tot_s else ""
            row["ratio_population"] = ""
            row["ratio_subset"] = ""
        else:
            cats = ["normal", "mild", "moderate", "severe"]
            row["pct_positive_population"] = ""
            row["pct_positive_subset"] = ""
            row["ratio_population"] = ":".join(str(bal_pop.get(c, 0)) for c in cats)
            row["ratio_subset"] = ":".join(str(bal_sub.get(c, 0)) for c in cats)
        balance_table_rows.append(row)
    if balance_table_rows:
        balance_df = pl.DataFrame(balance_table_rows)
        balance_df.write_csv(out_dir / "class_balance_by_task.csv")
        print("0b. Class balance table saved to", out_dir / "class_balance_by_task.csv")

    # --- 1. AUROC table with CIs (binary tasks) ---
    print("\n1. Computing AUROC with bootstrap CIs (binary tasks)...")
    tasks_in_results = df["task_name"].unique().to_list()
    binary_tasks = [
        t for t in tasks_in_results
        if t not in EXCLUDED_TASKS and t in BINARY_TASKS
    ]
    auroc_rows = []
    for i, task in enumerate(binary_tasks):
        print(f"   AUROC task {i+1}/{len(binary_tasks)}: {task}")
        sub = df.filter(pl.col("task_name") == task).drop_nulls(
            subset=["ground_truth_normalized", "llm_only_normalized", "llm_plus_tool_normalized"]
        )
        if sub.height < 10:
            continue
        y_true = to_binary_series(sub["ground_truth_normalized"])
        # Use continuous scores P(yes) when available for proper AUROC; else binary predictions
        def score_or_binary(score_col: str, norm_col: str, y_true_arr: np.ndarray) -> tuple:
            if score_col in sub.columns and sub[score_col].null_count() < sub.height:
                s = sub[score_col].fill_null(np.nan).to_numpy()
                mask = ~np.isnan(s)
                if mask.sum() >= 10:
                    return y_true_arr[mask], s[mask], True
            return y_true_arr, to_binary_series(sub[norm_col]), False

        y_true_llm, y_llm, _ = score_or_binary("llm_only_score", "llm_only_normalized", y_true)
        y_true_tool, y_tool, _ = score_or_binary("llm_plus_tool_score", "llm_plus_tool_normalized", y_true)
        if len(y_true_llm) < 10 or len(y_true_tool) < 10:
            continue

        try:
            auc_llm, lo_llm, hi_llm = bootstrap_auc(y_true_llm, y_llm)
            auc_tool, lo_tool, hi_tool = bootstrap_auc(y_true_tool, y_tool)
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

    # --- 5a2. Balanced accuracy (LLM vs LLM+tool) ---
    print("\n5a2. Balanced accuracy...")
    from sklearn.metrics import balanced_accuracy_score

    bal_acc_rows = []
    for task in df["task_name"].unique().to_list():
        sub = df.filter(pl.col("task_name") == task).drop_nulls(
            subset=["ground_truth_normalized", "llm_only_normalized", "llm_plus_tool_normalized"]
        )
        if sub.height < 5:
            continue
        y_true = sub["ground_truth_normalized"]
        y_llm = sub["llm_only_normalized"]
        y_tool = sub["llm_plus_tool_normalized"]
        if task in BINARY_TASKS:
            y_true_np = to_binary_series(y_true)
            y_llm_np = to_binary_series(y_llm)
            y_tool_np = to_binary_series(y_tool)
        else:
            y_true_np = np.array([to_cat_numeric(v) for v in y_true])
            y_llm_np = np.array([to_cat_numeric(v) for v in y_llm])
            y_tool_np = np.array([to_cat_numeric(v) for v in y_tool])
        mask = ~(np.isnan(y_true_np) | np.isnan(y_llm_np) | np.isnan(y_tool_np))
        if mask.sum() < 5:
            continue
        y_true_np = y_true_np[mask].astype(int)
        y_llm_np = y_llm_np[mask].astype(int)
        y_tool_np = y_tool_np[mask].astype(int)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bal_llm = balanced_accuracy_score(y_true_np, y_llm_np)
                bal_tool = balanced_accuracy_score(y_true_np, y_tool_np)
            bal_acc_rows.append({"task": task, "bal_acc_llm": bal_llm, "bal_acc_tool": bal_tool, "n": int(mask.sum())})
        except Exception:
            pass
    bal_acc_df = pl.DataFrame(bal_acc_rows)
    if bal_acc_df.height > 0:
        bal_acc_df.write_csv(out_dir / "balanced_accuracy_by_task.csv")
        if HAS_PLOT:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(bal_acc_df.height)
            w = 0.35
            ax.bar(x - w / 2, bal_acc_df["bal_acc_llm"].to_list(), w, label="LLM only", color="steelblue")
            ax.bar(x + w / 2, bal_acc_df["bal_acc_tool"].to_list(), w, label="LLM+tool", color="coral")
            ax.set_xticks(x)
            ax.set_xticklabels(bal_acc_df["task"].to_list(), rotation=45, ha="right")
            ax.set_ylabel("Balanced accuracy")
            ax.set_title("Balanced accuracy: LLM vs LLM+tool")
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "balanced_accuracy_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()
        print("5a2. Balanced accuracy saved to", out_dir / "balanced_accuracy_by_task.csv")

    # --- 5b. Tool executed / tool exposed (% of cases where tool was used) ---
    print("\n5b. Tool executed vs tool exposed...")
    if "tool_executions" in df.columns and HAS_PLOT:
        tool_pct = (
            df.group_by("task_name")
            .agg(
                pl.len().alias("exposed"),
                (pl.col("tool_executions") > 0).sum().alias("executed"),
            )
            .with_columns((100 * pl.col("executed") / pl.col("exposed")).round(1).alias("pct_executed"))
            .sort("task_name")
        )
        if tool_pct.height > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(tool_pct.height)
            ax.bar(x, tool_pct["pct_executed"].to_list(), color="steelblue", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(tool_pct["task_name"].to_list(), rotation=45, ha="right")
            ax.set_ylabel("% tool executed / tool exposed")
            ax.set_title("Percentage of cases where tool was executed (given tool exposed)")
            plt.tight_layout()
            plt.savefig(out_dir / "tool_executed_vs_exposed.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("5b. Tool executed vs exposed saved to", out_dir / "tool_executed_vs_exposed.png")

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
