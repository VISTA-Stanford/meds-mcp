#!/usr/bin/env python3
"""
Analyze vista_bench experiment results: LLM vs LLM+tool.
Produces balanced accuracy comparison and LLM vs tool disagreement (how often
the LLM final answer differs from the tool/ground-truth label).

Requirements: polars, scikit-learn, matplotlib, seaborn.
"""

import argparse
import json
import os
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
    TASK_TO_FILENAME,
    get_csv_path_for_task,
    get_labels_dir,
    is_binary_task,
)

# Tasks to exclude from all graphs (new_hyperlipidemia and new_hypertension are binary and included)
EXCLUDED_TASKS: set = set()

# For class balance: population (500) vs subset (100)
POPULATION_SIZE = 500
SUBSET_SIZE = 100


def _value_to_yes_no(raw: str) -> str:
    """Normalize CSV 'value' column (true/false/1/0) to yes/no for comparison with LLM output."""
    if raw is None or not str(raw).strip():
        return None
    lower = str(raw).strip().lower()
    if lower in ("true", "1", "yes"):
        return "yes"
    if lower in ("false", "0", "no"):
        return "no"
    return None


def load_tool_value_lookup(labels_dir: Path, task_names: list) -> pl.DataFrame:
    """Load CSV 'value' column per (patient_id, prediction_time, task_name) and normalize to yes/no.
    Used for LLM vs tool disagreement: compare LLM+tool output to what the tool (value) returned.
    """
    rows = []
    for task in task_names:
        if task not in TASK_TO_FILENAME:
            continue
        csv_path = labels_dir / TASK_TO_FILENAME[task]
        if not csv_path.exists():
            continue
        try:
            tbl = pl.read_csv(csv_path)
        except Exception:
            continue
        if "value" not in tbl.columns or "patient_id" not in tbl.columns:
            continue
        pred_col = "prediction_time" if "prediction_time" in tbl.columns else None
        if not pred_col:
            continue
        val_lower = pl.col("value").str.to_lowercase().str.strip_chars()
        tool_val = (
            pl.when(val_lower.is_in(["true", "1", "yes"]))
            .then(pl.lit("yes"))
            .when(val_lower.is_in(["false", "0", "no"]))
            .then(pl.lit("no"))
            .otherwise(pl.lit(None))
        )
        tbl = tbl.select(
            pl.col("patient_id").cast(pl.Utf8),
            pl.col(pred_col).cast(pl.Utf8).alias("prediction_time"),
            pl.lit(task).alias("task_name"),
            tool_val.alias("tool_value_normalized"),
        )
        rows.append(tbl)
    if not rows:
        return pl.DataFrame(schema={"patient_id": pl.Utf8, "prediction_time": pl.Utf8, "task_name": pl.Utf8, "tool_value_normalized": pl.Utf8})
    return pl.concat(rows)


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
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Labels dir for task CSVs (value column); used for LLM vs tool disagreement. Default: VISTA_LABELS_DIR or repo default.",
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

    # Option 2: load CSV 'value' (tool output) for LLM vs tool disagreement
    if args.labels_dir is not None:
        os.environ["VISTA_LABELS_DIR"] = str(args.labels_dir)
    labels_dir = args.labels_dir if args.labels_dir is not None else get_labels_dir()
    value_lookup = load_tool_value_lookup(labels_dir, df["task_name"].unique().to_list())
    df = df.with_columns(
        pl.col("patient_id").cast(pl.Utf8),
        pl.col("prediction_time").cast(pl.Utf8),
    ).join(value_lookup, on=["patient_id", "prediction_time", "task_name"], how="left")
    n_with_tool_value = df.filter(pl.col("tool_value_normalized").is_not_null()).height
    print(f"  Tool value lookup: {n_with_tool_value} rows with value from CSVs (labels_dir={labels_dir})")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns  # noqa: F401

        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False
        print("matplotlib not available; skipping plots")

    # LLM vs tool disagree = final answer differs from CSV 'value' (what the tool returns), only where value present
    llm_vs_tool_disagree = (pl.col("llm_plus_tool_normalized") != pl.col("tool_value_normalized")) & pl.col("tool_value_normalized").is_not_null()
    df = df.with_columns(
        (pl.col("llm_only_normalized") == pl.col("llm_plus_tool_normalized")).alias("agree"),
        (pl.col("llm_only_normalized") == pl.col("ground_truth_normalized")).alias("llm_correct"),
        (pl.col("llm_plus_tool_normalized") == pl.col("ground_truth_normalized")).alias("tool_correct"),
        llm_vs_tool_disagree.alias("llm_vs_tool_disagree"),
    )
    df = df.filter(~pl.col("task_name").is_in(list(EXCLUDED_TASKS)))
    disagree = df.filter(~pl.col("agree"))

    # --- 1. LLM final answer vs tool call: disagreements per task (vs CSV 'value') ---
    print("\n1. LLM vs tool disagreement (LLM+tool output != CSV value / tool return)...")
    llm_vs_tool = (
        df.group_by("task_name")
        .agg(
            pl.col("llm_vs_tool_disagree").sum().alias("disagree_count"),
            pl.col("tool_value_normalized").is_not_null().sum().alias("with_tool_value"),
        )
        .with_columns(
            pl.when(pl.col("with_tool_value") > 0)
            .then(100.0 * pl.col("disagree_count") / pl.col("with_tool_value"))
            .otherwise(None)
            .round(2)
            .alias("disagree_pct")
        )
    )
    llm_vs_tool.write_csv(out_dir / "llm_vs_tool_disagreement_by_task.csv")
    total_disagree = int(df["llm_vs_tool_disagree"].sum())
    print(f"   Total rows where LLM final answer != tool (CSV value): {total_disagree} / {n_with_tool_value} (rows with tool value)")
    if HAS_PLOT and llm_vs_tool.height > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(llm_vs_tool.height)
        ax.bar(x, llm_vs_tool["disagree_count"].to_list(), color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(llm_vs_tool["task_name"].to_list(), rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("LLM final answer disagrees with tool call (per task)")
        plt.tight_layout()
        plt.savefig(out_dir / "llm_vs_tool_disagreement.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("1. LLM vs tool disagreement plot saved to", out_dir / "llm_vs_tool_disagreement.png")

    # --- 2. Balanced accuracy (LLM vs LLM+tool) ---
    print("\n2. Balanced accuracy...")
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
        print("2. Balanced accuracy saved to", out_dir / "balanced_accuracy_by_task.csv")

    # --- 3. Summary ---
    print("\n3. Writing summary...")
    n_disagree = int(df["llm_vs_tool_disagree"].sum())
    summary = {
        "total_rows": df.height,
        "rows_with_tool_value": n_with_tool_value,
        "llm_vs_tool_disagree_count": n_disagree,
        "llm_vs_tool_disagree_pct": round(100.0 * n_disagree / n_with_tool_value, 2) if n_with_tool_value else 0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved to", out_dir / "summary.json")
    print("\nDone. Outputs in", out_dir)


if __name__ == "__main__":
    main()
