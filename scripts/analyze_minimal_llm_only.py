#!/usr/bin/env python3
"""
Quick analysis for minimal LLM-only experiment: compute balanced accuracy.

Reads JSONL from the minimal LLM-only run (patient_id, task_name, prediction_time,
ground_truth_normalized, llm_only_normalized) and prints/writes balanced accuracy.

Usage:
  uv run python scripts/analyze_minimal_llm_only.py results/minimal_llm_only_20260226_134453.jsonl
  uv run python scripts/analyze_minimal_llm_only.py results/minimal_llm_only_20260226_134453.jsonl --output-dir results/minimal_llm_only_20260226_134453
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import polars as pl
from sklearn.metrics import balanced_accuracy_score


def yes_no_to_binary_expr(col: pl.Expr) -> pl.Expr:
    """Map yes/true/1 -> 1, no/false/0 -> 0; null/invalid -> null."""
    lower = col.str.to_lowercase().str.strip_chars()
    return (
        pl.when(lower.is_in(["yes", "true", "1"]))
        .then(1)
        .when(lower.is_in(["no", "false", "0"]))
        .then(0)
        .otherwise(None)
    ).cast(pl.Int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute balanced accuracy for minimal LLM-only experiment JSONL"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to minimal LLM-only JSONL output",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, write balanced_accuracy.csv and summary to this directory",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pl.read_ndjson(input_path)

    if "ground_truth_normalized" not in df.columns or "llm_only_normalized" not in df.columns:
        print("Error: JSONL must contain ground_truth_normalized and llm_only_normalized", file=sys.stderr)
        sys.exit(1)

    # Drop error rows and nulls
    df = df.filter(
        pl.col("llm_only_normalized").is_not_null()
        & ~pl.col("llm_only_normalized").str.starts_with("[ERROR")
        & pl.col("ground_truth_normalized").is_not_null()
    )

    df = df.with_columns(
        _y_true=yes_no_to_binary_expr(pl.col("ground_truth_normalized")),
        _y_pred=yes_no_to_binary_expr(pl.col("llm_only_normalized")),
    )
    # Drop rows where either is null (invalid yes/no)
    df = df.filter(pl.col("_y_true").is_not_null() & pl.col("_y_pred").is_not_null())
    y_true = df["_y_true"].to_numpy()
    y_pred = df["_y_pred"].to_numpy()

    n = len(y_true)
    if n < 1:
        print("No valid rows (need ground_truth_normalized and llm_only_normalized as yes/no).")
        sys.exit(0)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    task = df["task_name"][0] if "task_name" in df.columns else "lab_thrombocytopenia"

    print(f"Task: {task}")
    print(f"N:    {n}")
    print(f"Balanced accuracy: {bal_acc:.4f}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = pl.DataFrame({
            "task": [task],
            "balanced_accuracy": [round(bal_acc, 4)],
            "n": [n],
        })
        summary.write_csv(out_dir / "balanced_accuracy.csv")
        print(f"\nWrote {out_dir / 'balanced_accuracy.csv'}")


if __name__ == "__main__":
    main()
