#!/usr/bin/env python3
"""
Compute accuracy per variant and flip / fix / hurt rates vs variant 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def correct(pred: str | None, label: str) -> bool:
    if pred is None:
        return False
    return str(pred).strip() == str(label).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze progression_subset experiment_results.jsonl")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs/experiment_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n = len(rows)
    if n == 0:
        print("No rows in input")
        return

    def acc(key: str) -> float:
        return sum(1 for r in rows if correct(r.get(key), r.get("label", ""))) / n

    acc1 = acc("pred_v1")
    acc2 = acc("pred_v2")
    acc3 = acc("pred_v3")

    w1 = sum(1 for r in rows if not correct(r.get("pred_v1"), r.get("label", "")))
    flip12 = sum(
        1
        for r in rows
        if not correct(r.get("pred_v1"), r.get("label", ""))
        and correct(r.get("pred_v2"), r.get("label", ""))
    )
    flip13 = sum(
        1
        for r in rows
        if not correct(r.get("pred_v1"), r.get("label", ""))
        and correct(r.get("pred_v3"), r.get("label", ""))
    )
    hurt12 = sum(
        1
        for r in rows
        if correct(r.get("pred_v1"), r.get("label", ""))
        and not correct(r.get("pred_v2"), r.get("label", ""))
    )
    hurt13 = sum(
        1
        for r in rows
        if correct(r.get("pred_v1"), r.get("label", ""))
        and not correct(r.get("pred_v3"), r.get("label", ""))
    )

    fix2 = (flip12 / w1 * 100) if w1 else None
    fix3 = (flip13 / w1 * 100) if w1 else None

    summary = {
        "n": n,
        "accuracy_v1": round(acc1, 4),
        "accuracy_v2": round(acc2, 4),
        "accuracy_v3": round(acc3, 4),
        "delta_acc_v2_minus_v1": round(acc2 - acc1, 4),
        "delta_acc_v3_minus_v1": round(acc3 - acc1, 4),
        "count_v1_wrong": w1,
        "flip_pct_all_rows_v1_wrong_v2_right": round(100.0 * flip12 / n, 4),
        "flip_pct_all_rows_v1_wrong_v3_right": round(100.0 * flip13 / n, 4),
        "fix_rate_given_v1_wrong_v2": round(fix2, 4) if fix2 is not None else None,
        "fix_rate_given_v1_wrong_v3": round(fix3, 4) if fix3 is not None else None,
        "hurt_pct_all_rows_v1_right_v2_wrong": round(100.0 * hurt12 / n, 4),
        "hurt_pct_all_rows_v1_right_v3_wrong": round(100.0 * hurt13 / n, 4),
    }

    out_path = args.output_dir / "analysis_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
