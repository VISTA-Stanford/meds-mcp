#!/usr/bin/env python3
"""
Merge experiment_results_<context>.jsonl files and compute summary metrics.

For each context variant present:
  - Overall accuracy, parse rate, confusion matrix over Yes/No.
  - Per-task accuracy (flagged SMALL_N when n < 10).

When a baseline run is present, compute flip / fix / hurt rates of each other
variant vs baseline, joined on (person_id, task).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.fewshot_with_labels import _paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SMALL_N_THRESHOLD = 10


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def key_of(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("person_id", "")), str(row.get("task", "")))


def _balanced_accuracy(rows: list[dict[str, Any]]) -> Optional[float]:
    """BA = (TPR + TNR) / 2 for binary Yes/No predictions."""
    tp = fn = tn = fp = 0
    for r in rows:
        true = r.get("true_yes_no")
        pred = r.get("pred")
        if true not in ("Yes", "No") or pred not in ("Yes", "No"):
            continue
        if true == "Yes" and pred == "Yes":
            tp += 1
        elif true == "Yes" and pred == "No":
            fn += 1
        elif true == "No" and pred == "No":
            tn += 1
        else:
            fp += 1
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    if tpr is None or tnr is None:
        return None
    return round((tpr + tnr) / 2, 4)


def per_variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    parsed = sum(1 for r in rows if r.get("pred") in ("Yes", "No"))
    correct = sum(1 for r in rows if r.get("correct"))

    # Confusion over parsed predictions; rows with None pred are counted separately.
    conf: Counter[tuple[str, str]] = Counter()
    unparsed = 0
    for r in rows:
        true = r.get("true_yes_no")
        pred = r.get("pred")
        if pred in ("Yes", "No") and true in ("Yes", "No"):
            conf[(true, pred)] += 1
        else:
            unparsed += 1

    per_task: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get("task", ""))].append(r)
    for task, trows in sorted(by_task.items()):
        tn = len(trows)
        tc = sum(1 for r in trows if r.get("correct"))
        per_task[task] = {
            "n": tn,
            "accuracy": round(tc / tn, 4) if tn else None,
            "balanced_accuracy": _balanced_accuracy(trows),
            "small_n": tn < SMALL_N_THRESHOLD,
        }

    return {
        "n": n,
        "parse_rate": round(parsed / n, 4) if n else None,
        "accuracy": round(correct / n, 4) if n else None,
        "balanced_accuracy": _balanced_accuracy(rows),
        "n_correct": correct,
        "n_unparsed": unparsed,
        "confusion": {f"true={t}|pred={p}": c for (t, p), c in sorted(conf.items())},
        "per_task": per_task,
    }


def compare_vs_baseline(
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_key = {key_of(r): r for r in baseline_rows}
    variant_by_key = {key_of(r): r for r in variant_rows}
    shared = sorted(set(baseline_by_key) & set(variant_by_key))
    n = len(shared)
    if n == 0:
        return {"n_shared": 0}

    baseline_wrong = 0
    flips = 0
    hurts = 0
    acc_b = 0
    acc_v = 0
    for k in shared:
        b = baseline_by_key[k]
        v = variant_by_key[k]
        bc = bool(b.get("correct"))
        vc = bool(v.get("correct"))
        if bc:
            acc_b += 1
        if vc:
            acc_v += 1
        if not bc:
            baseline_wrong += 1
            if vc:
                flips += 1
        if bc and not vc:
            hurts += 1

    shared_baseline = [baseline_by_key[k] for k in shared]
    shared_variant = [variant_by_key[k] for k in shared]
    ba_b = _balanced_accuracy(shared_baseline)
    ba_v = _balanced_accuracy(shared_variant)
    delta_ba = round(ba_v - ba_b, 4) if (ba_b is not None and ba_v is not None) else None

    return {
        "n_shared": n,
        "accuracy_baseline_on_shared": round(acc_b / n, 4),
        "accuracy_variant_on_shared": round(acc_v / n, 4),
        "delta_acc_variant_minus_baseline": round((acc_v - acc_b) / n, 4),
        "balanced_accuracy_baseline_on_shared": ba_b,
        "balanced_accuracy_variant_on_shared": ba_v,
        "delta_balanced_acc_variant_minus_baseline": delta_ba,
        "count_baseline_wrong_on_shared": baseline_wrong,
        "flip_pct_baseline_wrong_variant_right": round(100.0 * flips / n, 4),
        "fix_rate_given_baseline_wrong": (
            round(100.0 * flips / baseline_wrong, 4) if baseline_wrong else None
        ),
        "hurt_pct_baseline_right_variant_wrong": round(100.0 * hurts / n, 4),
    }


def _print_example_prompts(input_dir: Path, contexts: list[str]) -> None:
    """Print one saved example prompt per context variant."""
    shown = 0
    # Prefer the canonical zero-shot / fewshot pair when both are present.
    priority = ["baseline_vignette", "vignette", "baseline_timeline", "timeline"]
    ordered = [c for c in priority if c in contexts] + [c for c in contexts if c not in priority]
    for ctx in ordered:
        path = input_dir / f"example_prompt_{ctx}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        label = "ZERO-SHOT" if ctx.startswith("baseline") else "FEW-SHOT"
        print(f"\n{'='*70}")
        print(f"EXAMPLE CONTEXT — {ctx} ({label})")
        print("=" * 70)
        print(text)
        shown += 1
    if shown == 0:
        print(
            "\n[No example_prompt_<context>.txt files found in input_dir. "
            "Run run_experiment_vertex_batch.py to generate them.]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze fewshot_with_labels experiment_results_<context>.jsonl files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_paths.outputs_dir(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_paths.outputs_dir(),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("experiment_results_*.jsonl"))
    if not files:
        print(f"No experiment_results_*.jsonl files under {args.input_dir}")
        return

    by_context: dict[str, list[dict[str, Any]]] = {}
    for p in files:
        ctx = p.stem[len("experiment_results_"):]
        by_context[ctx] = load(p)

    per_variant = {ctx: per_variant_summary(rows) for ctx, rows in by_context.items()}

    # Each with-similars context (vignette / timeline) is paired 1:1 with its own
    # baseline (baseline_vignette / baseline_timeline) so the only difference
    # between paired prompts is the presence of the similars block.
    pair_map = {
        "vignette": "baseline_vignette",
        "timeline": "baseline_timeline",
    }
    comparisons: dict[str, Any] = {}
    for variant, baseline_name in pair_map.items():
        if variant in by_context and baseline_name in by_context:
            comparisons[variant] = {
                "baseline": baseline_name,
                **compare_vs_baseline(by_context[baseline_name], by_context[variant]),
            }

    # Cross-variant per-task matrix: for each task, list n + accuracy under each
    # context, plus best variant (ties broken by context order).
    contexts = sorted(by_context.keys())
    all_tasks = sorted({t for v in per_variant.values() for t in v["per_task"]})
    per_task_matrix: dict[str, Any] = {}
    for task in all_tasks:
        row: dict[str, Any] = {}
        for ctx in contexts:
            cell = per_variant[ctx]["per_task"].get(task)
            row[ctx] = cell  # {n, accuracy, small_n} or None
        accs = [
            (ctx, row[ctx]["accuracy"])
            for ctx in contexts
            if row.get(ctx) and row[ctx].get("accuracy") is not None
        ]
        if not accs:
            row["best_variant"] = None
        else:
            top = max(v for _, v in accs)
            winners = [ctx for ctx, v in accs if v == top]
            row["best_variant"] = winners[0] if len(winners) == 1 else "tie"
        per_task_matrix[task] = row

    summary = {
        "small_n_threshold": SMALL_N_THRESHOLD,
        "contexts": contexts,
        "per_variant": per_variant,
        "per_task_across_variants": per_task_matrix,
        "comparisons_vs_baseline": comparisons if comparisons else None,
    }

    out_path = args.output_dir / "analysis_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV dump of the per-task matrix for easy spreadsheet import.
    csv_path = args.output_dir / "per_task_accuracy.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["task"]
        for ctx in contexts:
            header.extend([f"{ctx}_n", f"{ctx}_acc", f"{ctx}_balanced_acc", f"{ctx}_small_n"])
        header.append("best_variant")
        writer.writerow(header)
        for task in all_tasks:
            row_cells: list[Any] = [task]
            for ctx in contexts:
                cell = per_task_matrix[task].get(ctx)
                if cell:
                    row_cells.extend([cell["n"], cell["accuracy"], cell.get("balanced_accuracy"), cell["small_n"]])
                else:
                    row_cells.extend(["", "", "", ""])
            row_cells.append(per_task_matrix[task].get("best_variant") or "")
            writer.writerow(row_cells)

    # Human-readable stdout table.
    print_per_task_table(per_task_matrix, contexts)

    # Overall line (compact).
    print("\nOverall accuracy:")
    for ctx in contexts:
        v = per_variant[ctx]
        print(
            f"  {ctx:9s}  n={v['n']:<5d}  acc={v['accuracy']}  "
            f"balanced_acc={v['balanced_accuracy']}  parse_rate={v['parse_rate']}"
        )

    if comparisons:
        print("\nPaired comparisons (with-similars vs matched baseline):")
        for ctx, c in comparisons.items():
            pair = c.get("baseline", "?")
            if c.get("n_shared", 0) == 0:
                print(f"  {ctx} vs {pair}: no shared rows")
                continue
            print(
                f"  {ctx:9s} vs {pair:<20s}  "
                f"delta_acc={c['delta_acc_variant_minus_baseline']:+.4f}  "
                f"delta_balanced_acc={c['delta_balanced_acc_variant_minus_baseline']}  "
                f"fix={c['fix_rate_given_baseline_wrong']}  "
                f"hurt={c['hurt_pct_baseline_right_variant_wrong']}"
            )

    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")

    _print_example_prompts(args.input_dir, contexts)


def print_per_task_table(
    per_task_matrix: dict[str, Any], contexts: list[str]
) -> None:
    if not per_task_matrix:
        return
    # Layout: task | ctx1 (n acc) | ctx2 ... | best
    task_w = max(len("task"), max(len(t) for t in per_task_matrix))
    cell_w = 14  # "  123 | 0.857"
    rows: list[str] = []
    header = f"{'task':<{task_w}} " + " ".join(f"| {ctx:^{cell_w}}" for ctx in contexts) + " | best"
    rows.append(header)
    rows.append("-" * len(header))
    for task in sorted(per_task_matrix):
        cells: list[str] = []
        for ctx in contexts:
            c = per_task_matrix[task].get(ctx)
            if c is None or c.get("accuracy") is None:
                cells.append(f"| {'-':^{cell_w}}")
            else:
                flag = "*" if c.get("small_n") else " "
                cells.append(f"| {c['n']:>4d} {c['accuracy']:.3f}{flag} ".ljust(cell_w + 2))
        best = per_task_matrix[task].get("best_variant") or "-"
        rows.append(f"{task:<{task_w}} " + "".join(cells) + f"| {best}")
    print("\nPer-task accuracy (n and acc; * = small_n flag):")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
