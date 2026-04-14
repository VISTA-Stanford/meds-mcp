#!/usr/bin/env python3
"""
Stratified sample of N unique rows from progression_subset.csv (equal rows per task).
Writes sampled_rows.csv and manifest JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _count_rows_per_task(csv_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["task"]] += 1
    return dict(counts)


def stratified_reservoir_sample(csv_path: Path, n: int, seed: int) -> list[dict]:
    """
    Two passes: (1) count rows per task, (2) reservoir sample k per task without holding the full CSV.
    """
    counts = _count_rows_per_task(csv_path)
    tasks = sorted(counts.keys())
    if not tasks:
        return []
    rng = random.Random(seed)
    per = n // len(tasks)
    rem = n % len(tasks)
    need: dict[str, int] = {}
    for i, task in enumerate(tasks):
        k = per + (1 if i < rem else 0)
        if counts[task] < k:
            raise ValueError(
                f"Task {task} has only {counts[task]} rows; need {k} for stratified sample of {n}."
            )
        need[task] = k

    pools: dict[str, list[dict]] = {t: [] for t in tasks}
    seen: dict[str, int] = {t: 0 for t in tasks}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["task"]
            k = need[t]
            seen[t] += 1
            r = dict(row)
            pool = pools[t]
            if len(pool) < k:
                pool.append(r)
            else:
                j = rng.randint(1, seen[t])
                if j <= k:
                    pool[j - 1] = r

    out: list[dict] = []
    for t in tasks:
        out.extend(pools[t])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified sample from progression_subset.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "data/collections/vista_bench/progression_subset.csv",
        help="Path to progression_subset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "experiments/progression_subset/outputs",
        help="Directory for sampled_rows.csv and manifest",
    )
    parser.add_argument("--n", type=int, default=100, help="Total rows (default 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampled = stratified_reservoir_sample(args.input, args.n, args.seed)

    out_csv = args.output_dir / "sampled_rows.csv"
    fieldnames = list(sampled[0].keys()) if sampled else [
        "person_id",
        "embed_time",
        "task",
        "label",
        "question",
        "label_description",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sampled:
            w.writerow(r)

    manifest = {
        "n_requested": args.n,
        "n_sampled": len(sampled),
        "seed": args.seed,
        "input": str(args.input),
        "output_csv": str(out_csv),
        "tasks": sorted({r["task"] for r in sampled}),
    }
    with open(args.output_dir / "sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(sampled)} rows to {out_csv}")
    print(f"Manifest: {args.output_dir / 'sample_manifest.json'}")


if __name__ == "__main__":
    main()
