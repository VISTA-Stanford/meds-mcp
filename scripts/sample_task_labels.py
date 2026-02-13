#!/usr/bin/env python3
"""
Sample N rows uniformly at random from each task labels CSV and write new CSVs.

Usage:
  python scripts/sample_task_labels.py --input-dir data/collections/vista_bench/labels --output-dir data/collections/vista_bench/labels_100
  python scripts/sample_task_labels.py --input-dir /path/to/labels --output-dir /path/to/labels_100 --n 100 --seed 42
"""

import argparse
import csv
import random
from pathlib import Path

from meds_mcp.experiments.task_config import TASK_TO_FILENAME, get_labels_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniformly randomly sample N rows from each task CSV and write new CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder containing one CSV per task (default: vista_bench labels dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where sampled CSVs will be written (created if missing)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows to sample per task (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    input_dir = args.input_dir or get_labels_dir()
    input_dir = input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    n = args.n
    for task_name, filename in TASK_TO_FILENAME.items():
        csv_path = input_dir / filename
        if not csv_path.exists():
            print(f"Skip {filename} (not found)")
            continue

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print(f"Skip {filename} (no header)")
                continue
            rows = list(reader)

        if not rows:
            print(f"Skip {filename} (empty)")
            continue

        sample_size = min(n, len(rows))
        sampled = random.sample(rows, sample_size)

        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled)

        print(f"Wrote {sample_size} rows to {out_path}")


if __name__ == "__main__":
    main()
