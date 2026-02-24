#!/usr/bin/env python3
"""
Sample N rows from each task labels CSV. With --balanced, stratified 50/50:
- Binary tasks: 50% positive, 50% negative.
- Lab (categorical) tasks: binarize to normal (0) vs abnormal (1,2,3), then 50 normal, 50 abnormal;
  output label column is "false"/"true" for downstream binary treatment.

Optional: --meds-dir or --meds-patient-ids-file to keep only patient_ids present in MEDS.
Optional: --one-label-per-patient to dedupe by patient_id (prefer positive/non-normal label if any).

Usage:
  python scripts/sample_task_labels.py --input-dir data/collections/vista_bench/labels --output-dir data/collections/vista_bench/labels_100 --balanced
  python scripts/sample_task_labels.py --input-dir data/collections/ehrshot/labels --output-dir data/collections/ehrshot/labels/labels_100 --balanced --meds-dir data/collections/ehrshot/meds_corpus --one-label-per-patient
  python scripts/sample_task_labels.py --input-dir /path/to/labels --output-dir /path/to/labels_100 --n 100 --seed 42  # uniform sampling
"""

import argparse
import copy
import csv
import random
from pathlib import Path
from typing import Optional, Set

from meds_mcp.experiments.task_config import (
    BINARY_TASKS,
    TASK_TO_FILENAME,
    get_labels_dir,
)

# Lab tasks: binarize to normal (0) vs abnormal (1,2,3) and sample 50/50 (output "false"/"true").
LAB_TASKS = {
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_thrombocytopenia",
}


def _is_positive_binary(label: str) -> bool:
    raw = (label or "").strip().lower()
    return raw in ("true", "1", "yes")


def _is_normal_lab(label: str) -> bool:
    raw = (label or "").strip().lower()
    return raw == "0"


def _is_positive_or_abnormal(label: str, task_name: str) -> bool:
    """True if label is positive (binary) or non-normal (lab). Used to pick one row per patient."""
    if task_name in LAB_TASKS:
        return not _is_normal_lab(label)
    return _is_positive_binary(label)


def get_meds_patient_ids(
    meds_dir: Optional[Path] = None,
    meds_patient_ids_file: Optional[Path] = None,
) -> Optional[Set[str]]:
    """Return set of patient IDs in MEDS, or None if neither source provided."""
    if meds_patient_ids_file is not None and meds_patient_ids_file.exists():
        ids = set()
        with open(meds_patient_ids_file, "r", encoding="utf-8") as f:
            for line in f:
                pid = line.strip()
                if pid:
                    ids.add(pid)
        return ids
    if meds_dir is not None:
        d = Path(meds_dir).resolve()
        if not d.is_dir():
            return None
        return {p.stem for p in d.glob("*.xml")}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample N rows per task. With --balanced: 50/50 pos/neg (binary) or 50/50 normal/abnormal (lab)."
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
        "--meds-dir",
        type=Path,
        default=None,
        help="MEDS corpus directory (glob *.xml stems = patient_ids). Keep only these patients.",
    )
    parser.add_argument(
        "--meds-patient-ids-file",
        type=Path,
        default=None,
        help="Text file with one patient_id per line. Keep only these patients. Overrides --meds-dir if both set.",
    )
    parser.add_argument(
        "--one-label-per-patient",
        action="store_true",
        help="Dedupe by patient_id: keep one row per patient (prefer positive/non-normal label if any).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows to sample per task (default: 100). With --balanced, n/2 per class.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Stratified sample: 50/50 for binary; lab tasks binarized to normal/abnormal then 50/50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    input_dir = args.input_dir or get_labels_dir()
    input_dir = Path(input_dir).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meds_ids: Optional[Set[str]] = None
    if args.meds_patient_ids_file is not None or args.meds_dir is not None:
        meds_ids = get_meds_patient_ids(args.meds_dir, args.meds_patient_ids_file)
        if not meds_ids:
            raise SystemExit("No MEDS patient IDs found from --meds-dir / --meds-patient-ids-file.")
        print(f"Using {len(meds_ids)} MEDS patient IDs for join")

    if args.seed is not None:
        random.seed(args.seed)

    n = args.n
    half = n // 2

    for task_name, filename in TASK_TO_FILENAME.items():
        csv_path = input_dir / filename
        if not csv_path.exists():
            print(f"Skip {filename} (not found)")
            continue

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames) if reader.fieldnames else []
            if not fieldnames:
                print(f"Skip {filename} (no header)")
                continue
            rows = list(reader)

        if not rows:
            print(f"Skip {filename} (empty)")
            continue

        # Label column: EHRSHOT uses "value", Vista uses "label"
        label_col = "value" if "value" in fieldnames else "label"
        if label_col not in fieldnames:
            print(f"Skip {filename} (no 'label' or 'value' column)")
            continue

        def get_label(r: dict) -> str:
            return (r.get(label_col) or "").strip()

        # Join: keep only rows whose patient_id is in MEDS (if meds_ids set)
        if meds_ids is not None:
            rows = [r for r in rows if (r.get("patient_id") or "").strip() in meds_ids]
            if not rows:
                print(f"Skip {filename} (no rows after MEDS join)")
                continue

        # One label per patient: group by patient_id, pick one row (prefer positive/non-normal)
        if args.one_label_per_patient:
            by_patient: dict = {}
            for r in rows:
                pid = (r.get("patient_id") or "").strip()
                if not pid:
                    continue
                if pid not in by_patient:
                    by_patient[pid] = []
                by_patient[pid].append(r)
            out_rows = []
            for pid, patient_rows in by_patient.items():
                positive_rows = [r for r in patient_rows if _is_positive_or_abnormal(get_label(r), task_name)]
                chosen = positive_rows[0] if positive_rows else patient_rows[0]
                out_rows.append(copy.copy(chosen))
            rows = out_rows
            if not rows:
                print(f"Skip {filename} (no rows after one-label-per-patient)")
                continue

        if args.balanced and task_name in LAB_TASKS:
            # Lab: 50/50 normal vs abnormal; output "false"/"true" in same column
            normal = [r for r in rows if _is_normal_lab(get_label(r))]
            abnormal = [r for r in rows if not _is_normal_lab(get_label(r))]
            n_norm = min(half, len(normal))
            n_abnorm = min(half, len(abnormal))
            sampled_norm = random.sample(normal, n_norm) if normal else []
            sampled_abnorm = random.sample(abnormal, n_abnorm) if abnormal else []
            out_rows = []
            for r in sampled_norm:
                out_r = copy.copy(r)
                out_r[label_col] = "false"
                out_rows.append(out_r)
            for r in sampled_abnorm:
                out_r = copy.copy(r)
                out_r[label_col] = "true"
                out_rows.append(out_r)
            sampled = out_rows
            random.shuffle(sampled)
            sample_size = len(sampled)
        elif args.balanced and task_name in BINARY_TASKS:
            pos = [r for r in rows if _is_positive_binary(get_label(r))]
            neg = [r for r in rows if not _is_positive_binary(get_label(r))]
            n_pos = min(half, len(pos))
            n_neg = min(half, len(neg))
            sampled_pos = random.sample(pos, n_pos) if pos else []
            sampled_neg = random.sample(neg, n_neg) if neg else []
            sampled = sampled_pos + sampled_neg
            random.shuffle(sampled)
            sample_size = len(sampled)
        else:
            sample_size = min(n, len(rows))
            sampled = random.sample(rows, sample_size)

        # Ensure output has "label" column for downstream (task_tools, experiment)
        out_fieldnames = list(fieldnames)
        if "label" not in out_fieldnames and label_col in out_fieldnames:
            out_fieldnames.append("label")
        for r in sampled:
            r["label"] = r.get(label_col, r.get("label", ""))

        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sampled)

        print(f"Wrote {sample_size} rows to {out_path}")


if __name__ == "__main__":
    main()
