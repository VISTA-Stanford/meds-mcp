#!/usr/bin/env python3
"""
Create labels_100 per task from subject_splits.parquet: stratified sample to match test-set
distribution, then flip 20% of labels to mimic ~80% accurate CLMBR tool.

- Eligible pool per task: MEDS ∩ task labels. Restrict to patients in split == "test".
- Sample 100 patients stratified to match test-set class ratio.
- Randomly flip 20 labels (so ~80% correct = tool output).
- Output: value (tool output), ground_truth (for evaluation). Tool uses value; experiment scores on ground_truth.

CheXpert: binarize as value 8192 = normal (false), else abnormal (true). One row per patient.

Usage:
  uv run python scripts/create_labels_100_from_splits.py \\
    --subject-splits data/collections/ehrshot/subject_splits.parquet \\
    --labels-dir data/collections/ehrshot/labels \\
    --meds-dir data/collections/ehrshot/meds_corpus \\
    --output-dir data/collections/ehrshot/labels/labels_100 \\
    --n 100 --flip-n 20 --seed 42
"""

import argparse
import copy
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from meds_mcp.experiments.task_config import TASK_TO_FILENAME


# CheXpert: 8192 = normal, anything else = abnormal
CHEXPERT_NORMAL_VALUE = 8192

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


def get_meds_patient_ids(
    meds_dir: Optional[Path] = None,
    meds_patient_ids_file: Optional[Path] = None,
) -> Set[str]:
    """Return set of patient IDs in MEDS."""
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
            return set()
        return {p.stem for p in d.glob("*.xml")}
    return set()


def _load_subject_splits_test(parquet_path: Path) -> Set[str]:
    """Load parquet, filter split == 'test', return set of patient IDs."""
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for parquet: pip install pandas pyarrow")
    df = pd.read_parquet(parquet_path)
    if "split" not in df.columns:
        raise SystemExit("subject_splits.parquet must have column 'split'")
    df = df[df["split"].astype(str).str.strip().str.lower() == "test"]
    for col in ("patient_id", "subject_id", "person_id"):
        if col in df.columns:
            pid_col = col
            break
    else:
        raise SystemExit(
            "subject_splits.parquet must have 'patient_id', 'subject_id', or 'person_id'"
        )
    return set(df[pid_col].astype(str).str.strip().unique())


def _chexpert_binarize(value_str: str) -> str:
    """8192 = normal (false), else abnormal (true)."""
    try:
        v = int(value_str.strip())
        return "false" if v == CHEXPERT_NORMAL_VALUE else "true"
    except (ValueError, TypeError):
        return "true"


def _get_binary_label_for_task(row: Dict[str, Any], task_name: str, label_col: str) -> str:
    """Return 'true' or 'false' for the task (binary)."""
    raw = (row.get(label_col) or row.get("value") or row.get("label") or "").strip().lower()
    if task_name == "chexpert":
        return _chexpert_binarize(row.get("value", raw) or raw)
    if task_name in LAB_TASKS:
        return "false" if _is_normal_lab(raw) else "true"
    return "true" if _is_positive_binary(raw) else "false"


def _load_task_rows(
    csv_path: Path,
    task_name: str,
    meds_ids: Set[str],
    test_patient_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Load task CSV, restrict to MEDS ∩ test, one row per patient (first row). Return list of row dicts."""
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)
    if not rows:
        return []
    if "patient_id" not in (fieldnames or []):
        return []
    label_col = "value" if "value" in (fieldnames or []) else "label"
    if label_col not in (fieldnames or []) and task_name != "chexpert":
        return []
    if task_name == "chexpert" and "value" not in (fieldnames or []):
        return []

    # Restrict to MEDS and test
    eligible = [r for r in rows if (r.get("patient_id") or "").strip() in meds_ids]
    eligible = [r for r in eligible if (r.get("patient_id") or "").strip() in test_patient_ids]
    if not eligible:
        return []

    # One row per patient (first occurrence)
    by_patient: Dict[str, Dict[str, Any]] = {}
    for r in eligible:
        pid = (r.get("patient_id") or "").strip()
        if not pid or pid in by_patient:
            continue
        by_patient[pid] = copy.copy(r)
    return list(by_patient.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create labels_100 from subject_splits (test set), stratified, 20%% labels flipped (CLMBR mimic)."
    )
    parser.add_argument(
        "--subject-splits",
        type=Path,
        default=Path("data/collections/ehrshot/subject_splits.parquet"),
        help="Path to subject_splits.parquet (must have 'split' and patient_id/subject_id/person_id)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/collections/ehrshot/labels"),
        help="Directory containing full task label CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for labels_100 CSVs",
    )
    parser.add_argument(
        "--meds-dir",
        type=Path,
        default=None,
        help="MEDS corpus directory (glob *.xml stems = patient_ids)",
    )
    parser.add_argument(
        "--meds-patient-ids-file",
        type=Path,
        default=None,
        help="Text file with one patient_id per line (overrides --meds-dir if set)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of patients to sample per task (default: 100)",
    )
    parser.add_argument(
        "--flip-n",
        type=int,
        default=20,
        help="Number of labels to flip to mimic ~80%% accurate tool (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meds_ids = get_meds_patient_ids(args.meds_dir, args.meds_patient_ids_file)
    if not meds_ids:
        raise SystemExit("No MEDS patient IDs found. Provide --meds-dir or --meds-patient-ids-file.")
    print(f"MEDS patient count: {len(meds_ids)}")

    test_patient_ids = _load_subject_splits_test(args.subject_splits.resolve())
    print(f"Test split patient count: {len(test_patient_ids)}")

    random.seed(args.seed)
    n = args.n
    flip_n = min(args.flip_n, n)

    for task_name, filename in TASK_TO_FILENAME.items():
        csv_path = labels_dir / filename
        rows = _load_task_rows(csv_path, task_name, meds_ids, test_patient_ids)
        if not rows:
            print(f"Skip {filename} (no eligible rows after MEDS ∩ test)")
            continue

        # Determine label column and get binary label per row
        label_col = "value" if "value" in (rows[0] or {}) else "label"
        for r in rows:
            r["_binary"] = _get_binary_label_for_task(r, task_name, label_col)

        # Test-set class distribution (on full eligible pool)
        pos = [r for r in rows if r["_binary"] == "true"]
        neg = [r for r in rows if r["_binary"] == "false"]
        n_pos_pool, n_neg_pool = len(pos), len(neg)
        if n_pos_pool + n_neg_pool == 0:
            print(f"Skip {filename} (no valid labels)")
            continue
        frac_pos = n_pos_pool / (n_pos_pool + n_neg_pool)

        # Stratified sample: match test ratio as closely as possible
        n_pos_target = round(n * frac_pos)
        n_neg_target = n - n_pos_target
        n_pos_target = min(n_pos_target, n_pos_pool)
        n_neg_target = min(n_neg_target, n_neg_pool)
        # If one class is short, take more from the other to reach n
        if n_pos_target + n_neg_target < n:
            if n_pos_pool - n_pos_target >= n_neg_pool - n_neg_target:
                n_pos_target = min(n_pos_pool, n_pos_target + (n - n_pos_target - n_neg_target))
            else:
                n_neg_target = min(n_neg_pool, n_neg_target + (n - n_pos_target - n_neg_target))

        sampled_pos = random.sample(pos, n_pos_target) if pos and n_pos_target else []
        sampled_neg = random.sample(neg, n_neg_target) if neg and n_neg_target else []
        sampled = sampled_pos + sampled_neg
        random.shuffle(sampled)

        if len(sampled) < n:
            print(f"Skip {filename} (only {len(sampled)} rows after stratified sample, need {n})")
            continue
        sampled = sampled[:n]

        # Ground truth = original binary label
        for r in sampled:
            r["ground_truth"] = r["_binary"]

        # Tool output: flip flip_n at random
        flip_indices = set(random.sample(range(len(sampled)), flip_n))
        for i, r in enumerate(sampled):
            if i in flip_indices:
                r["value"] = "false" if r["_binary"] == "true" else "true"
            else:
                r["value"] = r["_binary"]
            del r["_binary"]

        # Write CSV: patient_id, prediction_time, value (tool), ground_truth. No "label" so tool uses value.
        out_path = output_dir / filename
        fieldnames = ["patient_id", "prediction_time", "value", "ground_truth"]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in sampled:
                row_out = {
                    "patient_id": r.get("patient_id", ""),
                    "prediction_time": r.get("prediction_time", ""),
                    "value": r.get("value", ""),
                    "ground_truth": r.get("ground_truth", ""),
                }
                writer.writerow(row_out)
        print(f"Wrote {len(sampled)} rows to {out_path} (flipped {flip_n})")


if __name__ == "__main__":
    main()
