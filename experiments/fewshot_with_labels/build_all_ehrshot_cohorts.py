#!/usr/bin/env python3
"""
Build combined patients.jsonl + items.jsonl for all EHRSHOT tasks.

Reads labeled_patients.csv from each task subfolder under --benchmark-dir and
produces a single pair of JSONL files covering all tasks. PatientState records
are deduplicated by (person_id, embed_time) across tasks.

Usage:
  uv run python experiments/fewshot_with_labels/build_all_ehrshot_cohorts.py \\
    --benchmark-dir ~/data/EHRSHOT_ASSETS/benchmark \\
    --splits-csv ~/data/EHRSHOT_ASSETS/splits/person_id_map.csv \\
    --output-dir experiments/fewshot_with_labels/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity.cohort import LabeledItem, PatientState, utc_now_iso

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TASK_GROUPS: dict[str, str] = {
    "guo_icu": "guo",
    "guo_los": "guo",
    "guo_readmission": "guo",
    "lab_thrombocytopenia": "lab",
    "lab_hyperkalemia": "lab",
    "lab_hypoglycemia": "lab",
    "lab_hyponatremia": "lab",
    "lab_anemia": "lab",
    "new_hypertension": "new",
    "new_hyperlipidemia": "new",
    "new_pancan": "new",
    "new_celiac": "new",
    "new_lupus": "new",
    "new_acutemi": "new",
    "chexpert": "chexpert",
}

TASK_QUESTIONS: dict[str, str] = {
    "guo_readmission": "Will the patient be readmitted to the hospital within 30 days?",
    "guo_icu": "Will the patient be transferred to the intensive care unit?",
    "guo_los": "Will the patient stay in the hospital for more than 7 days?",
    "lab_thrombocytopenia": "Will the patient's thrombocytopenia lab come back as abnormal?",
    "lab_hyperkalemia": "Will the patient's hyperkalemia lab come back as abnormal?",
    "lab_hypoglycemia": "Will the patient's hypoglycemia lab come back as abnormal?",
    "lab_hyponatremia": "Will the patient's hyponatremia lab come back as abnormal?",
    "lab_anemia": "Will the patient's anemia lab come back as abnormal?",
    "new_hypertension": "Will the patient develop hypertension in the next year?",
    "new_hyperlipidemia": "Will the patient develop hyperlipidemia in the next year?",
    "new_pancan": "Will the patient develop pancreatic cancer in the next year?",
    "new_celiac": "Will the patient develop celiac disease in the next year?",
    "new_lupus": "Will the patient develop lupus in the next year?",
    "new_acutemi": "Will the patient develop an acute myocardial infarction in the next year?",
    "chexpert": "Will the patient's chest X-ray come back as abnormal?",
}

LABEL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "guo_readmission": ("not readmitted", "readmitted"),
    "guo_icu": ("no ICU transfer", "ICU transfer"),
    "guo_los": ("short stay", "prolonged stay"),
    "lab_thrombocytopenia": ("normal", "abnormal"),
    "lab_hyperkalemia": ("normal", "abnormal"),
    "lab_hypoglycemia": ("normal", "abnormal"),
    "lab_hyponatremia": ("normal", "abnormal"),
    "lab_anemia": ("normal", "abnormal"),
    "new_hypertension": ("not diagnosed", "diagnosed"),
    "new_hyperlipidemia": ("not diagnosed", "diagnosed"),
    "new_pancan": ("not diagnosed", "diagnosed"),
    "new_celiac": ("not diagnosed", "diagnosed"),
    "new_lupus": ("not diagnosed", "diagnosed"),
    "new_acutemi": ("not diagnosed", "diagnosed"),
    "chexpert": ("normal", "abnormal"),
}


def _iso_embed_time(prediction_time: str) -> str:
    return prediction_time.strip().replace(" ", "T")


def _parse_label(value: str, label_type: str) -> int:
    v = value.strip().lower()
    if label_type.strip().lower() == "categorical":
        # 0 = normal, anything else = abnormal
        return 0 if v == "0" else 1
    # boolean
    if v in ("true", "1"):
        return 1
    if v in ("false", "0"):
        return 0
    raise ValueError(f"Unexpected label value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path.home() / "data/EHRSHOT_ASSETS/benchmark",
        help="Path to EHRSHOT_ASSETS/benchmark (contains one subfolder per task).",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=Path.home() / "data/EHRSHOT_ASSETS/splits/person_id_map.csv",
        help="Path to person_id_map.csv (columns: split, omop_person_id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "experiments/fewshot_with_labels/outputs",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of tasks to include. Default: all recognized task folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Only include rows from these splits (e.g. train test). Default: all.",
    )
    args = parser.parse_args()

    # Load split map
    split_map: dict[str, str] = {}
    with open(args.splits_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split_map[str(int(row["omop_person_id"]))] = row["split"]
    logger.info("Loaded %d patient splits", len(split_map))

    tasks = args.tasks or list(TASK_GROUPS.keys())
    splits_filter = set(args.splits) if args.splits else None
    now = utc_now_iso()

    states: dict[tuple[str, str], PatientState] = {}
    items: list[LabeledItem] = []

    for task in tasks:
        labels_csv = args.benchmark_dir / task / "labeled_patients.csv"
        if not labels_csv.exists():
            logger.warning("No labeled_patients.csv for task %s at %s — skipping", task, labels_csv)
            continue

        question = TASK_QUESTIONS.get(task, "")
        task_group = TASK_GROUPS.get(task, "")
        neg_desc, pos_desc = LABEL_DESCRIPTIONS.get(task, ("no", "yes"))

        task_items = 0
        task_skipped = 0

        with open(labels_csv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = str(int(row["patient_id"]))
                embed_time = _iso_embed_time(row["prediction_time"])

                split = split_map.get(pid)
                if split is None:
                    task_skipped += 1
                    continue
                if splits_filter and split not in splits_filter:
                    task_skipped += 1
                    continue

                try:
                    label = _parse_label(row["value"], row.get("label_type", "boolean"))
                except ValueError:
                    task_skipped += 1
                    logger.warning("Bad label for patient %s task %s: %s", pid, task, row["value"])
                    continue

                state_key = (pid, embed_time)
                if state_key not in states:
                    states[state_key] = PatientState(
                        person_id=pid,
                        embed_time=embed_time,
                        split=split,
                        vignette="",
                        created_at=now,
                    )

                items.append(
                    LabeledItem(
                        person_id=pid,
                        task=task,
                        task_group=task_group,
                        question=question,
                        label=label,
                        label_description=pos_desc if label == 1 else neg_desc,
                        split=split,
                        embed_time=embed_time,
                        created_at=now,
                    )
                )
                task_items += 1

        logger.info("Task %-25s  items=%d  skipped=%d", task, task_items, task_skipped)

    logger.info("Total: %d unique (person_id, embed_time) states | %d items", len(states), len(items))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients_path = args.output_dir / "patients.jsonl"
    items_path = args.output_dir / "items.jsonl"

    tmp_p = patients_path.with_suffix(".jsonl.tmp")
    with open(tmp_p, "w", encoding="utf-8") as f:
        for s in states.values():
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    tmp_p.replace(patients_path)

    tmp_i = items_path.with_suffix(".jsonl.tmp")
    with open(tmp_i, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    tmp_i.replace(items_path)

    logger.info("Wrote %s (%d states)", patients_path, len(states))
    logger.info("Wrote %s (%d items)", items_path, len(items))


if __name__ == "__main__":
    main()
