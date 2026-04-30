#!/usr/bin/env python3
"""
Build patients.jsonl + items.jsonl for the CohortStore pipeline from EHRSHOT CSVs.

Reads:
  --labels-csv   EHRSHOT_ASSETS/benchmark/<task>/labeled_patients.csv
                 Columns: patient_id, prediction_time, value (True/False), label_type
  --splits-csv   EHRSHOT_ASSETS/splits/person_id_map.csv
                 Columns: split, omop_person_id

Writes:
  --output-dir / patients.jsonl   one PatientState per (person_id, prediction_time)
  --output-dir / items.jsonl      one LabeledItem per row in labels-csv

The vignette field in patients.jsonl is left empty; run
precompute_vignettes_vertex_batch.py afterwards to fill it in.

Usage:
  python experiments/fewshot_with_labels/build_ehrshot_cohort.py \\
    --labels-csv ~/LRRL_MEDS/data/EHRSHOT_ASSETS/benchmark/guo_readmission/labeled_patients.csv \\
    --splits-csv ~/LRRL_MEDS/data/EHRSHOT_ASSETS/splits/person_id_map.csv \\
    --task guo_readmission \\
    --output-dir ~/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot
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
for _p in (_REPO_ROOT / "src", _REPO_ROOT, _REPO_ROOT / "LRRL_MEDS"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity.cohort import LabeledItem, PatientState, utc_now_iso

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TASK_QUESTIONS: dict[str, str] = {
    "guo_icu": "Will the patient be transferred to the intensive care unit?",
    "guo_los": "Will the patient stay in the hospital for more than 7 days?",
    "guo_readmission": "Will the patient be readmitted to the hospital within 30 days?",
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


def _iso_embed_time(prediction_time: str) -> str:
    """Normalise 'YYYY-MM-DD HH:MM:SS' → 'YYYY-MM-DDTHH:MM:SS' ISO 8601."""
    return prediction_time.strip().replace(" ", "T")


def _parse_label(value: str) -> int:
    v = value.strip().lower()
    if v in ("true", "1"):
        return 1
    if v in ("false", "0"):
        return 0
    raise ValueError(f"Unexpected label value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CohortStore JSONLs from EHRSHOT CSVs.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="Path to labeled_patients.csv (patient_id, prediction_time, value, label_type)",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        required=True,
        help="Path to person_id_map.csv (split, omop_person_id)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_QUESTIONS.keys()),
        help="Task name (e.g. guo_readmission)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write patients.jsonl and items.jsonl",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Only include rows from these splits (e.g. train test). Default: all.",
    )
    args = parser.parse_args()

    # Load split map: person_id (str) → split
    split_map: dict[str, str] = {}
    with open(args.splits_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split_map[str(int(row["omop_person_id"]))] = row["split"]
    logger.info("Loaded %d patient splits", len(split_map))

    question = TASK_QUESTIONS[args.task]
    task_group = TASK_GROUPS[args.task]
    splits_filter = set(args.splits) if args.splits else None
    now = utc_now_iso()

    # Deduplicate patient states: (person_id, embed_time) → PatientState
    states: dict[tuple[str, str], PatientState] = {}
    items: list[LabeledItem] = []
    skipped_no_split = 0
    skipped_split_filter = 0
    skipped_bad_label = 0

    with open(args.labels_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = str(int(row["patient_id"]))
            prediction_time = row["prediction_time"]
            embed_time = _iso_embed_time(prediction_time)

            split = split_map.get(pid)
            if split is None:
                skipped_no_split += 1
                continue
            if splits_filter and split not in splits_filter:
                skipped_split_filter += 1
                continue

            try:
                label = _parse_label(row["value"])
            except ValueError:
                skipped_bad_label += 1
                logger.warning("Bad label for patient %s: %s", pid, row["value"])
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
                    task=args.task,
                    task_group=task_group,
                    question=question,
                    label=label,
                    label_description="readmitted" if label == 1 else "not readmitted",
                    split=split,
                    embed_time=embed_time,
                    created_at=now,
                )
            )

    logger.info(
        "States: %d unique (person_id, embed_time) pairs | Items: %d",
        len(states),
        len(items),
    )
    if skipped_no_split:
        logger.warning("Skipped %d rows — person_id not in splits CSV", skipped_no_split)
    if skipped_split_filter:
        logger.info("Skipped %d rows — split not in filter %s", skipped_split_filter, splits_filter)
    if skipped_bad_label:
        logger.warning("Skipped %d rows — unparseable label", skipped_bad_label)

    # Split breakdown
    from collections import Counter
    split_counts = Counter(s.split for s in states.values())
    label_counts = Counter(it.label for it in items)
    logger.info("Split breakdown: %s", dict(sorted(split_counts.items())))
    logger.info("Label breakdown (items): %s", dict(sorted(label_counts.items())))

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
    logger.info("Next step: precompute vignettes with precompute_vignettes_vertex_batch.py --patients %s --corpus-dir <lumia_xml_dir>", patients_path)


if __name__ == "__main__":
    main()
