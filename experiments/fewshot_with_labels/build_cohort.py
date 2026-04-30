#!/usr/bin/env python3
"""
Build the cohort store for the fewshot_with_labels experiment.

Reads the full vista-bench progression CSV, drops rows with label == -1, and
writes two JSONL files:

  outputs/patients.jsonl   one PatientState per unique person_id (no vignette yet)
  outputs/items.jsonl      one LabeledItem per (person_id, task) after filter

Vignettes are filled in later by precompute_vignettes.py. Timelines are never
stored here — they are regenerated on demand from the XML corpus at prompt-build
time.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import (
    CohortStore,
    LabeledItem,
    PatientState,
    utc_now_iso,
)
from experiments.fewshot_with_labels import _paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


REQUIRED_COLS = (
    "person_id",
    "split",
    "embed_time",
    "task",
    "task_group",
    "question",
    "label",
    "label_description",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cohort JSONLs for fewshot_with_labels")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_paths.cohort_csv(),
        help=(
            "Source CSV (columns: %s). Overridable via env var VISTA_COHORT_CSV."
            % ", ".join(REQUIRED_COLS)
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_paths.outputs_dir(),
        help="Override via env var VISTA_OUTPUTS_DIR.",
    )
    parser.add_argument(
        "--drop-label",
        type=int,
        action="append",
        default=None,
        help="Label values to drop (repeatable). Default: [-1].",
    )
    args = parser.parse_args()

    drop_labels = set(args.drop_label) if args.drop_label else {-1}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients_path = args.output_dir / "patients.jsonl"
    items_path = args.output_dir / "items.jsonl"

    created_at = utc_now_iso()

    n_rows = 0
    n_dropped_label = 0
    n_bad_label = 0
    split_rows = Counter()
    label_rows = Counter()
    task_rows = Counter()

    # Keyed by (person_id, embed_time). A patient can have multiple states if
    # their tasks use different prediction times.
    states: dict[tuple[str, str], PatientState] = {}
    items: list[LabeledItem] = []
    pid_split: dict[str, str] = {}  # for summary only

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV missing required columns: {missing}. Got: {reader.fieldnames}")

        for row in reader:
            n_rows += 1
            pid = str(row["person_id"] or "").strip()
            if not pid:
                continue

            try:
                label = int(str(row["label"] or "").strip())
            except ValueError:
                n_bad_label += 1
                continue

            split = str(row.get("split", "") or "").strip()
            embed_time = str(row.get("embed_time", "") or "").strip()
            task = str(row.get("task", "") or "").strip()
            task_group = str(row.get("task_group", "") or "").strip()
            question = str(row.get("question", "") or "").strip()
            label_desc = str(row.get("label_description", "") or "").strip()

            split_rows[split] += 1
            label_rows[label] += 1
            task_rows[task] += 1

            if not embed_time:
                # State needs an embed_time to be meaningful.
                continue

            key = (pid, embed_time)
            existing = states.get(key)
            if existing is None:
                states[key] = PatientState(
                    person_id=pid,
                    embed_time=embed_time,
                    split=split,
                    vignette="",  # filled in by precompute_vignettes.py
                    created_at=created_at,
                )
                pid_split.setdefault(pid, split)
            elif existing.split != split:
                logger.warning(
                    "Inconsistent split for %s at %s: existing=%r, row=%r. Keeping first.",
                    pid,
                    embed_time,
                    existing.split,
                    split,
                )

            if label in drop_labels:
                n_dropped_label += 1
                continue
            if not task:
                continue

            items.append(
                LabeledItem(
                    person_id=pid,
                    task=task,
                    task_group=task_group,
                    question=question,
                    label=label,
                    label_description=label_desc,
                    split=split,
                    embed_time=embed_time,
                    created_at=created_at,
                )
            )

    store = CohortStore(states.values(), items)
    store.save(patients_path, items_path)

    # Summary
    split_pids: Counter[str] = Counter()
    for pid, sp in pid_split.items():
        split_pids[sp] += 1

    states_per_pid = Counter()
    for (pid, _et) in states:
        states_per_pid[pid] += 1
    multi_state = sum(1 for n in states_per_pid.values() if n > 1)

    kept_per_task = Counter()
    for it in items:
        kept_per_task[it.task] += 1

    logger.info("Rows read:                       %d", n_rows)
    logger.info("Rows dropped (label in %s):      %d", sorted(drop_labels), n_dropped_label)
    logger.info("Rows with bad label:             %d", n_bad_label)
    logger.info("Items written:                   %d", len(items))
    logger.info("Unique (person_id, embed_time):  %d", len(states))
    logger.info("Unique person_ids:               %d", len(states_per_pid))
    logger.info("Patients with >1 embed_time:     %d", multi_state)
    logger.info("Patients by split:               %s", dict(split_pids))
    logger.info("Rows by label (all):             %s", dict(label_rows))
    logger.info("Tasks seen:                      %d", len(task_rows))
    logger.info(
        "Kept items per task              min=%d median=%d max=%d",
        min(kept_per_task.values(), default=0),
        sorted(kept_per_task.values())[len(kept_per_task) // 2] if kept_per_task else 0,
        max(kept_per_task.values(), default=0),
    )
    logger.info("Wrote %s", patients_path)
    logger.info("Wrote %s", items_path)


if __name__ == "__main__":
    main()
