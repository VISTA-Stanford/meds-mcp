#!/usr/bin/env python3
"""
Sample N unique person_ids from the valid split to form the evaluation pool.

Pool composition is deterministic given --seed and the sorted list of valid
person_ids present in patients.jsonl. Writes outputs/pool_valid_<N>.json with
a sorted list of selected person_ids plus a small sidecar manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from meds_mcp.similarity import CohortStore
from experiments.fewshot_with_labels import _paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample evaluation pool from valid split")
    parser.add_argument(
        "--patients",
        type=Path,
        default=_paths.patients_jsonl(),
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=_paths.items_jsonl(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_paths.outputs_dir(),
    )
    parser.add_argument("--n", type=int, default=100, help="Pool size (unique person_ids)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include every eligible patient in the split (no sampling). Overrides --n and --seed.",
    )
    parser.add_argument("--split", type=str, default="valid", help="Eligible split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-vignette",
        action="store_true",
        help="Only sample patients that already have a non-empty vignette.",
    )
    parser.add_argument(
        "--require-item",
        action="store_true",
        help="Only sample patients that have at least one LabeledItem after filtering.",
    )
    parser.add_argument(
        "--stratify-task",
        type=str,
        default=None,
        help=(
            "When set, sample patients stratified by binary label (0/1) for this task. "
            "Selects --n-per-label patients from each label class. "
            "Patients without a valid label for this task are excluded from the stratified pool."
        ),
    )
    parser.add_argument(
        "--n-per-label",
        type=int,
        default=None,
        help=(
            "Patients per label class when --stratify-task is set. "
            "Default: --n // number_of_label_classes (e.g. 50 each for binary)."
        ),
    )
    args = parser.parse_args()

    store = CohortStore.load(args.patients, args.items)

    # Sampling is patient-level (pid), not state-level. A patient is eligible
    # if they have at least one PatientState in the target split that
    # (optionally) satisfies --require-vignette, and (optionally) has at least
    # one LabeledItem.
    eligible_set: set[str] = set()
    for p in store.patient_states():
        if p.split != args.split:
            continue
        if args.require_vignette and not p.vignette.strip():
            continue
        if args.require_item and not store.items_for_patient(p.person_id):
            continue
        eligible_set.add(p.person_id)
    eligible_ids = sorted(eligible_set)

    rng = random.Random(args.seed)

    if args.all:
        chosen = eligible_ids
        pool_tag = "all"
    elif args.stratify_task:
        # Stratified sampling: sample evenly across label classes for one task.
        label_groups: dict[int, list[str]] = {}
        for pid in eligible_ids:
            items = [
                it
                for it in store.items_for_patient(pid)
                # Only look at items from the same split being sampled (test),
                # so train-split labels never influence test-pool stratification.
                if it.task == args.stratify_task and it.label != -1 and it.split == args.split
            ]
            if not items:
                continue
            label = int(items[0].label)
            label_groups.setdefault(label, []).append(pid)

        n_classes = len(label_groups)
        if n_classes == 0:
            logger.error("No patients with a valid label for task=%s in split=%s", args.stratify_task, args.split)
            return

        n_per_label = args.n_per_label or (args.n // n_classes)
        chosen = []
        for label in sorted(label_groups):
            pids = sorted(label_groups[label])
            if len(pids) < n_per_label:
                logger.warning(
                    "Label %d for task=%s has only %d patients (requested %d); taking all.",
                    label, args.stratify_task, len(pids), n_per_label,
                )
                chosen.extend(pids)
            else:
                chosen.extend(sorted(rng.sample(pids, n_per_label)))
        chosen = sorted(chosen)
        pool_tag = str(args.n)
        logger.info(
            "Stratified pool for task=%s: %s",
            args.stratify_task,
            {label: len(label_groups[label]) for label in sorted(label_groups)},
        )
    elif len(eligible_ids) < args.n:
        logger.warning(
            "Only %d eligible patients in split=%s (requested %d); returning all.",
            len(eligible_ids),
            args.split,
            args.n,
        )
        chosen = eligible_ids
        pool_tag = str(args.n)
    else:
        chosen = sorted(rng.sample(eligible_ids, args.n))
        pool_tag = str(args.n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / f"pool_{args.split}_{pool_tag}.json"
    manifest_path = args.output_dir / f"pool_{args.split}_{pool_tag}_manifest.json"

    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)

    manifest = {
        "pool_path": str(pool_path),
        "split": args.split,
        "all": args.all,
        "n_requested": None if args.all else args.n,
        "n_chosen": len(chosen),
        "n_eligible": len(eligible_ids),
        "seed": None if (args.all or args.stratify_task) else args.seed,
        "require_vignette": args.require_vignette,
        "require_item": args.require_item,
        "stratify_task": args.stratify_task,
        "n_per_label": args.n_per_label,
        "patients_source": str(args.patients),
        "items_source": str(args.items),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote %s (%d pids)", pool_path, len(chosen))
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
