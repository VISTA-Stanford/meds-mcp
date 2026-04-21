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

    if args.all:
        chosen = eligible_ids
        pool_tag = "all"
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
        rng = random.Random(args.seed)
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
        "seed": None if args.all else args.seed,
        "require_vignette": args.require_vignette,
        "require_item": args.require_item,
        "patients_source": str(args.patients),
        "items_source": str(args.items),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote %s (%d pids)", pool_path, len(chosen))
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
