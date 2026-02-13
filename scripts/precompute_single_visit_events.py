#!/usr/bin/env python3
"""
Precompute single-visit events for all (patient_id, prediction_time) pairs in task CSVs.
Run once before the experiment to avoid per-request XML parsing and speed up API calls.

Output: single_visit_events_cache.json (in --output-dir or results/)
  Keys: "patient_id|prediction_time"
  Values: list of event dicts (format compatible with cohort_chat)

Usage:
  python scripts/precompute_single_visit_events.py --config configs/vista.yaml
  python scripts/precompute_single_visit_events.py --config configs/vista.yaml --task guo_readmission --limit 100
  python scripts/precompute_single_visit_events.py --config configs/vista.yaml --output-dir results
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Precompute single-visit events for vista_bench (patient_id, prediction_time) pairs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vista.yaml",
        help="Path to config (for corpus_dir)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run only this task. Default: all tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows per task (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for cache file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output path for cache JSON. Overrides --output-dir.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.get("data", {}).get("corpus_dir") or os.getenv(
        "DATA_DIR", "data/collections/vista_bench/thoracic_cohort_lumia"
    )
    data_path = Path(data_dir)
    if not data_path.exists():
        logging.error(f"Corpus directory not found: {data_dir}")
        sys.exit(1)

    from meds_mcp.experiments.task_config import (
        ALL_TASKS,
        get_csv_path_for_task,
        get_labels_dir,
    )
    from meds_mcp.server.rag.visit_filter import get_events_for_single_visit_from_xml

    tasks_to_run = [args.task] if args.task else ALL_TASKS
    if args.task and args.task not in ALL_TASKS:
        logging.error(f"Unknown task: {args.task}")
        sys.exit(1)

    seen: set[tuple[str, str]] = set()
    cache: dict[str, list] = {}

    for task_name in tasks_to_run:
        csv_path = get_csv_path_for_task(task_name)
        if not csv_path.exists():
            logging.warning(f"CSV not found for {task_name}: {csv_path}, skipping")
            continue

        count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.limit and count >= args.limit:
                    break
                pid = row.get("patient_id", "").strip()
                pt = row.get("prediction_time", "").strip()
                if not pid or not pt:
                    continue
                if (pid, pt) in seen:
                    continue
                seen.add((pid, pt))
                count += 1

                events = get_events_for_single_visit_from_xml(pid, pt, str(data_path))
                key = f"{pid}|{pt}"
                cache[key] = events

        logging.info(f"Task {task_name}: processed {count} rows")

    output_path = Path(args.output) if args.output else Path(args.output_dir) / "single_visit_events_cache.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=None, default=str)

    logging.info(f"Precomputed {len(cache)} (patient_id, prediction_time) -> events. Saved to {output_path}")


if __name__ == "__main__":
    main()
