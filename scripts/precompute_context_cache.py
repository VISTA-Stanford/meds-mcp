#!/usr/bin/env python3
"""
Precompute formatted context (delta-encoded, 4096 tokens, lab-aware) for each
(patient_id, prediction_time, task_name) in the task label CSVs.

Output: context_cache.json
  Keys: "patient_id|prediction_time|task_name"
  Values: formatted context string (same format as cohort_chat uses)

Uses single-visit events: either from --events-cache (single_visit_events_cache.json)
or by parsing XML (requires --config with corpus_dir).

Usage:
  # From existing events cache (fast; run precompute_single_visit_events.py first)
  python scripts/precompute_context_cache.py --events-cache results/single_visit_events_cache.json --output-dir results

  # From XML (no events cache)
  python scripts/precompute_context_cache.py --config configs/vista.yaml --output-dir results

  python scripts/precompute_context_cache.py --config configs/vista.yaml --task guo_readmission --limit 10
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Precompute formatted context (delta-encoded, 4096 tokens) per (patient_id, prediction_time, task_name)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vista.yaml",
        help="Path to config (for corpus_dir when not using --events-cache)",
    )
    parser.add_argument(
        "--events-cache",
        type=str,
        default=None,
        help="Path to single_visit_events_cache.json; when set, events are loaded from here instead of XML",
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
        help="Output directory for context cache file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output path for context cache JSON. Overrides --output-dir.",
    )
    args = parser.parse_args()

    from meds_mcp.experiments.task_config import (
        ALL_TASKS,
        get_csv_path_for_task,
    )
    from meds_mcp.server.rag.context_formatter import format_patient_context

    tasks_to_run = [args.task] if args.task else ALL_TASKS
    if args.task and args.task not in ALL_TASKS:
        logging.error(f"Unknown task: {args.task}")
        sys.exit(1)

    # Load events cache if provided
    events_cache: dict[str, list] = {}
    if args.events_cache:
        cache_path = Path(args.events_cache)
        if not cache_path.exists():
            logging.error(f"Events cache not found: {cache_path}")
            sys.exit(1)
        with open(cache_path, "r", encoding="utf-8") as f:
            events_cache = json.load(f)
        logging.info(f"Loaded {len(events_cache)} event lists from {args.events_cache}")

    # When not using events cache, we need XML path for get_events_for_single_visit_from_xml
    data_path = None
    if not events_cache:
        config = load_config(args.config)
        data_dir = config.get("data", {}).get("corpus_dir") or os.getenv(
            "DATA_DIR", "data/collections/vista_bench/thoracic_cohort_lumia"
        )
        data_path = Path(data_dir)
        if not data_path.exists():
            logging.error(f"Corpus directory not found: {data_dir}. Use --events-cache or fix config.")
            sys.exit(1)
        from meds_mcp.server.rag.visit_filter import get_events_for_single_visit_from_xml

    context_cache: dict[str, str] = {}

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

                key = f"{pid}|{pt}|{task_name}"
                if key in context_cache:
                    count += 1
                    continue

                if events_cache:
                    events_key = f"{pid}|{pt}"
                    events = events_cache.get(events_key)
                    if events is None:
                        continue
                else:
                    events = get_events_for_single_visit_from_xml(pid, pt, str(data_path))

                if not events:
                    continue

                context_text = format_patient_context(
                    events,
                    patient_id=pid,
                    prediction_time=pt,
                    task_name=task_name,
                    max_tokens=4096,
                    include_event_key=True,
                )
                if context_text:
                    context_cache[key] = context_text
                count += 1

        logging.info(f"Task {task_name}: processed {count} rows")

    output_path = Path(args.output) if args.output else Path(args.output_dir) / "context_cache.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(context_cache, f, ensure_ascii=False, indent=None)

    logging.info(f"Precomputed {len(context_cache)} (patient_id|prediction_time|task_name) -> context text. Saved to {output_path}")


if __name__ == "__main__":
    main()
