#!/usr/bin/env python3
"""
Debug script: print the exact context (cohort block + user prompt) that would be
sent to the LLM for a small sample of rows. No API call.

Usage:
  uv run python scripts/debug_context.py --config configs/ehrshot.yaml --context-cache results/context_cache.json --task guo_icu --limit 2
  uv run python scripts/debug_context.py --config configs/vista.yaml --context-cache results/context_cache.json --task guo_readmission --limit 1
"""

import argparse
import csv
import json
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


def _label_to_yes_no(raw: str) -> str:
    raw = (raw or "").strip().lower()
    return "yes" if raw in ("true", "1", "yes") else "no"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print context (cohort block + user prompt) for a small sample; no API call.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ehrshot.yaml",
        help="Config path (used for labels_dir)",
    )
    parser.add_argument(
        "--context-cache",
        type=str,
        required=True,
        help="Path to context_cache.json (patient_id|prediction_time|task_name -> context string)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g. guo_icu, lab_anemia)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of rows to show (default: 2)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("labels_dir"):
        os.environ["VISTA_LABELS_DIR"] = config["labels_dir"]

    from meds_mcp.experiments.task_config import (
        ALL_TASKS,
        TASK_QUESTIONS,
        TASK_TO_FILENAME,
        get_csv_path_for_task,
    )
    from meds_mcp.experiments.formatters import RESPONSE_FORMAT_BINARY

    if args.task not in ALL_TASKS:
        print(f"Unknown task: {args.task}")
        sys.exit(1)

    cache_path = Path(args.context_cache)
    if not cache_path.exists():
        print(f"Context cache not found: {cache_path}")
        sys.exit(1)
    with open(cache_path, "r", encoding="utf-8") as f:
        context_cache = json.load(f)

    csv_path = get_csv_path_for_task(args.task)
    if not csv_path.exists():
        print(f"Labels CSV not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            rows.append(row)
    label_col = "value" if "value" in fieldnames else "label"

    if not rows:
        print("No rows in CSV (or limit 0)")
        sys.exit(0)

    question = TASK_QUESTIONS.get(args.task, "Answer based on the patient timeline.")
    tool_name = f"get_{args.task}_prediction"

    context_parts = []
    for row in rows:
        pid = row.get("patient_id", "").strip()
        pt = row.get("prediction_time", "").strip()
        if not pid or not pt:
            continue
        key = f"{pid}|{pt}|{args.task}"
        context_text = context_cache.get(key)
        if context_text is None:
            context_parts.append(f"Patient {pid}:\n[No context in cache for key {key}]")
            continue
        raw_label = row.get(label_col) or row.get("label", "")
        label = _label_to_yes_no(raw_label)
        block = f"Patient {pid}:\n{context_text}"
        block += f"\n\nThe {tool_name} tool was already called for this patient; result: {label}."
        context_parts.append(block)

    cohort_block = "\n\n".join(context_parts)
    user_prompt = f"""Here is a cohort of patients with selected events (most recent 4096 tokens per patient, strictly before prediction_time when set):

{cohort_block}

QUESTION:
{question}

{RESPONSE_FORMAT_BINARY}
"""

    sep = "=" * 60
    print(f"\n{sep} DEBUG CONTEXT (sample n={len(rows)}, task={args.task}) {sep}")
    print("--- COHORT CONTEXT BLOCK ---")
    print(cohort_block)
    print("\n--- USER PROMPT (what the LLM sees) ---")
    print(user_prompt)
    print(f"{sep} END DEBUG {sep}\n")


if __name__ == "__main__":
    main()
