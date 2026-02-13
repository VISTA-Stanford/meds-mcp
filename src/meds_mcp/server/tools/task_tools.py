"""
Task-specific prediction tools for vista_bench experiments.
Each task has a CSV with patient_id, prediction_time, label.
Tools look up the label and return it (for LLM support - tool as support, not authority).
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

from meds_mcp.experiments.task_config import (
    ALL_TASKS,
    BINARY_TASKS,
    CATEGORICAL_TASKS,
    get_csv_path_for_task,
    is_binary_task,
)
from meds_mcp.experiments.formatters import NUM_TO_STR


def _parse_prediction_time(ts: Optional[str]) -> Optional[str]:
    """Normalize prediction_time for comparison (handle ISO format)."""
    if not ts or not str(ts).strip():
        return None
    s = str(ts).strip()
    # Compare as string for exact match, or truncate to avoid microsecond mismatches
    if "T" in s:
        return s.split(".")[0] if "." in s else s
    return s


async def get_task_prediction(
    person_id: str,
    task_name: str,
    prediction_time: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Look up the label for a patient from the task's CSV.
    Used by experiment tools - returns ground truth for that (patient_id, prediction_time).
    Binary tasks: label is true/false -> returned as yes/no
    Categorical tasks: label is 0,1,2,3 -> returned as normal,mild,moderate,severe
    """
    if task_name not in ALL_TASKS:
        return {
            "error": f"Unknown task: {task_name}",
            "patient_id": person_id,
            "label": None,
        }
    path = Path(csv_path) if csv_path else get_csv_path_for_task(task_name)
    if not path.exists():
        return {
            "error": f"CSV not found: {path}",
            "patient_id": person_id,
            "label": None,
        }
    pred_norm = _parse_prediction_time(prediction_time) if prediction_time else None
    is_binary = is_binary_task(task_name)

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "patient_id" not in (reader.fieldnames or []):
                return {
                    "error": "CSV has no 'patient_id' column",
                    "patient_id": person_id,
                    "label": None,
                }
            for row in reader:
                if row.get("patient_id") != person_id:
                    continue
                if pred_norm:
                    row_pt = _parse_prediction_time(row.get("prediction_time"))
                    if row_pt != pred_norm:
                        continue
                raw_label = (row.get("label") or "").strip()
                if is_binary:
                    label_str = "yes" if raw_label.lower() in ("true", "1", "yes") else "no"
                    return {
                        "patient_id": person_id,
                        "label": label_str,
                        "task": task_name,
                    }
                # Categorical: 0,1,2,3 -> normal,mild,moderate,severe
                try:
                    n = int(raw_label)
                    label_str = NUM_TO_STR.get(n, str(n))
                    numeric = n
                except (ValueError, TypeError):
                    label_str = raw_label or None
                    numeric = None
                return {
                    "patient_id": person_id,
                    "label": label_str,
                    "numeric": numeric,
                    "task": task_name,
                }
    except OSError as e:
        return {
            "error": f"Failed to read CSV: {e}",
            "patient_id": person_id,
            "label": None,
        }

    return {
        "error": f"Patient {person_id} not found in {task_name} labels",
        "patient_id": person_id,
        "label": None,
    }


def get_task_tool_definition(
    task_name: str,
    prediction_time: Optional[str] = None,
) -> Dict[str, Any]:
    """OpenAI-format tool definition for a task. Includes prediction_time when provided."""
    from meds_mcp.experiments.task_config import TASK_DESCRIPTIONS

    desc = TASK_DESCRIPTIONS.get(task_name, task_name)
    tool_name = f"get_{task_name}_prediction"
    # Use a generic name that maps to our get_task_prediction dispatcher
    props = {
        "person_id": {
            "type": "string",
            "description": "Patient ID to look up.",
        }
    }
    required = ["person_id"]
    if prediction_time:
        props["prediction_time"] = {
            "type": "string",
            "description": "Prediction time (ISO format) to match the specific task instance.",
        }
        required.append("prediction_time")

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Look up the {desc} for a patient from the labels dataset. Use this to inform your answer. The result is for support—you may use your own judgment after seeing it.",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


# Map tool name (from LLM) -> task_name for execute_cohort_tool_call
def tool_name_to_task(tool_name: str) -> Optional[str]:
    """e.g. get_guo_readmission_prediction -> guo_readmission"""
    prefix = "get_"
    suffix = "_prediction"
    if tool_name.startswith(prefix) and tool_name.endswith(suffix):
        middle = tool_name[len(prefix) : -len(suffix)]
        return middle if middle in ALL_TASKS else None
    return None
