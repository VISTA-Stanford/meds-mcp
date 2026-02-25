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
    get_csv_path_for_task,
)


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
    Look up the label for a patient from the task's CSV (one row per patient in preprocessed data).
    Returns ground truth as yes/no for that (patient_id, prediction_time).
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
                raw_label = (row.get("value") or row.get("label") or "").strip().lower()
                label_str = "yes" if raw_label in ("true", "1", "yes") else "no"
                return {
                    "patient_id": person_id,
                    "label": label_str,
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
    """OpenAI-format tool definition for a task. Clear description of what the tool returns and what inputs it requires."""
    from meds_mcp.experiments.task_config import TASK_TOOL_RETURNS

    returns_phrase = TASK_TOOL_RETURNS.get(
        task_name,
        f"whether the specified patient has the outcome for task {task_name}",
    )
    tool_name = f"get_{task_name}_prediction"
    description = f"Returns a binary prediction indicating {returns_phrase}."
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {
                        "type": "string",
                        "description": "Unique identifier for the patient.",
                    },
                    "prediction_time": {
                        "type": "string",
                        "description": "Timestamp representing the time at which the prediction is made. Format: YYYY-MM-DD HH:MM",
                    },
                },
                "required": ["person_id", "prediction_time"],
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
