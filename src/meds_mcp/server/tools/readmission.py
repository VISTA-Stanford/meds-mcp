"""
Readmission prediction tool: look up the loaded patient's predicted readmission
label from the readmission_labels CSV.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Default CSV path relative to repo root (5 levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_CSV_PATH = _REPO_ROOT / "data" / "ehrshot" / "test" / "readmission_labels.csv"


def _get_csv_path(csv_path: Optional[str] = None) -> Path:
    """Resolve path to readmission CSV: arg > env > default."""
    if csv_path:
        return Path(csv_path)
    env_path = os.getenv("READMISSION_CSV")
    if env_path:
        return Path(env_path)
    return _DEFAULT_CSV_PATH


async def get_readmission_prediction(
    person_id: str,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For the given (loaded) patient, look up their predicted readmission from the
    readmission_labels CSV and return whether they are predicted to have readmission.

    Args:
        person_id: Patient ID (e.g. the currently loaded patient).
        csv_path: Optional path to the readmission CSV. If not set, uses
            READMISSION_CSV env var or default data/ehrshot/test/readmission_labels.csv.

    Returns:
        Dict with keys: patient_id, readmission (categorical label, e.g. "yes"/"no"),
        and optionally error if the CSV is missing or the patient is not in the CSV.
    """
    path = _get_csv_path(csv_path)
    if not path.exists():
        return {
            "error": f"Readmission CSV not found: {path}",
            "patient_id": person_id,
            "readmission": None,
        }

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "patient_id" not in (reader.fieldnames or []):
                return {
                    "error": "Readmission CSV has no 'patient_id' column",
                    "patient_id": person_id,
                    "readmission": None,
                }
            for row in reader:
                if row.get("patient_id") == person_id:
                    label = row.get("readmission", "").strip()
                    return {
                        "patient_id": person_id,
                        "readmission": label or None,
                        "predicted_readmission": label.lower() in ("yes", "1", "true") if label else None,
                    }
    except OSError as e:
        return {
            "error": f"Failed to read readmission CSV: {e}",
            "patient_id": person_id,
            "readmission": None,
        }

    return {
        "error": f"Patient {person_id} not found in readmission labels",
        "patient_id": person_id,
        "readmission": None,
    }
