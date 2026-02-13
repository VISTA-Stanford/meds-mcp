"""
Task configuration for vista_bench experiments.
Maps task names to CSV files, question templates, and task types.
"""

import csv as csv_module
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LABELS_DIR = _REPO_ROOT / "data" / "collections" / "vista_bench" / "labels"

# Task type: binary (yes/no) or categorical (severe/moderate/mild/normal)
# new_hyperlipidemia and new_hypertension are binary (first-diagnosis within one year), not lab severity.
BINARY_TASKS = [
    "new_celiac",
    "guo_icu",
    "guo_los",
    "new_lupus",
    "new_acutemi",
    "new_pancan",
    "guo_readmission",
    "new_hyperlipidemia",
    "new_hypertension",
]

CATEGORICAL_TASKS = [
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hypoatremia",
    "lab_thrombocytopenia",
]

ALL_TASKS = BINARY_TASKS + CATEGORICAL_TASKS

# Task name -> CSV filename (in vista_bench/labels/)
TASK_TO_FILENAME: Dict[str, str] = {
    "guo_readmission": "labels_guo_readmission.csv",
    "new_celiac": "labels_new_celiac.csv",
    "guo_icu": "labels_guo_icu.csv",
    "guo_los": "labels_guo_los.csv",
    "new_lupus": "labels_new_lupus.csv",
    "new_acutemi": "labels_new_acutemi.csv",
    "new_pancan": "labels_new_pancan.csv",
    "new_hyperlipidemia": "labels_new_hyperlipidemia.csv",
    "new_hypertension": "labels_new_hypertension.csv",
    "lab_anemia": "labels_lab_anemia.csv",
    "lab_hyperkalemia": "labels_lab_hyperkalemia.csv",
    "lab_hypoglycemia": "labels_lab_hypoglycemia.csv",
    "lab_hypoatremia": "labels_lab_hypoatremia.csv",
    "lab_thrombocytopenia": "labels_lab_thrombocytopenia.csv",
}

# Task -> human-readable description for tool/system prompt (used in cohort_chat when task_name is set)
TASK_DESCRIPTIONS: Dict[str, str] = {
    "guo_los": (
        "This is a probabilistic model trained to predict whether a patient will have a long "
        "hospital length of stay (≥7 days) at admission time. The assistant may choose to use this "
        "tool if estimating risk of extended admission would improve reasoning, but it is not required."
    ),
    "guo_readmission": (
        "This is a probabilistic model trained to predict 30-day hospital readmission at discharge. "
        "The assistant may use this tool if post-discharge risk estimation is useful, but it does not "
        "need to invoke it unless beneficial."
    ),
    "guo_icu": (
        "This is a probabilistic model trained to predict ICU transfer during an admission. "
        "The assistant may call this tool if ICU risk estimation would improve its answer, but it is optional."
    ),
    "lab_thrombocytopenia": (
        "This is a probabilistic model trained to predict the severity category of thrombocytopenia prior "
        "to the next platelet lab result. The model outputs class probabilities (normal, mild, moderate, severe). "
        "The assistant may use this tool if anticipating lab severity improves reasoning, but it is not mandatory."
    ),
    "lab_hyperkalemia": (
        "This is a probabilistic model trained to predict hyperkalemia severity before the next potassium "
        "lab result. Use is optional and only recommended if lab forecasting supports the reasoning process."
    ),
    "lab_hypoatremia": (
        "This is a probabilistic model trained to predict hyponatremia severity before the next sodium lab "
        "value. The assistant may call this tool if anticipating electrolyte abnormalities is useful, but it is not required."
    ),
    "lab_anemia": (
        "This is a probabilistic model trained to predict anemia severity prior to the next hemoglobin result. "
        "The tool outputs probabilities over severity categories. The assistant may use this tool if "
        "anticipating hematologic abnormalities improves its reasoning."
    ),
    "new_hypertension": (
        "This is a probabilistic model trained to predict first diagnosis of hypertension within one year "
        "post-discharge. The assistant may invoke this tool if estimating incident hypertension risk is useful."
    ),
    "new_hyperlipidemia": (
        "This is a probabilistic model trained to predict first diagnosis of hyperlipidemia within one year "
        "after discharge. The assistant may choose to use this tool when long-term cardiometabolic risk estimation is relevant."
    ),
    "new_pancan": (
        "This is a probabilistic model trained to predict first diagnosis of pancreatic cancer within one year "
        "post-discharge. The assistant may call this tool if rare malignancy risk estimation is helpful."
    ),
    "new_celiac": (
        "This is a probabilistic model trained to predict first diagnosis of celiac disease within one year "
        "of discharge. The assistant may invoke this tool if autoimmune risk prediction is useful, but it is optional."
    ),
    "new_lupus": (
        "This is a probabilistic model trained to predict first diagnosis of lupus within one year post-discharge. "
        "The assistant may use this tool when systemic autoimmune risk estimation would improve its reasoning."
    ),
    "new_acutemi": (
        "This is a probabilistic model trained to predict first diagnosis of acute myocardial infarction within "
        "one year of discharge. The assistant may invoke this tool when estimating cardiovascular event risk is helpful."
    ),
    "lab_hypoglycemia": (
        "This is a probabilistic model trained to predict hypoglycemia severity before the next glucose "
        "measurement. The assistant may use this tool if anticipating glucose abnormalities improves reasoning, but it is optional."
    ),
}

# Task -> question template (one patient)
# Categorical tasks (lab_*) expect answer in: severe, moderate, mild, or normal.
TASK_QUESTIONS: Dict[str, str] = {
    "guo_readmission": "Based on this patient's health status and discharge profile, is it likely they will require rehospitalization soon after discharge?",
    "guo_los": "Based on this patient's condition at admission, is it likely that this hospitalization will require a prolonged stay?",
    "guo_icu": "Given this patient's presentation and early hospital course, is it likely they will deteriorate and require transfer to intensive care?",
    "new_hypertension": "Based on this patient's clinical history to date, is it likely they will develop chronic high blood pressure within the next year?",
    "new_hyperlipidemia": "Based on this patient's current health profile, is it likely they will be diagnosed with elevated cholesterol or lipid abnormalities within the next year?",
    "new_pancan": "Based on this patient's history and risk factors, is it likely they will be diagnosed with pancreatic cancer within the next year?",
    "new_celiac": "Based on this patient's medical history and symptoms, is it likely they will be diagnosed with celiac disease within the next year?",
    "new_lupus": "Based on this patient's clinical trajectory, is it likely they will be diagnosed with systemic lupus within the next year?",
    "new_acutemi": "Based on this patient's cardiovascular risk profile and history, is it likely they will experience a heart attack within the next year?",
    "lab_thrombocytopenia": "Based on this patient's current condition, what is the expected severity of their next platelet count result?",
    "lab_hyperkalemia": "Based on this patient's clinical status and medications, what is the expected severity of their next potassium level?",
    "lab_hypoglycemia": "Based on this patient's metabolic status, what is the expected severity of their next glucose level?",
    "lab_hypoatremia": "Based on this patient's fluid and electrolyte balance, what is the expected severity of their next sodium level?",
    "lab_anemia": "Based on this patient's hematologic status, what is the expected severity of their next hemoglobin level?",
}


def get_labels_dir() -> Path:
    """Return the labels directory. Override via VISTA_LABELS_DIR env var."""
    import os
    env_path = os.getenv("VISTA_LABELS_DIR")
    if env_path:
        return Path(env_path)
    return _LABELS_DIR


def get_csv_path_for_task(task_name: str) -> Path:
    """Get full path to the CSV for a task."""
    filename = TASK_TO_FILENAME.get(task_name)
    if not filename:
        raise ValueError(f"Unknown task: {task_name}")
    return get_labels_dir() / filename


def is_binary_task(task_name: str) -> bool:
    return task_name in BINARY_TASKS


def is_categorical_task(task_name: str) -> bool:
    return task_name in CATEGORICAL_TASKS


def get_patient_ids_from_task_csvs(
    tasks: List[str],
    limit_per_task: Optional[int] = None,
) -> List[str]:
    """
    Collect all unique patient_ids from the given task CSV files.
    Used to load only the subset of patients needed for the experiment.
    """
    patient_ids = []
    seen = set()
    labels_dir = get_labels_dir()

    for task_name in tasks:
        filename = TASK_TO_FILENAME.get(task_name)
        if not filename:
            continue
        csv_path = labels_dir / filename
        if not csv_path.exists():
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            count = 0
            for row in reader:
                if limit_per_task and count >= limit_per_task:
                    break
                pid = row.get("patient_id", "").strip()
                if pid and pid not in seen:
                    seen.add(pid)
                    patient_ids.append(pid)
                count += 1

    return patient_ids
