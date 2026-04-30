"""
Task configuration for vista_bench experiments.
Maps task names to CSV files, question templates, and task types.
"""

import csv as csv_module
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LABELS_DIR = _REPO_ROOT / "data" / "collections" / "ehrshot" / "labels" / "labels_100_0pct_flip"

# All tasks are binary (yes/no). Lab tasks are binarized to normal vs abnormal in label CSVs.
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
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_thrombocytopenia",
    "chexpert",
]

CATEGORICAL_TASKS: List[str] = []  # Kept for backward compat; all tasks treated as binary

# Lab task names (for context formatting: value/unit, collapse by day)
LAB_TASK_NAMES = {
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_thrombocytopenia",
}

ALL_TASKS = BINARY_TASKS

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
    "lab_hyponatremia": "labels_lab_hyponatremia.csv",
    "lab_thrombocytopenia": "labels_lab_thrombocytopenia.csv",
    "chexpert": "labels_chexpert.csv",
}

# Task -> short focus statement for vignette generation (1-2 sentences).
# Substituted into the {TASK_FOCUS} placeholder of configs/prompts/vignette_prompt.example.txt
# so the LLM steers the vignette toward information most relevant to the downstream task.
TASK_DESCRIPTIONS: Dict[str, str] = {
    # EHRSHOT admin / lab / incident-disease tasks
    "guo_los": "Highlight admission acuity, comorbidity burden, prior length-of-stay patterns, and any factors likely to delay discharge.",
    "guo_readmission": "Highlight discharge stability, recent hospitalizations, unresolved issues at discharge, and post-discharge support.",
    "guo_icu": "Highlight early hospital course severity, vital trends, escalating oxygen or pressor support, and signs of clinical deterioration.",
    "new_hypertension": "Highlight blood-pressure trajectory, cardiovascular risk factors, family history, and lifestyle factors.",
    "new_hyperlipidemia": "Highlight lipid-panel trends, cardiovascular risk factors, diet and lifestyle, and family history.",
    "new_pancan": "Highlight pancreatic-cancer risk factors (smoking, family history, new-onset diabetes), abdominal symptoms, and abnormal pancreatic imaging or labs.",
    "new_celiac": "Highlight GI symptoms, autoimmune comorbidities, family history, anemia, and any tTG-IgA results.",
    "new_lupus": "Highlight autoimmune symptoms (joint pain, rash, serositis), ANA / dsDNA results, cytopenias, and family history.",
    "new_acutemi": "Highlight cardiovascular risk factors, prior MI / CAD, recent chest pain, troponin / EKG abnormalities, and lipid and diabetes control.",
    "lab_thrombocytopenia": "Highlight platelet trajectory, recent transfusions, marrow-suppressing drugs, infections, and any bleeding.",
    "lab_hyperkalemia": "Highlight potassium trajectory, renal function, ACEi / ARB / spironolactone use, and acid-base status.",
    "lab_hypoglycemia": "Highlight glucose trajectory, insulin or sulfonylurea use, recent oral intake, and hepatic / renal function.",
    "lab_hyponatremia": "Highlight sodium trajectory, volume status, diuretic use, SIADH context, and renal function.",
    "lab_anemia": "Highlight hemoglobin trajectory, recent bleeding, transfusions, marrow-suppressing therapy, and iron / B12 status.",
    "chexpert": "Highlight pulmonary symptoms, recent infections, oxygen requirement, prior chest-imaging findings, and cardiac comorbidities.",
}


# ---------------------------------------------------------------------------
# Thoracic-oncology outcome tasks present in
# experiments/fewshot_with_labels/outputs/items.jsonl. Eight families × five
# horizons (1, 2, 3, 4, 5 years) = 40 entries, generated from a template per
# family so the wording stays consistent.
# ---------------------------------------------------------------------------

_THORACIC_HORIZON_TASK_TEMPLATES: Dict[str, str] = {
    "died_any_cause": "Highlight stage at diagnosis, treatment response, performance status, comorbidity burden, and any progression or deterioration relevant to {n}-year overall survival.",
    "died_of_cancer": "Highlight cancer stage, histology, treatment response, sites of disease, and progression markers relevant to {n}-year cancer-specific mortality.",
    "died_other_cause": "Highlight non-cancer comorbidities (cardiopulmonary, vascular, frailty), competing causes, and overall medical status — separate from the cancer trajectory — relevant to {n}-year non-cancer mortality.",
    "has_progression_nonrecurrence": "Highlight current disease status, treatment received, response, and any imaging or biomarker signals of progression within {n} year{s}.",
    "has_recurrence": "Highlight curative treatment received, response status, surveillance findings, and risk factors for relapse within {n} year{s}.",
    "has_stable_disease": "Highlight current treatment, recent response, surveillance findings, and stability markers over the {n}-year window.",
    "is_cured_by_horizon": "Highlight curative treatment, response, recurrence-free interval, and absence of progression markers relevant to cure at the {n}-year horizon.",
    "progression_recurrence_free_survival": "Highlight stage, curative treatment, surveillance findings, and progression / recurrence markers relevant to {n}-year disease-free survival.",
}

for _family, _template in _THORACIC_HORIZON_TASK_TEMPLATES.items():
    for _n in (1, 2, 3, 4, 5):
        TASK_DESCRIPTIONS[f"{_family}_{_n}_yr"] = _template.format(
            n=_n, s="" if _n == 1 else "s",
        )

del _family, _template, _n

# Task -> clause for tool description: "the likelihood that {phrase}" (no "whether").
TASK_TOOL_RETURNS: Dict[str, str] = {
    "guo_icu": "the specified patient will require ICU admission within 24 hours of the provided prediction time",
    "guo_los": "the specified patient will have a prolonged hospital length of stay (≥7 days) from admission",
    "guo_readmission": "the specified patient will be rehospitalized within 30 days of discharge",
    "new_hypertension": "the specified patient will receive a first-time diagnosis of hypertension within the 1-year period following the prediction time",
    "new_hyperlipidemia": "the specified patient will receive a first-time diagnosis of hyperlipidemia within the 1-year period following the prediction time",
    "new_pancan": "the specified patient will receive a first-time diagnosis of pancreatic cancer within the 1-year period following the prediction time",
    "new_celiac": "the specified patient will receive a first-time diagnosis of celiac disease within the 1-year period following the prediction time",
    "new_lupus": "the specified patient will receive a first-time diagnosis of lupus within the 1-year period following the prediction time",
    "new_acutemi": "the specified patient will experience an acute myocardial infarction within the 1-year period following the prediction time",
    "lab_thrombocytopenia": "the specified patient's next platelet count will be abnormal (prior to the next lab result)",
    "lab_hyperkalemia": "the specified patient's next potassium level will be abnormal (prior to the next lab result)",
    "lab_hypoglycemia": "the specified patient's next glucose level will be abnormal (prior to the next lab result)",
    "lab_hyponatremia": "the specified patient's next sodium level will be abnormal (prior to the next lab result)",
    "lab_anemia": "the specified patient's next hemoglobin level will be abnormal (prior to the next lab result)",
    "chexpert": "the specified patient's chest imaging will show an abnormal finding",
}

# Task -> prediction target phrasing for user prompt (time horizon + target).
TASK_PREDICTION_TARGET: Dict[str, str] = {
    "guo_icu": "will require ICU admission within 24 hours of the prediction time",
    "guo_los": "will have a prolonged hospital length of stay (≥7 days) from admission",
    "guo_readmission": "will be rehospitalized within 30 days of discharge",
    "new_hypertension": "will receive a first-time diagnosis of hypertension within the 1-year period following the prediction time",
    "new_hyperlipidemia": "will receive a first-time diagnosis of hyperlipidemia within the 1-year period following the prediction time",
    "new_pancan": "will receive a first-time diagnosis of pancreatic cancer within the 1-year period following the prediction time",
    "new_celiac": "will receive a first-time diagnosis of celiac disease within the 1-year period following the prediction time",
    "new_lupus": "will receive a first-time diagnosis of lupus within the 1-year period following the prediction time",
    "new_acutemi": "will experience an acute myocardial infarction (heart attack) within the 1-year period following the prediction time",
    "lab_thrombocytopenia": "will have an abnormal next platelet count (prior to the next lab result)",
    "lab_hyperkalemia": "will have an abnormal next potassium level (prior to the next lab result)",
    "lab_hypoglycemia": "will have an abnormal next glucose level (prior to the next lab result)",
    "lab_hyponatremia": "will have an abnormal next sodium level (prior to the next lab result)",
    "lab_anemia": "will have an abnormal next hemoglobin level (prior to the next lab result)",
    "chexpert": "will have chest imaging showing an abnormal finding",
}

# Task -> question template (one patient). All questions are binary (yes/no).
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
    "lab_thrombocytopenia": "Based on this patient's current condition, is it likely their next platelet count will be abnormal?",
    "lab_hyperkalemia": "Based on this patient's clinical status and medications, is it likely their next potassium level will be abnormal?",
    "lab_hypoglycemia": "Based on this patient's metabolic status, is it likely their next glucose level will be abnormal?",
    "lab_hyponatremia": "Based on this patient's fluid and electrolyte balance, is it likely their next sodium level will be abnormal?",
    "lab_anemia": "Based on this patient's hematologic status, is it likely their next hemoglobin level will be abnormal?",
    "chexpert": "Based on this patient's clinical data, is it likely their chest imaging will show an abnormal finding?",
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
