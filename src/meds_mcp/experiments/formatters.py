"""
Response formatters for vista_bench experiment output normalization.
Binary tasks: yes / no
Categorical tasks: severe / moderate / mild / normal (3, 2, 1, 0)
"""

import json
import re
from typing import Optional

STR_TO_NUM = {"severe": 3, "moderate": 2, "mild": 1, "normal": 0}
NUM_TO_STR = {3: "severe", 2: "moderate", 1: "mild", 0: "normal"}

RESPONSE_FORMAT_BINARY = (
    'You must respond with exactly one word: either "yes" or "no". '
    "Do not include any reasoning, explanation, or discussion—only the single-word answer."
)
RESPONSE_FORMAT_CATEGORICAL = (
    "You must respond with exactly one of: severe, moderate, mild, or normal. "
    "Do not include any reasoning, explanation, or discussion—only the single-word answer."
)


def normalize_binary_from_json(raw: Optional[str]) -> Optional[str]:
    """If raw is JSON with an 'outcome' field (e.g. {"outcome": "no", "reasoning": "..."}), return outcome normalized to 'yes' or 'no'."""
    if raw is None or not str(raw).strip():
        return None
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        outcome = obj.get("outcome")
        if outcome is None:
            return None
        lower = str(outcome).strip().lower()
        if lower in ("yes", "true", "1", "positive", "y"):
            return "yes"
        if lower in ("no", "false", "0", "negative", "n"):
            return "no"
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def normalize_binary(raw: Optional[str]) -> Optional[str]:
    """Normalize LLM output to 'yes' or 'no' for binary tasks.
    If the response is JSON with an 'outcome' field, uses that; otherwise extracts from free text.
    """
    if raw is None:
        return None
    # Prefer JSON {outcome: ..., reasoning: ...} when present
    from_json = normalize_binary_from_json(raw)
    if from_json is not None:
        return from_json
    lower = str(raw).strip().lower()
    if lower in ("yes", "true", "1", "positive", "y"):
        return "yes"
    if lower in ("no", "false", "0", "negative", "n"):
        return "no"
    # Extract from longer response
    text = str(raw).lower()
    yes_pat = r"\b(?:answer is|answer:|conclusion:|therefore|i (?:would )?say|final answer:?)\s*(yes|y)\b"
    no_pat = r"\b(?:answer is|answer:|conclusion:|therefore|i (?:would )?say|final answer:?)\s*(no|n)\b"
    for pattern, result in [(yes_pat, "yes"), (no_pat, "no")]:
        m = re.search(pattern, text)
        if m:
            return "yes" if result == "yes" else "no"
    tail = text[-80:] if len(text) > 80 else text
    if re.search(r"\byes\b", tail) and not re.search(r"\bno\b", tail):
        return "yes"
    if re.search(r"\bno\b", tail) and not re.search(r"\byes\b", tail):
        return "no"
    return None


def normalize_categorical(raw: Optional[str]) -> Optional[str]:
    """Normalize LLM output to severe, moderate, mild, or normal for categorical tasks.
    Extracts answer from reasoning (e.g. 'The severity appears to be moderate.').
    """
    if raw is None:
        return None
    lower = str(raw).strip().lower()
    if lower in STR_TO_NUM:
        return lower
    if lower in ("critical", "3", "high", "severe"):
        return "severe"
    if lower in ("mod", "2", "medium", "moderate"):
        return "moderate"
    if lower in ("1", "low", "mild"):
        return "mild"
    if lower in ("0", "nl", "normal", "none"):
        return "normal"
    # Extract from longer response
    text = str(raw).lower()
    patterns = [
        (r"\b(?:severity is|answer is|answer:|classification:?|level:?)\s*(severe|moderate|mild|normal)\b", 1),
        (r"\b(?:severity of |level of )?(severe|moderate|mild|normal)\b", 1),
        (r"\b(severe|moderate|mild|normal)\s*(?:severity|level)?\s*\.", 1),
    ]
    for pattern, group in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(group).lower()
    for label in ("severe", "moderate", "mild", "normal"):
        if re.search(rf"\b{label}\b", text):
            return label
    return None


def response_to_numeric(label: Optional[str]) -> Optional[int]:
    """Map categorical label (severe/moderate/mild/normal) to 0-3."""
    if label is None:
        return None
    return STR_TO_NUM.get(str(label).strip().lower())


def ground_truth_to_normalized(label_value: str, is_binary: bool) -> Optional[str]:
    """
    Convert ground truth from CSV to normalized string format.
    Binary: 'true'/'false' -> 'yes'/'no'
    Categorical: '0'/'1'/'2'/'3' -> 'normal'/'mild'/'moderate'/'severe'
    """
    if not label_value or not str(label_value).strip():
        return None
    raw = str(label_value).strip().lower()
    if is_binary:
        if raw in ("true", "1", "yes"):
            return "yes"
        if raw in ("false", "0", "no"):
            return "no"
        return None
    # Categorical: 0,1,2,3
    try:
        n = int(raw)
        return NUM_TO_STR.get(n)
    except ValueError:
        if raw in STR_TO_NUM:
            return raw
        return None
