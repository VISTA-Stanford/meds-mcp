"""Secure LLM adapter for task-aware vignette generation.

Single source of truth for the vignette system prompt:
``configs/prompts/vignette_prompt.txt`` (or ``vignette_prompt.example.txt``
as a tracked default). The template MUST contain ``{TASK_QUESTION}`` and
``{TASK_FOCUS}`` placeholders. There is no hardcoded fallback prompt.
"""

from typing import Any, Dict, Optional
from pathlib import Path

from meds_mcp.server.llm import (
    get_llm_client,
    extract_response_content,
    get_default_generation_config,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"
_PROMPT_FILE = _PROMPTS_DIR / "vignette_prompt.txt"
_PROMPT_EHRSHOT_FILE = _PROMPTS_DIR / "vignette_prompt_EHRSHOT.txt"
_PROMPT_EXAMPLE_FILE = _PROMPTS_DIR / "vignette_prompt.example.txt"

# EHRSHOT task names — these use the EHRSHOT-specific prompt template.
_EHRSHOT_TASKS = {
    "guo_icu", "guo_los", "guo_readmission",
    "new_celiac", "new_lupus", "new_acutemi", "new_pancan",
    "new_hyperlipidemia", "new_hypertension",
    "lab_anemia", "lab_hyperkalemia", "lab_hypoglycemia",
    "lab_hyponatremia", "lab_thrombocytopenia",
    "chexpert",
}


def _validate_template(text: str, path: Path) -> str:
    for placeholder in ("{TASK_QUESTION}", "{TASK_FOCUS}"):
        if placeholder not in text:
            raise ValueError(
                f"Prompt template {path} is missing required placeholder {placeholder}."
            )
    return text


def load_vignette_prompt() -> str:
    """Load the default (VISTA/thoracic) vignette prompt template.

    Prefers ``vignette_prompt.txt`` (gitignored, for local customization) and
    falls back to the tracked ``vignette_prompt.example.txt``.
    """
    for candidate in (_PROMPT_FILE, _PROMPT_EXAMPLE_FILE):
        if candidate.exists():
            return _validate_template(candidate.read_text().strip(), candidate)
    raise FileNotFoundError(
        f"No vignette prompt template found at {_PROMPT_FILE} or {_PROMPT_EXAMPLE_FILE}."
    )


def load_vignette_prompt_for_task(task: str) -> str:
    """Load the appropriate vignette prompt template for a given task.

    EHRSHOT tasks (guo_*, lab_*, new_*, chexpert) use
    ``vignette_prompt_EHRSHOT.txt``. All other tasks (VISTA/thoracic horizon
    tasks) use ``vignette_prompt.txt`` / ``vignette_prompt.example.txt``.
    In both cases ``vignette_prompt.txt`` takes priority as a local override.
    """
    if task in _EHRSHOT_TASKS:
        for candidate in (_PROMPT_FILE, _PROMPT_EHRSHOT_FILE, _PROMPT_EXAMPLE_FILE):
            if candidate.exists():
                return _validate_template(candidate.read_text().strip(), candidate)
    else:
        for candidate in (_PROMPT_FILE, _PROMPT_EXAMPLE_FILE):
            if candidate.exists():
                return _validate_template(candidate.read_text().strip(), candidate)
    raise FileNotFoundError(
        f"No vignette prompt template found for task '{task}' under {_PROMPTS_DIR}."
    )


class SecureLLMSummarizer:
    """LLM adapter that renders a task-aware vignette prompt for a single patient.

    The system prompt is selected per task (EHRSHOT vs VISTA) with
    ``{TASK_QUESTION}`` and ``{TASK_FOCUS}`` filled per call.
    The user message is the patient timeline.
    """

    def __init__(
        self,
        model: str,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.client = get_llm_client(model_name=model)
        self.model = model
        # Pre-load both templates at init so file I/O happens once.
        self._ehrshot_template = (
            _validate_template(_PROMPT_EHRSHOT_FILE.read_text().strip(), _PROMPT_EHRSHOT_FILE)
            if _PROMPT_EHRSHOT_FILE.exists() else load_vignette_prompt()
        )
        self._default_template = load_vignette_prompt()
        self.gen_config = get_default_generation_config(generation_overrides)

    def _template_for_task(self, task: Optional[str]) -> str:
        if task and task in _EHRSHOT_TASKS:
            return self._ehrshot_template
        return self._default_template

    def render_system_prompt(self, task_question: str, task_focus: str, task: Optional[str] = None) -> str:
        """Substitute the task fields into the appropriate template."""
        return self._template_for_task(task).format(
            TASK_QUESTION=task_question.strip(),
            TASK_FOCUS=task_focus.strip(),
        )

    def summarize(
        self,
        text: str,
        *,
        task_question: str,
        task_focus: str,
        task: Optional[str] = None,
    ) -> str:
        """Generate a task-aware clinical vignette.

        Args:
            text: Patient timeline (with demographics prelude).
            task_question: The exact downstream-task question. Substituted into
                ``{TASK_QUESTION}``.
            task_focus: 1-2 sentence focus statement for the task. Substituted
                into ``{TASK_FOCUS}``.
            task: Task name used to select the correct prompt template
                (EHRSHOT vs VISTA). If None, defaults to the VISTA template.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.render_system_prompt(task_question, task_focus, task),
                },
                {"role": "user", "content": text},
            ],
            **self.gen_config,
        )
        return extract_response_content(response)
