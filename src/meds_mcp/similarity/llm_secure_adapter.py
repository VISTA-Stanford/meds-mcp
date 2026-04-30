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
_PROMPT_EXAMPLE_FILE = _PROMPTS_DIR / "vignette_prompt.example.txt"


def load_vignette_prompt() -> str:
    """Load the vignette prompt template.

    Prefers ``vignette_prompt.txt`` (gitignored, for local customization) and
    falls back to the tracked ``vignette_prompt.example.txt``. Raises if
    neither exists or the template is missing the required placeholders.
    """
    for candidate in (_PROMPT_FILE, _PROMPT_EXAMPLE_FILE):
        if candidate.exists():
            text = candidate.read_text().strip()
            for placeholder in ("{TASK_QUESTION}", "{TASK_FOCUS}"):
                if placeholder not in text:
                    raise ValueError(
                        f"Prompt template {candidate} is missing required "
                        f"placeholder {placeholder}."
                    )
            return text
    raise FileNotFoundError(
        f"No vignette prompt template found at {_PROMPT_FILE} or {_PROMPT_EXAMPLE_FILE}."
    )


class SecureLLMSummarizer:
    """LLM adapter that renders a task-aware vignette prompt for a single patient.

    The system prompt is the loaded template with ``{TASK_QUESTION}`` and
    ``{TASK_FOCUS}`` filled per call. The user message is the patient timeline.
    """

    def __init__(
        self,
        model: str,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.client = get_llm_client(model_name=model)
        self.model = model
        self.template = load_vignette_prompt()
        self.gen_config = get_default_generation_config(generation_overrides)

    def render_system_prompt(self, task_question: str, task_focus: str) -> str:
        """Substitute the task fields into the template."""
        return self.template.format(
            TASK_QUESTION=task_question.strip(),
            TASK_FOCUS=task_focus.strip(),
        )

    def summarize(
        self,
        text: str,
        *,
        task_question: str,
        task_focus: str,
    ) -> str:
        """Generate a task-aware clinical vignette.

        Args:
            text: Patient timeline (with demographics prelude).
            task_question: The exact downstream-task question. Substituted into
                ``{TASK_QUESTION}``.
            task_focus: 1-2 sentence focus statement for the task. Substituted
                into ``{TASK_FOCUS}``.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.render_system_prompt(task_question, task_focus),
                },
                {"role": "user", "content": text},
            ],
            **self.gen_config,
        )
        return extract_response_content(response)
