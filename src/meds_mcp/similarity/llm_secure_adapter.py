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
_PROMPT_EHRSHOT_FILE = _PROMPTS_DIR / "vignette_prompt_EHRSHOT.txt"
_PROMPT_VISTA_FILE = _PROMPTS_DIR / "vignette_prompt_VISTA.txt"
_PROMPT_GENERIC_FILE = _PROMPTS_DIR / "vignette_prompt_generic.txt"

# Imported lazily to avoid circular imports at module load time.
def _get_task_sets() -> tuple[frozenset, frozenset]:
    from meds_mcp.experiments.task_config import BINARY_TASKS, TASK_DESCRIPTIONS
    ehrshot = frozenset(BINARY_TASKS)
    vista = frozenset(t for t in TASK_DESCRIPTIONS if t not in ehrshot)
    return ehrshot, vista


def load_vignette_prompt_for_task(task: str) -> str:
    """Return the correct vignette prompt template for ``task``.

    - EHRSHOT tasks  → ``vignette_prompt_EHRSHOT.txt``
    - VISTA tasks    → ``vignette_prompt_VISTA.txt``
    - Anything else  → ``vignette_prompt_generic.txt``
    """
    ehrshot_tasks, vista_tasks = _get_task_sets()
    if task in ehrshot_tasks:
        target = _PROMPT_EHRSHOT_FILE
    elif task in vista_tasks:
        target = _PROMPT_VISTA_FILE
    else:
        target = _PROMPT_GENERIC_FILE
    if not target.exists():
        raise FileNotFoundError(f"Prompt template not found: {target}")
    return target.read_text().strip()


# Keep the old name as an alias so existing callers don't break.
def load_vignette_prompt() -> str:
    """Load the VISTA prompt template (legacy entry point)."""
    if not _PROMPT_VISTA_FILE.exists():
        raise FileNotFoundError(f"Prompt template not found: {_PROMPT_VISTA_FILE}")
    return _PROMPT_VISTA_FILE.read_text().strip()


class SecureLLMSummarizer:
    """LLM adapter that renders a task-aware vignette prompt for a single patient.

    Selects EHRSHOT, VISTA, or generic template based on the task name.
    ``{TASK_QUESTION}`` and ``{TASK_FOCUS}`` are filled per call for templates
    that contain those placeholders; the generic template is used as-is.
    """

    def __init__(
        self,
        model: str,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.client = get_llm_client(model_name=model)
        self.model = model
        # Pre-load all three templates at init so file I/O happens once.
        self._templates = {
            "ehrshot": _PROMPT_EHRSHOT_FILE.read_text().strip() if _PROMPT_EHRSHOT_FILE.exists() else "",
            "vista": _PROMPT_VISTA_FILE.read_text().strip() if _PROMPT_VISTA_FILE.exists() else "",
            "generic": _PROMPT_GENERIC_FILE.read_text().strip() if _PROMPT_GENERIC_FILE.exists() else "",
        }
        self._ehrshot_tasks, self._vista_tasks = _get_task_sets()
        self.gen_config = get_default_generation_config(generation_overrides)

    def _template_for_task(self, task: Optional[str]) -> str:
        if task and task in self._ehrshot_tasks:
            return self._templates["ehrshot"]
        if task and task in self._vista_tasks:
            return self._templates["vista"]
        return self._templates["generic"]

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
            task_question: Substituted into ``{TASK_QUESTION}`` (no-op for generic template).
            task_focus: Substituted into ``{TASK_FOCUS}`` (no-op for generic template).
            task: Task name used to select EHRSHOT / VISTA / generic template.
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
