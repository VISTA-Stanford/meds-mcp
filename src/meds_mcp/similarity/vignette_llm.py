"""LLM-based vignette generator (decorator over a base generator)."""

from typing import Optional

from .llm_secure_adapter import SecureLLMSummarizer
from .vignette_base import BaseVignetteGenerator


class LLMVignetteGenerator(BaseVignetteGenerator):
    """Wraps a base vignette generator with task-aware LLM summarization."""

    def __init__(
        self,
        base_generator: BaseVignetteGenerator,
        llm: SecureLLMSummarizer,
        task_question: str,
        task_focus: str,
    ):
        self.base = base_generator
        self.llm = llm
        self.task_question = task_question
        self.task_focus = task_focus

    def generate(
        self,
        patient_id: str,
        cutoff_date: Optional[str] = None,
        n_encounters: Optional[int] = None,
    ) -> str:
        base_text = self.base.generate(
            patient_id,
            cutoff_date=cutoff_date,
            n_encounters=n_encounters,
        )
        if not base_text.strip():
            return base_text
        return self.llm.summarize(
            base_text,
            task_question=self.task_question,
            task_focus=self.task_focus,
        )
