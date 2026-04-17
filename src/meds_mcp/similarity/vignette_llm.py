"""LLM-based vignette generator (decorator over a base generator)."""

from typing import Optional

from .vignette_base import BaseVignetteGenerator


class LLMVignetteGenerator(BaseVignetteGenerator):
    """Wraps a base vignette generator with LLM summarization.

    The LLM must expose a ``summarize(text: str) -> str`` method
    (e.g. SecureLLMSummarizer).
    """

    def __init__(self, base_generator: BaseVignetteGenerator, llm):
        self.base = base_generator
        self.llm = llm

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
        return self.llm.summarize(base_text)
