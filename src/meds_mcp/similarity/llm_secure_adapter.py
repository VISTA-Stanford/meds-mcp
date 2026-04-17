"""Secure LLM adapter for vignette summarization."""

from typing import Optional, Dict, Any
from pathlib import Path

from meds_mcp.server.llm import (
    get_llm_client,
    extract_response_content,
    get_default_generation_config,
)

# Path to external prompt file
PROMPT_FILE = Path(__file__).resolve().parents[3] / "configs" / "prompts" / "vignette_prompt.txt"


def load_vignette_prompt() -> Optional[str]:
    """Load vignette prompt from external file if it exists."""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text().strip()
    return None


class SecureLLMSummarizer:
    """
    Adapter to use secure-llm with LLMVignetteGenerator.
    Exposes summarize(text: str) -> str.
    """

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.client = get_llm_client(model_name=model)
        self.model = model

        # Load prompt from external file, fall back to provided or default
        self.system_prompt = system_prompt or load_vignette_prompt() or self._default_prompt()

        self.gen_config = get_default_generation_config(generation_overrides)

    @staticmethod
    def _default_prompt() -> str:
        """Fallback prompt if no external file or system_prompt provided."""
        return (
            "You are generating a clinical vignette for patient similarity retrieval to support tumor board discussion in thoracic oncology. "
            "Rewrite the following patient timeline into a concise, factual clinical vignette written in clinical language appropriate for a multidisciplinary tumor board. "
            "Focus on key clinical landmarks relevant to thoracic cancer, including diagnosis and staging (when stated), treatments and treatment changes, response or progression events, major comorbidities, and clinically significant laboratory, imaging, or pathology results when explicitly mentioned. "
            "Preserve all explicitly stated information and do not add, infer, or reinterpret any details. Do not speculate, assign causality, or introduce unstated clinical reasoning. "
            "Exclude administrative details and minor events unless clinically relevant. Include laboratory or imaging results only if they are explicitly reported and materially relevant to disease status or treatment decisions. "
            "Do not describe the timeline itself. Prefer concrete clinical events and states, and maintain chronological ordering when possible. "
            "The output should be a single neutral paragraph of 3–5 sentences, suitable for review by a tumor board."
        )

    def summarize(self, text: str) -> str:
        """Generate a clinical vignette from patient timeline.

        Args:
            text: Patient timeline with clinical events

        Returns:
            Clinical vignette (3-5 sentences)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            **self.gen_config,
        )
        return extract_response_content(response)
