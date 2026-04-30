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
            "You are generating a clinical vignette for patient similarity retrieval in thoracic oncology. "
            "Output exactly one paragraph (4-8 sentences), narrative prose only. "
            "Hard constraints: "
            "1) Findings-first: do not produce a procedure list. Every mentioned imaging/lab/path study must include its clinically relevant result when present. "
            "2) No speculation: use only explicit facts from the source; never infer diagnoses or intent. "
            "3) Prioritize disease status and trajectory: diagnosis/stage/histology (if stated), treatment course, response/progression, key abnormal findings, and clinically meaningful negatives (for example no metastatic disease). "
            "4) Compress repetitive surveillance into trajectory statements rather than enumerating every repeated test. "
            "5) If a result is not documented, mention that briefly once; do not repeat placeholders. "
            "6) No bullets, no headers, no timeline meta-commentary, no dates, and no administrative details unless clinically relevant."
        )

    @staticmethod
    def task_focus_suffix(task_description: str) -> str:
        """System-prompt suffix appended when generating a task-conditioned vignette.

        Kept as a small, stable block so prompt-cache prefixes still hit on the
        base ``vignette_prompt.txt`` portion across patients within the same task.
        """
        return (
            "\n\nTASK FOCUS\n"
            "This vignette is being prepared for the following downstream prediction task:\n"
            f"{task_description.strip()}\n"
            "Within all rules above, give priority to information that is most "
            "informative for the task (relevant comorbidities, treatments, lab "
            "trajectories, prior events, and findings/negatives that bear on the "
            "task target). Do not invent or speculate — use only what the source "
            "timeline states. Length, format, demographics rules, no-dates rule, "
            "and the findings-first requirement are unchanged."
        )

    def build_system_prompt(self, task_description: Optional[str] = None) -> str:
        """Compose the system prompt for a single call, optionally task-conditioned."""
        if task_description and task_description.strip():
            return self.system_prompt + self.task_focus_suffix(task_description)
        return self.system_prompt

    def summarize(
        self,
        text: str,
        task_description: Optional[str] = None,
    ) -> str:
        """Generate a clinical vignette from patient timeline.

        Args:
            text: Patient timeline with clinical events.
            task_description: Optional downstream-task description. When given,
                a TASK FOCUS block is appended to the system prompt so the LLM
                steers details toward the task without violating the base
                vignette rules.

        Returns:
            Clinical vignette (4-8 sentences).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.build_system_prompt(task_description)},
                {"role": "user", "content": text},
            ],
            **self.gen_config,
        )
        return extract_response_content(response)
