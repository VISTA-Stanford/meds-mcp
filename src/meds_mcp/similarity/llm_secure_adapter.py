from typing import Optional, Dict, Any
from pathlib import Path
import sys


# Ensure the examples chat package is on the import path
EXAMPLES_CHAT_DIR = Path(__file__).resolve().parents[3] / "examples" / "mcp_chat_demo" / "chat"
if EXAMPLES_CHAT_DIR.is_dir():
    sys.path.insert(0, str(EXAMPLES_CHAT_DIR))

from llm.secure_llm_client import (
    get_llm_client,
    extract_response_content,
    get_default_generation_config,
)


class SecureLLMSummarizer:
    """
    Adapter to use secure-llm with LLMVignetteGenerator.
    """

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.client = get_llm_client(model_name=model)
        self.model = model
        self.system_prompt = system_prompt or (
            "You are generating a clinical vignette for patient similarity retrieval to support tumor board discussion in thoracic oncology. "
            "Rewrite the following patient timeline into a concise, factual clinical vignette written in professional clinical language appropriate for a multidisciplinary tumor board. "
            "Focus on key clinical landmarks relevant to thoracic cancer, including diagnosis and staging (when stated), treatments and treatment changes, response or progression events, major comorbidities, and clinically significant laboratory, imaging, or pathology results when explicitly mentioned. "
            "Preserve all explicitly stated information and do not add, infer, or reinterpret any details. Do not speculate, assign causality, or introduce unstated clinical reasoning. "
            "Exclude administrative details and minor events unless clinically relevant. Include laboratory or imaging results only if they are explicitly reported and materially relevant to disease status or treatment decisions. "
            "Do not describe the timeline itself. Prefer concrete clinical events and states, and maintain chronological ordering when possible. "
            "The output should be a single neutral paragraph of 3–5 sentences, suitable for review by a tumor board."
        )

        self.gen_config = get_default_generation_config(generation_overrides)

    def summarize(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            **self.gen_config,
        )
        return extract_response_content(response)
