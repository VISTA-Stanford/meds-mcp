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
            "You are generating a clinical vignette for patient similarity retrieval. "
            "Rewrite the following patient timeline into a concise, factual clinical vignette written in plain clinical language. "
            "Preserve all explicitly stated clinical information, including conditions, symptoms, procedures, medications, and key findings. "
            "Do not describe the timeline itself, and do not add, infer, or reinterpret any information. "
            "Avoid speculation, causal claims, or diagnostic reasoning beyond what is stated. "
            "Prefer concrete clinical events and states, and maintain chronological ordering when possible. "
            "The output should be a short paragraph of 3–5 sentences in neutral, note-like clinical language."
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
