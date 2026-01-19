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
            "You are generating a clinical vignette for patient similarity retrieval.\n\n"
            "Rewrite the following patient timeline into a concise clinical vignette.\n\n"
            "Rules:\n"
            "- Preserve clinical content (conditions, procedures, medications, findings).\n"
            "- Do NOT describe the timeline itself.\n"
            "- Do NOT add new information or interpretations.\n"
            "- Do NOT speculate or diagnose.\n"
            "- Prefer concrete events over commentary.\n"
            "- Keep temporal ordering when possible.\n\n"
            "Output:\n"
            "- 3–5 short sentences or bullet points.\n"
            "- Plain clinical language."
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
