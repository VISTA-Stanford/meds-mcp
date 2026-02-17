from typing import Optional, Dict, Any
from pathlib import Path
import sys


# Ensure the examples chat package is on the import path
EXAMPLES_CHAT_DIR = Path(__file__).resolve().parents[3] / "examples" / "mcp_chat_demo" / "chat"
if EXAMPLES_CHAT_DIR.is_dir():
    sys.path.insert(0, str(EXAMPLES_CHAT_DIR))

# Path to external prompt file
PROMPT_FILE = Path(__file__).resolve().parents[3] / "configs" / "prompts" / "vignette_prompt.txt"


def load_vignette_prompt() -> Optional[str]:
    """Load vignette prompt from external file if it exists."""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text().strip()
    return None


from llm.secure_llm_client import (
    get_llm_client,
    extract_response_content,
    get_default_generation_config,
)

# Optional DSPy import
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


# Only define DSPy adapter when DSPy is available
if DSPY_AVAILABLE:
    class DSPySecureLLM(dspy.LM):
        """DSPy language model adapter for SecureLLM"""

        def __init__(self, secure_llm_summarizer):
            super().__init__(model=secure_llm_summarizer.model)
            self.summarizer = secure_llm_summarizer

        def basic_request(self, prompt: str, **kwargs):
            """Basic request interface for DSPy"""
            response = self.summarizer.client.chat.completions.create(
                model=self.summarizer.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.summarizer.gen_config, **kwargs}
            )
            return extract_response_content(response)

        def __call__(self, prompt=None, messages=None, **kwargs):
            """Main interface for DSPy LM"""
            if messages:
                # Handle chat-style messages
                prompt_text = "\n".join([
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in messages
                ])
            else:
                prompt_text = prompt or ""

            return self.basic_request(prompt_text, **kwargs)
else:
    DSPySecureLLM = None  # Placeholder when DSPy not available


class SecureLLMSummarizer:
    """
    Adapter to use secure-llm with LLMVignetteGenerator.
    Supports both standard prompting and DSPy-based structured prompting.
    """

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
        use_dspy: bool = False,
    ):
        self.client = get_llm_client(model_name=model)
        self.model = model
        self.use_dspy = use_dspy

        # Load prompt from external file, fall back to provided or default
        self.system_prompt = system_prompt or load_vignette_prompt() or self._default_prompt()

        self.gen_config = get_default_generation_config(generation_overrides)

        # Initialize DSPy LM wrapper if requested (for direct prompting)
        self.dspy_lm = None
        if self.use_dspy:
            if not DSPY_AVAILABLE:
                raise ImportError(
                    "DSPy is not installed. Install with: pip install dspy-ai\n"
                    "Or set use_dspy=False to use standard prompting."
                )
            self.dspy_lm = DSPySecureLLM(self)

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
        if self.use_dspy and self.dspy_lm:
            # Use DSPy LM wrapper for direct prompting
            try:
                prompt = (
                    f"{self.system_prompt}\n\n"
                    f"Timeline:\n{text}\n\n"
                    "Vignette:"
                )
                return self.dspy_lm(prompt)
            except Exception as e:
                print(f"⚠️  DSPy prompting failed, falling back to standard prompt: {e}")

        # Use standard prompting
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            **self.gen_config,
        )
        return extract_response_content(response)
