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

# Optional DSPy import
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class DSPySecureLLM(dspy.LM if DSPY_AVAILABLE else object):
    """DSPy language model adapter for SecureLLM"""

    def __init__(self, secure_llm_summarizer):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not installed. Install with: pip install dspy-ai")
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


class ClinicalVignetteSummary(dspy.Signature if DSPY_AVAILABLE else object):
    """Generate a clinical vignette from patient timeline for tumor board discussion.

    Rewrites patient timelines into concise, factual clinical vignettes using professional
    clinical language. Focuses on key landmarks relevant to thoracic cancer including
    diagnosis/staging, treatments, disease progression, major comorbidities, and clinically
    significant results.
    """

    timeline = dspy.InputField(
        desc="Patient clinical timeline with chronological events including diagnoses, "
             "treatments, lab results, imaging findings, and clinical assessments"
    )

    vignette = dspy.OutputField(
        desc="Concise clinical vignette (3-5 sentences) in professional medical language, "
             "suitable for tumor board review. Preserves factual information, maintains "
             "chronological order, excludes administrative details, avoids speculation"
    )


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

        self.system_prompt = system_prompt or (
            # "You are generating a clinical vignette for patient similarity retrieval to support tumor board discussion in thoracic oncology. "
            # "Rewrite the following patient timeline into a concise, factual clinical vignette written in professional clinical language appropriate for a multidisciplinary tumor board. "
            # "Focus on key clinical landmarks relevant to thoracic cancer, including diagnosis and staging (when stated), treatments and treatment changes, response or progression events, major comorbidities, and clinically significant laboratory, imaging, or pathology results when explicitly mentioned. "
            # "Preserve all explicitly stated information and do not add, infer, or reinterpret any details. Do not speculate, assign causality, or introduce unstated clinical reasoning. "
            # "Exclude administrative details and minor events unless clinically relevant. Include laboratory or imaging results only if they are explicitly reported and materially relevant to disease status or treatment decisions. "
            # "Do not describe the timeline itself. Prefer concrete clinical events and states, and maintain chronological ordering when possible. "
            # "The output should be a single neutral paragraph of 3–5 sentences, suitable for review by a tumor board."

            "You are generating a clinical vignette for patient similarity retrieval to support tumor board discussion in thoracic oncology. "
            "Rewrite the following patient timeline into a concise, factual clinical vignette written in professional clinical language appropriate for a multidisciplinary tumor board. "
            "Focus on key clinical landmarks relevant to thoracic cancer, including diagnosis and staging (when stated), cancer-related imaging or pathology findings, oncologic treatments and treatment changes, and response or progression events. "
            "Preserve all explicitly stated information and do not add, infer, or reinterpret any details. Do not speculate or use interpretive language (e.g., 'suggestive of', 'consistent with') unless explicitly stated. "
            "Do not include administrative, scheduling, or case-management encounters unless they directly affect cancer diagnosis, staging, or treatment decisions. "
            "Include laboratory or imaging results only if they are explicitly reported and materially relevant to disease status or treatment decisions. "
            "If no thoracic malignancy, cancer-related imaging finding, or oncologic treatment is explicitly stated, produce a minimal vignette that only reports the absence of oncologic information. "
            "Do not describe the timeline itself. Prefer concrete clinical events and states, and maintain chronological ordering when possible. "
            "The output should be a single neutral paragraph of 4–6 sentences, suitable for tumor board review."

        )

        self.gen_config = get_default_generation_config(generation_overrides)

        # Initialize DSPy if requested
        self.dspy_predictor = None
        if self.use_dspy:
            if not DSPY_AVAILABLE:
                raise ImportError(
                    "DSPy is not installed. Install with: pip install dspy-ai\n"
                    "Or set use_dspy=False to use standard prompting."
                )

            # Configure DSPy with our secure LLM
            self.dspy_lm = DSPySecureLLM(self)
            dspy.settings.configure(lm=self.dspy_lm)
            # Avoid DSPy JSON parsing by calling the LM directly with a strict prompt
            self.dspy_predictor = None

    def summarize(self, text: str) -> str:
        """Generate a clinical vignette from patient timeline.

        Args:
            text: Patient timeline with clinical events

        Returns:
            Clinical vignette (3-5 sentences)
        """
        if self.use_dspy:
            # Use DSPy LM directly to avoid adapter parsing issues
            try:
                dspy_prompt = (
                    "Rewrite the patient timeline into a concise clinical vignette. "
                    "Return only the vignette text as a single paragraph (3–5 sentences).\n\n"
                    f"Timeline:\n{text}\n\n"
                    "Vignette:"
                )
                return self.dspy_lm(dspy_prompt)
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
