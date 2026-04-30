"""Test LLM vignette generation (requires VAULT_SECRET_KEY)."""

import os
from pathlib import Path

import pytest

from meds_mcp.similarity.deterministic_linearization import (
    DeterministicTimelineLinearizationGenerator,
)
from meds_mcp.similarity.vignette_llm import LLMVignetteGenerator
from meds_mcp.similarity.llm_secure_adapter import SecureLLMSummarizer

CORPUS_DIR = "data/collections/dev-corpus"

_TEST_TASK_QUESTION = (
    "Summarize this thoracic oncology patient's clinical trajectory for similarity retrieval."
)
_TEST_TASK_FOCUS = "Highlight diagnosis, stage, treatment received, response, and disease trajectory."


def _make_generator() -> tuple[LLMVignetteGenerator, str]:
    xml_files = sorted(Path(CORPUS_DIR).glob("*.xml"))
    if not xml_files:
        pytest.skip("No XML files in corpus")
    patient_id = xml_files[0].stem

    base_vg = DeterministicTimelineLinearizationGenerator(xml_dir=CORPUS_DIR)
    secure_llm = SecureLLMSummarizer(
        model="apim:gpt-4.1-mini",
        generation_overrides={"max_tokens": 512, "temperature": 0.1},
    )
    return (
        LLMVignetteGenerator(
            base_generator=base_vg,
            llm=secure_llm,
            task_question=_TEST_TASK_QUESTION,
            task_focus=_TEST_TASK_FOCUS,
        ),
        patient_id,
    )


def _require_llm_env():
    if not os.getenv("VAULT_SECRET_KEY"):
        pytest.skip("VAULT_SECRET_KEY not set")
    if not Path(CORPUS_DIR).exists():
        pytest.skip(f"Corpus dir not found: {CORPUS_DIR}")


@pytest.mark.integration
def test_llm_vignette_with_cutoff():
    _require_llm_env()
    vg, pid = _make_generator()
    out = vg.generate(pid, cutoff_date="2023-10-31")
    assert isinstance(out, str)
    assert len(out) > 50, f"LLM output too short ({len(out)} chars): {out!r}"


@pytest.mark.integration
def test_llm_vignette_full_timeline():
    """Defaults (no cutoff_date, no n_encounters) must yield the full-timeline vignette."""
    _require_llm_env()
    vg, pid = _make_generator()
    out = vg.generate(pid)
    assert isinstance(out, str)
    assert len(out) > 50, f"LLM output too short ({len(out)} chars): {out!r}"


@pytest.mark.integration
def test_llm_vignette_last_n_encounters_only():
    """n_encounters=1 with cutoff_date=None must still return a non-empty vignette."""
    _require_llm_env()
    vg, pid = _make_generator()
    out = vg.generate(pid, n_encounters=1)
    assert isinstance(out, str)
    assert len(out) > 0
