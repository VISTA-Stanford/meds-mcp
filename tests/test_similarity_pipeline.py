"""Integration smoke tests for PatientSimilarityPipeline (no LLM required).

Uses ``use_llm_vignettes=False`` so the BM25 index is built over raw linearized
timelines. Verifies that the full-timeline default (cutoff_date=None,
n_encounters=None) works end-to-end through PatientRecord, build_index, and
find_similar.
"""

from pathlib import Path

import pytest

from meds_mcp.similarity import (
    PatientRecord,
    PatientSimilarityPipeline,
)

CORPUS_DIR = "data/collections/dev-corpus"


@pytest.fixture(scope="module")
def patient_ids() -> list[str]:
    if not Path(CORPUS_DIR).exists():
        pytest.skip(f"Corpus dir not found: {CORPUS_DIR}")
    xml_files = sorted(Path(CORPUS_DIR).glob("*.xml"))
    if len(xml_files) < 2:
        pytest.skip("Need at least 2 patients in corpus for similarity test")
    return [p.stem for p in xml_files]


@pytest.mark.integration
def test_pipeline_full_timeline_defaults(patient_ids: list[str]) -> None:
    """PatientRecord(cutoff_date=None) must flow through build_index + find_similar."""
    pipeline = PatientSimilarityPipeline(xml_dir=CORPUS_DIR, use_llm_vignettes=False)
    records = [PatientRecord(person_id=pid) for pid in patient_ids]

    index = pipeline.build_index(records)
    assert index.size >= 2

    query_pid = patient_ids[0]
    results = pipeline.find_similar(query_pid, top_k=3)
    assert len(results) <= 3
    assert all(r.person_id != query_pid for r in results), "query patient must be excluded"


@pytest.mark.integration
def test_pipeline_cutoff_and_n_encounters_optional(patient_ids: list[str]) -> None:
    """Both cutoff_date and n_encounters must be pure opt-ins; omitting them
    returns the same-or-longer text than passing either filter."""
    pipeline = PatientSimilarityPipeline(xml_dir=CORPUS_DIR, use_llm_vignettes=False)
    pid = patient_ids[0]

    full = pipeline.generate_vignette(pid)
    capped = pipeline.generate_vignette(pid, n_encounters=1)
    cut = pipeline.generate_vignette(pid, cutoff_date="1900-01-01")

    assert len(capped) <= len(full)
    assert len(cut) <= len(full)
