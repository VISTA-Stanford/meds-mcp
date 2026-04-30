"""Integration test for patient similarity BM25 pipeline."""

import pytest
from pathlib import Path

from meds_mcp.similarity.deterministic_linearization import (
    DeterministicTimelineLinearizationGenerator,
)
from meds_mcp.similarity.bm25_retrieval import PatientBM25Index

CORPUS_DIR = Path("data/collections/dev-corpus")


@pytest.mark.integration
def test_bm25_similarity_pipeline():
    """Build index from deterministic vignettes and retrieve similar patients."""
    if not CORPUS_DIR.exists():
        pytest.skip(f"Corpus dir not found: {CORPUS_DIR}")

    xml_files = sorted(CORPUS_DIR.glob("*.xml"))[:10]
    if len(xml_files) < 3:
        pytest.skip("Need at least 3 XML files for similarity test")

    gen = DeterministicTimelineLinearizationGenerator(xml_dir=str(CORPUS_DIR))

    records = []
    for xml_path in xml_files:
        pid = xml_path.stem
        vignette = gen.generate(patient_id=pid)
        if vignette.strip():
            records.append({"person_id": pid, "vignette": vignette})

    assert len(records) >= 3, f"Only {len(records)} non-empty vignettes from {len(xml_files)} files"

    index = PatientBM25Index.from_vignettes(records)
    assert index.size == len(records)

    # Search using the first patient as query
    query_pid = records[0]["person_id"]
    results = index.search(
        query_vignette=records[0]["vignette"],
        top_k=3,
        exclude_person_id=query_pid,
    )

    assert len(results) > 0, "No results returned"
    assert all(r.person_id != query_pid for r in results), "Query patient in results"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score"


@pytest.mark.integration
def test_empty_query_returns_empty():
    """An empty query vignette should return no results."""
    if not CORPUS_DIR.exists():
        pytest.skip(f"Corpus dir not found: {CORPUS_DIR}")

    xml_files = sorted(CORPUS_DIR.glob("*.xml"))[:3]
    if len(xml_files) < 1:
        pytest.skip("Need at least 1 XML file")

    gen = DeterministicTimelineLinearizationGenerator(xml_dir=str(CORPUS_DIR))
    records = []
    for xml_path in xml_files:
        pid = xml_path.stem
        vignette = gen.generate(patient_id=pid)
        if vignette.strip():
            records.append({"person_id": pid, "vignette": vignette})

    if not records:
        pytest.skip("No non-empty vignettes")

    index = PatientBM25Index.from_vignettes(records)
    results = index.search(query_vignette="", top_k=5)
    assert results == []
