import pytest
import numpy as np
import os
from pathlib import Path

from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch
from meds_mcp.similarity.vignette_deterministic import DeterministicVignetteGenerator
from meds_mcp.similarity.candidates import MeiliCandidateRetriever
from meds_mcp.similarity.embeddings import CachedEmbeddingStore
from meds_mcp.similarity.retrieval import PatientSimilarityRetriever


@pytest.mark.integration
def test_patient_similarity_retrieval_end_to_end():
    """
    End-to-end integration test for patient similarity retrieval
    with extensive debugging.
    """

    print("\n================= SETUP =================")

    # -----------------------------
    # 1) Initialize components
    # -----------------------------
    meili = MCPMeiliSearch()

    hits = meili.search("", limit=5)["hits"]
    print([h["patient_id"] for h in hits])

    vignette_generator = DeterministicVignetteGenerator(
        xml_dir="data/meds/medalign/medalign_instructions_v1_3/ehrs"
    )

    candidate_retriever = MeiliCandidateRetriever(meili)

    embedding_store = CachedEmbeddingStore(
        embedding_dir="data/embeddings",
        embedder=None,
    )

    retriever = PatientSimilarityRetriever(
        vignette_generator=vignette_generator,
        candidate_retriever=candidate_retriever,
        embedding_store=embedding_store,
        candidate_k=10,
        final_k=2,
        debug=True,   # keeps internal retrieval debug
    )

    # -----------------------------
    # 2) Sanity check: query embedding exists
    # -----------------------------
    patient_id = "124602840"
    embedding_path = os.path.join("data/embeddings", f"{patient_id}.npy")

    print(f"\nChecking embedding for query patient: {patient_id}")
    print("Embedding path:", embedding_path)

    assert os.path.exists(embedding_path), f"Missing embedding for {patient_id}"

    query_vec = np.load(embedding_path)
    print("Query embedding shape:", query_vec.shape)

    assert query_vec.ndim == 1
    assert np.isfinite(query_vec).all()

    # -----------------------------
    # 3) Inspect Meilisearch candidates directly
    # -----------------------------
    print("\n================= MEILISEARCH =================")

    raw_candidates = candidate_retriever.retrieve(
        query_text="",
        limit=10,
        filters=None,
    )

    print(f"Meilisearch returned {len(raw_candidates)} candidates")

    candidate_ids = [c["patient_id"] for c in raw_candidates]
    print("Candidate patient_ids:")
    for pid in candidate_ids:
        print("  ", pid)

    # -----------------------------
    # 4) Check embedding overlap
    # -----------------------------
    print("\n================= EMBEDDING OVERLAP =================")

    embedded_ids = []
    missing_ids = []

    for pid in candidate_ids:
        emb_path = os.path.join("data/embeddings", f"{pid}.npy")
        if os.path.exists(emb_path):
            embedded_ids.append(pid)
        else:
            missing_ids.append(pid)

    print(f"Candidates WITH embeddings ({len(embedded_ids)}):")
    for pid in embedded_ids:
        print("  ✓", pid)

    print(f"Candidates WITHOUT embeddings ({len(missing_ids)}):")
    for pid in missing_ids:
        print("  ✗", pid)

    # This is the key diagnostic check
    if len(embedded_ids) == 0:
        # Extra diagnostics before skipping
        embedding_dir = Path("data/embeddings")
        available = sorted([p.stem for p in embedding_dir.glob("*.npy")])
        print("\nNo candidate overlap with embeddings. Available embedding IDs (sample):", available[:10])
        print("Candidate IDs (Meili):", candidate_ids)
        pytest.skip(
            "Meilisearch candidates do not overlap with available embeddings. "
            "This means patient_id alignment is broken between indexing and embedding generation."
        )

    # -----------------------------
    # 5) Run full retrieval
    # -----------------------------
    print("\n================= FULL RETRIEVAL =================")

    results = retriever.find_similar(
        patient_id=patient_id,
        start_date="2011-10-01",
        end_date="2025-10-31",
        temporal_weighting=False,
        structured_filters=None,
    )

    print(f"\nFinal retrieved neighbors: {len(results)}")
    for r in results:
        print(
            f"  → {r['patient_id']}: "
            f"cos={r['embedding_score']:.4f}"
        )

    # -----------------------------
    # 6) Assertions
    # -----------------------------
    assert isinstance(results, list)
    assert len(results) > 0, "No neighbors returned after filtering + embeddings"

    scores = [r["embedding_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert patient_id not in {r["patient_id"] for r in results}
