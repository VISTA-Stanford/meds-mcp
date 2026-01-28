import os
import numpy as np
from numpy.linalg import norm
from typing import Optional


class PatientSimilarityRetriever:
    """
    Two-stage patient similarity retrieval:
      1) embedding-based retrieval (fast)
      2) optional LLM reranking (slow, precise)
    """

    def __init__(
        self,
        vignette_generator,
        candidate_retriever,
        embedding_store,
        candidate_k: int = 100,
        final_k: int = 10,
        llm_reranker=None,
        llm_top_k: int = 10,
        debug: bool = False,
    ):
        self.vg = vignette_generator
        self.cr = candidate_retriever
        self.es = embedding_store
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.llm_reranker = llm_reranker
        self.llm_top_k = llm_top_k
        self.debug = debug

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

    def find_similar(
        self,
        patient_id: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        temporal_weighting: bool = False,
        structured_filters: Optional[str] = None,
    ):
        # -----------------------------
        # 1) Query vignette
        # -----------------------------
        query_vignette = self.vg.generate(
            patient_id,
            start_date=start_date,
            end_date=end_date,
            temporal_weighting=temporal_weighting,
        )

        if self.debug:
            print("\n=== QUERY VIGNETTE ===")
            print(query_vignette[:1200])
            print("=====================\n")

        # -----------------------------
        # 2) Candidate retrieval
        # -----------------------------
        # Use empty query to pull candidates by filters only; avoids Meili text mismatch
        candidates = self.cr.retrieve(
            query_text="",
            limit=self.candidate_k,
            filters=structured_filters,
        )

        if not candidates:
            return []

        candidate_ids = [
            c["patient_id"]
            for c in candidates
            if self.es.exists(c["patient_id"])
        ]

        print("Meili candidates:", [c["patient_id"] for c in candidates][:10])
        print("Embeddings exist for:",
              [pid for pid in [c["patient_id"] for c in candidates]
               if os.path.exists(f"data/embeddings/{pid}.npy")])

        if patient_id not in candidate_ids:
            candidate_ids.append(patient_id)

        texts = {
            pid: self.vg.generate(
                pid,
                start_date=start_date,
                end_date=end_date,
                temporal_weighting=temporal_weighting,
            )
            for pid in candidate_ids
        }

        embeddings = self.es.batch_get(candidate_ids, texts)
        query_vec = embeddings[patient_id]

        print("Candidates from Meili:", candidate_ids)
        print("Embeddings available:", list(embeddings.keys())[:10])

        # -----------------------------
        # 3) Embedding rerank
        # -----------------------------
        scored = []
        for c in candidates:
            pid = c["patient_id"]
            if pid not in embeddings:
                continue

            sim = self._cosine(query_vec, embeddings[pid])

            scored.append(
                {
                    "patient_id": pid,
                    "embedding_score": sim,
                    "meili_score": c.get("meili_score", 0.0),
                    "final_score": sim,
                }
            )

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        # -----------------------------
        # 4) Optional LLM rerank
        # -----------------------------
        if self.llm_reranker:
            top = scored[: self.llm_top_k]

            if self.debug:
                print(f"LLM reranking top {len(top)} candidates")

            for r in top:
                pid = r["patient_id"]
                r["llm_score"] = self.llm_reranker.score(
                    query_vignette,
                    texts[pid],
                )

            top.sort(key=lambda x: x["llm_score"], reverse=True)
            return top[: self.final_k]

        # -----------------------------
        # 5) Default return
        # -----------------------------
        return scored[: self.final_k]
