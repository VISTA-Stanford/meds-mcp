"""BM25 patient similarity retrieval using llama-index BM25Retriever."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document
from llama_index.retrievers.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


@dataclass
class SimilarPatient:
    """A patient returned by similarity search."""

    person_id: str
    score: float
    vignette: str = ""


class PatientBM25Index:
    """BM25 index over patient vignettes for similarity retrieval.

    Wraps llama-index BM25Retriever with patient-ID-aware search
    (excludes self from results).
    """

    def __init__(self, documents: List[Document]):
        """Build the index from a list of Documents.

        Each Document.metadata must contain ``person_id``.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list")
        self._documents = documents
        self._retriever = BM25Retriever(
            nodes=documents,
            similarity_top_k=len(documents),
        )

    @classmethod
    def from_vignettes(cls, records: List[Dict[str, Any]]) -> PatientBM25Index:
        """Build index from a list of dicts with ``person_id`` and ``vignette``.

        Skips records with empty vignettes.
        """
        documents: List[Document] = []
        for rec in records:
            vignette = (rec.get("vignette") or "").strip()
            if not vignette:
                continue
            pid = str(rec["person_id"])
            doc = Document(
                text=vignette,
                metadata={"person_id": pid},
            )
            documents.append(doc)
        return cls(documents)

    def search(
        self,
        query_vignette: str,
        top_k: int = 5,
        exclude_person_id: Optional[str] = None,
    ) -> List[SimilarPatient]:
        """Find the top-k most similar patients by BM25 score.

        Args:
            query_vignette: The query patient's vignette text.
            top_k: Number of results to return.
            exclude_person_id: Patient ID to exclude (typically the query patient).

        Returns:
            List of SimilarPatient ordered by descending score.
        """
        if not query_vignette.strip():
            return []

        results = self._retriever.retrieve(query_vignette)

        out: List[SimilarPatient] = []
        for r in results:
            pid = r.metadata.get("person_id", "")
            if exclude_person_id and pid == exclude_person_id:
                continue
            out.append(
                SimilarPatient(
                    person_id=pid,
                    score=r.score or 0.0,
                    vignette=r.text,
                )
            )
            if len(out) >= top_k:
                break
        return out

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._documents)
