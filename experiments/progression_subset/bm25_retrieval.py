"""
BM25 over vignette text using bm25s (project dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

try:
    import bm25s
except ImportError as e:
    raise ImportError("bm25s is required. Install with: pip install bm25s PyStemmer") from e


@dataclass
class BM25VignetteIndex:
    retriever: Any
    person_ids: List[str]


def build_vignette_index(records: list[dict[str, Any]]) -> BM25VignetteIndex:
    """records: list of {person_id, embed_time, vignette}."""
    corpus = []
    person_ids = []
    for r in records:
        v = (r.get("vignette") or "").strip()
        if not v:
            continue
        corpus.append(v)
        person_ids.append(str(r["person_id"]))
    if not corpus:
        raise ValueError("No non-empty vignettes to index")
    tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    return BM25VignetteIndex(retriever=retriever, person_ids=person_ids)


def retrieve_similar(
    index: BM25VignetteIndex,
    query_vignette: str,
    query_pid: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not query_vignette.strip():
        return []
    q_tok = bm25s.tokenize([query_vignette])
    # bm25s.tokenize on a list of strings -> list[list[str]]; single query is q_tok[0]
    query_tokens = q_tok[0] if q_tok else []
    if not query_tokens:
        return []
    scores = index.retriever.get_scores(query_tokens)
    scores = np.asarray(scores).ravel()
    order = np.argsort(-scores)
    out: list[dict[str, Any]] = []
    for idx in order:
        i = int(idx)
        pid = index.person_ids[i]
        if str(pid) == str(query_pid):
            continue
        out.append({"person_id": pid, "score": float(scores[i])})
        if len(out) >= top_k:
            break
    return out
