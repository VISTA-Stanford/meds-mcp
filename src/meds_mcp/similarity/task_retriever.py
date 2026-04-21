"""Task-aware BM25 retrieval over a ``CohortStore``.

BM25 similarity is computed on **vignettes only**, never on raw timelines —
regardless of how the downstream prompt renders the retrieved neighbors.

Instantiates one ``PatientBM25Index`` per task. Each per-task index:

- contains only candidates from a fixed ``candidate_split`` (default
  ``"train"``) whose label for that task is not in ``drop_labels``
  (default ``{-1}``), and
- uses each candidate's **task-aligned** vignette — the ``PatientState``
  stored at ``(candidate.person_id, candidate.item.embed_time)``. When the
  same patient has different embed_times across tasks, the vignette used
  for BM25 (and later shown in the LLM context) is the one built at the
  embed_time that matches the task being searched.

Callers retrieve with the query's **own task-aligned vignette** (the vignette
at the query's ``embed_time`` for the item being asked about).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bm25_retrieval import PatientBM25Index
from .cohort import CohortStore, LabeledItem, PatientState

logger = logging.getLogger(__name__)

__all__ = ["SimilarNeighbor", "TaskAwareRetriever"]


@dataclass(frozen=True)
class SimilarNeighbor:
    """One BM25 hit, joined with its task-aligned patient state and labeled item."""

    patient: PatientState
    item: LabeledItem
    score: float


# BM25 doc identity must disambiguate per-(pid, embed_time) when multiple
# PatientStates exist for a patient. We encode both into the BM25 metadata
# under a composite ``person_id`` string so exclude-by-pid continues to work
# while still resolving back to the right state.
def _composite_id(person_id: str, embed_time: str) -> str:
    return f"{person_id}|{embed_time}"


def _split_composite(composite: str) -> Tuple[str, str]:
    if "|" in composite:
        pid, et = composite.split("|", 1)
        return pid, et
    return composite, ""


class TaskAwareRetriever:
    """Per-task BM25 retrieval restricted to one split, excluding dropped labels.

    Usage::

        retriever = TaskAwareRetriever(store)
        query_state = store.get(query_pid, query_item.embed_time)
        hits = retriever.retrieve(
            query_vignette=query_state.vignette,
            task=query_item.task,
            top_k=3,
            exclude_pid=query_pid,
        )
    """

    def __init__(
        self,
        store: CohortStore,
        *,
        candidate_split: str = "train",
        drop_labels: tuple[int, ...] = (-1,),
    ) -> None:
        self._store = store
        self._candidate_split = candidate_split
        self._drop_labels = set(drop_labels)

        self._per_task_index: Dict[str, PatientBM25Index] = {}
        # (task, composite_id) -> LabeledItem, where composite_id encodes
        # (person_id, embed_time) for the row that contributed the vignette.
        self._per_task_items: Dict[str, Dict[str, LabeledItem]] = {}
        self._per_task_counts: Dict[str, int] = {}

        for task in store.tasks():
            eligible = [
                it
                for it in store.items_for(task)
                if it.split == candidate_split and it.label not in self._drop_labels
            ]
            docs = []
            items_by_composite: Dict[str, LabeledItem] = {}
            for it in eligible:
                state = store.get_or_none(it.person_id, it.embed_time)
                if state is None or not state.vignette.strip():
                    continue
                composite = _composite_id(it.person_id, it.embed_time)
                if composite in items_by_composite:
                    continue  # defensive: one (pid, task) per row, so one (pid, et) too
                docs.append({"person_id": composite, "vignette": state.vignette})
                items_by_composite[composite] = it
            self._per_task_counts[task] = len(docs)
            if docs:
                self._per_task_index[task] = PatientBM25Index.from_vignettes(docs)
                self._per_task_items[task] = items_by_composite

        if self._per_task_counts:
            logger.info(
                "TaskAwareRetriever built: %d tasks with indices (min=%d, max=%d candidates).",
                len(self._per_task_index),
                min(self._per_task_counts.values()),
                max(self._per_task_counts.values()),
            )
        else:
            logger.info("TaskAwareRetriever built: 0 tasks (no eligible candidates).")

    @property
    def candidate_split(self) -> str:
        return self._candidate_split

    def tasks(self) -> List[str]:
        return sorted(self._per_task_index.keys())

    def candidate_count(self, task: str) -> int:
        return self._per_task_counts.get(task, 0)

    def retrieve(
        self,
        *,
        query_vignette: str,
        task: str,
        top_k: int,
        exclude_pid: Optional[str] = None,
    ) -> List[SimilarNeighbor]:
        idx = self._per_task_index.get(task)
        if idx is None:
            logger.warning("No BM25 index for task=%s (no eligible candidates).", task)
            return []
        if not query_vignette or not query_vignette.strip():
            return []

        # Overshoot because we may filter post-hoc on pid (composite carries pid+et,
        # and multiple composites can share the same pid).
        raw = idx.search(
            query_vignette=query_vignette,
            top_k=max(top_k * 4, top_k + 8),
            exclude_person_id=None,
        )
        items_by_composite = self._per_task_items[task]
        out: List[SimilarNeighbor] = []
        for h in raw:
            pid, et = _split_composite(h.person_id)
            if exclude_pid is not None and pid == exclude_pid:
                continue
            item = items_by_composite.get(h.person_id)
            if item is None:
                continue
            state = self._store.get_or_none(pid, et)
            if state is None:
                continue
            # Defensive invariants.
            assert state.split == self._candidate_split, (
                f"candidate {pid} has split={state.split!r}, expected {self._candidate_split!r}"
            )
            assert item.label not in self._drop_labels, (
                f"candidate {pid} has dropped label {item.label}"
            )
            out.append(SimilarNeighbor(patient=state, item=item, score=h.score))
            if len(out) >= top_k:
                break
        return out
