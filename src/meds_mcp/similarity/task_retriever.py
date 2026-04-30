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
        class_balanced: bool = True,
    ) -> List[SimilarNeighbor]:
        """Retrieve up to ``top_k`` similar candidates for ``task``.

        When ``class_balanced=True`` (default), the returned list is split as
        evenly as possible across the available labels (e.g. for the binary
        Yes/No tasks here: ``floor(top_k/2)`` for one class, ``ceil(top_k/2)``
        for the other; the extra slot for odd ``top_k`` goes to whichever class
        has the higher top-scored hit, with deterministic alphabetical
        tie-break on the label string). If one class has fewer hits than its
        allocation, the deficit is filled from the other class so the total
        count is preserved when possible. Within each class, candidates are
        sorted by BM25 score descending. The returned list is interleaved by
        class (highest-allocated class first by score) so the prompt sees
        Yes/No exposure rather than a same-label run.

        When ``class_balanced=False``, the original score-ordered top-k
        behavior is preserved.

        Single-class fallback: if a task has zero hits in one class, returns
        plain top-k from the available class (warning logged once per task).
        """
        idx = self._per_task_index.get(task)
        if idx is None:
            logger.warning("No BM25 index for task=%s (no eligible candidates).", task)
            return []
        if not query_vignette or not query_vignette.strip():
            return []

        # Overshoot because we may filter post-hoc on pid (composite carries pid+et,
        # and multiple composites can share the same pid). When class-balancing,
        # we additionally need enough from the minority class — bump the
        # multiplier so we have headroom.
        overshoot = max(top_k * 8, top_k + 16) if class_balanced else max(top_k * 4, top_k + 8)
        raw = idx.search(
            query_vignette=query_vignette,
            top_k=overshoot,
            exclude_person_id=None,
        )
        items_by_composite = self._per_task_items[task]

        # Resolve raw BM25 hits into validated SimilarNeighbor objects, in
        # score order. Skip excluded pid, missing items, and missing states.
        # Defensive invariants are kept.
        ordered: List[SimilarNeighbor] = []
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
            assert state.split == self._candidate_split, (
                f"candidate {pid} has split={state.split!r}, expected {self._candidate_split!r}"
            )
            assert item.label not in self._drop_labels, (
                f"candidate {pid} has dropped label {item.label}"
            )
            ordered.append(SimilarNeighbor(patient=state, item=item, score=h.score))

        if not class_balanced:
            return ordered[:top_k]

        # Bucket by label, preserving score-descending order within each class.
        by_label: Dict[int, List[SimilarNeighbor]] = {}
        for n in ordered:
            by_label.setdefault(int(n.item.label), []).append(n)

        if len(by_label) <= 1:
            # Single class available for this task — fall back to plain top-k
            # rather than failing. Log once per task to avoid log spam.
            if not getattr(self, "_warned_single_class", set()).__contains__(task):
                logger.warning(
                    "Task %s has only one label class among retrieved candidates; "
                    "falling back to plain top-k (no class balancing possible).",
                    task,
                )
                self._warned_single_class = (
                    getattr(self, "_warned_single_class", set()) | {task}
                )
            return ordered[:top_k]

        # Allocate slots: floor(k/2) and ceil(k/2). The extra slot for odd k
        # goes to whichever class has the higher top-scored hit (deterministic
        # alphabetical tie-break on str(label)).
        labels_sorted = sorted(by_label.keys())  # deterministic order
        # Score the top hit in each label bucket; pick the one with the higher
        # top score to receive the ceil(k/2) allocation. This keeps the
        # higher-similarity class slightly favored when k is odd.
        top_score_by_label = {lbl: by_label[lbl][0].score for lbl in labels_sorted}
        # Sort labels by (-top_score, str(label)) so the highest-scored class
        # is first; ties broken alphabetically for determinism.
        ranked_labels = sorted(
            labels_sorted,
            key=lambda lbl: (-top_score_by_label[lbl], str(lbl)),
        )
        primary, secondary = ranked_labels[0], ranked_labels[1]

        ceil_k = (top_k + 1) // 2
        floor_k = top_k // 2
        primary_alloc = ceil_k
        secondary_alloc = floor_k

        primary_pool = by_label[primary]
        secondary_pool = by_label[secondary]

        primary_take = primary_pool[:primary_alloc]
        secondary_take = secondary_pool[:secondary_alloc]

        # Deficit fill: if one bucket fell short of its allocation, top up from
        # the other bucket's remaining hits to preserve total count.
        primary_deficit = primary_alloc - len(primary_take)
        secondary_deficit = secondary_alloc - len(secondary_take)
        if primary_deficit > 0:
            extras = secondary_pool[len(secondary_take): len(secondary_take) + primary_deficit]
            secondary_take = secondary_take + extras
        if secondary_deficit > 0:
            extras = primary_pool[len(primary_take): len(primary_take) + secondary_deficit]
            primary_take = primary_take + extras

        # Interleave: primary, secondary, primary, secondary, ...
        # Primary's bucket is the higher-scored class; interleaving prevents a
        # same-label run at the start of the prompt.
        interleaved: List[SimilarNeighbor] = []
        for i in range(max(len(primary_take), len(secondary_take))):
            if i < len(primary_take):
                interleaved.append(primary_take[i])
            if i < len(secondary_take):
                interleaved.append(secondary_take[i])
        return interleaved[:top_k]
