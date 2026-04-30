"""Tests for class-balanced retrieval in TaskAwareRetriever."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from meds_mcp.similarity.cohort import CohortStore, LabeledItem, PatientState
from meds_mcp.similarity.task_retriever import TaskAwareRetriever


TASK = "died_any_cause_1_yr"
QUESTION = "Will the patient die within 1 year?"
QUERY_VIGNETTE = (
    "A 65-year-old man with a history of smoking and lung adenocarcinoma "
    "underwent serial chest CT showing stable pulmonary nodules."
)


def _make_state(pid: str, vignette: str, split: str = "train") -> PatientState:
    return PatientState(
        person_id=pid,
        embed_time="2024-01-01",
        split=split,
        vignette=vignette,
        created_at="2024-01-01T00:00:00+00:00",
    )


def _make_item(pid: str, label: int, split: str = "train") -> LabeledItem:
    return LabeledItem(
        person_id=pid,
        task=TASK,
        task_group="mortality",
        question=QUESTION,
        label=label,
        label_description="Yes" if label == 1 else "No",
        split=split,
        embed_time="2024-01-01",
        created_at="2024-01-01T00:00:00+00:00",
    )


def _make_store(rows: List[Tuple[str, int, str]]) -> CohortStore:
    """Build a CohortStore from a list of (pid, label, vignette_text)."""
    states = [_make_state(pid, vig) for pid, _, vig in rows]
    items = [_make_item(pid, lbl) for pid, lbl, _ in rows]
    return CohortStore(states, items)


def _label_counts(neighbors) -> dict[int, int]:
    counts: dict[int, int] = {}
    for n in neighbors:
        counts[int(n.item.label)] = counts.get(int(n.item.label), 0) + 1
    return counts


def test_class_balanced_even_split():
    """5 Yes + 5 No candidates, top_k=4 -> exactly 2 Yes + 2 No."""
    rows = [
        (f"yes_{i}", 1, f"lung cancer patient yes_{i} chest CT pulmonary nodule")
        for i in range(5)
    ] + [
        (f"no_{i}", 0, f"lung cancer patient no_{i} chest CT pulmonary nodule")
        for i in range(5)
    ]
    store = _make_store(rows)
    retriever = TaskAwareRetriever(store)

    neighbors = retriever.retrieve(
        query_vignette=QUERY_VIGNETTE,
        task=TASK,
        top_k=4,
    )
    assert len(neighbors) == 4
    assert _label_counts(neighbors) == {1: 2, 0: 2}


def test_class_balanced_odd_top_k():
    """top_k=3 -> ceil(3/2)=2 in primary class, floor(3/2)=1 in secondary."""
    rows = [
        (f"yes_{i}", 1, f"lung cancer patient yes_{i} chest CT pulmonary nodule")
        for i in range(5)
    ] + [
        (f"no_{i}", 0, f"lung cancer patient no_{i} chest CT pulmonary nodule")
        for i in range(5)
    ]
    store = _make_store(rows)
    retriever = TaskAwareRetriever(store)

    neighbors = retriever.retrieve(
        query_vignette=QUERY_VIGNETTE,
        task=TASK,
        top_k=3,
    )
    assert len(neighbors) == 3
    counts = _label_counts(neighbors)
    # One class gets 2, the other gets 1 — depends on which class's top hit
    # scored higher. Just assert split is {2,1}.
    assert sorted(counts.values()) == [1, 2]


def test_class_balanced_deficit_fill():
    """1 Yes + 5 No, top_k=4 -> 1 Yes + 3 No (deficit filled from larger class)."""
    rows = [
        ("yes_0", 1, "lung cancer patient yes_0 chest CT pulmonary nodule"),
    ] + [
        (f"no_{i}", 0, f"lung cancer patient no_{i} chest CT pulmonary nodule")
        for i in range(5)
    ]
    store = _make_store(rows)
    retriever = TaskAwareRetriever(store)

    neighbors = retriever.retrieve(
        query_vignette=QUERY_VIGNETTE,
        task=TASK,
        top_k=4,
    )
    assert len(neighbors) == 4
    assert _label_counts(neighbors) == {1: 1, 0: 3}


def test_class_balanced_single_class_fallback(caplog):
    """0 Yes + 5 No, top_k=3 -> falls back to top-3 No, warning logged."""
    rows = [
        (f"no_{i}", 0, f"lung cancer patient no_{i} chest CT pulmonary nodule")
        for i in range(5)
    ]
    store = _make_store(rows)
    retriever = TaskAwareRetriever(store)

    with caplog.at_level("WARNING", logger="meds_mcp.similarity.task_retriever"):
        neighbors = retriever.retrieve(
            query_vignette=QUERY_VIGNETTE,
            task=TASK,
            top_k=3,
        )
    assert len(neighbors) == 3
    assert _label_counts(neighbors) == {0: 3}
    assert any("only one label class" in m for m in caplog.messages), (
        "Expected single-class fallback warning"
    )


def test_class_balanced_disabled_preserves_topk():
    """class_balanced=False returns the score-ordered top-k regardless of label."""
    # Make all No vignettes share the query's distinctive token so they rank
    # higher than the single Yes vignette by BM25.
    rows = [
        ("yes_0", 1, "completely unrelated text about cardiology"),
    ] + [
        (f"no_{i}", 0, f"lung cancer patient no_{i} chest CT pulmonary nodule")
        for i in range(5)
    ]
    store = _make_store(rows)
    retriever = TaskAwareRetriever(store)

    neighbors = retriever.retrieve(
        query_vignette=QUERY_VIGNETTE,
        task=TASK,
        top_k=3,
        class_balanced=False,
    )
    assert len(neighbors) == 3
    # All three should be No since they BM25-match the query much better.
    assert _label_counts(neighbors) == {0: 3}


def test_exclude_pid_respected_under_balancing():
    """exclude_pid still drops the matching candidate even with balancing."""
    rows = [
        (f"yes_{i}", 1, f"lung cancer patient yes_{i} chest CT pulmonary nodule")
        for i in range(3)
    ] + [
        (f"no_{i}", 0, f"lung cancer patient no_{i} chest CT pulmonary nodule")
        for i in range(3)
    ]
    store = _make_store(rows)
    retriever = TaskAwareRetriever(store)

    neighbors = retriever.retrieve(
        query_vignette=QUERY_VIGNETTE,
        task=TASK,
        top_k=4,
        exclude_pid="yes_0",
    )
    pids = {n.patient.person_id for n in neighbors}
    assert "yes_0" not in pids
    assert len(neighbors) == 4
    assert _label_counts(neighbors) == {1: 2, 0: 2}
