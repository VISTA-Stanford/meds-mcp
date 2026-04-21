"""Normalized cohort storage for label-bearing patient items.

Splits the cohort into two JSONL-backed shapes:

- ``PatientState`` — keyed by ``(person_id, embed_time)``. Carries the
  split, the precomputed vignette built from that patient's timeline chopped
  at ``embed_time``, and a creation timestamp. A single ``person_id`` may
  have multiple ``PatientState`` entries when different tasks use different
  prediction times.
- ``LabeledItem`` — one per ``(person_id, task)`` after dropping rows with
  ``label == -1``. Each item carries its own ``embed_time`` and therefore
  resolves to a specific ``PatientState``.

Timelines are NOT stored here — they are regenerated on demand from the XML
corpus via ``DeterministicTimelineLinearizationGenerator``.

``CohortStore`` loads both JSONLs, offers state/task/item lookups, and exposes
a ``join(pid, task) -> CohortItem`` view (patient state + item joined at the
item's ``embed_time``) matching the "one object per (patient, task)" shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

__all__ = [
    "PatientState",
    "LabeledItem",
    "CohortItem",
    "CohortStore",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


StateKey = Tuple[str, str]  # (person_id, embed_time)


@dataclass(frozen=True)
class PatientState:
    """One vignette-bearing record per (person_id, embed_time)."""

    person_id: str
    embed_time: str
    split: str
    vignette: str
    created_at: str

    @property
    def key(self) -> StateKey:
        return (self.person_id, self.embed_time)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatientState":
        return cls(
            person_id=str(d["person_id"]),
            embed_time=str(d.get("embed_time", "") or ""),
            split=str(d.get("split", "") or ""),
            vignette=str(d.get("vignette", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabeledItem:
    """One row per (person_id, task) after filtering label != -1.

    ``embed_time`` may differ across tasks for the same ``person_id``; the
    corresponding PatientState is resolved via ``(person_id, embed_time)``.
    """

    person_id: str
    task: str
    task_group: str
    question: str
    label: int
    label_description: str
    split: str
    embed_time: str
    created_at: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LabeledItem":
        return cls(
            person_id=str(d["person_id"]),
            task=str(d["task"]),
            task_group=str(d.get("task_group", "") or ""),
            question=str(d.get("question", "") or ""),
            label=int(d["label"]),
            label_description=str(d.get("label_description", "") or ""),
            split=str(d.get("split", "") or ""),
            embed_time=str(d.get("embed_time", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CohortItem:
    """Joined view: patient state (at item.embed_time) + labeled item."""

    state: PatientState
    item: LabeledItem

    @property
    def person_id(self) -> str:
        return self.state.person_id


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class CohortStore:
    """In-memory store of ``PatientState`` + ``LabeledItem`` keyed by
    ``(person_id, embed_time)`` / ``(person_id, task)`` respectively."""

    def __init__(
        self,
        patients: Iterable[PatientState],
        items: Iterable[LabeledItem],
    ) -> None:
        self._states: Dict[StateKey, PatientState] = {}
        for p in patients:
            key = p.key
            if key in self._states:
                raise ValueError(
                    f"Duplicate PatientState for {key}; primary key is (person_id, embed_time)"
                )
            self._states[key] = p

        self._items: List[LabeledItem] = list(items)
        self._by_pid_task: Dict[Tuple[str, str], LabeledItem] = {
            (it.person_id, it.task): it for it in self._items
        }
        self._by_task: Dict[str, List[LabeledItem]] = {}
        self._by_pid: Dict[str, List[LabeledItem]] = {}
        for it in self._items:
            self._by_task.setdefault(it.task, []).append(it)
            self._by_pid.setdefault(it.person_id, []).append(it)

    # ------------------------- constructors -------------------------

    @classmethod
    def load(
        cls,
        patients_path: Union[str, Path],
        items_path: Union[str, Path],
    ) -> "CohortStore":
        patients = [PatientState.from_dict(d) for d in _read_jsonl(Path(patients_path))]
        items = [LabeledItem.from_dict(d) for d in _read_jsonl(Path(items_path))]
        return cls(patients, items)

    def save(
        self,
        patients_path: Union[str, Path],
        items_path: Union[str, Path],
    ) -> None:
        _write_jsonl(Path(patients_path), (p.to_dict() for p in self._states.values()))
        _write_jsonl(Path(items_path), (it.to_dict() for it in self._items))

    # ------------------------- state lookups -------------------------

    def get(self, person_id: str, embed_time: str) -> PatientState:
        return self._states[(person_id, embed_time)]

    def get_or_none(self, person_id: str, embed_time: str) -> Optional[PatientState]:
        return self._states.get((person_id, embed_time))

    def has(self, person_id: str, embed_time: str) -> bool:
        return (person_id, embed_time) in self._states

    def patient_states(self) -> List[PatientState]:
        return list(self._states.values())

    def person_ids(self) -> List[str]:
        return sorted({k[0] for k in self._states})

    def states_for_patient(self, person_id: str) -> List[PatientState]:
        return [s for s in self._states.values() if s.person_id == person_id]

    def update_patient(self, state: PatientState) -> None:
        """Replace a patient state in-place (used by precompute_vignettes).

        Raises KeyError if no state exists at ``state.key``.
        """
        if state.key not in self._states:
            raise KeyError(f"No PatientState to update for {state.key}")
        self._states[state.key] = state

    # ------------------------- item lookups -------------------------

    def items(self) -> List[LabeledItem]:
        return list(self._items)

    def tasks(self) -> List[str]:
        return sorted(self._by_task.keys())

    def items_for(self, task: str) -> List[LabeledItem]:
        return list(self._by_task.get(task, []))

    def items_for_patient(self, person_id: str) -> List[LabeledItem]:
        return list(self._by_pid.get(person_id, []))

    def join(self, person_id: str, task: str) -> CohortItem:
        item = self._by_pid_task[(person_id, task)]
        state = self._states[(person_id, item.embed_time)]
        return CohortItem(state=state, item=item)

    def __iter__(self) -> Iterator[LabeledItem]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
