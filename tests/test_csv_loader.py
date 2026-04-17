"""Unit tests for load_patient_records_from_csv."""

from pathlib import Path

import pytest

from meds_mcp.similarity import PatientRecord, load_patient_records_from_csv


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_basic_load(tmp_path: Path) -> None:
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "person_id,embed_time",
        "pid_1,2020-01-15",
        "pid_2,2021-06-01",
    ])

    records = load_patient_records_from_csv(csv_path)
    assert records == [
        PatientRecord("pid_1", cutoff_date="2020-01-15"),
        PatientRecord("pid_2", cutoff_date="2021-06-01"),
    ]


def test_first_row_wins_for_duplicate_person_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "person_id,embed_time",
        "pid_1,2020-01-15",
        "pid_1,2099-12-31",  # duplicate — should be ignored
    ])

    records = load_patient_records_from_csv(csv_path)
    assert len(records) == 1
    assert records[0].cutoff_date == "2020-01-15"


def test_blank_cutoff_becomes_none(tmp_path: Path) -> None:
    """Blank cutoff cells must yield cutoff_date=None (full timeline)."""
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "person_id,embed_time",
        "pid_1,",
        "pid_2,2020-01-15",
    ])

    records = load_patient_records_from_csv(csv_path)
    assert records[0].person_id == "pid_1"
    assert records[0].cutoff_date is None
    assert records[1].cutoff_date == "2020-01-15"


def test_custom_column_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "patient,landmark,extra",
        "p1,2022-03-03,x",
    ])

    records = load_patient_records_from_csv(
        csv_path, person_id_col="patient", cutoff_col="landmark"
    )
    assert records == [PatientRecord("p1", cutoff_date="2022-03-03")]


def test_cutoff_col_none_returns_all_with_no_landmark(tmp_path: Path) -> None:
    """Passing cutoff_col=None yields records without any landmark filter."""
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "person_id,embed_time",
        "pid_1,2020-01-15",
        "pid_2,2021-06-01",
    ])

    records = load_patient_records_from_csv(csv_path, cutoff_col=None)
    assert all(r.cutoff_date is None for r in records)
    assert [r.person_id for r in records] == ["pid_1", "pid_2"]


def test_require_all_columns_populated_excludes_patient(tmp_path: Path) -> None:
    """A patient appearing in any row with a blank column must be excluded entirely."""
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "person_id,embed_time,label",
        "pid_1,2020-01-15,positive",
        "pid_2,2021-06-01,",  # blank label → exclude pid_2
        "pid_3,2022-07-07,negative",
    ])

    records = load_patient_records_from_csv(
        csv_path, require_all_columns_populated=True
    )
    assert [r.person_id for r in records] == ["pid_1", "pid_3"]


def test_missing_person_id_is_skipped(tmp_path: Path) -> None:
    csv_path = tmp_path / "cohort.csv"
    _write_csv(csv_path, [
        "person_id,embed_time",
        ",2020-01-15",
        "pid_1,2021-06-01",
    ])

    records = load_patient_records_from_csv(csv_path)
    assert [r.person_id for r in records] == ["pid_1"]
