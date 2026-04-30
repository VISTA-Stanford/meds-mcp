"""Tests for deterministic timeline linearization."""

import pytest
from pathlib import Path
from datetime import datetime

from meds_mcp.similarity.deterministic_linearization import (
    DeterministicTimelineLinearizationGenerator,
    normalize_cutoff_datetime,
    _parse_entry_ts,
)

CORPUS_DIR = "data/collections/dev-corpus"


# --- Unit tests for timestamp parsing ---


class TestParseEntryTs:
    def test_iso_datetime(self):
        dt = _parse_entry_ts("2023-10-15T14:30:00")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 10
        assert dt.day == 15
        assert dt.hour == 14

    def test_space_separated_datetime(self):
        dt = _parse_entry_ts("2023-10-15 14:30")
        assert dt is not None
        assert dt.year == 2023
        assert dt.hour == 14

    def test_date_only(self):
        dt = _parse_entry_ts("2023-10-15")
        assert dt is not None
        assert dt.year == 2023
        assert dt.hour == 0  # no time → midnight

    def test_none_input(self):
        assert _parse_entry_ts(None) is None

    def test_empty_string(self):
        assert _parse_entry_ts("") is None

    def test_whitespace_only(self):
        assert _parse_entry_ts("   ") is None

    def test_garbage_returns_none(self):
        assert _parse_entry_ts("not-a-date") is None


class TestNormalizeCutoffDatetime:
    def test_date_only_becomes_end_of_day(self):
        dt = normalize_cutoff_datetime("2023-10-15")
        assert dt.hour == 23
        assert dt.minute == 59
        assert dt.second == 59
        assert dt.day == 15

    def test_full_datetime_preserved(self):
        dt = normalize_cutoff_datetime("2023-10-15T14:30:00")
        assert dt.hour == 14
        assert dt.minute == 30

    def test_whitespace_stripped(self):
        dt = normalize_cutoff_datetime("  2023-10-15  ")
        assert dt.day == 15
        assert dt.hour == 23


# --- Integration tests (require dev-corpus XML files) ---


@pytest.mark.integration
class TestGenerateTimeline:
    @pytest.fixture(autouse=True)
    def _check_corpus(self):
        if not Path(CORPUS_DIR).exists():
            pytest.skip(f"Corpus dir not found: {CORPUS_DIR}")

    @pytest.fixture
    def generator(self):
        return DeterministicTimelineLinearizationGenerator(xml_dir=CORPUS_DIR)

    @pytest.fixture
    def patient_id(self):
        """Pick the first available patient from the corpus."""
        xml_files = sorted(Path(CORPUS_DIR).glob("*.xml"))
        if not xml_files:
            pytest.skip("No XML files in corpus")
        return xml_files[0].stem

    def test_full_timeline(self, generator, patient_id):
        text = generator.generate(patient_id)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_cutoff_reduces_output(self, generator, patient_id):
        full = generator.generate(patient_id)
        cut = generator.generate(patient_id, cutoff_date="2015-01-01")
        # A very early cutoff should produce fewer or equal events
        assert len(cut) <= len(full)

    def test_n_encounters_limits_output(self, generator, patient_id):
        full = generator.generate(patient_id)
        limited = generator.generate(patient_id, n_encounters=1)
        assert len(limited) <= len(full)

    def test_cutoff_and_n_encounters_combined(self, generator, patient_id):
        full = generator.generate(patient_id)
        both = generator.generate(
            patient_id, cutoff_date="2020-01-01", n_encounters=1
        )
        assert len(both) <= len(full)

    def test_missing_patient_raises(self, generator):
        with pytest.raises(FileNotFoundError):
            generator.generate("nonexistent_patient_id_12345")
