"""
Tests for context formatter: value/unit parsing and display in patient context.

Run with:
    pytest tests/test_context_formatter.py -v
"""

import pytest

from meds_mcp.server.rag.context_formatter import (
    _format_event_line,
    _delta_encode_events,
    format_patient_context,
    filter_events_before_prediction_time,
)


@pytest.mark.unit
class TestFormatEventLineUnits:
    """Ensure value and unit (and unit fallbacks) are formatted correctly in context lines."""

    def test_value_and_unit_top_level(self):
        ev = {
            "code": "LOINC/8478-0",
            "name": "Mean blood pressure",
            "value": "93",
            "unit": "mmHg",
        }
        line = _format_event_line(ev, is_lab=False, event_key="p1:ev0")
        assert "93" in line
        assert "mmHg" in line
        assert line == "LOINC/8478-0 | Mean blood pressure | 93 mmHg [[p1:ev0]]"

    def test_value_and_unit_source_value(self):
        """unit_source_value is used when unit is missing (e.g. OMOP-style XML)."""
        ev = {
            "code": "LOINC/LP21258-6",
            "name": "Oxygen saturation",
            "value": "98",
            "unit_source_value": "%",
        }
        line = _format_event_line(ev, is_lab=False, event_key="")
        assert "98" in line and "%" in line
        assert "98 %" in line or "98  %" in line
        assert line.strip().endswith("98 %")

    def test_value_and_units_plural(self):
        """'units' attribute is used when unit/unit_source_value are missing."""
        ev = {
            "code": "LOINC/8867-4",
            "name": "Heart rate",
            "value": "72",
            "units": "/min",
        }
        line = _format_event_line(ev, is_lab=False, event_key="p2:ev1")
        assert "72" in line and "/min" in line
        assert "72 /min" in line

    def test_unit_from_metadata(self):
        """Unit in metadata (e.g. from xml_loader) is used."""
        ev = {
            "code": "LOINC/8462-4",
            "name": "Diastolic blood pressure",
            "value": "60",
            "metadata": {"unit": "mmHg"},
        }
        line = _format_event_line(ev, is_lab=False, event_key="")
        assert "60" in line and "mmHg" in line

    def test_unit_source_value_from_metadata(self):
        """unit_source_value in metadata is used."""
        ev = {
            "code": "LOINC/49701-6",
            "name": "pH of Blood",
            "value": "7.4",
            "metadata": {"unit_source_value": ""},
        }
        # Empty unit_source_value: we still get value
        line = _format_event_line(ev, is_lab=False, event_key="")
        assert "7.4" in line

    def test_prefer_unit_over_unit_source_value(self):
        """When both unit and unit_source_value exist, unit is used (first in fallback chain)."""
        ev = {
            "code": "LOINC/8480-6",
            "name": "Systolic blood pressure",
            "value": "120",
            "unit": "mmHg",
            "unit_source_value": "mm[Hg]",
        }
        line = _format_event_line(ev, is_lab=False, event_key="")
        assert "120 mmHg" in line
        assert "mm[Hg]" not in line

    def test_no_value_no_unit(self):
        """Event with only code and name: no value/unit segment."""
        ev = {"code": "RxNorm/866508", "name": "metoprolol tartrate Injection"}
        line = _format_event_line(ev, is_lab=False, event_key="p:ev0")
        assert "| 93" not in line
        assert "RxNorm/866508 | metoprolol" in line

    def test_value_only_no_unit(self):
        """Value without unit still shows (e.g. oxygen saturation 98)."""
        ev = {"code": "LOINC/LP21258-6", "name": "Oxygen saturation", "value": "98"}
        line = _format_event_line(ev, is_lab=False, event_key="")
        assert "98" in line
        assert "LOINC/LP21258-6 | Oxygen saturation | 98" in line


@pytest.mark.unit
class TestDeltaEncodeUnits:
    """Full delta-encoded context includes value and unit in event lines."""

    def test_delta_encode_includes_value_unit(self):
        events = [
            {
                "event_id": "p1_enc0_ent0_ev0",
                "timestamp": "2023-01-15 10:00",
                "code": "LOINC/8478-0",
                "name": "Mean blood pressure",
                "value": "93",
                "unit": "mmHg",
            },
        ]
        out = _delta_encode_events(events, patient_id="p1", is_lab_task=False, include_event_key=True)
        assert "93 mmHg" in out
        assert "LOINC/8478-0 | Mean blood pressure | 93 mmHg" in out

    def test_format_patient_context_includes_unit(self):
        events = [
            {
                "event_id": "e1",
                "timestamp": "2023-01-15 10:00",
                "code": "LOINC/LP21258-6",
                "name": "Oxygen saturation",
                "value": "98",
                "unit_source_value": "%",
            },
        ]
        out = format_patient_context(
            events,
            patient_id="p1",
            prediction_time=None,
            task_name=None,
            max_tokens=4096,
            include_event_key=True,
        )
        assert "98 %" in out or "98  %" in out
        assert "Oxygen saturation" in out


@pytest.mark.unit
class TestVisitFilterUnitParsing:
    """visit_filter._event_elem_to_dict uses unit, unit_source_value, units from XML."""

    def test_event_elem_unit_source_value(self):
        from lxml import etree
        from meds_mcp.server.rag.visit_filter import _event_elem_to_dict

        # XML with unit_source_value (no unit attribute)
        xml = (
            '<event type="measurement" code="LOINC/LP21258-6" name="Oxygen saturation" '
            'unit_source_value="%">98</event>'
        )
        elem = etree.fromstring(xml)
        d = _event_elem_to_dict(elem, "2023-01-15 10:00", "p1", 0, 0, 0)
        assert d["value"] == "98"
        assert d["unit"] == "%"

    def test_event_elem_units_fallback(self):
        from lxml import etree
        from meds_mcp.server.rag.visit_filter import _event_elem_to_dict

        xml = (
            '<event type="measurement" code="LOINC/8867-4" name="Heart rate" units="/min">72</event>'
        )
        elem = etree.fromstring(xml)
        d = _event_elem_to_dict(elem, "2023-01-15 10:00", "p1", 0, 0, 0)
        assert d["value"] == "72"
        assert d["unit"] == "/min"

    def test_event_elem_unit_preferred(self):
        from lxml import etree
        from meds_mcp.server.rag.visit_filter import _event_elem_to_dict

        xml = (
            '<event type="measurement" code="LOINC/8480-6" name="Systolic blood pressure" '
            'unit="mmHg" unit_source_value="mm[Hg]">120</event>'
        )
        elem = etree.fromstring(xml)
        d = _event_elem_to_dict(elem, "2023-01-15 10:00", "p1", 0, 0, 0)
        assert d["unit"] == "mmHg"


@pytest.mark.unit
class TestPatientGetEventsUnits:
    """Patient.get_events() exposes unit from unit_source_value and units in metadata."""

    def test_get_events_unit_from_metadata_unit_source_value(self):
        from llama_index.core.schema import TextNode
        from meds_mcp.server.rag.simple_storage import Patient

        patient = Patient("p1")
        node = TextNode(
            text="<event/>",
            id_="p1_ev0",
            metadata={
                "timestamp": "2023-01-15 10:00",
                "event_type": "measurement",
                "code": "LOINC/LP21258-6",
                "name": "Oxygen saturation",
                "value": "98",
                "unit_source_value": "%",
            },
        )
        patient.add_node(node)
        events = patient.get_events()
        assert len(events) == 1
        assert events[0]["value"] == "98"
        assert events[0]["unit"] == "%"

    def test_get_events_unit_from_metadata_units(self):
        from llama_index.core.schema import TextNode
        from meds_mcp.server.rag.simple_storage import Patient

        patient = Patient("p1")
        node = TextNode(
            text="<event/>",
            id_="p1_ev0",
            metadata={
                "timestamp": "2023-01-15 10:00",
                "code": "LOINC/8478-0",
                "name": "Mean blood pressure",
                "value": "93",
                "units": "mmHg",
            },
        )
        patient.add_node(node)
        events = patient.get_events()
        assert events[0]["unit"] == "mmHg"
