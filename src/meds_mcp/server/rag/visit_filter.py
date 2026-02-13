"""
Direct XML parsing to restrict patient context to a single encounter (visit).
Used by cohort_chat when prediction_time is set—single-visit filtering is the default.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from lxml import etree


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse timestamp string to datetime."""
    if not ts_str or not str(ts_str).strip():
        return None
    s = str(ts_str).replace("Z", "+00:00").split(".")[0]
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return None


def _event_elem_to_dict(
    event_elem: Any,
    entry_ts_str: Optional[str],
    person_id: str,
    encounter_idx: int,
    entry_idx: int,
    event_idx: int,
) -> Dict[str, Any]:
    """Convert an XML <event> element to the dict format expected by cohort_chat."""
    raw_eid = f"{person_id}_enc{encounter_idx}_ent{entry_idx}_ev{event_idx}"
    event_text = event_elem.text.strip() if event_elem.text else ""
    return {
        "id": raw_eid,
        "event_id": raw_eid,
        "content": etree.tostring(event_elem, encoding="unicode", with_tail=False),
        "metadata": {
            "timestamp": entry_ts_str,
            "person_id": person_id,
        },
        "timestamp": entry_ts_str,
        "event_type": event_elem.get("type") or event_elem.get("category") or event_elem.tag,
        "code": event_elem.get("code"),
        "name": event_elem.get("name"),
        "value": event_text,
        "unit": event_elem.get("unit") or event_elem.get("unit_source_value") or event_elem.get("units"),
        "text": event_text,
        "person_id": person_id,
    }


def get_events_for_single_visit_from_xml(
    person_id: str,
    prediction_time_str: str,
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Parse patient XML and return only events from the single encounter (visit)
    that contains prediction_time, truncated to prediction_time.

    The "encounter containing prediction_time" is the encounter whose most recent
    entry timestamp is <= prediction_time and is the latest among all encounters.

    Returns events in the same format as Patient.get_events() for cohort_chat compatibility.
    """
    data_path = Path(data_dir)
    xml_path = data_path / f"{person_id}.xml"
    if not xml_path.exists():
        return []

    cutoff = _parse_timestamp(prediction_time_str)
    if cutoff is None:
        return []

    try:
        root = etree.parse(str(xml_path)).getroot()
    except etree.XMLSyntaxError as e:
        print(f"XML is empty or invalid for patient {person_id} ({xml_path}): {e}. Cache will not be created for this patient.")
        return []
    best_encounter_events: List[Dict[str, Any]] = []
    best_max_ts: Optional[datetime] = None

    for enc_idx, encounter in enumerate(root.findall("encounter")):
        events_elem = encounter.find("events")
        if events_elem is None:
            continue

        encounter_events: List[Dict[str, Any]] = []
        encounter_max_ts: Optional[datetime] = None

        for ent_idx, entry in enumerate(events_elem.findall("entry")):
            entry_ts_str = entry.get("timestamp")
            entry_ts = _parse_timestamp(entry_ts_str)

            if entry_ts is None or entry_ts > cutoff:
                continue

            for ev_idx, event in enumerate(entry.findall("event")):
                ev_dict = _event_elem_to_dict(
                    event, entry_ts_str, person_id, enc_idx, ent_idx, ev_idx
                )
                encounter_events.append(ev_dict)
                if entry_ts is not None:
                    if encounter_max_ts is None or entry_ts > encounter_max_ts:
                        encounter_max_ts = entry_ts

        if encounter_events and encounter_max_ts is not None:
            if best_max_ts is None or encounter_max_ts > best_max_ts:
                best_encounter_events = encounter_events
                best_max_ts = encounter_max_ts

    return best_encounter_events
