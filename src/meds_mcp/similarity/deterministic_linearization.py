"""
Deterministic timeline linearization from XML.

Extracts and formats patient event data from XML files into structured
text. Supports temporal filtering via a cutoff date (landmark date) and
optional last-N-encounters limiting.

Timestamp parsing and cutoff logic ported from
experiments/progression_subset/timeline.py for robustness.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Set
from datetime import datetime, time

from lxml import etree

from .vignette_base import BaseVignetteGenerator


# Default code-prefix whitelist for thoracic-oncology vignettes.
# Keeps clinically salient events and drops admin/measurement noise.
THORACIC_CODE_PREFIXES: Set[str] = {
    "STANFORD_ONCOLOGY",   # staging, biomarkers
    "NAACCR",              # cancer registry (histology, stage)
    "SNOMED",              # diagnoses/conditions
    "STANFORD_SHC_DRUG",   # medications
    "STANFORD_PROC",       # procedures (surgery, imaging orders)
    "STANFORD_NOTE",       # notes — further filtered by note_title
    "LOINC",               # labs — noisy; callers may drop this prefix
}

# Default note-title whitelist, applied only to STANFORD_NOTE events.
# Drops telephone/letter/charge/patient-instruction noise.
THORACIC_NOTE_TITLES: Set[str] = {
    "pathology",
    "imaging",
    "progress notes",
    "consults",
    "h&p",
    "radiation completion notes",
}


def _parse_entry_ts(raw: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp string into a datetime, with two fallbacks.

    Handles both ISO format ("2023-10-15T14:30:00") and space-separated
    ("2023-10-15 14:30"). Returns None if unparseable.
    """
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s.replace(" ", "T"))
    except ValueError:
        return None


def normalize_cutoff_datetime(cutoff: str) -> datetime:
    """Normalize a cutoff date string to a datetime.

    If cutoff is date-only (YYYY-MM-DD), returns end-of-day (23:59:59)
    so that events on that calendar day are included. Otherwise parses
    as a full ISO datetime.
    """
    s = cutoff.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = datetime.fromisoformat(s).date()
        return datetime.combine(d, time(23, 59, 59))
    return datetime.fromisoformat(s)


class DeterministicTimelineLinearizationGenerator(BaseVignetteGenerator):
    """Deterministic timeline linearizer that emits ordered event rows.

    Given a patient XML file, extracts events in chronological order.
    Supports filtering by a cutoff date and limiting to the last N
    encounters.
    """

    def __init__(
        self,
        xml_dir: str,
        code_prefix_whitelist: Optional[Iterable[str]] = None,
        note_title_whitelist: Optional[Iterable[str]] = None,
    ):
        self.xml_dir = Path(xml_dir)
        self.code_prefix_whitelist = (
            set(code_prefix_whitelist) if code_prefix_whitelist is not None else None
        )
        self.note_title_whitelist = (
            {t.lower() for t in note_title_whitelist}
            if note_title_whitelist is not None
            else None
        )

    def _event_is_allowed(self, event: etree._Element) -> bool:
        if self.code_prefix_whitelist is None:
            return True
        code = event.attrib.get("code", "")
        prefix = code.split("/", 1)[0] if "/" in code else code
        if prefix not in self.code_prefix_whitelist:
            return False
        if prefix == "STANFORD_NOTE" and self.note_title_whitelist is not None:
            title = (event.attrib.get("note_title") or "").strip().lower()
            if title not in self.note_title_whitelist:
                return False
        return True

    def _load_xml(self, patient_id: str) -> etree._Element:
        path = self.xml_dir / f"{patient_id}.xml"
        if not path.exists():
            raise FileNotFoundError(f"Missing XML for {patient_id}")
        return etree.parse(str(path)).getroot()

    def generate(
        self,
        patient_id: str,
        cutoff_date: Optional[str] = None,
        n_encounters: Optional[int] = None,
    ) -> str:
        """Generate linearized timeline text for a patient.

        Args:
            patient_id: Patient ID (maps to {patient_id}.xml)
            cutoff_date: Landmark date; only events on or before this date
                are included. Supports YYYY-MM-DD (inclusive of full day)
                and ISO datetime. If None, all events are included.
            n_encounters: If set, only take the last N qualifying encounters
                (those with at least one event <= cutoff). If None, all
                qualifying encounters are used.

        Returns:
            Newline-separated event lines: "[timestamp] type | code | name"
        """
        root = self._load_xml(patient_id)

        cutoff = normalize_cutoff_datetime(cutoff_date) if cutoff_date else None

        # Collect qualifying encounters (those with at least one entry <= cutoff)
        encounters = root.findall("encounter")
        if cutoff is not None:
            eligible = []
            for enc in encounters:
                events_elem = enc.find("events")
                if events_elem is None:
                    continue
                for entry in events_elem.findall("entry"):
                    ts = _parse_entry_ts(entry.attrib.get("timestamp"))
                    if ts is None or ts <= cutoff:
                        eligible.append(enc)
                        break
        else:
            eligible = [
                enc for enc in encounters
                if enc.find("events") is not None
            ]

        # Limit to last N encounters if requested
        if n_encounters is not None and n_encounters > 0:
            eligible = eligible[-n_encounters:]

        # Emit event lines
        lines: List[str] = []
        for encounter in eligible:
            events_elem = encounter.find("events")
            if events_elem is None:
                continue

            for entry in events_elem.findall("entry"):
                ts_raw = entry.attrib.get("timestamp", "UNK_TIME")
                ts = _parse_entry_ts(entry.attrib.get("timestamp"))

                # Skip entries after cutoff (entries with no timestamp are kept)
                if cutoff is not None and ts is not None and ts > cutoff:
                    continue

                for event in entry.findall("event"):
                    if not self._event_is_allowed(event):
                        continue
                    etype = event.attrib.get("type", "")
                    code = event.attrib.get("code", "")
                    name = event.attrib.get("name", "")

                    parts = [p for p in [etype, code, name] if p]
                    if parts:
                        lines.append(f"[{ts_raw}] " + " | ".join(parts))

        return "\n".join(lines)
