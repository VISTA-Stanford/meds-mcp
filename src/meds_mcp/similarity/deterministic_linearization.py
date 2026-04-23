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


_MISSING_CONCEPT_MARKERS = {
    "no matching concept",
    "unknown",
    "none",
    "",
}


def _clean_demographic_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower() in _MISSING_CONCEPT_MARKERS:
        return None
    return s


def _parse_birthdate(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None


def _age_years(birthdate: datetime, at: datetime) -> Optional[int]:
    if birthdate > at:
        return None
    yrs = at.year - birthdate.year
    if (at.month, at.day) < (birthdate.month, birthdate.day):
        yrs -= 1
    return yrs if yrs >= 0 else None


def demographics_block(
    xml_dir: str,
    patient_id: str,
    cutoff_date: Optional[str] = None,
) -> str:
    """Deterministic demographic prelude for a patient.

    Reads ``<person><birthdate>`` and ``<person><demographics>`` from the
    patient's XML and computes age at ``cutoff_date`` (if provided, else uses
    the latest entry timestamp; falls back to the `<age><years>` value if
    birthdate is missing).

    Returns a short, well-labelled block that the LLM summarizer can quote
    verbatim for the opening sentence:

        PATIENT DEMOGRAPHICS AT PREDICTION TIME:
        Age: 58 years
        Sex: female
        Race: Asian
        Ethnicity: Not Hispanic or Latino

    Unknown / "No matching concept" fields are silently omitted. Returns an
    empty string if the ``<person>`` block is missing entirely.
    """
    path = Path(xml_dir) / f"{patient_id}.xml"
    if not path.exists():
        return ""

    try:
        root = etree.parse(str(path)).getroot()
    except Exception:
        return ""

    person = root.find(".//person")
    if person is None:
        return ""

    # --- age ---
    age_years: Optional[int] = None
    birth = _parse_birthdate(person.findtext("birthdate"))
    if birth is not None:
        cutoff_dt = (
            normalize_cutoff_datetime(cutoff_date) if cutoff_date else None
        )
        if cutoff_dt is None:
            # Fallback: latest entry timestamp in the file.
            tss = [
                _parse_entry_ts(e.attrib.get("timestamp", ""))
                for e in root.iter("entry")
            ]
            tss = [t for t in tss if t is not None]
            cutoff_dt = max(tss) if tss else None
        if cutoff_dt is not None:
            age_years = _age_years(birth, cutoff_dt)
        # If we have a birthdate and cutoff but age is None (cutoff predates
        # birth, or some other anomaly), LEAVE age_years as None — do not
        # fall through to the XML-embedded <age> which is the patient's
        # present-day age and would be wrong for a historical prediction time.
    else:
        # Only use the XML-embedded age when birthdate is missing entirely.
        raw_years = person.findtext(".//age/years")
        try:
            age_years = int(str(raw_years).strip()) if raw_years else None
        except ValueError:
            age_years = None

    sex = _clean_demographic_value(person.findtext(".//gender"))
    if sex is not None:
        sex = sex.lower()  # "female"/"male"
    race = _clean_demographic_value(person.findtext(".//race"))
    ethnicity = _clean_demographic_value(person.findtext(".//ethnicity"))

    lines: list[str] = ["PATIENT DEMOGRAPHICS AT PREDICTION TIME:"]
    if age_years is not None:
        lines.append(f"Age: {age_years} years")
    if sex:
        lines.append(f"Sex: {sex}")
    if race:
        lines.append(f"Race: {race}")
    if ethnicity:
        lines.append(f"Ethnicity: {ethnicity}")

    # If we only have the header and nothing else, return an empty string
    # so the caller knows there's nothing usable.
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


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
