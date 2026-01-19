# src/meds_mcp/similarity/vignette.py

"""
Vignette generation from XML (MEDS-aware, minimal)
This module provides a deterministic vignette generator that
extracts and formats patient event data from XML files into
a structured text vignette. It supports optional temporal filtering
and weighting of events based on recency.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
from lxml import etree

from .vignette_base import BaseVignetteGenerator


class DeterministicVignetteGenerator(BaseVignetteGenerator):
    def __init__(self, xml_dir: str):
        self.xml_dir = Path(xml_dir)

    def _load_xml(self, patient_id: str):
        path = self.xml_dir / f"{patient_id}.xml"
        if not path.exists():
            raise FileNotFoundError(f"Missing XML for {patient_id}")
        return etree.parse(str(path)).getroot()

    def _parse_time(self, ts: Optional[str]) -> Optional[str]:
        """
        Normalize timestamp to ISO date if possible.
        Example: '2013-10-10 19:08' -> '2013-10-10'
        """
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts).date().isoformat()
        except Exception:
            return ts

    def _in_window(
        self,
        t: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> bool:
        if not t:
            return True
        ts = datetime.fromisoformat(t)
        if start and ts < datetime.fromisoformat(start):
            return False
        if end and ts > datetime.fromisoformat(end):
            return False
        return True

    def generate(
        self,
        patient_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        temporal_weighting: bool = False,
    ) -> str:
        root = self._load_xml(patient_id)

        lines: List[str] = []

        # Walk encounters → entries → events
        for encounter in root.findall("encounter"):
            events_elem = encounter.find("events")
            if events_elem is None:
                continue

            for entry in events_elem.findall("entry"):
                entry_ts = self._parse_time(entry.attrib.get("timestamp"))

                if not self._in_window(entry_ts, start_date, end_date):
                    continue

                for event in entry.findall("event"):
                    etype = event.attrib.get("type", "")
                    code = event.attrib.get("code", "")
                    name = event.attrib.get("name", "")

                    label = ""
                    if temporal_weighting and end_date and entry_ts:
                        if entry_ts >= end_date:
                            label = "[RECENT] "
                        else:
                            label = "[HISTORY] "

                    parts = [p for p in [etype, code, name] if p]
                    if parts:
                        ts = entry_ts or "UNK_TIME"
                        lines.append(
                            f"{label}[{ts}] " + " | ".join(parts)
                        )

        return "\n".join(lines)
