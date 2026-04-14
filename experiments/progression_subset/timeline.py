"""
Lumia XML: chop timelines at a cutoff, extract last-N-encounters text, or full linearized timeline.
Cutoff is inclusive of the calendar day when embed_time is date-only (YYYY-MM-DD).
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import List, Optional

from lxml import etree


def _parse_entry_ts(raw: Optional[str]) -> Optional[datetime]:
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


def normalize_cutoff_datetime(embed_time: str) -> datetime:
    """
    If embed_time is YYYY-MM-DD, use end of that day (inclusive) for comparisons.
    Otherwise parse as ISO datetime.
    """
    s = embed_time.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = datetime.fromisoformat(s).date()
        return datetime.combine(d, time(23, 59, 59))
    return datetime.fromisoformat(s)


def _entry_includes_before_cutoff(entry, cutoff: datetime) -> bool:
    ts = _parse_entry_ts(entry.attrib.get("timestamp"))
    if ts is None:
        return True
    return ts <= cutoff


def extract_last_n_encounters_text(
    xml_path: Path,
    n_encounters: int,
    embed_time: str,
) -> str:
    """
    Chop timeline at embed_time, then take the last n_encounters that contain
    at least one entry on or before the cutoff; only emit events on or before cutoff.
    """
    if n_encounters < 1:
        return ""
    cutoff = normalize_cutoff_datetime(embed_time)
    root = etree.parse(str(xml_path)).getroot()
    encounters = root.findall("encounter")
    eligible: List = []
    for enc in encounters:
        events_elem = enc.find("events")
        if events_elem is None:
            continue
        has_any = False
        for entry in events_elem.findall("entry"):
            if _entry_includes_before_cutoff(entry, cutoff):
                has_any = True
                break
        if has_any:
            eligible.append(enc)
    chosen = eligible[-n_encounters:] if len(eligible) >= n_encounters else eligible
    lines: List[str] = []
    for encounter in chosen:
        events_elem = encounter.find("events")
        if events_elem is None:
            continue
        for entry in events_elem.findall("entry"):
            ts_raw = entry.attrib.get("timestamp", "UNK_TIME")
            ts = _parse_entry_ts(entry.attrib.get("timestamp"))
            if ts is not None and ts > cutoff:
                continue
            for event in entry.findall("event"):
                etype = event.attrib.get("type", "")
                code = event.attrib.get("code", "")
                name = event.attrib.get("name", "")
                parts = [p for p in [etype, code, name] if p]
                if parts:
                    lines.append(f"[{ts_raw}] " + " | ".join(parts))
    return "\n".join(lines)


def linearize_timeline_until_cutoff(xml_path: Path, embed_time: str) -> str:
    """
    All events from XML with entry timestamp <= embed_time cutoff (inclusive of calendar day).
    """
    cutoff = normalize_cutoff_datetime(embed_time)
    root = etree.parse(str(xml_path)).getroot()
    lines: List[str] = []
    for encounter in root.findall("encounter"):
        events_elem = encounter.find("events")
        if events_elem is None:
            continue
        for entry in events_elem.findall("entry"):
            ts_raw = entry.attrib.get("timestamp", "UNK_TIME")
            ts = _parse_entry_ts(entry.attrib.get("timestamp"))
            if ts is not None and ts > cutoff:
                continue
            for event in entry.findall("event"):
                etype = event.attrib.get("type", "")
                code = event.attrib.get("code", "")
                name = event.attrib.get("name", "")
                parts = [p for p in [etype, code, name] if p]
                if parts:
                    lines.append(f"[{ts_raw}] " + " | ".join(parts))
    return "\n".join(lines)
