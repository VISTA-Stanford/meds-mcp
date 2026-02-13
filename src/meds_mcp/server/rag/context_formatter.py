"""
Format patient event lists for LLM context: filter by prediction_time, delta encoding,
lab-specific value/unit and collapse-by-day, and truncate to last N tokens.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple

# Lab tasks: add value/unit and collapse by day
try:
    from meds_mcp.experiments.task_config import CATEGORICAL_TASKS
    LAB_TASKS = set(CATEGORICAL_TASKS)  # lab_anemia, lab_hyperkalemia, etc.
except Exception:
    LAB_TASKS = set()

DEFAULT_MAX_TOKENS = 4096


def _parse_ts(value: Any) -> Optional[datetime.datetime]:
    """Parse timestamp from event (string or datetime) to comparable datetime."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00").split(".")[0]
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        try:
            return datetime.datetime.strptime(str(value), "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return None


def _ts_to_date_key(ts: Optional[datetime.datetime]) -> str:
    """Format datetime as date-only key for collapse-by-day (e.g. '2017-11-25')."""
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%d")


def _ts_to_display(ts: Optional[datetime.datetime]) -> str:
    """Format timestamp for display in delta output (e.g. '2017-11-25 06:09')."""
    if ts is None:
        return ""
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M")


def filter_events_before_prediction_time(
    events: List[Dict[str, Any]],
    prediction_time: Optional[str],
) -> List[Dict[str, Any]]:
    """Keep only events with timestamp < prediction_time (strictly before)."""
    if not prediction_time:
        return list(events)
    cutoff = _parse_ts(prediction_time)
    if cutoff is None:
        return list(events)
    out = []
    for ev in events:
        ts = ev.get("timestamp") or ev.get("event_time")
        ev_ts = _parse_ts(ts)
        if ev_ts is not None and ev_ts < cutoff:
            out.append(ev)
    return out


def _event_sort_key(ev: Dict[str, Any]) -> Tuple[datetime.datetime, str]:
    """Sort key: (timestamp, event_id) for stable order."""
    ts = _parse_ts(ev.get("timestamp") or ev.get("event_time"))
    if ts is None:
        ts = datetime.datetime.min
    eid = ev.get("event_id") or ev.get("id") or ""
    return (ts, str(eid))


def _count_tokens(text: str) -> int:
    """Count tokens; use tiktoken if available else heuristic."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _truncate_to_last_n_tokens(text: str, n: int) -> str:
    """Keep only the last n tokens of text (most recent in time)."""
    if n <= 0:
        return ""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= n:
            return text
        truncated = enc.decode(tokens[-n:])
        return truncated
    except Exception:
        approx = n * 4
        if len(text) <= approx:
            return text
        return text[-approx:]


def _format_event_line(
    ev: Dict[str, Any],
    is_lab: bool,
    event_key: str,
) -> str:
    """One line: CODE | Name [| value unit]; optional [[event_key]]. Value/unit shown whenever present."""
    code = ev.get("code") or ""
    name = ev.get("name") or ev.get("event_type") or ""
    parts = [f"{code} | {name}".strip() or "?"]
    value = ev.get("value") or ev.get("metadata", {}).get("value") or ""
    meta = ev.get("metadata") or {}
    unit = (
        ev.get("unit")
        or meta.get("unit")
        or ev.get("unit_source_value")
        or meta.get("unit_source_value")
        or ev.get("units")
        or meta.get("units")
        or ""
    )
    if value or unit:
        parts.append(f"{value} {unit}".strip())
    line = " | ".join(parts)
    if event_key:
        line = f"{line} [[{event_key}]]"
    return line


def _delta_encode_events(
    events: List[Dict[str, Any]],
    patient_id: str,
    is_lab_task: bool,
    include_event_key: bool = True,
) -> str:
    """
    Delta encoding: emit timestamp when it changes, then event lines.
    For lab tasks: collapse by calendar day (one header per day) and add value/unit.
    """
    if not events:
        return ""
    lines: List[str] = []
    # Sort by timestamp
    sorted_events = sorted(events, key=_event_sort_key)

    if is_lab_task:
        # Group by date (collapse by day)
        from collections import defaultdict
        by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ev in sorted_events:
            ts = _parse_ts(ev.get("timestamp") or ev.get("event_time"))
            date_key = _ts_to_date_key(ts)
            by_date[date_key].append(ev)
        for date_key in sorted(by_date.keys()):
            lines.append(date_key)  # timestamp line (day only)
            for ev in by_date[date_key]:
                raw_eid = ev.get("event_id") or ev.get("id") or ""
                event_key = f"{patient_id}:{raw_eid}" if include_event_key else ""
                lines.append(_format_event_line(ev, is_lab=True, event_key=event_key))
    else:
        # Group by exact timestamp
        last_ts: Optional[datetime.datetime] = None
        for ev in sorted_events:
            ts = _parse_ts(ev.get("timestamp") or ev.get("event_time"))
            if ts != last_ts:
                last_ts = ts
                lines.append(_ts_to_display(ts))
            raw_eid = ev.get("event_id") or ev.get("id") or ""
            event_key = f"{patient_id}:{raw_eid}" if include_event_key else ""
            lines.append(_format_event_line(ev, is_lab=False, event_key=event_key))

    return "\n".join(lines)


def format_patient_context(
    events: List[Dict[str, Any]],
    patient_id: str,
    prediction_time: Optional[str] = None,
    task_name: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    include_event_key: bool = True,
) -> str:
    """
    Build context string for one patient:
    - Filter to timestamp < prediction_time
    - Sort by timestamp ascending
    - Delta-encode; for lab tasks add value/unit and collapse by day
    - Truncate to the last max_tokens tokens (most recent w.r.t. prediction_time)
    """
    events = filter_events_before_prediction_time(events, prediction_time)
    if not events:
        return ""
    is_lab = task_name in LAB_TASKS if task_name else False
    text = _delta_encode_events(
        events,
        patient_id=patient_id,
        is_lab_task=is_lab,
        include_event_key=include_event_key,
    )
    if not text:
        return ""
    if max_tokens > 0:
        text = _truncate_to_last_n_tokens(text, max_tokens)
    return text
