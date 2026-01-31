# src/meds_mcp/server/api/cohort_chat.py

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from typing import List, Optional, Dict, Any
import json
import datetime as dt

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import logging

from meds_mcp.server.rag.simple_storage import (
    get_all_patient_events,
)
from meds_mcp.server.tools.readmission import get_readmission_prediction

# Add examples directory to path for secure_llm_client import
# This allows the import to work when running from the server
# From src/meds_mcp/server/api/cohort_chat.py, we need to go up 5 levels to reach project root
_project_root = Path(__file__).parent.parent.parent.parent.parent
_examples_path = _project_root / "examples" / "mcp_chat_demo"
if str(_examples_path) not in sys.path:
    sys.path.insert(0, str(_examples_path))

from chat.llm.secure_llm_client import (
    get_llm_client,
    extract_response_content,
    get_default_generation_config,
)
from chat.llm.chat import (
    get_calculator_tool_definition,
    execute_tool_call,
    _is_simple_calculation,
)


def get_readmission_tool_definition() -> Dict[str, Any]:
    """OpenAI-format tool definition for readmission prediction lookup."""
    return {
        "type": "function",
        "function": {
            "name": "get_readmission_prediction",
            "description": (
                "Look up the predicted readmission label for a patient in the cohort from the readmission labels dataset. "
                "Use this when the user asks whether a patient (or patients) is predicted to have readmission, "
                "or about readmission risk/prediction for the cohort. Pass the patient_id (one of the cohort patient IDs)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {
                        "type": "string",
                        "description": "Patient ID to look up (use one of the patient IDs in the cohort, e.g. from the cohort data).",
                    }
                },
                "required": ["person_id"],
            },
        },
    }


async def execute_cohort_tool_call(
    tool_call_dict: Dict[str, Any],
    patient_ids: List[str],
) -> str:
    """
    Execute a tool call for cohort chat. Handles get_readmission_prediction (async)
    and delegates others (e.g. calculator) to the sync execute_tool_call.
    """
    name = (tool_call_dict.get("function") or {}).get("name", "")
    raw_args = (tool_call_dict.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        return f"Error: Invalid arguments for {name}"

    if name == "get_readmission_prediction":
        person_id = args.get("person_id") or (patient_ids[0] if patient_ids else None)
        if not person_id:
            return json.dumps({"error": "No person_id provided and cohort has no patient IDs.", "readmission": None})
        result = await get_readmission_prediction(person_id)
        return json.dumps(result)

    return execute_tool_call(tool_call_dict)


def _json_default(obj):
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    # fallback:
    return str(obj)

logger = logging.getLogger(__name__)
router = APIRouter()


class CohortChatRequest(BaseModel):
    question: str = Field(..., description="User question about the cohort")
    patient_ids: List[str] = Field(..., description="List of patient IDs in the cohort")
    event_query: Optional[str] = Field(
        None,
        description="Optional text filter to select only certain events (e.g., 'echocardiogram')",
    )
    max_events_per_patient: int = Field(
        50,
        description="Maximum number of events to include per patient",
    )
    model: Optional[str] = Field(
        None,
        description="Optional model name for the LLM (if not provided, use default)",
    )
    generation_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for generation parameters (temperature, top_p, max_tokens, etc.)",
    )


class CohortChatResponse(BaseModel):
    answer: str
    used_patient_ids: List[str]
    num_events_used: int
    debug_context_size: int

    # Evidence & event lookup for UI
    evidence_data: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map from event_key to evidence snippets used in the answer",
    )
    event_index: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Flat index of events keyed by event_key (e.g. '123456:evt_001')",
    )


def _filter_events(events: List[Dict[str, Any]], query: Optional[str]) -> List[Dict[str, Any]]:
    """Very simple filter: keep events whose text or name contains the query string."""
    if not query:
        return events

    q = query.lower()
    filtered = []
    for ev in events:
        text = (ev.get("text") or ev.get("content") or "").lower()
        name = (ev.get("name") or ev.get("event_type") or "").lower()
        if q in text or q in name:
            filtered.append(ev)
    return filtered


@router.post("/cohort-chat", response_model=CohortChatResponse)
async def cohort_chat(payload: CohortChatRequest):
    if not payload.patient_ids:
        raise HTTPException(status_code=400, detail="No patient_ids provided")

    # 1) Gather events for each patient
    cohort_context: List[Dict[str, Any]] = []
    total_events = 0

    for pid in payload.patient_ids:
        try:
            # ✅ get_all_patient_events is async → await it
            events = await get_all_patient_events(pid)
        except Exception:
            # Skip patients that fail; log if desired
            continue

        events = _filter_events(events, payload.event_query)
        if payload.max_events_per_patient:
            events = events[: payload.max_events_per_patient]

        if events:
            total_events += len(events)
            cohort_context.append(
                {
                    "patient_id": pid,
                    "events": events,
                }
            )

    if not cohort_context:
        raise HTTPException(
            status_code=400,
            detail="No events found for the selected patients (after filtering)",
        )

    # 2) Build a compact prompt for the LLM
    import textwrap
    import json

    context_snippets = []

    # evidence + event index for frontend
    evidence_data: Dict[str, List[str]] = {}
    event_index: Dict[str, Dict[str, Any]] = {}

    for entry in cohort_context:
        pid = entry["patient_id"]
        events = entry["events"]

        simplified_events = []
        for idx, ev in enumerate(events):
            # Try to get a stable event id if available
            raw_eid = (
                ev.get("event_id")
                or ev.get("id")
                or ev.get("_id")
                or ev.get("event_uid")
            )

            # Fallback: synthesize one if missing
            if raw_eid is None:
                raw_eid = f"ev{idx}"

            # 🔑 event_key is what the LLM will cite and what the UI will look up
            event_key = f"{pid}:{raw_eid}"

            ts = ev.get("timestamp") or ev.get("event_time")
            ev_type = ev.get("event_type") or ev.get("type")
            text = ev.get("text") or ev.get("content") or ""
            name = ev.get("name")

            # Compact snippet for evidence pane
            snippet_bits = []
            if ts:
                snippet_bits.append(str(ts))
            if ev_type:
                snippet_bits.append(str(ev_type))
            if name:
                snippet_bits.append(str(name))
            if text:
                snippet_bits.append(text[:200])

            snippet = " | ".join(snippet_bits) or "(no details)"

            # Store in evidence_data
            evidence_data.setdefault(event_key, []).append(snippet)

            # Store raw-ish event for modal
            event_index[event_key] = {
                "patient_id": pid,
                "event_key": event_key,
                "raw_event_id": raw_eid,
                "timestamp": ts,
                "type": ev_type,
                "code": ev.get("code"),
                "name": name,
                "value": ev.get("value"),
                "unit": ev.get("unit"),
                "text": text,
                # keep full original too if you want
                "raw": ev,
            }

            simplified_events.append(
                {
                    "event_key": event_key,
                    "timestamp": ts,
                    "type": ev_type,
                    "code": ev.get("code"),
                    "name": name,
                    "value": ev.get("value"),
                    "unit": ev.get("unit"),
                    "text": text,
                }
            )

        context_snippets.append(
            {
                "patient_id": pid,
                "events": simplified_events,
            }
        )


    cohort_json = json.dumps(
        context_snippets,
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )

    system_prompt = (
        "You are a clinical data analyst reviewing a cohort of patients. "
        "You will be given a list of patients with selected events from their timelines, "
        "and a question about this pool. Answer using trends, similarities, and differences "
        "across patients. Do not hallucinate diagnoses or outcomes not supported by the data.\n\n"
        "Each event in the JSON includes an 'event_key' field like '123456:ev42'. "
        "Whenever you make a statement that is directly supported by a specific event, "
        "append a citation in the form [[event_key]]. For example: "
        "\"Patient 123456 had a high creatinine [[123456:ev42]]\".\n"
        "Use citations when possible, but you may omit them for very high-level summaries."
    )


    user_prompt = textwrap.dedent(
        f"""
        Here is a cohort of patients with selected events:

        COHORT DATA (JSON):
        {cohort_json}

        QUESTION ABOUT THIS COHORT:
        {payload.question}

        Please provide:
        - A concise summary of key patterns across these patients.
        - Any notable differences between them (if visible).
        - Brief mention of limitations (e.g., missing labs, limited time span) if relevant.
        """
    )

    # 3) Call secure LLM (via secure-llm client)
    client = get_llm_client(payload.model)
    model_name = payload.model or "apim:gpt-4.1-mini"  # adapt default to your registry

    # Merge default generation config with user overrides
    gen_cfg = get_default_generation_config(payload.generation_config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **gen_cfg,
        )
        answer_text = extract_response_content(completion)
    except Exception as e:
        msg = str(e)
        logger.exception("LLM generation failed")

        if "Read timed out" in msg or "ConnectTimeout" in msg:
            raise HTTPException(
                status_code=504,
                detail=(
                    "LLM backend at apim.stanfordhealthcare.org timed out. "
                    "This is likely an APIM / model deployment issue, not a bug in the cohort API."
                ),
            )

        raise HTTPException(status_code=500, detail=f"LLM generation failed: {msg}")

    used_ids = [entry["patient_id"] for entry in cohort_context]

    return CohortChatResponse(
        answer=answer_text,
        used_patient_ids=used_ids,
        num_events_used=total_events,
        debug_context_size=len(cohort_context),
        evidence_data=evidence_data,
        event_index=event_index,
    )
