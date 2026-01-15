# src/meds_mcp/server/api/cohort_chat.py

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import datetime as dt

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import logging

from meds_mcp.server.rag.simple_storage import (
    get_all_patient_events,
)

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
)

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

    # Define available tools
    tools = [get_calculator_tool_definition()]
    print(f"🔧 [Cohort Chat] Tools registered: {[t.get('function', {}).get('name') for t in tools]}")

    try:
        # Try to call with tools - some models/APIs might not support it
        # Note: secure-llm's APIM provider may not handle tool calls properly
        # If it fails, we'll fall back to regular calls
        try:
            print("🔧 [Cohort Chat] Sending request with tools parameter")
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **gen_cfg,
            )
            print("🔧 [Cohort Chat] API call with tools succeeded")
        except (TypeError, ValueError) as e:
            # If tools parameter is not supported or secure-llm can't parse tool responses
            error_msg = str(e)
            if "Failed to parse OpenAI response" in error_msg or "NoneType" in error_msg:
                print(f"🔧 [Cohort Chat] secure-llm cannot handle tool calls (expected): {error_msg}")
                print("🔧 [Cohort Chat] Falling back to regular API call without tools")
            else:
                print(f"🔧 [Cohort Chat] Tools parameter not supported: {e}")
                print("🔧 [Cohort Chat] Falling back to regular API call without tools")
            # Fall back to regular call without tools
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **gen_cfg,
            )
        except Exception as e:
            print(f"🔧 [Cohort Chat] Unexpected error calling API with tools: {e}")
            # Try fallback before giving up
            try:
                print("🔧 [Cohort Chat] Attempting fallback to regular API call")
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **gen_cfg,
                )
            except Exception as fallback_error:
                print(f"🔧 [Cohort Chat] Fallback also failed: {fallback_error}")
                raise

        # Handle tool calls if present
        max_tool_iterations = 5
        iteration = 0
        
        while iteration < max_tool_iterations:
            # Extract response content and tool calls
            if isinstance(completion, dict):
                choices = completion.get("choices", [])
            else:
                try:
                    choices = completion.choices if hasattr(completion, "choices") else []
                except AttributeError:
                    choices = []
            
            if not choices:
                break
            
            # Extract message from choice
            if isinstance(choices[0], dict):
                message = choices[0].get("message", {})
            else:
                message = choices[0].message if hasattr(choices[0], "message") else {}
            
            # Extract tool_calls
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
            else:
                tool_calls = getattr(message, "tool_calls", None)
            
            # If no tool calls, break and process the response normally
            if not tool_calls:
                print("🔧 [Cohort Chat] No tool calls detected in response, processing normally")
                break
            
            # Convert tool_calls to list if needed
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls] if tool_calls else []
            
            # Extract message content
            if isinstance(message, dict):
                message_content = message.get("content")
            else:
                message_content = getattr(message, "content", None)
            
            # Add assistant message with tool calls to history
            assistant_message = {
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls
            }
            messages.append(assistant_message)
            
            # Execute all tool calls
            print(f"🔧 [Cohort Chat] Executing {len(tool_calls)} tool call(s)")
            for tool_call in tool_calls:
                # Convert tool_call to dict if it's an object
                if not isinstance(tool_call, dict):
                    tool_call_dict = {
                        "id": getattr(tool_call, "id", ""),
                        "function": {
                            "name": getattr(tool_call.function, "name", "") if hasattr(tool_call, "function") else "",
                            "arguments": getattr(tool_call.function, "arguments", "{}") if hasattr(tool_call, "function") else "{}"
                        }
                    }
                else:
                    tool_call_dict = tool_call
                
                tool_result = execute_tool_call(tool_call_dict)
                tool_call_id = tool_call_dict.get("id", "")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result
                })
            
            # Continue conversation with tool results
            iteration += 1
            print(f"🔄 [Cohort Chat] Tool call iteration {iteration}, continuing conversation...")
            
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **gen_cfg,
                )
            except TypeError:
                # Fallback if tools not supported in continuation
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
