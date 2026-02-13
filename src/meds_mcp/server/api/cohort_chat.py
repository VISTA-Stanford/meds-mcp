# src/meds_mcp/server/api/cohort_chat.py

import json
import os
import datetime as dt
import logging
import textwrap
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from meds_mcp.server.rag.simple_storage import get_all_patient_events
from meds_mcp.server.llm import (
    get_llm_client,
    extract_response_content,
    get_default_generation_config,
)
from meds_mcp.server.tools.calculator import (
    get_calculator_tool_definition,
    execute_tool_call,
)
from meds_mcp.server.tools.readmission import (
    get_readmission_prediction,
    get_readmission_tool_definition,
)
from meds_mcp.server.tools.task_tools import (
    get_task_prediction,
    get_task_tool_definition,
    tool_name_to_task,
)
from meds_mcp.experiments.task_config import (
    TASK_DESCRIPTIONS,
    TASK_QUESTIONS,
    is_binary_task,
)
from meds_mcp.experiments.formatters import (
    RESPONSE_FORMAT_BINARY,
    RESPONSE_FORMAT_CATEGORICAL,
)

def _json_default(obj):
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    return str(obj)


def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
    """Normalize tool_call from API response (dict or object) to dict."""
    if isinstance(tool_call, dict):
        return tool_call
    fn = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", ""),
        "function": {
            "name": getattr(fn, "name", "") if fn else "",
            "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
        },
    }


def _tool_error(name: str, message: str) -> str:
    """Return a structured error string for tool failures."""
    return json.dumps({"error": message, "tool": name})


async def execute_cohort_tool_call(
    tool_call_dict: Dict[str, Any],
    patient_ids: List[str],
    prediction_time: Optional[str] = None,
    task_name: Optional[str] = None,
) -> str:
    """
    Execute a tool call for cohort chat. Handles get_readmission_prediction (async),
    task-specific get_*_prediction tools (vista_bench), and delegates others (e.g. calculator).
    """
    try:
        name = (tool_call_dict.get("function") or {}).get("name", "")
        raw_args = (tool_call_dict.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return _tool_error(name, f"Invalid arguments for {name}")

        # Task-specific tools (get_{task}_prediction)
        resolved_task = task_name or tool_name_to_task(name)
        if resolved_task:
            person_id = args.get("person_id") or (patient_ids[0] if patient_ids else None)
            pred_time = args.get("prediction_time") or prediction_time
            if not person_id:
                return json.dumps({
                    "error": "No person_id provided and cohort has no patient IDs.",
                    "label": None,
                })
            result = await get_task_prediction(person_id, resolved_task, pred_time)
            return json.dumps(result)

        if name == "get_readmission_prediction":
            person_id = args.get("person_id") or (patient_ids[0] if patient_ids else None)
            if not person_id:
                return json.dumps({
                    "error": "No person_id provided and cohort has no patient IDs.",
                    "readmission": None,
                })
            result = await get_readmission_prediction(person_id)
            return json.dumps(result)

        return execute_tool_call(tool_call_dict)
    except Exception as e:
        logger.exception("Cohort tool execution failed")
        name = (tool_call_dict.get("function") or {}).get("name", "?")
        return _tool_error(name, str(e))


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
    use_tools: bool = Field(
        True,
        description="If True, enable calculator and readmission tools; if False, LLM answers without tools.",
    )
    prediction_time: Optional[str] = Field(
        None,
        description="If set, truncate events to only those with timestamp <= prediction_time (ISO format).",
    )
    task_name: Optional[str] = Field(
        None,
        description="If set, use only this task's tool (vista_bench experiment mode). Overrides default tools.",
    )
    precomputed_context: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="If provided, keyed by patient_id; use these events instead of fetching. "
        "Use when events were precomputed (single-visit extraction). When prediction_time "
        "is set and this is absent, single-visit XML parsing is used by default.",
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


def _truncate_events_to_prediction_time(
    events: List[Dict[str, Any]],
    prediction_time_str: Optional[str],
) -> List[Dict[str, Any]]:
    """Filter events to only those with timestamp <= prediction_time."""
    if not prediction_time_str or not str(prediction_time_str).strip():
        return events
    try:
        cutoff = dt.datetime.fromisoformat(
            str(prediction_time_str).replace("Z", "+00:00").split(".")[0]
        )
    except (ValueError, TypeError):
        return events
    from meds_mcp.server.tools.search import parse_timestamp_from_metadata

    filtered = []
    for ev in events:
        ts = parse_timestamp_from_metadata(ev.get("timestamp"))
        if ts is None or ts <= cutoff:
            filtered.append(ev)
    return filtered


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
            if payload.precomputed_context is not None and pid in payload.precomputed_context:
                events = payload.precomputed_context[pid]
            elif payload.prediction_time:
                # Single-visit by default when prediction_time is set
                from meds_mcp.server.rag.simple_storage import get_document_store
                from meds_mcp.server.rag.visit_filter import get_events_for_single_visit_from_xml

                store = get_document_store()
                if store is not None:
                    events = get_events_for_single_visit_from_xml(
                        person_id=pid,
                        prediction_time_str=payload.prediction_time,
                        data_dir=str(store.data_dir),
                    )
                else:
                    events = await get_all_patient_events(pid)
                    events = _truncate_events_to_prediction_time(events, payload.prediction_time)
            else:
                events = await get_all_patient_events(pid)
                events = _truncate_events_to_prediction_time(events, payload.prediction_time)
        except Exception:
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

    # Minimal timeline format for vista_bench: "timestamp | code | name" per line (reduces tokens)
    def _to_minimal_timeline(snippets: List[Dict[str, Any]]) -> str:
        lines = []
        for entry in snippets:
            pid = entry.get("patient_id", "")
            events = entry.get("events", [])
            if len(snippets) > 1:
                lines.append(f"Patient {pid}:")
            for ev in events:
                ts = ev.get("timestamp") or ev.get("event_time") or ""
                code = ev.get("code") or ev.get("type") or ev.get("event_type") or ""
                name = ev.get("name") or ""
                parts = [str(p) for p in (ts, code, name) if p]
                if parts:
                    lines.append(" | ".join(parts))
        return "\n".join(lines) if lines else "(no events)"

    cohort_minimal = _to_minimal_timeline(context_snippets)

    # Task-specific mode (vista_bench experiment)
    task_name = payload.task_name
    if task_name:
        task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
        is_binary = is_binary_task(task_name)
        format_instruction = RESPONSE_FORMAT_BINARY if is_binary else RESPONSE_FORMAT_CATEGORICAL

        system_prompt_base = (
            "You are a clinical data analyst. Answer the user's question based on the patient timeline data provided.\n\n"
            f"Response format: {format_instruction}\n\n"
        )
        if payload.use_tools:
            system_prompt = (
                system_prompt_base
                + f"You have access to a tool that can look up {task_desc} for this patient. "
                "Use it to supplement the timeline data, then output only your final one-word answer with no reasoning.\n\n"
                f"Response format: {format_instruction}\n\n"
            )
        else:
            system_prompt = system_prompt_base

        user_suffix = (
            "You may use the available tool to look up {task} information for this patient. "
            "When calling the tool, use person_id from the cohort data and prediction_time: {pred_time}. "
            "After receiving the tool result, output only your one-word answer. "
            "Do not include any reasoning, explanation, or justification—only the answer.\n\n"
            "Answer with {format_instruction}."
        )
        pred_time_str = payload.prediction_time or "N/A"
        user_suffix = user_suffix.format(
            task=task_desc, pred_time=pred_time_str, format_instruction=format_instruction
        )
        if not payload.use_tools:
            user_suffix = f"Answer with {format_instruction}. Do not include reasoning—only the one-word answer."

        user_prompt = textwrap.dedent(
            f"""
            Here is the patient's timeline (events up to the prediction time), one event per line: timestamp | code | name

            TIMELINE:
            {cohort_minimal}

            QUESTION: {payload.question}

            {user_suffix}
            """
        )
    else:
        # Default cohort chat mode
        system_prompt_base = (
            "You are a clinical data analyst with access to a cohort of patients and their events. "
            "Answer the user's question directly. When the question is about the cohort data, "
            "use trends, similarities, and differences across patients; do not hallucinate diagnoses or outcomes not supported by the data.\n\n"
            "Each event in the JSON includes an 'event_key' field like '123456:ev42'. "
            "When you cite a specific event, append a citation in the form [[event_key]]. For example: "
            "\"Patient 123456 had a high creatinine [[123456:ev42]]\".\n\n"
        )
        if payload.use_tools:
            system_prompt = (
                system_prompt_base
                + "You have access to tools: a calculator (for any arithmetic or numerical computation) and "
                "get_readmission_prediction (to look up predicted readmission for a patient in the cohort). "
                "Use the calculator for ANY math; use get_readmission_prediction when asked about readmission risk or prediction."
            )
        else:
            system_prompt = system_prompt_base

        user_prompt = textwrap.dedent(
            f"""
            Here is a cohort of patients with selected events:

            COHORT DATA (JSON):
            {cohort_json}

            QUESTION:
            {payload.question}

            Answer the question directly. Use the cohort data only when the question is about these patients. For simple factual or arithmetic questions, answer briefly without adding unrelated cohort analysis.
            """
        )

    # 3) Call secure LLM with tool-calling
    client = get_llm_client(payload.model)
    model_name = payload.model or "apim:gpt-4.1-mini"  # adapt default to your registry
    gen_cfg = get_default_generation_config(payload.generation_config)
    # Lower temperature for vista_bench experiments: faster, more deterministic answers
    if task_name:
        gen_cfg = {**gen_cfg, "temperature": 0.0}
        if payload.generation_config:
            gen_cfg.update(payload.generation_config)

    if task_name:
        tools = [get_task_tool_definition(task_name, payload.prediction_time)]
    else:
        tools = [
            get_readmission_tool_definition(),
            get_calculator_tool_definition(),
        ]
    used_ids = [entry["patient_id"] for entry in cohort_context]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # DEBUG: Print full context before API call (set DEBUG_COHORT_CONTEXT=1 to enable)
    if os.environ.get("DEBUG_COHORT_CONTEXT", "").strip() in ("1", "true", "yes"):
        print("\n" + "=" * 80)
        print(
            f"FULL CONTEXT SENT TO LLM (task_name={task_name}, use_tools={payload.use_tools}, "
            f"patients={used_ids})"
        )
        print("=" * 80)
        print("\n--- SYSTEM PROMPT ---\n")
        print(system_prompt)
        print("\n--- USER PROMPT ---\n")
        print(user_prompt)
        if payload.use_tools and tools:
            print("\n--- TOOLS ---\n")
            for t in tools:
                print(json.dumps(t, indent=2, default=_json_default))
        print("=" * 80 + "\n")

    max_tool_iterations = 5
    response = None
    use_tools = payload.use_tools

    try:
        if not use_tools:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **gen_cfg,
            )
            logger.info("Cohort chat: request sent without tools (use_tools=False)")
        else:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **gen_cfg,
                )
                logger.info(
                    "Cohort chat: request sent with tools=%s",
                    [t.get("function", {}).get("name") for t in tools],
                )
            except (TypeError, ValueError, Exception) as tools_err:
                err_msg = str(tools_err)
                if "tool" in err_msg.lower() or "parse" in err_msg.lower() or "NoneType" in err_msg:
                    logger.warning(
                        "Cohort chat: API failed with tools, falling back to no tools: %s",
                        err_msg[:200],
                    )
                    use_tools = False
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **gen_cfg,
                    )
                else:
                    raise

        iteration = 0
        while use_tools and iteration < max_tool_iterations:
            if isinstance(response, dict):
                choices = response.get("choices", [])
            else:
                choices = getattr(response, "choices", []) or []

            if not choices:
                break

            if isinstance(choices[0], dict):
                message = choices[0].get("message", {})
            else:
                message = getattr(choices[0], "message", {})

            tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
            if not tool_calls:
                # Normal: model returned final text answer (no more tool use)
                if iteration > 0:
                    logger.info(
                        "Cohort chat: model returned final answer after %s tool round(s)",
                        iteration,
                    )
                break
            logger.info(
                "Cohort chat: executing %s tool call(s)",
                len(tool_calls) if isinstance(tool_calls, list) else 1,
            )

            tool_calls_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
            message_content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)

            # Normalize tool_calls for API (some clients expect list of dicts with function.name, function.arguments)
            tool_calls_for_api = []
            for tc in tool_calls_list:
                d = _tool_call_to_dict(tc)
                tool_calls_for_api.append({
                    "id": d.get("id", ""),
                    "type": "function",
                    "function": {"name": d["function"]["name"], "arguments": d["function"].get("arguments", "{}")},
                })

            messages.append({
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls_for_api,
            })

            for tc in tool_calls_list:
                tc_dict = _tool_call_to_dict(tc)
                tool_name = tc_dict.get("function", {}).get("name", "?")
                tool_args = tc_dict.get("function", {}).get("arguments", "{}")
                if not payload.task_name:
                    print(f"Cohort chat: executing tool: {tool_name} (args: {tool_args})")
                    logger.info("Cohort chat: executing tool: %s with args: %s", tool_name, tool_args)
                else:
                    logger.debug("Cohort chat: executing tool: %s", tool_name)
                result = await execute_cohort_tool_call(
                    tc_dict, used_ids,
                    prediction_time=payload.prediction_time,
                    task_name=payload.task_name,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_dict.get("id", ""),
                    "content": result,
                })

            iteration += 1
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **gen_cfg,
            )

        answer_text = extract_response_content(response)
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

    return CohortChatResponse(
        answer=answer_text,
        used_patient_ids=used_ids,
        num_events_used=total_events,
        debug_context_size=len(cohort_context),
        evidence_data=evidence_data,
        event_index=event_index,
    )
