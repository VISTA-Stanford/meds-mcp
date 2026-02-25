# src/meds_mcp/server/api/cohort_chat.py

import json
import datetime as dt
import logging
import textwrap
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from meds_mcp.server.rag.simple_storage import get_all_patient_events
from meds_mcp.server.rag.context_formatter import (
    filter_events_before_prediction_time,
    format_patient_context,
)
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

try:
    from meds_mcp.experiments.task_config import (
        ALL_TASKS,
        is_binary_task,
    )
    from meds_mcp.experiments.formatters import RESPONSE_FORMAT_BINARY
except Exception:
    ALL_TASKS = set()  # type: ignore[misc, assignment]
    is_binary_task = lambda _: False  # type: ignore[misc, assignment]
    RESPONSE_FORMAT_BINARY = "Answer with yes or no only. Do not include reasoning."

# When debug=True, truncate message content beyond this for full-request JSON (readable stdout)
_DEBUG_MAX_CONTENT_LEN = 2500

def _json_default(obj):
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    return str(obj)


def _debug_request_payload(
    model_name: str,
    messages: List[Dict[str, Any]],
    effective_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the request payload for debug print (truncate long content).
    effective_tools = what is actually sent to the API ([] for LLM-only or Option A).
    """
    msg_copy: List[Dict[str, Any]] = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, str) and len(content) > _DEBUG_MAX_CONTENT_LEN:
            m["content"] = content[:_DEBUG_MAX_CONTENT_LEN] + "\n\n... [truncated for debug]"
        msg_copy.append(m)
    payload: Dict[str, Any] = {"model": model_name, "messages": msg_copy}
    payload["tools"] = effective_tools
    if effective_tools:
        payload["tool_choice"] = "auto"
    return payload


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
    task_name: Optional[str] = None,
    prediction_time: Optional[str] = None,
) -> str:
    """
    Execute a tool call for cohort chat. Handles Vista task tools (get_<task>_prediction)
    via get_task_prediction (CSVs in data/collections/vista_bench/labels), get_readmission_prediction,
    and delegates others (e.g. calculator) to the sync execute_tool_call.
    """
    try:
        name = (tool_call_dict.get("function") or {}).get("name", "")
        raw_args = (tool_call_dict.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return _tool_error(name, f"Invalid arguments for {name}")

        person_id = args.get("person_id") or (patient_ids[0] if patient_ids else None)
        pred_time = args.get("prediction_time") or prediction_time

        # Vista task tools: only allow the tool for the current task (model must not use other tasks' tools)
        resolved_task = tool_name_to_task(name)
        if resolved_task is not None:
            if task_name is None:
                return _tool_error(name, "Task prediction tools are not available for this request.")
            if resolved_task != task_name:
                return _tool_error(
                    name,
                    f"Tool not available for this request. This request is for task '{task_name}' only.",
                )
            if not person_id:
                return json.dumps({
                    "error": "No person_id provided and cohort has no patient IDs.",
                    "label": None,
                })
            result = await get_task_prediction(
                person_id=person_id,
                task_name=resolved_task,
                prediction_time=pred_time,
            )
            return json.dumps(result)

        if name == "get_readmission_prediction":
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
        description="Maximum number of events to include per patient (ignored when using formatter with prediction_time)",
    )
    prediction_time: Optional[str] = Field(
        None,
        description="If set, only events with timestamp < prediction_time are used; context is last 4096 tokens before this time.",
    )
    task_name: Optional[str] = Field(
        None,
        description="Task name (e.g. lab_anemia); when a lab task, events show value/unit and are collapsed by day.",
    )
    precomputed_context: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="Optional map patient_id -> list of events; when set, used instead of fetching from store.",
    )
    precomputed_context_text: Optional[Dict[str, str]] = Field(
        None,
        description="Optional map patient_id -> preformatted context string (delta-encoded, 4096 tokens). When set for a patient, used as-is and event fetch/format is skipped.",
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
        description="If True, enable tool calling (calculator, get_readmission_prediction). If False, LLM-only response.",
    )
    inject_tool_results: bool = Field(
        False,
        description="If True, pre-fetch task prediction per patient and inject 'tool already called; result: X' into context; do not expose the task tool (single-turn).",
    )
    debug: bool = Field(
        False,
        description="If true, print the full context (cohort block + user prompt) to stdout before the LLM/tool call.",
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
    tool_executions: int = Field(
        default=0,
        description="Number of times the task prediction tool was executed in this request (for Vista bench summary).",
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

    # 1) Gather context per patient: either precomputed text or events to format
    cohort_context: List[Dict[str, Any]] = []
    total_events = 0

    for pid in payload.patient_ids:
        if payload.precomputed_context_text and pid in payload.precomputed_context_text and payload.precomputed_context_text[pid]:
            cohort_context.append({
                "patient_id": pid,
                "context_text": payload.precomputed_context_text[pid],
            })
            continue

        if payload.precomputed_context and pid in payload.precomputed_context:
            events = payload.precomputed_context.get(pid) or []
        else:
            try:
                events = await get_all_patient_events(pid)
            except Exception:
                continue

        events = _filter_events(events, payload.event_query)
        if not payload.prediction_time and payload.max_events_per_patient:
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

    # Pre-fetch task prediction per patient when inject_tool_results (single-turn: result in context, no tool exposed)
    payload_task = (payload.task_name or "").strip()
    tool_results_by_patient: Dict[str, tuple] = {}
    if payload.inject_tool_results and payload_task and payload_task in ALL_TASKS:
        tool_def = get_task_tool_definition(payload_task, payload.prediction_time)
        tool_name = tool_def.get("function", {}).get("name", f"get_{payload_task}_prediction")
        for entry in cohort_context:
            pid = entry["patient_id"]
            result = await get_task_prediction(pid, payload_task, payload.prediction_time)
            label = result.get("label") if result and "error" not in result else None
            tool_results_by_patient[pid] = (tool_name, label)

    # 2) Build context: delta-encoded, filtered by prediction_time, last 4096 tokens per patient
    evidence_data: Dict[str, List[str]] = {}
    event_index: Dict[str, Dict[str, Any]] = {}
    context_parts: List[str] = []

    for entry in cohort_context:
        pid = entry["patient_id"]

        if "context_text" in entry:
            # Precomputed formatted context (e.g. from context cache)
            if entry["context_text"]:
                block = f"Patient {pid}:\n{entry['context_text']}"
                # Option A: tool result is injected via multi-turn dialog, not in user text
                context_parts.append(block)
            continue

        events = entry["events"]
        # Events that go into the formatter (strictly before prediction_time when set)
        events_for_context = filter_events_before_prediction_time(
            events, payload.prediction_time
        )

        # Build event_index and evidence_data from events that could appear in context
        for idx, ev in enumerate(events_for_context):
            raw_eid = (
                ev.get("event_id")
                or ev.get("id")
                or ev.get("_id")
                or ev.get("event_uid")
            )
            if raw_eid is None:
                raw_eid = f"ev{idx}"
            event_key = f"{pid}:{raw_eid}"

            ts = ev.get("timestamp") or ev.get("event_time")
            ev_type = ev.get("event_type") or ev.get("type")
            text = ev.get("text") or ev.get("content") or ""
            name = ev.get("name")

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
            evidence_data.setdefault(event_key, []).append(snippet)

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
                "raw": ev,
            }

        # Delta-encoded context, lab value/unit and collapse-by-day when task is lab, last 4096 tokens
        context_text = format_patient_context(
            events_for_context,
            patient_id=pid,
            prediction_time=payload.prediction_time,
            task_name=payload.task_name,
            max_tokens=4096,
            include_event_key=False,
        )
        if context_text:
            block = f"Patient {pid}:\n{context_text}"
            # Option A: tool result is injected via multi-turn dialog, not in user text
            context_parts.append(block)

    cohort_context_block = "\n\n".join(context_parts) if context_parts else "(No events in window.)"

    task_name = payload_task
    is_vista_task = task_name and task_name in ALL_TASKS

    system_prompt = (
        "You are a clinical data analyst reviewing a cohort of patients. "
        "You will be given a list of patients with timeline events (delta-encoded: timestamp then code | name lines).\n\n"
    )
    if is_vista_task:
        system_prompt += (
            "For this task you must reply with ONLY a single word: either \"yes\" or \"no\". "
            "Do not include any reasoning, explanation, or discussion.\n\n"
        )
    # Same tool line when the model has the task tool OR when Option A (synthetic tool call): payload must look like a real tool call.
    show_task_tool_line = (
        is_vista_task
        and task_name
        and task_name in ALL_TASKS
        and (payload.use_tools or payload.inject_tool_results)
    )
    if show_task_tool_line:
        tool_def = get_task_tool_definition(task_name, payload.prediction_time)
        tool_name = tool_def.get("function", {}).get("name", f"get_{task_name}_prediction")
        system_prompt += (
            f"You have access to the {tool_name} tool. Use it when appropriate to inform your answer.\n\n"
        )
    else:
        system_prompt += (
            "Answer based on the information in the conversation only. Do not use any tools.\n\n"
        )

    # For Vista/EHRSHOT tasks: binary yes/no only; otherwise cohort summary.
    if is_vista_task:
        question_instruction = RESPONSE_FORMAT_BINARY
    else:
        question_instruction = (
            "Please provide:\n"
            "- A concise summary of key patterns across these patients.\n"
            "- Any notable differences between them (if visible).\n"
            "- Brief mention of limitations (e.g., missing labs, limited time span) if relevant."
        )

    user_prompt = textwrap.dedent(
        f"""
        Here is a cohort of patients with selected events (most recent 4096 tokens per patient, strictly before prediction_time when set):

        {cohort_context_block}

        QUESTION:
        {payload.question}

        {question_instruction}
        """
    )

    # 3) Call secure LLM with tool-calling
    client = get_llm_client(payload.model)
    model_name = payload.model or "apim:gpt-4.1-mini"  # adapt default to your registry
    gen_cfg = get_default_generation_config(payload.generation_config)

    # Vista tasks only: expose task-specific tool only (no calculator for experiment). When inject_tool_results, do not expose task tool.
    if payload_task and payload_task in ALL_TASKS and not payload.inject_tool_results:
        tools = [get_task_tool_definition(payload_task, payload.prediction_time)]
    elif payload.inject_tool_results:
        tools = []
    else:
        tools = []
    used_ids = [entry["patient_id"] for entry in cohort_context]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Option A: when inject_tool_results, append synthetic assistant + tool messages (multi-turn dialog)
    injected_tool_turn = False
    if (
        payload.inject_tool_results
        and payload_task
        and payload_task in ALL_TASKS
        and tool_results_by_patient
    ):
        # One synthetic tool call per patient that has a result (cohort order)
        patients_with_results: List[tuple] = []
        for entry in cohort_context:
            pid = entry["patient_id"]
            if pid not in tool_results_by_patient:
                continue
            tname, label = tool_results_by_patient[pid]
            if label is None:
                continue
            patients_with_results.append((pid, tname, label))
        if patients_with_results:
            tool_def = get_task_tool_definition(payload_task, payload.prediction_time)
            tool_name = tool_def.get("function", {}).get("name", f"get_{payload_task}_prediction")
            synthetic_tool_calls: List[Dict[str, Any]] = []
            for i, (pid, _tname, _label) in enumerate(patients_with_results):
                call_id = f"call_{i + 1:03d}"
                args: Dict[str, Any] = {"person_id": pid}
                if payload.prediction_time:
                    args["prediction_time"] = payload.prediction_time
                synthetic_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
                })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": synthetic_tool_calls,
            })
            for i, (pid, _tname, label) in enumerate(patients_with_results):
                call_id = f"call_{i + 1:03d}"
                result = {"patient_id": pid, "label": label, "task": payload_task}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                })
            injected_tool_turn = True

    max_tool_iterations = 5
    response = None
    use_tools = payload.use_tools and len(tools) > 0 and not injected_tool_turn
    tool_executions = 0

    if payload.debug:
        effective_tools = [] if injected_tool_turn else (tools if use_tools else [])
        request_payload = _debug_request_payload(
            model_name, messages, effective_tools
        )
        print("\n" + "FULL REQUEST PAYLOAD (debug):\n")
        print(json.dumps(request_payload, indent=2, default=_json_default))
        print()

    try:
        if injected_tool_turn:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **gen_cfg,
            )
            logger.info(
                "Cohort chat: injected tool turn, single completion for final answer"
            )
        elif use_tools:
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
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **gen_cfg,
            )
            logger.info("Cohort chat: request sent without tools (LLM-only)")

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
                logger.info("Cohort chat: executing tool: %s with args: %s", tool_name, tool_args)
                if payload.debug:
                    print(
                        f"[DEBUG tool calling] task={payload.task_name or '(none)'} | "
                        f"executing tool={tool_name} | args={tool_args}"
                    )
                # Count task prediction tool executions for this request (Vista bench summary)
                resolved_task = tool_name_to_task(tool_name)
                if resolved_task is not None and payload.task_name and resolved_task == payload.task_name:
                    tool_executions += 1
                result = await execute_cohort_tool_call(
                    tc_dict,
                    used_ids,
                    task_name=payload.task_name,
                    prediction_time=payload.prediction_time,
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
        tool_executions=tool_executions,
    )
