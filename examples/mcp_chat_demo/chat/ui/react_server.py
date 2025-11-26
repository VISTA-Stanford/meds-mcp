"""
FastAPI server for React-based chat interface.

This server provides:
- API endpoints for chat, patient loading, model selection
- Static file serving for the React app
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chat.config import parse_args, get_defaults, generate_system_prompt
from chat.core.session import session_state
from chat.core.patient import load_patient_async
from chat.llm.chat import stream_chat_response
from chat.llm.secure_llm_client import get_llm_client, get_available_models
from chat.mcp_client.client import get_event_by_id, test_connection_async

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MCP Chat React Interface")

# Enable CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_llm_client = None
_mcp_url = None
_args = None
_initialized = False

def set_server_args(args):
    """Set server arguments from external caller (e.g., react_chat_demo.py)."""
    global _args, _mcp_url, _llm_client
    _args = args
    _mcp_url = args.mcp_url
    try:
        _llm_client = get_llm_client(args.model)
        logger.info(f"Initialized React server with MCP URL: {_mcp_url}")
        logger.info(f"Default model: {args.model}")
    except Exception as e:
        logger.warning(f"Could not initialize LLM client: {e}")
        _llm_client = None

def _initialize_server():
    """Lazy initialization - only called when needed."""
    global _llm_client, _mcp_url, _args, _initialized

    if _initialized:
        return

    # If args were already set externally, use them
    if _args is not None:
        _initialized = True
        return

    # Try to initialize with command line args, but allow defaults
    try:
        _args = parse_args()
        _mcp_url = _args.mcp_url
        _llm_client = get_llm_client(_args.model)
        logger.info(f"Initialized React server with MCP URL: {_mcp_url}")
        logger.info(f"Default model: {_args.model}")
    except SystemExit:
        # parse_args() raises SystemExit when --help is used or on error
        # Use defaults instead
        _mcp_url = "http://localhost:8000/mcp"
        try:
            _llm_client = get_llm_client("apim:gpt-4o-mini")
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}")
            _llm_client = None
    except Exception as e:
        logger.warning(f"Error initializing server: {e}, using defaults")
        _mcp_url = _mcp_url or "http://localhost:8000/mcp"
        try:
            _llm_client = _llm_client or get_llm_client("apim:gpt-4o-mini")
        except Exception:
            _llm_client = None

    _initialized = True


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]
    model: str
    system_prompt: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 8192
    timeline_mode: bool = True
    max_input_length: int = 8192
    use_cache: bool = False


class ChatResponse(BaseModel):
    history: List[Dict[str, str]]
    evidence_data: Dict[str, List[str]]
    event_ids: List[str]

class CohortChatRequest(BaseModel):
    question: str
    patient_ids: List[str]
    event_query: Optional[str] = None
    max_events_per_patient: int = 50
    model: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None


class CohortChatProxyResponse(BaseModel):
    answer: str
    used_patient_ids: List[str]
    num_events_used: int
    debug_context_size: int


class LoadPatientRequest(BaseModel):
    patient_id: str


class LoadPatientResponse(BaseModel):
    success: bool
    message: str
    patient_id: Optional[str] = None
    datetime_str: Optional[str] = None


class EventDetailResponse(BaseModel):
    success: bool
    event: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    """Serve the React HTML app."""
    html_path = Path(__file__).parent / "react_index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="React app not found")
    return FileResponse(html_path)


@app.get("/api/models")
async def get_models():
    """Get list of available models from secure-llm."""
    _initialize_server()
    try:
        models = get_available_models()
        return {"models": models, "default": _args.model if _args else "apim:gpt-4o-mini"}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {"models": [], "default": "apim:gpt-4o-mini"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat message and return response with evidence."""
    _initialize_server()
    try:
        if not _llm_client:
            raise HTTPException(
                status_code=500,
                detail="LLM client not initialized. Please check server configuration."
            )

        if not session_state.timeline_loaded or not session_state.current_patient_id:
            raise HTTPException(
                status_code=400,
                detail="No patient data loaded. Please load a patient first."
            )

        # Normalize model name (remove duplicate prefixes like "apim:apim:" -> "apim:")
        model_name = request.model
        if model_name:
            # Remove duplicate prefixes (e.g., "apim:apim:gpt-4.1" -> "apim:gpt-4.1")
            parts = model_name.split(":")
            if len(parts) > 2 and parts[0] == parts[1]:
                # Duplicate prefix detected, remove one
                model_name = ":".join([parts[0]] + parts[2:])
                logger.warning(f"Normalized duplicate model prefix: {request.model} -> {model_name}")

        # Use system prompt from request or generate default
        system_prompt = request.system_prompt or generate_system_prompt()

        # Get defaults for prompt template
        defaults = get_defaults()
        prompt_template = defaults.get("prompt_template", "{context}\n\n{question}")

        # Call chat function
        history, state_new, fig = stream_chat_response(
            user_input=request.message,
            history=request.history,
            system_prompt=system_prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            timeline_mode=request.timeline_mode,
            model_name=model_name,
            max_input_length=request.max_input_length,
            prompt_template=prompt_template,
            use_cache=request.use_cache,
            mcp_url=_mcp_url,
            llm_client=_llm_client,
        )

        # Get evidence data from session state
        evidence_data = session_state.last_evidence_data or {}
        event_ids = list(evidence_data.keys()) if evidence_data else []

        return ChatResponse(
            history=history,
            evidence_data=evidence_data,
            event_ids=event_ids,
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cohort-chat", response_model=CohortChatProxyResponse)
async def cohort_chat_proxy(request: CohortChatRequest):
    """
    Proxy cohort-level chat to the meds_mcp backend's /api/cohort/cohort-chat endpoint.
    Reuses the same model handling / config as the single-patient /api/chat endpoint.
    """
    _initialize_server()

    # Basic validation: must have at least one patient
    if not request.patient_ids:
        raise HTTPException(
            status_code=400,
            detail="No patient_ids provided for cohort chat.",
        )

    if not _mcp_url:
        raise HTTPException(
            status_code=500,
            detail="MCP URL not configured. Please check server configuration.",
        )

    # Example: mcp_url = "http://localhost:8000/mcp" -> backend_base = "http://localhost:8000"
    if _mcp_url.endswith("/mcp"):
        backend_base = _mcp_url[: -len("/mcp")]
    else:
        backend_base = _mcp_url.rsplit("/", 1)[0]

    # This is the actual FastAPI cohort endpoint in meds_mcp.server.api.cohort_chat
    cohort_url = f"{backend_base}/api/cohort/cohort-chat"

    # Use default model if none provided, same logic as /api/chat
    model_name = request.model or (_args.model if _args and _args.model else "apim:gpt-4o-mini")

    # Normalize model name (remove duplicate prefixes like "apim:apim:gpt-4.1" -> "apim:gpt-4.1")
    if model_name:
        parts = model_name.split(":")
        if len(parts) > 2 and parts[0] == parts[1]:
            model_name = ":".join([parts[0]] + parts[2:])
            logger.warning(f"Normalized duplicate model prefix in cohort chat: {request.model} -> {model_name}")

    payload = {
        "question": request.question,
        "patient_ids": request.patient_ids,
        "event_query": request.event_query,
        "max_events_per_patient": request.max_events_per_patient,
        "model": model_name,
        "generation_config": request.generation_config,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(cohort_url, json=payload)

        if resp.status_code != 200:
            logger.error(f"Backend cohort-chat error ({resp.status_code}): {resp.text}")
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Backend cohort-chat error: {resp.text}",
            )

        data = resp.json()
        return CohortChatProxyResponse(**data)

    except httpx.TimeoutException as e:
        logger.error(f"Cohort chat timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="Cohort chat backend timed out. Please try again or reduce cohort size.",
        )
    except Exception as e:
        logger.error(f"Error in cohort_chat_proxy: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cohort chat failed: {str(e)}",
        )



@app.post("/api/load-patient", response_model=LoadPatientResponse)
async def load_patient(request: LoadPatientRequest):
    """Load patient data."""
    _initialize_server()
    try:
        if not _mcp_url:
            return LoadPatientResponse(
                success=False,
                message="MCP URL not configured. Please check server configuration.",
            )

        patient_id, message, fig, datetime_str, timeline_visible, success = await load_patient_async(
            request.patient_id, _mcp_url
        )

        return LoadPatientResponse(
            success=success,
            message=message,
            patient_id=patient_id if success else None,
            datetime_str=datetime_str if success else None,
        )

    except Exception as e:
        logger.error(f"Error loading patient: {e}", exc_info=True)
        return LoadPatientResponse(
            success=False,
            message=f"Error: {str(e)}",
        )


@app.get("/api/event/{event_id}", response_model=EventDetailResponse)
async def get_event(event_id: str):
    """Get event details by ID."""
    _initialize_server()
    try:
        if not _mcp_url:
            return EventDetailResponse(
                success=False,
                error="MCP URL not configured"
            )

        success, event_data, error = await get_event_by_id(event_id, _mcp_url)

        if success:
            return EventDetailResponse(success=True, event=event_data)
        else:
            return EventDetailResponse(success=False, error=error or "Event not found")

    except Exception as e:
        logger.error(f"Error getting event {event_id}: {e}")
        return EventDetailResponse(success=False, error=str(e))


@app.get("/api/test-connection")
async def test_connection():
    """Test MCP server connection."""
    _initialize_server()
    try:
        if not _mcp_url:
            return {"success": False, "message": "MCP URL not configured"}

        success, message = await test_connection_async(_mcp_url)
        return {"success": success, "message": message}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.get("/api/patient-status")
async def get_patient_status():
    """Get current patient status."""
    _initialize_server()

    # Get auto-load patient ID from args if available
    auto_load_id = None
    if _args and hasattr(_args, 'patient_id') and _args.patient_id:
        auto_load_id = _args.patient_id

    return {
        "patient_id": session_state.current_patient_id,
        "timeline_loaded": session_state.timeline_loaded,
        "query_datetime": (
            session_state.query_datetime.strftime("%Y-%m-%d %H:%M:%S")
            if session_state.query_datetime
            else None
        ),
        "auto_load_patient_id": auto_load_id,
    }


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    set_server_args(args)

    port = getattr(args, "port", 8080)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

