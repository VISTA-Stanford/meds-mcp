# React Chat Interface

A lightweight, modern React-based chat interface for interacting with patient records via the MCP server.

## Features

- **Chat Interface**: Clean, modern chat UI with message history
- **Clickable Citations**: Click on `[[event_id]]` citations in responses to view detailed event information from the patient timeline
- **Model Selection**: Choose from available LLM models integrated via secure-llm
- **Patient Loading**: Load patient data directly from the interface
- **Evidence Viewing**: View evidence snippets and full event details in a modal

## Usage

### Start the React Server

```bash
# From the project root
uv run python examples/mcp_chat_demo/chat/ui/run_react_app.py \
    --model "apim:gpt-4.1" \
    --mcp_url "http://localhost:8000/mcp" \
    --patient_id 127672063
```

Or use the server directly:

```bash
python examples/mcp_chat_demo/chat/ui/react_server.py \
    --model "apim:gpt-4.1" \
    --mcp_url "http://localhost:8000/mcp"
```

### Available Models

**APIM Models:**
- `apim:gpt-4.1`, `apim:gpt-4.1-mini`, `apim:gpt-4.1-nano`
- `apim:o3-mini`
- `apim:claude-3.5`, `apim:claude-3.7`
- `apim:gemini-2.0-flash`, `apim:gemini-2.5-pro-preview-05-06`
- `apim:llama-3.3-70b`, `apim:llama-4-maverick-17b`, `apim:llama-4-scout-17b`
- `apim:deepseek-chat`

**Nero Models** (requires GCP/Nero Vertex API):
- `nero:gemini-2.0-flash`, `nero:gemini-2.5-pro`, `nero:gemini-2.5-flash`, `nero:gemini-2.5-flash-lite`

**Recommended:** `apim:gpt-4.1-mini`, `apim:o3-mini`, or `apim:gpt-4.1` for best balance of speed and quality.

### Access the Interface

Open your browser to: **http://localhost:8080**

## How It Works

1. **Load a Patient**: Enter a patient ID in the header and click "Load"
2. **Select a Model**: Choose from available models in the dropdown (populated from secure-llm)
3. **Chat**: Type messages and get responses with citations
4. **View Evidence**: Click on any `[[event_id]]` citation to see:
   - Full event content
   - Event metadata (timestamp, etc.)
   - Evidence snippets that support the claim

## Architecture

- **Backend**: FastAPI server (`react_server.py`) providing REST API endpoints
- **Frontend**: Single-page React app (`react_index.html`) using React via CDN
- **API Endpoints**:
  - `GET /` - Serves the React app
  - `GET /api/models` - Get available LLM models
  - `POST /api/chat` - Send chat message and get response
  - `POST /api/load-patient` - Load patient data
  - `GET /api/event/{event_id}` - Get event details
  - `GET /api/patient-status` - Get current patient status

## Citation Format

The LLM returns responses with citations in the format `[[event_id]]`. These are automatically parsed and made clickable in the React interface. When clicked, a modal shows:

- The full event content from the patient timeline
- Event metadata (timestamp, etc.)
- Evidence snippets that were used to support the claim

## Requirements

- Python 3.10+
- FastAPI
- uvicorn
- All dependencies from the main project (secure-llm, mcp, etc.)

The React app uses CDN-hosted libraries, so no Node.js or build step is required.

