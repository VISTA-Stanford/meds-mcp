# Quick Start Guide: React Chat Interface

This guide shows you how to run the React chat interface with the MCP backend.

## Prerequisites

1. **MCP Backend Server** must be running (port 8000)
2. **React Frontend Server** will run on port 8080
3. Make sure you have the required dependencies installed

## Step-by-Step Instructions

### Step 1: Start the MCP Backend Server

Open a **first terminal window** and run:

```bash
# From the project root directory
python src/meds_mcp/server/main.py --config configs/medalign.yaml
```

Or if using `uv`:

```bash
uv run python src/meds_mcp/server/main.py --config configs/medalign.yaml
```

You should see:
```
Starting MEDS MCP server on 0.0.0.0:8000
```

**Keep this terminal running** - the MCP server must stay active.

### Step 2: Start the React Frontend Server

Open a **second terminal window** and run:

```bash
# From the project root directory (recommended)
uv run python examples/mcp_chat_demo/react_chat_demo.py \
    --model "apim:gpt-4.1" \
    --mcp_url "http://localhost:8000/mcp" \
    --patient_id 127672063
```

Or using the alternative launcher:

```bash
uv run python examples/mcp_chat_demo/chat/ui/run_react_app.py \
    --model "apim:gpt-4.1" \
    --mcp_url "http://localhost:8000/mcp" \
    --patient_id 127672063
```

**Available models** (adjust based on what you have access to):

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

You should see:
```
============================================================
Starting React Chat Interface
============================================================
MCP URL: http://localhost:8000/mcp
Default Model: apim:gpt-4o-mini
Auto-load Patient: 127672063
============================================================

Open your browser to: http://localhost:8080
Press Ctrl+C to stop the server
```

### Step 3: Open the Chat Interface

Open your web browser and navigate to:

**http://localhost:8080**

You should see the React chat interface!

## Using the Interface

1. **Load a Patient**: If you didn't use `--patient_id`, enter a patient ID in the header and click "Load"
2. **Select a Model**: Choose from available models in the dropdown (populated from secure-llm)
3. **Start Chatting**: Type a message and press Enter or click "Send"
4. **View Evidence**: Click on any `[[event_id]]` citation in responses to see detailed event information

## Troubleshooting

### "Connection failed" or "MCP URL not configured"
- Make sure the MCP backend server is running on port 8000
- Check that the `--mcp_url` matches your MCP server URL

### "No patient data loaded"
- Load a patient using the patient ID input in the header
- Make sure the patient ID exists in your data

### "LLM client not initialized"
- Check that you have `VAULT_SECRET_KEY` set in your environment
- Verify that secure-llm is properly installed and configured

### Port already in use
- If port 8000 is in use, change the MCP server port in the config file
- If port 8080 is in use, modify `run_react_app.py` to use a different port

## Running Without Auto-Load Patient

If you want to load patients manually through the UI:

```bash
uv run python examples/mcp_chat_demo/react_chat_demo.py \
    --model "apim:gpt-4o-mini" \
    --mcp_url "http://localhost:8000/mcp"
```

(Just omit the `--patient_id` parameter)

## Stopping the Servers

- Press `Ctrl+C` in each terminal window to stop the respective servers
- Stop the React frontend first, then the MCP backend

