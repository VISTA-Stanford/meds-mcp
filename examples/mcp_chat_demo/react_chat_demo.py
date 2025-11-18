"""
React Chat Demo - Lightweight React-based chat interface

A modern, lightweight React frontend for interacting with patient records via the MCP server.

Key Features:
- Clean, modern chat interface
- Clickable citations that show timeline event details
- Model selection from available secure-llm models
- Patient loading and management
- Evidence viewing with event details

Usage:
    uv run python examples/mcp_chat_demo/react_chat_demo.py \
        --model "apim:gpt-4.1" \
        --mcp_url "http://localhost:8000/mcp" \
        --patient_id 127672063

Or without auto-loading a patient:
    uv run python examples/mcp_chat_demo/react_chat_demo.py \
        --model "apim:gpt-4.1" \
        --mcp_url "http://localhost:8000/mcp"

Available models for --model parameter:
    APIM: apim:gpt-4.1, apim:gpt-4.1-mini, apim:gpt-4.1-nano, apim:o3-mini,
          apim:claude-3.5, apim:claude-3.7, apim:gemini-2.0-flash,
          apim:gemini-2.5-pro-preview-05-06, apim:llama-3.3-70b,
          apim:llama-4-maverick-17b, apim:llama-4-scout-17b, apim:deepseek-chat
    Nero: nero:gemini-2.0-flash, nero:gemini-2.5-pro, nero:gemini-2.5-flash,
          nero:gemini-2.5-flash-lite
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

# Import configuration
from chat.config import parse_args

# Note: react_server and uvicorn are imported later to show progress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Print immediate startup message
    print("=" * 60, flush=True)
    print("Starting React Chat Interface...", flush=True)
    print("=" * 60, flush=True)
    
    # Parse command line arguments
    print("📋 Parsing arguments...", flush=True)
    args = parse_args()
    
    print("📦 Loading modules (this may take a moment)...", flush=True)
    
    # Import the React server app (this may be slow)
    try:
        from chat.ui.react_server import app, set_server_args
        # Pass the parsed args to the server so it knows about patient_id
        set_server_args(args)
        print("✅ Modules loaded", flush=True)
    except Exception as e:
        print(f"❌ Error loading modules: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    import uvicorn
    
    # Print startup banner
    print("\n" + "=" * 60, flush=True)
    print("React Chat Interface", flush=True)
    print("=" * 60, flush=True)
    print(f"MCP URL: {args.mcp_url}", flush=True)
    print(f"Default Model: {args.model}", flush=True)
    if args.patient_id:
        print(f"Auto-load Patient: {args.patient_id}", flush=True)
    print("=" * 60, flush=True)
    print("\n🌐 Open your browser to: http://localhost:8080", flush=True)
    print("Press Ctrl+C to stop the server\n", flush=True)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )

