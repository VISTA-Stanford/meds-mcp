#!/usr/bin/env python3
"""
Launcher script for the React-based chat interface.

Usage:
    python examples/mcp_chat_demo/chat/ui/run_react_app.py \
        --model "nero:gemini-2.0-flash" \
        --mcp_url "http://localhost:8000/mcp" \
        --patient_id 127672063
"""

import sys
import os
from pathlib import Path

# Add parent directories to path so we can import chat module
# run_react_app.py is at: examples/mcp_chat_demo/chat/ui/run_react_app.py
# We need to add examples/mcp_chat_demo to path so 'chat' module is importable
mcp_chat_demo_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mcp_chat_demo_dir))

from chat.ui.react_server import app
import uvicorn
from chat.config import parse_args

if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("Starting React Chat Interface")
    print("=" * 60)
    print(f"MCP URL: {args.mcp_url}")
    print(f"Default Model: {args.model}")
    if args.patient_id:
        print(f"Auto-load Patient: {args.patient_id}")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:8080")
    print("Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )

