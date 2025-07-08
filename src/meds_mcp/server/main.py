import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from meds_mcp.server.rag.simple_storage import initialize_document_store

# Initialize document store with cache and data directories
doc_store = initialize_document_store(
    cache_dir="data/scratch/cache",
    data_dir="data/collections/dev-corpus"
)

# Import the unified tools module - this will register all tools with the FastMCP instance
import meds_mcp.server.tools

# Get the FastMCP instance from the tools module
mcp = meds_mcp.server.tools.mcp

if __name__ == "__main__":
    import uvicorn
    # Get the StreamableHTTP app from FastMCP
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8000) 