#!/usr/bin/env python3
"""
MEDS MCP Server - Main entry point.
"""

import os
import asyncio
import uvicorn
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP instance
mcp = FastMCP(name="meds-mcp-server")

# Import all tool functions
from meds_mcp.server.tools.search import search_patient_events, get_events_by_type, get_historical_values
from meds_mcp.server.tools.ontologies import get_code_metadata, get_ancestor_subgraph, get_descendant_subgraph
from meds_mcp.server.rag.simple_storage import (
    load_patient_xml, load_patient_timeline, get_patient_event, list_patients,
    get_document_store_stats, list_all_node_ids, list_patient_node_ids, get_patient_events
)

# Register all tools with decorators
search_patient_events_tool = mcp.tool("search_patient_events")(search_patient_events)
get_events_by_type_tool = mcp.tool("get_events_by_type")(get_events_by_type)
get_historical_values_tool = mcp.tool("get_historical_values")(get_historical_values)
get_code_metadata_tool = mcp.tool("get_code_metadata")(get_code_metadata)
get_ancestor_subgraph_tool = mcp.tool("get_ancestor_subgraph")(get_ancestor_subgraph)
get_descendant_subgraph_tool = mcp.tool("get_descendant_subgraph")(get_descendant_subgraph)
load_patient_xml_tool = mcp.tool("load_patient_xml")(load_patient_xml)
load_patient_timeline_tool = mcp.tool("load_patient_timeline")(load_patient_timeline)
get_patient_event_tool = mcp.tool("get_patient_event")(get_patient_event)
list_patients_tool = mcp.tool("list_patients")(list_patients)
get_document_store_stats_tool = mcp.tool("get_document_store_stats")(get_document_store_stats)
list_all_node_ids_tool = mcp.tool("list_all_node_ids")(list_all_node_ids)
list_patient_node_ids_tool = mcp.tool("list_patient_node_ids")(list_patient_node_ids)
get_patient_events_tool = mcp.tool("get_patient_events")(get_patient_events)

def initialize_server():
    """Initialize the server with required components."""
    # Initialize document store
    from meds_mcp.server.rag.simple_storage import initialize_document_store
    data_dir = os.getenv("DATA_DIR", "data/collections/dev-corpus")
    initialize_document_store(data_dir)
    
    # Initialize Athena ontology
    from meds_mcp.server.globals import initialize_athena_ontology
    ontology_dir = os.getenv("ONTOLOGY_DIR", "data/athena_omop_ontologies")
    initialize_athena_ontology(ontology_dir)

def main():
    """Main entry point for the server."""
    initialize_server()
    
    # Run the server
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 