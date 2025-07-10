#!/usr/bin/env python3
"""
MEDS MCP Server - Main entry point.


Example usage:

python src/meds_mcp/server/main.py \
--config configs/local.yaml

"""

import os
import asyncio
import uvicorn
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP instance
mcp = FastMCP(name="meds-mcp-server")

# Import all tool functions
from meds_mcp.server.tools.search import (
    search_patient_events,
    get_events_by_type,
    get_historical_values,
)
from meds_mcp.server.tools.ontologies import (
    get_code_metadata,
    get_ancestor_subgraph,
    get_descendant_subgraph,
)
from meds_mcp.server.rag.simple_storage import (
    load_patient_xml,
    load_patient_timeline,
    get_patient_event,
    list_patients,
    get_document_store_stats,
    list_all_node_ids,
    list_patient_node_ids,
    get_patient_events,
)

# Register all tools with decorators
search_patient_events_tool = mcp.tool("search_patient_events")(search_patient_events)
get_events_by_type_tool = mcp.tool("get_events_by_type")(get_events_by_type)
get_historical_values_tool = mcp.tool("get_historical_values")(get_historical_values)
get_code_metadata_tool = mcp.tool("get_code_metadata")(get_code_metadata)
get_ancestor_subgraph_tool = mcp.tool("get_ancestor_subgraph")(get_ancestor_subgraph)
get_descendant_subgraph_tool = mcp.tool("get_descendant_subgraph")(
    get_descendant_subgraph
)
load_patient_xml_tool = mcp.tool("load_patient_xml")(load_patient_xml)
load_patient_timeline_tool = mcp.tool("load_patient_timeline")(load_patient_timeline)
get_patient_event_tool = mcp.tool("get_patient_event")(get_patient_event)
list_patients_tool = mcp.tool("list_patients")(list_patients)
get_document_store_stats_tool = mcp.tool("get_document_store_stats")(
    get_document_store_stats
)
list_all_node_ids_tool = mcp.tool("list_all_node_ids")(list_all_node_ids)
list_patient_node_ids_tool = mcp.tool("list_patient_node_ids")(list_patient_node_ids)
get_patient_events_tool = mcp.tool("get_patient_events")(get_patient_events)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
            format=config.get("logging", {}).get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        )

        return config
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return {}


def initialize_server(config: Dict[str, Any]):
    """Initialize the server with required components."""
    # Initialize document store
    from meds_mcp.server.rag.simple_storage import initialize_document_store

    # Get corpus directory from config or environment variable
    data_dir = config.get("data", {}).get("corpus_dir") or os.getenv(
        "DATA_DIR", "data/collections/dev-corpus"
    )
    logging.info(f"Initializing document store with data directory: {data_dir}")
    initialize_document_store(data_dir)

    # Initialize Athena ontology
    from meds_mcp.server.globals import initialize_athena_ontology

    # Get ontology directory from config or environment variable
    ontology_dir = config.get("data", {}).get("ontology_dir") or os.getenv(
        "ONTOLOGY_DIR", "data/athena_omop_ontologies"
    )
    use_lazy_ontology = (
        config.get("data", {}).get("use_lazy_ontology", False)
        or os.getenv("USE_LAZY_ONTOLOGY", "false").lower() == "true"
    )

    logging.info(
        f"Initializing Athena ontology with directory: {ontology_dir}, lazy loading: {use_lazy_ontology}"
    )
    initialize_athena_ontology(ontology_dir, use_lazy=use_lazy_ontology)


def main():
    """Main entry point for the server."""
    # Load configuration
    config = load_config()

    # Initialize server components
    initialize_server(config)

    # Get server settings from config
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    logging.info(f"Starting MEDS MCP server on {host}:{port}")

    # Run the server
    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
