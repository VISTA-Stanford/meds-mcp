#!/usr/bin/env python3
"""
MEDS MCP Server - Main entry point.


Example usage:

python src/meds_mcp/server/main.py \
--config configs/medalign.yaml

"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # Suppress TPU warnings

import asyncio
import uvicorn
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI

# Tool imports will be done in main() to avoid import-time issues


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
    # Get cache directory from config or environment variable
    cache_dir = config.get("data", {}).get("cache_dir") or os.getenv(
        "CACHE_DIR", "cache"
    )
    # Get load_all_patients option from config or environment variable
    load_all_patients = (
        config.get("data", {}).get("load_all_patients", False)
        or os.getenv("LOAD_ALL_PATIENTS", "false").lower() == "true"
    )

    # Validate and report on corpus directory
    data_path = Path(data_dir)
    logging.info(f"Checking corpus directory: {data_dir}")

    if not data_path.exists():
        logging.error(f"❌ Corpus directory does not exist: {data_dir}")
        logging.error(
            "Please create the directory or update the corpus_dir configuration"
        )
        raise FileNotFoundError(f"Corpus directory not found: {data_dir}")

    if not data_path.is_dir():
        logging.error(f"❌ Corpus path is not a directory: {data_dir}")
        raise NotADirectoryError(f"Corpus path is not a directory: {data_dir}")

    # Count XML files in the directory
    xml_files = list(data_path.glob("*.xml"))
    logging.info(f"✅ Corpus directory exists: {data_dir}")
    logging.info(f"📁 Found {len(xml_files)} XML files to index")

    if len(xml_files) == 0:
        logging.warning("⚠️  No XML files found in corpus directory")
        logging.warning("Queries will fail until documents are added to the corpus")
    else:
        # Show sample of file names (first 5)
        sample_files = [f.name for f in xml_files[:5]]
        logging.info(f"📄 Sample files: {sample_files}")
        if len(xml_files) > 5:
            logging.info(f"... and {len(xml_files) - 5} more files")

    # Validate and create cache directory
    cache_path = Path(cache_dir)
    logging.info(f"Using cache directory: {cache_dir}")

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"✅ Cache directory ready: {cache_dir}")
    except Exception as e:
        logging.error(f"❌ Failed to create cache directory: {e}")
        raise
    
    # Initialize document store
    if load_all_patients:
        logging.info(
            "🚀 Initializing document store with ALL patients loaded (development mode)..."
        )
    else:
        logging.info("🚀 Initializing document store (lazy loading mode)...")

    try:
        initialize_document_store(data_dir, cache_dir, load_all_patients)
        logging.info("✅ Document store initialized successfully")
    except Exception as e:
        logging.error(f"❌ Failed to initialize document store: {e}")
        raise

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MEDS MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Import all tool functions after config is loaded
    from meds_mcp.server.tools.search import (
        search_patient_events,
        get_events_by_type,
        get_historical_values,
    )
    from meds_mcp.server.tools.ontologies import (
        get_code_metadata,
        get_ancestor_subgraph,
        get_descendant_subgraph,
        search_codes,
    )
    from meds_mcp.server.rag.simple_storage import (
        load_patient_xml,
        load_patient_timeline,
        get_patient_timeline,
        get_patient_event,
        list_patients,
        get_document_store_stats,
        list_all_node_ids,
        list_patient_node_ids,
        get_all_patient_events,
    )

    # Initialize server components
    initialize_server(config)

    # Initialize the FastMCP instance after config is loaded
    mcp = FastMCP(name="meds-mcp-server")

    # Register all tools with decorators
    search_patient_events_tool = mcp.tool("search_patient_events")(search_patient_events)
    get_events_by_type_tool = mcp.tool("get_events_by_type")(get_events_by_type)
    get_historical_values_tool = mcp.tool("get_historical_values")(get_historical_values)
    get_code_metadata_tool = mcp.tool("get_code_metadata")(get_code_metadata)
    get_ancestor_subgraph_tool = mcp.tool("get_ancestor_subgraph")(get_ancestor_subgraph)
    get_descendant_subgraph_tool = mcp.tool("get_descendant_subgraph")(
        get_descendant_subgraph
    )
    search_codes_tool = mcp.tool("search_codes")(search_codes)
    load_patient_xml_tool = mcp.tool("load_patient_xml")(load_patient_xml)
    load_patient_timeline_tool = mcp.tool("load_patient_timeline")(load_patient_timeline)
    get_patient_timeline_tool = mcp.tool("get_patient_timeline")(get_patient_timeline)
    get_patient_event_tool = mcp.tool("get_patient_event")(get_patient_event)
    list_patients_tool = mcp.tool("list_patients")(list_patients)
    get_document_store_stats_tool = mcp.tool("get_document_store_stats")(
        get_document_store_stats
    )
    list_all_node_ids_tool = mcp.tool("list_all_node_ids")(list_all_node_ids)
    list_patient_node_ids_tool = mcp.tool("list_patient_node_ids")(list_patient_node_ids)
    get_all_patient_events_tool = mcp.tool("get_all_patient_events")(get_all_patient_events)

    # Get server settings from config
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    logging.info(f"Starting MEDS MCP server on {host}:{port}")

    # Get the Starlette app from MCP
    app = mcp.streamable_http_app()

    # Create a FastAPI app for faceted search
    from meds_mcp.server.api import faceted_search

    search_api = FastAPI()
    search_api.include_router(faceted_search.router, prefix="/faceted-search", tags=["search"])
    app.mount("/api", search_api)

    # Run the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
