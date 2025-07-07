""" 
Tools for querying ontologies.

"""

import os
import mcp
from dotenv import load_dotenv
from meds_mcp.server.tools.athena import AthenaOntology 
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tools")

load_dotenv()

# Initialize ontology once at module level
# You can set this via environment variable or modify the path directly
PARQUET_PATH = os.getenv("ATHENA_VOCABULARIES_PATH", "data/athena_omop_ontologies/")

# Load ontology at module import time (startup)
print(f"Loading Athena ontology from {PARQUET_PATH}...")
_ontology = AthenaOntology.load_from_parquet(PARQUET_PATH)
print("Athena ontology loaded successfully!")

@mcp.tool()
def get_code_metadata(code: str) -> dict:
    """
    Get the metadata for a code.
    """
    return {"text":_ontology.get_description(code)}

@mcp.tool()
def get_ancestor_subgraph(code: str, vocabularies: list[str] = None) -> dict:
    """
    Get the ancestor subgraph of a code, optionally restricted to specific vocabularies.
    
    Args:
        code: The starting medical code
        vocabularies: List of allowed vocabularies (e.g., ['RxNorm', 'ATC']). 
                     Use '*' to allow all vocabularies. Default is None (all vocabularies).
    """
    G = _ontology.get_ancestor_subgraph(code, vocabularies)
    return _ontology.get_graph_metadata(G)


@mcp.tool()
def get_descendant_subgraph(code: str, vocabularies: list[str] = None) -> dict:
    """
    Get the descendant subgraph of a code.
    """
    G = _ontology.get_descendant_subgraph(code, vocabularies)
    return _ontology.get_graph_metadata(G)

if __name__ == "__main__":
    code = "SNOMED/363358000"
    vocabularies = [code.split("/")[0]]
    print("=== Testing full subgraph (all vocabularies) ===")
    print("ancestor", len(get_ancestor_subgraph(code)))
    print("descendant", len(get_descendant_subgraph(code)))
    print("=== Testing filtered subgraph (SNOMED only) ===")
    print("ancestor", len(get_ancestor_subgraph(code, vocabularies=vocabularies)))
    print("descendant", len(get_descendant_subgraph(code, vocabularies=vocabularies)))
    