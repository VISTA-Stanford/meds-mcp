""" 
Tools for querying ontologies.

"""

import os
import mcp
from dotenv import load_dotenv
from meds2text.ontology.athena import AthenaOntology  # type: ignore
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
    # Handle vocabulary filtering
    if vocabularies is None or '*' in vocabularies:
        # No filtering - get full subgraph
        G = _ontology.get_subgraph_to_roots(code)
    else:
        # Filter by vocabularies
        G = _get_filtered_subgraph(code, vocabularies)
    
    # Print all node names and their text descriptions
    print(f"\n=== Subgraph for code: {code} ===")
    if vocabularies and '*' not in vocabularies:
        print(f"Filtered to vocabularies: {vocabularies}")
    print("Nodes and their descriptions:")
    for node in G.nodes():
        description = _ontology.get_description(node)
        print(f"  {node}: {description}")
    print("=" * 50)
    
    # Convert DiGraph to dictionary format for return
    return {
        "nodes": list(G.nodes()),
        "edges": list(G.edges())
    }


def _get_filtered_subgraph(code: str, allowed_vocabularies: list[str]):
    """
    Get a filtered subgraph that only includes nodes from allowed vocabularies.
    
    Args:
        code: Starting code
        allowed_vocabularies: List of vocabulary prefixes to allow
        
    Returns:
        NetworkX DiGraph containing only nodes from allowed vocabularies
    """
    import networkx as nx
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    def is_allowed_node(node: str) -> bool:
        """Check if a node belongs to an allowed vocabulary."""
        for vocab in allowed_vocabularies:
            if node.startswith(f"{vocab}/"):
                return True
        return False
    
    def add_filtered_paths(current_code: str, visited: set):
        """Recursively add paths that only go through allowed vocabularies."""
        if current_code in visited:
            return
        
        visited.add(current_code)
        
        # Add the current node if it's allowed
        if is_allowed_node(current_code):
            G.add_node(current_code)
            
            # Get parents and filter them
            parents = _ontology.get_parents(current_code)
            for parent in parents:
                if is_allowed_node(parent):
                    G.add_edge(current_code, parent)
                    add_filtered_paths(parent, visited)
                else:
                    # If parent is not allowed, try to find allowed ancestors
                    add_filtered_paths(parent, visited)
        else:
            # Current node is not allowed, but we still need to explore its parents
            parents = _ontology.get_parents(current_code)
            for parent in parents:
                add_filtered_paths(parent, visited)
    
    # Start the recursive exploration
    add_filtered_paths(code, set())
    
    return G
    

@mcp.tool()
def get_descendant_subgraph(code: str) -> dict:
    """
    Get the descendant subgraph of a code.
    """
    return None #_ontology.get_descendant_subgraph(code)


if __name__ == "__main__":
    print("=== Testing full subgraph (all vocabularies) ===")
    print(get_ancestor_subgraph("RxNorm/308189"))
    
    print("\n=== Testing filtered subgraph (RxNorm only) ===")
    print(get_ancestor_subgraph("RxNorm/308189", vocabularies=["RxNorm"]))
    
    print("\n=== Testing filtered subgraph (ATC only) ===")
    print(get_ancestor_subgraph("RxNorm/308189", vocabularies=["ATC"]))
    
    print("\n=== Testing filtered subgraph (RxNorm and ATC) ===")
    print(get_ancestor_subgraph("RxNorm/308189", vocabularies=["RxNorm", "ATC"]))
    
    print("\n=== Testing wildcard (all vocabularies) ===")
    print(get_ancestor_subgraph("RxNorm/308189", vocabularies=["*"]))