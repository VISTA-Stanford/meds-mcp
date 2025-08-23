"""
Global state management for the MEDS MCP server.
"""

import os
from pathlib import Path
from typing import Optional, Union

# Global variables
athena_ontology: Optional[Union['AthenaOntology', 'LazyAthenaOntology']] = None

def initialize_athena_ontology(ontology_dir: str, use_lazy: bool = False):
    """
    Initialize the Athena ontology from the specified directory.
    
    Args:
        ontology_dir: Directory containing Athena ontology parquet files (for regular)
                      or path to Athena snapshot directory/zip (for lazy)
        use_lazy: If True, use LazyAthenaOntology for memory-efficient on-demand queries
    """
    global athena_ontology
    
    if use_lazy:
        print(f"Loading Lazy Athena ontology from {ontology_dir}...")
        try:
            # Import here to avoid circular imports
            from meds_mcp.server.tools.ontologies import LazyAthenaOntology
            
            athena_ontology = LazyAthenaOntology.load_from_athena_snapshot(ontology_dir)
            print("Lazy Athena ontology loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load Lazy Athena ontology: {e}")
            athena_ontology = None
    else:
        print(f"Loading Athena ontology from {ontology_dir}...")
        try:
            # Import here to avoid circular imports - use the new AthenaOntology with BM25 search
            from meds_mcp.server.tools.athena import AthenaOntology
            
            ontology_path = Path(ontology_dir)
            if ontology_path.exists():
                athena_ontology = AthenaOntology.load_from_parquet(str(ontology_path))
                print("Athena ontology with BM25 search loaded successfully!")
            else:
                print(f"Warning: Athena ontology directory not found: {ontology_path}")
                athena_ontology = None
        except Exception as e:
            print(f"Warning: Failed to load Athena ontology: {e}")
            athena_ontology = None 