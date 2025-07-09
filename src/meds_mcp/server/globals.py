"""
Global state management for the MEDS MCP server.
"""

import os
from pathlib import Path
from typing import Optional

# Global variables
athena_ontology: Optional[object] = None

def initialize_athena_ontology(ontology_dir: str):
    """
    Initialize the Athena ontology from the specified directory.
    
    Args:
        ontology_dir: Directory containing Athena ontology files
    """
    global athena_ontology
    
    print(f"Loading Athena ontology from {ontology_dir}...")
    
    try:
        # Import here to avoid circular imports
        from tools.ontologies import AthenaOntology
        
        ontology_path = Path(ontology_dir)
        if ontology_path.exists():
            athena_ontology = AthenaOntology(str(ontology_path))
            print("Athena ontology loaded successfully!")
        else:
            print(f"Warning: Athena ontology directory not found: {ontology_path}")
            athena_ontology = None
    except Exception as e:
        print(f"Warning: Failed to load Athena ontology: {e}")
        athena_ontology = None 