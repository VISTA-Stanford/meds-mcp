#!/usr/bin/env python3
"""
Rebuild sparse graphs from Athena snapshot - ONE MATRIX PER RELATIONSHIP TYPE.

Creates a directory with:
- codes.parquet: Shared code index
- {relationship_type}.npz: One sparse matrix per relationship type
- manifest.txt: List of available relationship types

This is MUCH faster to load than a single matrix + huge dictionary.
Only load the relationship types you need!
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild sparse graphs from Athena snapshot (one matrix per relationship type)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rebuild sparse graphs
  python scripts/rebuild_sparse_graph.py \\
    --athena_path data/athena_ontologies_snapshot.zip \\
    --parquet_path data/athena_omop_ontologies

Output directory structure:
  ontology_graphs/
    codes.parquet        # Shared code index
    is_a.npz             # Sparse matrix for "Is a" edges  
    maps_to.npz          # Sparse matrix for "Maps to" edges
    rxnorm_has_ing.npz   # etc.
    manifest.txt         # List of available types
        """
    )
    
    parser.add_argument(
        "--athena_path",
        type=str,
        required=True,
        help="Path to Athena snapshot (zip file or directory)"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Directory containing descriptions.parquet (for metadata queries)"
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default=None,
        help="Directory to save sparse graphs (defaults to parquet_path/ontology_graphs)"
    )
    
    args = parser.parse_args()
    
    graph_path = args.graph_path or f"{args.parquet_path}/ontology_graphs"
    
    print("="*80)
    print("Rebuilding Sparse Graphs (One Matrix Per Relationship Type)")
    print("="*80)
    print(f"Athena snapshot: {args.athena_path}")
    print(f"Parquet path: {args.parquet_path}")
    print(f"Output directory: {graph_path}")
    print("="*80)
    print()
    
    # Build sparse graphs from Athena snapshot
    ontology = SparseGraphOntology.build_from_athena_snapshot(
        athena_path=args.athena_path,
        parquet_path=args.parquet_path,
        graph_path=graph_path,
    )
    
    print()
    print("="*80)
    print("✅ Sparse graphs rebuilt successfully!")
    print("="*80)
    print(f"Relationship types available: {len(ontology._relationship_matrices)}")
    print(f"Types: {list(ontology._relationship_matrices.keys())[:10]}...")
    print()
    print("Usage: Only load the relationship types you need for fast startup!")
    print("  e.g., load_from_parquet(..., relationship_types=['Is a', 'Maps to'])")


if __name__ == "__main__":
    main()
