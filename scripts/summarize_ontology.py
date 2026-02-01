#!/usr/bin/env python3
"""
Summarize ontology statistics: vocabulary distribution, relationships, etc.
"""

import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import pyarrow.parquet as pq


def summarize_vocabularies(parquet_path: str):
    """Summarize vocabulary distribution."""
    print("="*80)
    print("Vocabulary Distribution")
    print("="*80)
    
    descriptions_path = Path(parquet_path) / "descriptions.parquet"
    if not descriptions_path.exists():
        print(f"Error: {descriptions_path} not found")
        return
    
    # Load concepts
    concepts_df = pl.scan_parquet(str(descriptions_path))
    
    # Extract vocabulary from code
    vocab_df = (
        concepts_df
        .with_columns([
            pl.col("code").str.split("/").list.first().alias("vocabulary")
        ])
        .select(["vocabulary", "code"])
        .collect()
    )
    
    # Count by vocabulary
    vocab_counts = vocab_df.group_by("vocabulary").agg(pl.len().alias("count")).sort("count", descending=True)
    
    print(f"\nTotal concepts: {vocab_df.height:,}")
    print(f"Unique vocabularies: {vocab_counts.height}")
    print("\nTop 20 vocabularies:")
    print(f"{'Vocabulary':<20} {'Count':>15} {'%':>10}")
    print("-" * 45)
    
    total = vocab_df.height
    for row in vocab_counts.head(20).rows():
        vocab, count = row
        pct = (count / total) * 100
        print(f"{vocab:<20} {count:>15,} {pct:>9.2f}%")


def summarize_relationships(parquet_path: str):
    """Summarize relationship types."""
    print("\n" + "="*80)
    print("Relationship Distribution")
    print("="*80)
    
    parents_path = Path(parquet_path) / "parents.parquet"
    if not parents_path.exists():
        print(f"Error: {parents_path} not found")
        return
    
    # Load parents parquet - structure: code (string), parents (list of strings)
    parents_df = pl.scan_parquet(str(parents_path))
    
    # Count parent relationships
    parent_counts = (
        parents_df
        .with_columns([
            pl.col("parents").list.len().alias("parent_count")
        ])
        .select(["code", "parent_count"])
        .filter(pl.col("parent_count") > 0)  # Filter out codes with no parents
        .collect()
    )
    
    print(f"\nTotal codes with parents: {parent_counts.height:,}")
    
    # Distribution of parent counts
    parent_dist = (
        parent_counts
        .group_by("parent_count")
        .agg(pl.len().alias("code_count"))
        .sort("parent_count")
    )
    
    print("\nParent count distribution:")
    print(f"{'Parent Count':<15} {'Codes':>15}")
    print("-" * 30)
    for row in parent_dist.head(20).rows():
        parent_count, code_count = row
        print(f"{parent_count:<15} {code_count:>15,}")
    
    # Average parents per code
    avg_parents = parent_counts["parent_count"].mean()
    print(f"\nAverage parents per code: {avg_parents:.2f}")


def summarize_relationship_types(athena_path: Optional[str] = None):
    """Summarize relationship types from Athena snapshot."""
    print("\n" + "="*80)
    print("Relationship Types Distribution")
    print("="*80)
    
    if not athena_path:
        print("\n(Athena snapshot path not provided - skipping relationship type analysis)")
        print("To see relationship types, provide --athena_path")
        return
    
    # Import directly to avoid loading the full LazyAthenaOntology class
    # Import AthenaFileReader - it's defined before LazyAthenaOntology
    # We need to import it carefully to avoid the List type issue in LazyAthenaOntology
    try:
        from meds_mcp.server.tools.ontologies import AthenaFileReader
    except NameError:
        # Fallback: import the file and extract just AthenaFileReader
        import sys
        import importlib
        ontologies_module = importlib.import_module('meds_mcp.server.tools.ontologies')
        AthenaFileReader = ontologies_module.AthenaFileReader
    
    try:
        with AthenaFileReader(athena_path) as reader:
            # Load CONCEPT_RELATIONSHIP.csv
            relationships_df = reader.read_csv("CONCEPT_RELATIONSHIP.csv")
            
            # Filter out invalid relationships
            valid_relationships = relationships_df.filter(
                (pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
                & (pl.col("concept_id_1") != pl.col("concept_id_2"))  # Exclude self-loops
            )
            
            # Count by relationship type
            rel_type_counts = (
                valid_relationships
                .group_by("relationship_id")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .collect()
            )
            
            total_relationships = valid_relationships.select(pl.len()).collect().row(0)[0]
            
            print(f"\nTotal valid relationships: {total_relationships:,}")
            print(f"Unique relationship types: {rel_type_counts.height}")
            print("\nRelationship type distribution:")
            print(f"{'Relationship Type':<30} {'Count':>15} {'%':>10}")
            print("-" * 55)
            
            for row in rel_type_counts.rows():
                rel_type, count = row
                pct = (count / total_relationships) * 100
                print(f"{rel_type:<30} {count:>15,} {pct:>9.2f}%")
            
            # Highlight key relationship types
            print("\nKey relationship types:")
            key_types = ["Is a", "Maps to", "Subsumes", "Mapped from", "Concept same as"]
            for rel_type in key_types:
                matching = rel_type_counts.filter(pl.col("relationship_id") == rel_type)
                if matching.height > 0:
                    count = matching.row(0)[1]
                    pct = (count / total_relationships) * 100
                    status = "✅ INCLUDED" if rel_type in ["Is a", "Maps to"] else "❌ NOT INCLUDED"
                    print(f"  • {rel_type:<20} {count:>12,} ({pct:>5.2f}%) {status}")
                else:
                    status = "❌ NOT INCLUDED"
                    print(f"  • {rel_type:<20} {'N/A':>12} {status}")
            
    except Exception as e:
        print(f"\nError loading relationship types: {e}")
        print("Make sure --athena_path points to a valid Athena snapshot")


def summarize_graph_structure(parquet_path: str, graph_path: str):
    """Summarize graph structure if graph file exists."""
    print("\n" + "="*80)
    print("Graph Structure")
    print("="*80)
    
    from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology
    
    try:
        ontology = SparseGraphOntology.load_from_parquet(
            parquet_path,
            graph_path=graph_path,
            load_graph=True,
        )
        
        matrix = ontology.parent_matrix
        print(f"\nSparse matrix:")
        print(f"  Shape: {matrix.shape[0]:,} x {matrix.shape[1]:,}")
        print(f"  Non-zero entries: {matrix.nnz:,}")
        print(f"  Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.6f}%")
        
        # Sample some statistics
        print(f"\nSampling node statistics...")
        sample_size = min(1000, matrix.shape[0])
        import numpy as np
        np.random.seed(42)
        sample_indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
        
        parent_counts = []
        for idx in sample_indices:
            parent_count = len(matrix[idx].indices)
            parent_counts.append(parent_count)
        
        print(f"  Sample size: {sample_size}")
        print(f"  Average parents (sample): {np.mean(parent_counts):.2f}")
        print(f"  Max parents (sample): {np.max(parent_counts)}")
        print(f"  Min parents (sample): {np.min(parent_counts)}")
        
    except Exception as e:
        print(f"Could not load graph: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ontology statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="data/athena_omop_ontologies",
        help="Path to parquet files directory"
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="data/athena_omop_ontologies/ontology_graph_sparse.npz",
        help="Path to sparse graph file (optional)"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph structure analysis"
    )
    parser.add_argument(
        "--athena_path",
        type=str,
        help="Path to Athena snapshot (zip or directory) for relationship type analysis"
    )
    
    args = parser.parse_args()
    
    summarize_vocabularies(args.parquet_path)
    summarize_relationships(args.parquet_path)
    summarize_relationship_types(args.athena_path)
    
    if not args.skip_graph:
        summarize_graph_structure(args.parquet_path, args.graph_path)
    
    print("\n" + "="*80)
    print("Summary complete")
    print("="*80)


if __name__ == "__main__":
    main()
