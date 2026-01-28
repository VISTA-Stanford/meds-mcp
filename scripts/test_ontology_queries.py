#!/usr/bin/env python3
"""
Test script for ontology path queries.

Tests:
1. get_parents() for ICD10CM/E11.9
2. get_ancestor_subgraph() restricted to ICD10CM vocabulary

Usage:
    # Use pregenerated parquet files (fastest)
    python scripts/test_ontology_queries.py --parquet_path data/athena_omop_ontologies
    
    # Use Athena snapshot (slower, but works with raw data)
    python scripts/test_ontology_queries.py --athena_path data/athena_ontologies_snapshot.zip
    
    # Use sample data for testing
    python scripts/test_ontology_queries.py --sample
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.ontologies import LazyAthenaOntology
from meds_mcp.server.tools.athena import AthenaOntology


def create_sample_athena_data(tmp_path: Path) -> Path:
    """Create sample Athena data for testing."""
    # Create sample CONCEPT.csv with ICD10CM/E11.9 and related concepts
    concept_data = [
        ["concept_id", "vocabulary_id", "concept_code", "concept_name", "invalid_reason", "standard_concept"],
        ["1", "SNOMED", "73211009", "Diabetes mellitus", "", "S"],
        ["2", "SNOMED", "44054006", "Type 2 diabetes mellitus", "", "S"],
        ["3", "ICD10CM", "E11.9", "Type 2 diabetes mellitus without complications", "", ""],
        ["4", "ICD10CM", "E11", "Type 2 diabetes mellitus", "", ""],
        ["5", "ICD10CM", "E08", "Diabetes mellitus due to underlying condition", "", ""],
        ["6", "ICD10CM", "E10", "Type 1 diabetes mellitus", "", ""],
        ["7", "SNOMED", "46635009", "Diabetes mellitus type 1", "", "S"],
    ]
    
    concept_content = "\n".join(["\t".join(row) for row in concept_data])
    (tmp_path / "CONCEPT.csv").write_text(concept_content)
    
    # Create CONCEPT_RELATIONSHIP.csv
    # ICD10CM/E11.9 maps to SNOMED concepts
    relationship_data = [
        ["concept_id_1", "concept_id_2", "relationship_id"],
        ["3", "2", "Maps to"],  # E11.9 -> Type 2 diabetes mellitus (SNOMED)
        ["4", "2", "Maps to"],  # E11 -> Type 2 diabetes mellitus (SNOMED)
        ["4", "1", "Maps to"],  # E11 -> Diabetes mellitus (SNOMED)
        ["6", "7", "Maps to"],  # E10 -> Type 1 diabetes (SNOMED)
    ]
    
    relationship_content = "\n".join(["\t".join(row) for row in relationship_data])
    (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_content)
    
    # Create CONCEPT_ANCESTOR.csv
    # E11.9 is a descendant of E11
    ancestor_data = [
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
        ["4", "3", "1"],  # E11 -> E11.9 (direct parent)
        ["1", "2", "1"],  # Diabetes mellitus -> Type 2 diabetes (SNOMED hierarchy)
        ["1", "3", "2"],  # Diabetes mellitus -> E11.9 (through E11 and Maps to)
    ]
    
    ancestor_content = "\n".join(["\t".join(row) for row in ancestor_data])
    (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_content)
    
    return tmp_path


def test_get_parents(ontology: Union[LazyAthenaOntology, AthenaOntology], code: str):
    """Test get_parents() method."""
    print(f"\n{'='*80}")
    print(f"Testing get_parents() for: {code}")
    print(f"Note: Returns ALL parents regardless of vocabulary")
    print(f"{'='*80}")
    
    parents = ontology.get_parents(code)
    
    print(f"\nFound {len(parents)} parent(s):")
    if parents:
        for parent in sorted(parents):
            description = ontology.get_description(parent)
            vocab = parent.split("/")[0] if "/" in parent else "Unknown"
            print(f"  - {parent} [{vocab}]: {description}")
    else:
        print("  (no parents found)")
    
    return parents


def test_get_ancestor_subgraph(ontology: Union[LazyAthenaOntology, AthenaOntology], code: str, vocabularies: list[str] = None):
    """Test get_ancestor_subgraph() method."""
    print(f"\n{'='*80}")
    print(f"Testing get_ancestor_subgraph() for: {code}")
    if vocabularies:
        print(f"Restricted to vocabularies: {vocabularies}")
    print(f"{'='*80}")
    
    import networkx as nx
    
    G = ontology.get_ancestor_subgraph(code, vocabularies=vocabularies)
    
    print(f"\nSubgraph contains {G.number_of_nodes()} node(s) and {G.number_of_edges()} edge(s)")
    
    if G.number_of_nodes() > 0:
        print("\nNodes in subgraph:")
        for node in sorted(G.nodes()):
            description = ontology.get_description(node)
            node_data = G.nodes[node]
            is_starting = node_data.get("is_starting_node", False)
            marker = " [STARTING]" if is_starting else ""
            print(f"  - {node}: {description}{marker}")
        
        if G.number_of_edges() > 0:
            print("\nEdges in subgraph (child -> parent):")
            for source, target in sorted(G.edges()):
                source_desc = ontology.get_description(source) or source
                target_desc = ontology.get_description(target) or target
                print(f"  {source} -> {target}")
                print(f"    ({source_desc[:50]} -> {target_desc[:50]})")
    
    return G


def main():
    parser = argparse.ArgumentParser(
        description="Test ontology path queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pregenerated parquet files (fastest - recommended)
  python scripts/test_ontology_queries.py --parquet_path data/athena_omop_ontologies
  
  # Test with sample data
  python scripts/test_ontology_queries.py --sample
  
  # Test with real Athena snapshot (slower, uses LazyAthenaOntology)
  python scripts/test_ontology_queries.py --athena_path data/athena_ontologies_snapshot.zip
  
  # Test with directory containing Athena CSV files
  python scripts/test_ontology_queries.py --athena_path data/athena_snapshot/
        """
    )
    
    parser.add_argument(
        "--parquet_path",
        type=str,
        help="Path to directory containing pregenerated parquet files (descriptions.parquet, parents.parquet)"
    )
    
    parser.add_argument(
        "--athena_path",
        type=str,
        help="Path to Athena snapshot directory or zip archive (uses LazyAthenaOntology)"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample test data instead of real Athena snapshot"
    )
    
    parser.add_argument(
        "--code",
        type=str,
        default="ICD10CM/E11.9",
        help="Code to test (default: ICD10CM/E11.9)"
    )
    
    args = parser.parse_args()
    
    ontology = None
    
    # Determine data source - prioritize parquet (fastest)
    if args.parquet_path:
        parquet_path = Path(args.parquet_path)
        if not parquet_path.exists():
            print(f"Error: Parquet path does not exist: {parquet_path}")
            sys.exit(1)
        
        descriptions_file = parquet_path / "descriptions.parquet"
        parents_file = parquet_path / "parents.parquet"
        
        if not descriptions_file.exists() or not parents_file.exists():
            print(f"Error: Required parquet files not found in {parquet_path}")
            print(f"  Expected: {descriptions_file}")
            print(f"  Expected: {parents_file}")
            sys.exit(1)
        
        print(f"Loading AthenaOntology from parquet files: {parquet_path}")
        print("  (Using pregenerated parquet - fastest option)")
        ontology = AthenaOntology.load_from_parquet(str(parquet_path))
        print(f"Loaded ontology with {len(ontology.description_map)} concepts")
        
    elif args.sample:
        print("Creating sample Athena data...")
        # Use a persistent temp directory since LazyAthenaOntology uses lazy evaluation
        tmpdir = tempfile.mkdtemp(prefix="test_ontology_")
        try:
            athena_path = create_sample_athena_data(Path(tmpdir))
            print(f"Sample data created at: {athena_path}")
            print("\nLoading LazyAthenaOntology from sample data...")
            print("  (Note: Using LazyAthenaOntology - data files must remain accessible)")
            ontology = LazyAthenaOntology.load_from_athena_snapshot(
                str(athena_path),
                ignore_invalid=True
            )
            # Note: We don't clean up tmpdir here because LazyAthenaOntology uses lazy evaluation
            # The directory will be cleaned up on script exit or manually
        except Exception as e:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
            
    elif args.athena_path:
        athena_path = Path(args.athena_path)
        if not athena_path.exists():
            print(f"Error: Athena snapshot path does not exist: {athena_path}")
            sys.exit(1)
        
        print(f"Loading LazyAthenaOntology from: {athena_path}")
        print("  (Using raw snapshot - slower but works with raw CSV data)")
        ontology = LazyAthenaOntology.load_from_athena_snapshot(
            str(athena_path),
            ignore_invalid=True
        )
        
    else:
        parser.print_help()
        print("\nError: Must specify one of: --parquet_path, --athena_path, or --sample")
        print("\nRecommended: Use --parquet_path with pregenerated parquet files for fastest loading")
        sys.exit(1)
    
    if ontology is None:
        print("Error: Failed to load ontology")
        sys.exit(1)
    
    # Run tests
    test_get_parents(ontology, args.code)
    test_get_ancestor_subgraph(ontology, args.code, vocabularies=["ICD10CM"])
    
    print(f"\n{'='*80}")
    print("Tests completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
