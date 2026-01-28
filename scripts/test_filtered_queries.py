#!/usr/bin/env python3
"""
Test script for filtered ontology queries.

Demonstrates:
1. Filtering by relationship type (Is a vs Maps to)
2. Filtering by vocabulary
3. Combining filters
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology
from meds_mcp.server.tools.athena import AthenaOntology


def test_filtered_queries(ontology, code: str = "ICD10CM/C34.02"):
    """Test various filtering options."""
    
    print(f"\n{'='*80}")
    print(f"Testing filtered queries for: {code}")
    print(f"{'='*80}\n")
    
    # 1. All parents (no filters)
    print("1. All parents (no filters):")
    parents_all = ontology.get_parents(code)
    print(f"   Found {len(parents_all)} parents:")
    for parent in sorted(parents_all):
        desc = ontology.get_description(parent)
        vocab = parent.split("/")[0] if "/" in parent else "Unknown"
        print(f"     - {parent} [{vocab}]: {desc}")
    
    # 2. Only ICD10CM vocabulary
    print("\n2. Parents filtered to ICD10CM vocabulary only:")
    parents_icd10cm = ontology.get_parents(code, vocabularies=["ICD10CM"])
    print(f"   Found {len(parents_icd10cm)} parents:")
    for parent in sorted(parents_icd10cm):
        desc = ontology.get_description(parent)
        print(f"     - {parent}: {desc}")
    
    # 3. Only SNOMED vocabulary
    print("\n3. Parents filtered to SNOMED vocabulary only:")
    parents_snomed = ontology.get_parents(code, vocabularies=["SNOMED"])
    print(f"   Found {len(parents_snomed)} parents:")
    for parent in sorted(parents_snomed):
        desc = ontology.get_description(parent)
        print(f"     - {parent}: {desc}")
    
    # 4. Ancestor subgraph with ICD10CM only
    print("\n4. Ancestor subgraph (ICD10CM vocabulary only, max_depth=2):")
    if hasattr(ontology, 'get_ancestor_subgraph'):
        # For NetworkX-based ontologies
        subgraph_nx = ontology.get_ancestor_subgraph(
            code,
            vocabularies=["ICD10CM"],
            max_depth=2
        )
        print(f"   Subgraph contains {subgraph_nx.number_of_nodes()} nodes:")
        for node in sorted(subgraph_nx.nodes()):
            desc = ontology.get_description(node)
            parents = list(subgraph_nx.predecessors(node))
            print(f"     - {node}: {desc} ({len(parents)} parents)")
            if parents:
                for parent in sorted(parents):
                    print(f"         -> {parent}")
    else:
        # For dict-based ontologies (sparse matrix)
        subgraph = ontology.get_ancestor_subgraph(
            code,
            vocabularies=["ICD10CM"],
            max_depth=2
        )
        print(f"   Subgraph contains {len(subgraph)} nodes:")
        for node in sorted(subgraph.keys()):
            desc = ontology.get_description(node)
            parent_count = len(subgraph[node])
            print(f"     - {node}: {desc} ({parent_count} parents)")
            if subgraph[node]:
                for parent in sorted(subgraph[node]):
                    print(f"         -> {parent}")
    
    # 5. Test relationship type filtering (if supported)
    print("\n5. Parents filtered to 'Is a' relationships only:")
    if hasattr(ontology.get_parents, '__code__'):
        # Check if method supports relationship_types parameter
        import inspect
        sig = inspect.signature(ontology.get_parents)
        if 'relationship_types' in sig.parameters:
            parents_isa = ontology.get_parents(code, relationship_types=["Is a"])
            print(f"   Found {len(parents_isa)} parents (Is a only):")
            for parent in sorted(parents_isa):
                desc = ontology.get_description(parent)
                vocab = parent.split("/")[0] if "/" in parent else "Unknown"
                print(f"     - {parent} [{vocab}]: {desc}")
        else:
            print("   (Relationship type filtering not supported in this ontology type)")
    else:
        print("   (Relationship type filtering not supported in this ontology type)")
    
    # 5. Test with different code
    if code != "ICD10CM/C34":
        print("\n5. Testing parent hierarchy for ICD10CM/C34:")
        parents_c34 = ontology.get_parents("ICD10CM/C34", vocabularies=["ICD10CM"])
        print(f"   Found {len(parents_c34)} ICD10CM parents:")
        for parent in sorted(parents_c34):
            desc = ontology.get_description(parent)
            print(f"     - {parent}: {desc}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test filtered ontology queries")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="data/athena_omop_ontologies",
        help="Path to parquet files directory"
    )
    parser.add_argument(
        "--code",
        type=str,
        default="ICD10CM/C34.02",
        help="Code to test"
    )
    parser.add_argument(
        "--use_sparse",
        action="store_true",
        help="Use SparseGraphOntology (default: AthenaOntology)"
    )
    
    args = parser.parse_args()
    
    if args.use_sparse:
        print("Using SparseGraphOntology")
        ontology = SparseGraphOntology.load_from_parquet(
            args.parquet_path,
            graph_path=f"{args.parquet_path}/ontology_graph_sparse.pkl.gz",
            load_graph=True,
        )
    else:
        print("Using AthenaOntology")
        ontology = AthenaOntology.load_from_parquet(args.parquet_path)
    
    test_filtered_queries(ontology, args.code)


if __name__ == "__main__":
    main()
