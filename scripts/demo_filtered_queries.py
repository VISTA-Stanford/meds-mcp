#!/usr/bin/env python3
"""
Demonstration of filtered ontology queries.

Shows how to use:
1. Vocabulary filtering
2. Relationship type filtering (Is a vs Maps to)
3. Combined filters
4. Depth limiting
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology
from meds_mcp.server.tools.athena import AthenaOntology


def demo_filtering(ontology, code: str = "ICD10CM/C34.02"):
    """Demonstrate various filtering options."""
    
    print(f"\n{'='*80}")
    print(f"Filtered Query Demonstrations for: {code}")
    print(f"{'='*80}\n")
    
    # 1. All parents
    print("1️⃣  ALL PARENTS (no filters)")
    print("-" * 80)
    parents_all = ontology.get_parents(code)
    print(f"Found {len(parents_all)} parents:")
    for parent in sorted(parents_all):
        desc = ontology.get_description(parent)
        vocab = parent.split("/")[0]
        print(f"  • {parent:25} [{vocab:10}] {desc}")
    
    # 2. Vocabulary filter: ICD10CM only
    print("\n2️⃣  VOCABULARY FILTER: ICD10CM only")
    print("-" * 80)
    parents_icd10cm = ontology.get_parents(code, vocabularies=["ICD10CM"])
    print(f"Found {len(parents_icd10cm)} ICD10CM parents:")
    for parent in sorted(parents_icd10cm):
        desc = ontology.get_description(parent)
        print(f"  • {parent}: {desc}")
    
    # 3. Vocabulary filter: SNOMED only
    print("\n3️⃣  VOCABULARY FILTER: SNOMED only")
    print("-" * 80)
    parents_snomed = ontology.get_parents(code, vocabularies=["SNOMED"])
    print(f"Found {len(parents_snomed)} SNOMED parents:")
    for parent in sorted(parents_snomed):
        desc = ontology.get_description(parent)
        print(f"  • {parent}: {desc}")
    
    # 4. Relationship type filter (if supported)
    print("\n4️⃣  RELATIONSHIP TYPE FILTER: 'Is a' only")
    print("-" * 80)
    try:
        parents_isa = ontology.get_parents(code, relationship_types=["Is a"])
        print(f"Found {len(parents_isa)} parents (Is a relationships only):")
        for parent in sorted(parents_isa):
            desc = ontology.get_description(parent)
            vocab = parent.split("/")[0]
            print(f"  • {parent:25} [{vocab:10}] {desc}")
    except TypeError:
        print("  (Relationship type filtering not supported in this ontology type)")
        print("  Use LazyAthenaOntology for relationship type filtering")
    
    # 5. Combined filters: ICD10CM + Is a
    print("\n5️⃣  COMBINED FILTERS: ICD10CM vocabulary + 'Is a' relationships")
    print("-" * 80)
    try:
        parents_filtered = ontology.get_parents(
            code,
            relationship_types=["Is a"],
            vocabularies=["ICD10CM"]
        )
        print(f"Found {len(parents_filtered)} parents:")
        for parent in sorted(parents_filtered):
            desc = ontology.get_description(parent)
            print(f"  • {parent}: {desc}")
    except TypeError:
        print("  (Combined filtering not supported in this ontology type)")
    
    # 6. Ancestor subgraph with filters
    print("\n6️⃣  ANCESTOR SUBGRAPH: ICD10CM only, max_depth=2")
    print("-" * 80)
    if hasattr(ontology, 'get_ancestor_subgraph'):
        subgraph = ontology.get_ancestor_subgraph(
            code,
            vocabularies=["ICD10CM"],
            max_depth=2
        )
        
        if hasattr(subgraph, 'number_of_nodes'):
            # NetworkX graph
            print(f"Subgraph contains {subgraph.number_of_nodes()} nodes:")
            for node in sorted(subgraph.nodes()):
                desc = ontology.get_description(node) or "N/A"
                parents_list = list(subgraph.predecessors(node))
                marker = " [START]" if node == code else ""
                print(f"  • {node:25} {desc}{marker}")
                if parents_list:
                    for parent in sorted(parents_list):
                        print(f"      └─> {parent}")
        else:
            # Dict-based subgraph
            print(f"Subgraph contains {len(subgraph)} nodes:")
            for node in sorted(subgraph.keys()):
                desc = ontology.get_description(node) or "N/A"
                marker = " [START]" if node == code else ""
                print(f"  • {node:25} {desc}{marker}")
                if subgraph[node]:
                    for parent in sorted(subgraph[node]):
                        print(f"      └─> {parent}")
    
    # 7. Compare different relationship types
    print("\n7️⃣  COMPARING RELATIONSHIP TYPES")
    print("-" * 80)
    try:
        parents_isa_only = ontology.get_parents(code, relationship_types=["Is a"])
        parents_maps_to_only = ontology.get_parents(code, relationship_types=["Maps to"])
        
        print(f"'Is a' relationships: {len(parents_isa_only)}")
        for parent in sorted(parents_isa_only):
            print(f"  • {parent}")
        
        print(f"\n'Maps to' relationships: {len(parents_maps_to_only)}")
        for parent in sorted(parents_maps_to_only):
            print(f"  • {parent}")
        
        # Show overlap
        overlap = parents_isa_only & parents_maps_to_only
        if overlap:
            print(f"\nOverlap (in both): {len(overlap)}")
            for parent in sorted(overlap):
                print(f"  • {parent}")
    except TypeError:
        print("  (Relationship type filtering not supported in this ontology type)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate filtered ontology queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use sparse matrix format (fastest)
  python scripts/demo_filtered_queries.py --use_sparse --code "ICD10CM/C34.02"
  
  # Use regular ontology
  python scripts/demo_filtered_queries.py --code "ICD10CM/C34.02"
  
  # Test different code
  python scripts/demo_filtered_queries.py --code "SNOMED/73211009"
        """
    )
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
        print("Using SparseGraphOntology (sparse matrix format)")
        ontology = SparseGraphOntology.load_from_parquet(
            args.parquet_path,
            graph_path=f"{args.parquet_path}/ontology_graph_sparse.pkl.gz",
            load_graph=True,
        )
    else:
        print("Using AthenaOntology (regular format)")
        ontology = AthenaOntology.load_from_parquet(args.parquet_path)
    
    demo_filtering(ontology, args.code)


if __name__ == "__main__":
    main()
