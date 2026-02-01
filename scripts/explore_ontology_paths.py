#!/usr/bin/env python3
"""
Explore ontology paths from leaf nodes to ancestors.

Samples leaf nodes (nodes with no children) for target vocabularies and
generates paths to their ancestors.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Set, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology


def find_leaf_nodes(ontology: SparseGraphOntology, vocabularies: List[str], sample_size: int = 10) -> List[str]:
    """Find leaf nodes (nodes with no children) in specified vocabularies."""
    import polars as pl
    
    print(f"Finding leaf nodes in vocabularies: {vocabularies}...")
    
    # Query metadata to get codes in vocabularies
    vocab_filters = [pl.col("code").str.starts_with(vocab) for vocab in vocabularies]
    result = (
        ontology.concepts_df
        .filter(pl.any_horizontal(vocab_filters))
        .select("code")
        .collect()
    )
    
    all_codes = result["code"].to_list()
    print(f"Found {len(all_codes)} codes in vocabularies")
    
    # Sample and check if they're leaves
    import random
    random.seed(42)  # Reproducible
    sampled = random.sample(all_codes, min(sample_size * 3, len(all_codes)))
    
    leaf_nodes = []
    for code in sampled:
        children = ontology.get_children(code)
        if not children:
            leaf_nodes.append(code)
            if len(leaf_nodes) >= sample_size:
                break
    
    print(f"Found {len(leaf_nodes)} leaf nodes")
    return leaf_nodes


def explore_paths(
    ontology: SparseGraphOntology,
    code: str,
    longest_only: bool = True,
    same_vocabulary: bool = True,
    max_depth: Optional[int] = None,
):
    """Explore paths from a code to its ancestors."""
    vocab = code.split("/")[0] if "/" in code else None
    
    vocabularies = [vocab] if same_vocabulary and vocab else None
    
    print(f"\n{'='*80}")
    print(f"Exploring paths from: {code}")
    print(f"{'='*80}")
    
    # Get parents
    parents = ontology.get_parents(code, vocabularies=vocabularies)
    print(f"\nImmediate parents ({len(parents)}):")
    for parent in sorted(parents):
        desc = ontology.get_description(parent)
        print(f"  • {parent}: {desc}")
    
    if not parents:
        print("  (No parents found)")
        return
    
    # Get subgraph
    if longest_only:
        print(f"\nSubgraph (longest paths only, same_vocab={same_vocabulary}):")
        subgraph = ontology.get_ancestor_subgraph_filtered(
            code,
            vocabularies=vocabularies,
            remove_redundant_edges=True,
            max_depth=max_depth,
        )
    else:
        print(f"\nSubgraph (all paths, same_vocab={same_vocabulary}):")
        subgraph = ontology.get_ancestor_subgraph(
            code,
            vocabularies=vocabularies,
            max_depth=max_depth,
        )
    
    print(f"Subgraph contains {len(subgraph)} nodes:")
    for node in sorted(subgraph.keys()):
        desc = ontology.get_description(node) or "N/A"
        marker = " [START]" if node == code else ""
        print(f"  • {node:30} {desc}{marker}")
        if subgraph[node]:
            for parent in sorted(subgraph[node]):
                print(f"      └─> {parent}")
    
    # For each parent, show all paths
    if not longest_only:
        print(f"\nAll paths to ancestors:")
        for parent in sorted(parents):
            if same_vocabulary and vocab:
                # Check if parent is in same vocabulary
                parent_vocab = parent.split("/")[0] if "/" in parent else None
                if parent_vocab != vocab:
                    continue
            
            paths = ontology.get_all_paths_to_ancestor(
                code,
                parent,
                vocabularies=vocabularies,
                max_depth=max_depth,
            )
            
            if paths:
                print(f"\n  Paths to {parent}:")
                for i, path in enumerate(paths, 1):
                    path_str = " -> ".join(path)
                    print(f"    {i}. [{len(path)-1} steps] {path_str}")
            else:
                print(f"\n  No paths found to {parent}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore ontology paths from leaf nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 5 leaf nodes from ICD10CM and show longest paths
  python scripts/explore_ontology_paths.py --vocabularies ICD10CM --sample 5 --longest
  
  # Show all paths (not just longest) for SNOMED leaf nodes
  python scripts/explore_ontology_paths.py --vocabularies SNOMED --sample 3 --all-paths
  
  # Explore specific code with all paths, allow cross-vocabulary
  python scripts/explore_ontology_paths.py --code "ICD10CM/C34.02" --all-paths --cross-vocab
        """
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
        help="Path to sparse graph file"
    )
    parser.add_argument(
        "--vocabularies",
        type=str,
        nargs="+",
        help="Target vocabularies to sample from (e.g., ICD10CM SNOMED)"
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Specific code to explore (overrides --vocabularies)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of leaf nodes to sample"
    )
    parser.add_argument(
        "--longest",
        action="store_true",
        help="Show only longest paths (remove redundant edges)"
    )
    parser.add_argument(
        "--all-paths",
        action="store_true",
        help="Show all paths (not just longest)"
    )
    parser.add_argument(
        "--cross-vocab",
        action="store_true",
        help="Allow cross-vocabulary paths (default: same vocabulary only)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth to traverse"
    )
    
    args = parser.parse_args()
    
    # Load ontology
    print("Loading ontology...")
    ontology = SparseGraphOntology.load_from_parquet(
        args.parquet_path,
        graph_path=args.graph_path,
        load_graph=True,
    )
    print("Ontology loaded.\n")
    
    # Determine longest_only
    longest_only = not args.all_paths
    
    # Explore specific code or sample leaf nodes
    if args.code:
        explore_paths(
            ontology,
            args.code,
            longest_only=longest_only,
            same_vocabulary=not args.cross_vocab,
            max_depth=args.max_depth,
        )
    elif args.vocabularies:
        leaf_nodes = find_leaf_nodes(ontology, args.vocabularies, args.sample)
        
        for code in leaf_nodes:
            explore_paths(
                ontology,
                code,
                longest_only=longest_only,
                same_vocabulary=not args.cross_vocab,
                max_depth=args.max_depth,
            )
    else:
        parser.error("Must specify either --code or --vocabularies")


if __name__ == "__main__":
    main()
