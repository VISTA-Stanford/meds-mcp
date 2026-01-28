#!/usr/bin/env python3
"""
Benchmark different ontology loading approaches.

Compares:
1. Current AthenaOntology.load_from_parquet() - loads everything into dicts
2. FastHybridOntology - uses Polars + pre-computed graph
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.athena import AthenaOntology
from meds_mcp.server.tools.fast_ontology import FastHybridOntology


def benchmark_current_approach(parquet_path: str, test_code: str):
    """Benchmark current AthenaOntology approach."""
    print("\n" + "="*80)
    print("Benchmarking: Current AthenaOntology.load_from_parquet()")
    print(f"Test code: {test_code}")
    print("="*80)
    
    start = time.time()
    ontology = AthenaOntology.load_from_parquet(parquet_path)
    load_time = time.time() - start
    
    print(f"Load time: {load_time:.2f}s")
    print(f"Concepts loaded: {len(ontology.description_map):,}")
    
    # Check if code exists
    description = ontology.get_description(test_code)
    if description:
        print(f"Code found: {test_code} -> {description}")
    else:
        print(f"Warning: Code not found: {test_code}")
    
    # Test query performance
    query_start = time.time()
    parents = ontology.get_parents(test_code)
    query_time = time.time() - query_start
    
    print(f"\nQuery time (get_parents): {query_time*1000:.2f}ms")
    print(f"Parents found: {len(parents)}")
    if parents:
        print("Parent codes:")
        for parent in sorted(parents):
            parent_desc = ontology.get_description(parent)
            print(f"  - {parent}: {parent_desc}")
    
    # Test graph traversal
    vocab = test_code.split("/")[0] if "/" in test_code else None
    graph_start = time.time()
    G = ontology.get_ancestor_subgraph(test_code, vocabularies=[vocab] if vocab else None)
    graph_time = time.time() - graph_start
    
    print(f"\nGraph traversal time: {graph_time*1000:.2f}ms")
    print(f"Nodes in subgraph: {G.number_of_nodes()}")
    if G.number_of_nodes() > 0:
        print("Subgraph nodes:")
        for node in sorted(G.nodes())[:10]:  # Show first 10
            node_desc = ontology.get_description(node) or "N/A"
            print(f"  - {node}: {node_desc}")
        if G.number_of_nodes() > 10:
            print(f"  ... and {G.number_of_nodes() - 10} more nodes")
    
    return {
        "load_time": load_time,
        "query_time": query_time,
        "graph_time": graph_time,
    }


def benchmark_fast_approach(parquet_path: str, test_code: str, graph_path: str = None):
    """Benchmark FastHybridOntology approach."""
    print("\n" + "="*80)
    print("Benchmarking: FastHybridOntology (Polars + Graph)")
    print(f"Test code: {test_code}")
    print("="*80)
    
    # First load (without graph)
    start = time.time()
    ontology = FastHybridOntology.load_from_parquet(
        parquet_path,
        graph_path=graph_path,
        load_graph=False,  # Don't load graph yet
    )
    load_time = time.time() - start
    
    print(f"Initial load time (metadata only): {load_time:.2f}s")
    print(f"Concepts available: {len(ontology):,}")
    
    # Check if code exists
    description = ontology.get_description(test_code)
    if description:
        print(f"Code found: {test_code} -> {description}")
    else:
        print(f"Warning: Code not found: {test_code}")
    
    # Load graph (one-time cost)
    if graph_path and Path(graph_path).exists():
        graph_start = time.time()
        _ = ontology.graph  # Trigger lazy loading
        graph_load_time = time.time() - graph_start
        print(f"Graph load time (from disk): {graph_load_time:.2f}s")
    else:
        graph_start = time.time()
        ontology = FastHybridOntology.load_from_parquet(
            parquet_path,
            graph_path=graph_path,
            load_graph=True,  # Build graph
        )
        graph_load_time = time.time() - graph_start
        print(f"Graph build time (one-time): {graph_load_time:.2f}s")
    
    # Test query performance
    query_start = time.time()
    parents = ontology.get_parents(test_code)
    query_time = time.time() - query_start
    
    print(f"\nQuery time (get_parents): {query_time*1000:.2f}ms")
    print(f"Parents found: {len(parents)}")
    if parents:
        print("Parent codes:")
        for parent in sorted(parents):
            parent_desc = ontology.get_description(parent)
            print(f"  - {parent}: {parent_desc}")
    
    # Test graph traversal
    vocab = test_code.split("/")[0] if "/" in test_code else None
    graph_start = time.time()
    G = ontology.get_ancestor_subgraph(test_code, vocabularies=[vocab] if vocab else None)
    graph_time = time.time() - graph_start
    
    print(f"\nGraph traversal time: {graph_time*1000:.2f}ms")
    print(f"Nodes in subgraph: {G.number_of_nodes()}")
    if G.number_of_nodes() > 0:
        print("Subgraph nodes:")
        for node in sorted(G.nodes())[:10]:  # Show first 10
            node_desc = ontology.get_description(node) or "N/A"
            print(f"  - {node}: {node_desc}")
        if G.number_of_nodes() > 10:
            print(f"  ... and {G.number_of_nodes() - 10} more nodes")
    
    return {
        "load_time": load_time,
        "graph_load_time": graph_load_time,
        "query_time": query_time,
        "graph_time": graph_time,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark ontology loading approaches")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="data/athena_omop_ontologies",
        help="Path to parquet files directory"
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="data/athena_omop_ontologies/ontology_graph.pkl.gz",
        help="Path to save/load graph file"
    )
    parser.add_argument(
        "--skip_current",
        action="store_true",
        help="Skip benchmarking current approach (faster)"
    )
    parser.add_argument(
        "--code",
        type=str,
        default="ICD10CM/C34.02",
        help="Code to test (default: ICD10CM/C34.02)"
    )
    
    args = parser.parse_args()
    
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        print(f"Error: Parquet path does not exist: {parquet_path}")
        sys.exit(1)
    
    results = {}
    
    if not args.skip_current:
        try:
            results["current"] = benchmark_current_approach(str(parquet_path), args.code)
        except Exception as e:
            print(f"Error benchmarking current approach: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        results["fast"] = benchmark_fast_approach(str(parquet_path), args.code, args.graph_path)
    except Exception as e:
        print(f"Error benchmarking fast approach: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if "current" in results and "fast" in results:
        current = results["current"]
        fast = results["fast"]
        
        print(f"\nLoad Time:")
        print(f"  Current: {current['load_time']:.2f}s")
        print(f"  Fast (metadata): {fast['load_time']:.2f}s")
        print(f"  Fast (with graph): {fast['load_time'] + fast.get('graph_load_time', 0):.2f}s")
        print(f"  Speedup (metadata only): {current['load_time'] / fast['load_time']:.1f}x")
        
        print(f"\nQuery Time (get_parents):")
        print(f"  Current: {current['query_time']*1000:.2f}ms")
        print(f"  Fast: {fast['query_time']*1000:.2f}ms")
        if fast['query_time'] > 0:
            print(f"  Speedup: {current['query_time'] / fast['query_time']:.1f}x")
        
        print(f"\nGraph Traversal Time:")
        print(f"  Current: {current['graph_time']*1000:.2f}ms")
        print(f"  Fast: {fast['graph_time']*1000:.2f}ms")
        if fast['graph_time'] > 0:
            print(f"  Speedup: {current['graph_time'] / fast['graph_time']:.1f}x")


if __name__ == "__main__":
    main()
