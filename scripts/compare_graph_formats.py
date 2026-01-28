#!/usr/bin/env python3
"""
Compare NetworkX vs Sparse Matrix graph storage formats.

Tests:
1. File size
2. Load time
3. Query performance
"""

import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.fast_ontology import FastHybridOntology
from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology


def compare_formats(parquet_path: str, test_code: str = "ICD10CM/C34.02"):
    """Compare NetworkX vs Sparse Matrix formats."""
    
    print("="*80)
    print("Comparing Graph Storage Formats")
    print("="*80)
    print(f"Test code: {test_code}\n")
    
    nx_graph_path = os.path.join(parquet_path, "ontology_graph.pkl.gz")
    sparse_graph_path = os.path.join(parquet_path, "ontology_graph_sparse.pkl.gz")
    
    # Test NetworkX format
    print("1. NetworkX Graph Format (current)")
    print("-" * 80)
    
    if os.path.exists(nx_graph_path):
        file_size_mb = os.path.getsize(nx_graph_path) / 1024 / 1024
        print(f"File size: {file_size_mb:.2f} MB")
        
        start = time.time()
        ontology_nx = FastHybridOntology.load_from_parquet(
            parquet_path,
            graph_path=nx_graph_path,
            load_graph=True,
        )
        load_time_nx = time.time() - start
        
        print(f"Load time: {load_time_nx:.2f}s")
        
        # Test query
        query_start = time.time()
        parents_nx = ontology_nx.get_parents(test_code)
        query_time_nx = time.time() - query_start
        
        print(f"Query time (get_parents): {query_time_nx*1000:.3f}ms")
        print(f"Parents found: {len(parents_nx)}")
    else:
        print("NetworkX graph not found - skipping")
        load_time_nx = None
        query_time_nx = None
    
    print("\n2. Sparse Matrix Format (optimized)")
    print("-" * 80)
    
    start = time.time()
    ontology_sparse = SparseGraphOntology.load_from_parquet(
        parquet_path,
        graph_path=sparse_graph_path,
        load_graph=True,
    )
    load_time_sparse = time.time() - start
    
    if os.path.exists(sparse_graph_path):
        file_size_mb = os.path.getsize(sparse_graph_path) / 1024 / 1024
        print(f"File size: {file_size_mb:.2f} MB")
    else:
        print("Sparse graph not found - will be created")
    
    print(f"Load time: {load_time_sparse:.2f}s")
    
    # Test query
    query_start = time.time()
    parents_sparse = ontology_sparse.get_parents(test_code)
    query_time_sparse = time.time() - query_start
    
    print(f"Query time (get_parents): {query_time_sparse*1000:.3f}ms")
    print(f"Parents found: {len(parents_sparse)}")
    if parents_sparse:
        print("Parent codes:")
        for parent in sorted(parents_sparse):
            parent_desc = ontology_sparse.get_description(parent)
            print(f"  - {parent}: {parent_desc}")
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison")
    print("="*80)
    
    if load_time_nx:
        speedup = load_time_nx / load_time_sparse
        print(f"\nLoad Time Speedup: {speedup:.1f}x faster")
        print(f"  NetworkX: {load_time_nx:.2f}s")
        print(f"  Sparse:   {load_time_sparse:.2f}s")
    
    if query_time_nx:
        query_speedup = query_time_nx / query_time_sparse if query_time_sparse > 0 else float('inf')
        print(f"\nQuery Time Speedup: {query_speedup:.1f}x faster")
        print(f"  NetworkX: {query_time_nx*1000:.3f}ms")
        print(f"  Sparse:   {query_time_sparse*1000:.3f}ms")
    
    if os.path.exists(nx_graph_path) and os.path.exists(sparse_graph_path):
        size_nx = os.path.getsize(nx_graph_path)
        size_sparse = os.path.getsize(sparse_graph_path)
        size_ratio = size_nx / size_sparse
        print(f"\nFile Size Ratio: {size_ratio:.2f}x")
        print(f"  NetworkX: {size_nx / 1024 / 1024:.2f} MB")
        print(f"  Sparse:   {size_sparse / 1024 / 1024:.2f} MB")
    
    # Verify results match
    if load_time_nx and parents_nx == parents_sparse:
        print("\n✅ Results match between formats")
    elif load_time_nx:
        print("\n⚠️  Results differ:")
        print(f"  NetworkX parents: {sorted(parents_nx)}")
        print(f"  Sparse parents:   {sorted(parents_sparse)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare graph storage formats")
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
    
    args = parser.parse_args()
    
    compare_formats(args.parquet_path, args.code)


if __name__ == "__main__":
    main()
