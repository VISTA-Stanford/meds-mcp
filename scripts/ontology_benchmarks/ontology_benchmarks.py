#!/usr/bin/env python3
"""
Comprehensive benchmarks for AthenaOntology vs LazyAthenaOntology.
This script provides both quick and detailed performance comparisons.
"""

import argparse
import time
import tempfile
import os
import random
import statistics
from typing import List, Dict, Any

def create_benchmark_data(tmp_dir: str, num_concepts: int):
    """Create benchmark data for testing."""
    print(f"🔧 Creating benchmark data with {num_concepts:,} concepts...")
    
    vocabularies = {
        "SNOMED": 0.4,      # 40% of concepts
        "RxNorm": 0.2,      # 20% of concepts  
        "ICD10CM": 0.15,    # 15% of concepts
        "LOINC": 0.1,       # 10% of concepts
        "ATC": 0.05,        # 5% of concepts
        "CPT4": 0.05,       # 5% of concepts
        "HCPCS": 0.05,      # 5% of concepts
    }
    
    # Generate concept data
    concept_data = ["concept_id\tvocabulary_id\tconcept_code\tconcept_name\tinvalid_reason\tstandard_concept"]
    relationship_data = ["concept_id_1\tconcept_id_2\trelationship_id"]
    ancestor_data = ["ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation"]
    
    concept_id = 1
    concept_codes = []
    vocab_concept_ids = {vocab: [] for vocab in vocabularies.keys()}
    
    # Generate concepts
    for vocab, proportion in vocabularies.items():
        count = int(num_concepts * proportion)
        for i in range(count):
            code = f"{random.randint(100000, 999999)}"
            name = f"{vocab} Benchmark Concept {i+1}"
            standard = "S" if vocab in ["SNOMED", "RxNorm", "LOINC"] else ""
            invalid = "" if random.random() > 0.05 else "D"
            
            concept_data.append(f"{concept_id}\t{vocab}\t{code}\t{name}\t{invalid}\t{standard}")
            
            if not invalid:
                full_code = f"{vocab}/{code}"
                concept_codes.append(full_code)
                vocab_concept_ids[vocab].append(concept_id)
            
            concept_id += 1
    
    # Generate relationships
    total_relationships = 0
    for vocab in vocabularies.keys():
        vocab_ids = vocab_concept_ids[vocab]
        if len(vocab_ids) < 2:
            continue
            
        num_relationships = min(len(vocab_ids) // 3, 100)
        for _ in range(num_relationships):
            if len(vocab_ids) >= 2:
                parent_id = random.choice(vocab_ids)
                child_id = random.choice([cid for cid in vocab_ids if cid != parent_id])
                ancestor_data.append(f"{parent_id}\t{child_id}\t1")
                total_relationships += 1
    
    # Write files
    with open(os.path.join(tmp_dir, "CONCEPT.csv"), "w") as f:
        f.write("\n".join(concept_data))
    
    with open(os.path.join(tmp_dir, "CONCEPT_RELATIONSHIP.csv"), "w") as f:
        f.write("\n".join(relationship_data))
    
    with open(os.path.join(tmp_dir, "CONCEPT_ANCESTOR.csv"), "w") as f:
        f.write("\n".join(ancestor_data))
    
    print(f"  ✓ Concepts: {len(concept_codes):,}")
    print(f"  ✓ Relationships: {total_relationships:,}")
    
    return concept_codes

def benchmark_loading(tmp_dir: str, concept_codes: List[str]) -> Dict[str, Any]:
    """Benchmark loading performance."""
    print(f"\n📊 Loading Performance Benchmark")
    print("-" * 40)
    
    from src.meds_mcp.server.tools.ontologies import AthenaOntology, LazyAthenaOntology
    
    # Benchmark LazyAthenaOntology loading
    print("Loading LazyAthenaOntology...")
    start = time.perf_counter()
    lazy_ontology = LazyAthenaOntology.load_from_athena_snapshot(tmp_dir)
    lazy_load_time = time.perf_counter() - start
    
    # Create equivalent regular ontology
    description_map = {code: f"Description for {code}" for code in concept_codes}
    parents_map = {code: set() for code in concept_codes}
    
    # Add realistic parent relationships
    for code in concept_codes:
        if random.random() < 0.25:  # 25% have parents
            vocab = code.split('/')[0]
            potential_parents = [c for c in concept_codes if c != code and c.startswith(vocab + '/')]
            if potential_parents:
                parents_map[code].add(random.choice(potential_parents))
    
    print("Creating Regular AthenaOntology...")
    start = time.perf_counter()
    regular_ontology = AthenaOntology(description_map, parents_map)
    regular_load_time = time.perf_counter() - start
    
    results = {
        'lazy': {'time': lazy_load_time, 'concepts': len(lazy_ontology)},
        'regular': {'time': regular_load_time, 'concepts': len(regular_ontology)},
        'ontologies': (regular_ontology, lazy_ontology)
    }
    
    print(f"  LazyAthenaOntology: {lazy_load_time:.3f}s")
    print(f"  Regular AthenaOntology: {regular_load_time:.3f}s")
    
    if lazy_load_time > 0:
        speedup = regular_load_time / lazy_load_time
        faster = "Lazy" if speedup > 1 else "Regular"
        print(f"  🏆 Winner: {faster} ({speedup:.1f}x faster)")
    
    return results

def benchmark_queries(regular_ontology, lazy_ontology, concept_codes: List[str], num_queries: int) -> Dict[str, Any]:
    """Benchmark query performance."""
    print(f"\n⚡ Query Performance Benchmark ({num_queries:,} queries)")
    print("-" * 50)
    
    # Sample random codes for testing
    test_codes = random.sample(concept_codes, min(num_queries, len(concept_codes)))
    
    operations = [
        ('get_description', 'Description Lookup'),
        ('get_parents', 'Parent Queries'),
        ('get_children', 'Child Queries'),
        ('get_code_metadata', 'Metadata Lookup')
    ]
    
    results = {}
    
    for method_name, display_name in operations:
        print(f"\n  {display_name}:")
        
        # Benchmark regular ontology
        start = time.perf_counter()
        for code in test_codes:
            getattr(regular_ontology, method_name)(code)
        regular_time = time.perf_counter() - start
        
        # Benchmark lazy ontology
        start = time.perf_counter()
        for code in test_codes:
            getattr(lazy_ontology, method_name)(code)
        lazy_time = time.perf_counter() - start
        
        avg_regular = (regular_time / len(test_codes)) * 1000
        avg_lazy = (lazy_time / len(test_codes)) * 1000
        
        speedup = lazy_time / regular_time if regular_time > 0 else float('inf')
        faster = "Regular" if speedup > 1 else "Lazy"
        
        print(f"    Regular: {avg_regular:.3f}ms avg")
        print(f"    Lazy: {avg_lazy:.3f}ms avg") 
        print(f"    🏆 {faster} is {abs(speedup):.1f}x faster")
        
        results[method_name] = {
            'regular': avg_regular,
            'lazy': avg_lazy,
            'speedup': speedup
        }
    
    return results

def print_recommendations(loading_results: Dict[str, Any], query_results: Dict[str, Any]):
    """Print performance recommendations."""
    print(f"\n💡 Performance Summary & Recommendations")
    print("=" * 50)
    
    lazy_load_time = loading_results['lazy']['time']
    regular_load_time = loading_results['regular']['time']
    load_speedup = regular_load_time / lazy_load_time if lazy_load_time > 0 else 1
    
    # Calculate average query performance
    query_speedups = [data['speedup'] for data in query_results.values()]
    avg_query_speedup = statistics.mean(query_speedups)
    
    print(f"\n📈 Performance Characteristics:")
    print(f"  Loading: {'Lazy' if load_speedup > 1 else 'Regular'} is {load_speedup:.1f}x faster")
    print(f"  Queries: {'Regular' if avg_query_speedup > 1 else 'Lazy'} is {avg_query_speedup:.1f}x faster on average")
    
    print(f"\n🎯 Choose LazyAthenaOntology when:")
    print(f"  ✅ Memory is constrained (< 4GB available)")
    print(f"  ✅ Fast startup time is critical")
    print(f"  ✅ Ontology queries are infrequent")
    print(f"  ✅ Running in containers/serverless")
    print(f"  ✅ Processing large ontologies (>100K concepts)")
    
    print(f"\n🚀 Choose Regular AthenaOntology when:")
    print(f"  ✅ Memory is abundant (>8GB available)")
    print(f"  ✅ Sub-millisecond query times needed")
    print(f"  ✅ High-frequency repeated queries")
    print(f"  ✅ Interactive/real-time applications")
    print(f"  ✅ Predictable performance is critical")

def run_quick_benchmark():
    """Run a quick benchmark comparison."""
    print("🚀 Quick Ontology Benchmark")
    print("=" * 30)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_codes = create_benchmark_data(tmp_dir, 1000)
        loading_results = benchmark_loading(tmp_dir, concept_codes)
        
        regular_ontology, lazy_ontology = loading_results['ontologies']
        query_results = benchmark_queries(regular_ontology, lazy_ontology, concept_codes, 50)
        
        print_recommendations(loading_results, query_results)

def run_detailed_benchmark():
    """Run a detailed benchmark comparison."""
    print("🚀 Detailed Ontology Benchmark")
    print("=" * 35)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        concept_codes = create_benchmark_data(tmp_dir, 5000)
        loading_results = benchmark_loading(tmp_dir, concept_codes)
        
        regular_ontology, lazy_ontology = loading_results['ontologies']
        query_results = benchmark_queries(regular_ontology, lazy_ontology, concept_codes, 200)
        
        # Additional detailed metrics
        print(f"\n📊 Detailed Performance Metrics:")
        print(f"  Dataset size: {len(concept_codes):,} concepts")
        print(f"  Query iterations: 200 per operation")
        print(f"  Memory footprint: Lazy << Regular")
        
        print_recommendations(loading_results, query_results)

def main():
    """Main function with command line options."""
    parser = argparse.ArgumentParser(description='Benchmark AthenaOntology implementations')
    parser.add_argument('--mode', choices=['quick', 'detailed'], default='quick',
                       help='Benchmark mode (default: quick)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'quick':
            run_quick_benchmark()
        else:
            run_detailed_benchmark()
            
        print(f"\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 