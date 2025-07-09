#!/usr/bin/env python3
"""
Quick speed comparison between AthenaOntology and LazyAthenaOntology.
This script provides a concise performance overview and practical recommendations.
"""

import time
import tempfile
import os
import random
import statistics
from typing import List

def create_quick_test_data(tmp_dir: str, num_concepts: int = 2000):
    """Create test data for quick performance evaluation."""
    print(f"Creating test data with {num_concepts} concepts...")
    
    # Generate realistic concept data
    concept_data = ["concept_id\tvocabulary_id\tconcept_code\tconcept_name\tinvalid_reason\tstandard_concept"]
    relationship_data = ["concept_id_1\tconcept_id_2\trelationship_id"]
    ancestor_data = ["ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation"]
    
    vocabularies = ["SNOMED", "RxNorm", "ICD10CM", "LOINC", "ATC"]
    concept_codes = []
    
    for i in range(1, num_concepts + 1):
        vocab = random.choice(vocabularies)
        code = f"{random.randint(100000, 999999)}"
        name = f"{vocab} Concept {i}"
        standard = "S" if vocab in ["SNOMED", "RxNorm"] else ""
        
        concept_data.append(f"{i}\t{vocab}\t{code}\t{name}\t\t{standard}")
        concept_codes.append(f"{vocab}/{code}")
        
        # Add some relationships
        if i > 1 and random.random() < 0.3:  # 30% chance of parent relationship
            parent_id = random.randint(1, i - 1)
            ancestor_data.append(f"{parent_id}\t{i}\t1")
    
    # Write files
    with open(os.path.join(tmp_dir, "CONCEPT.csv"), "w") as f:
        f.write("\n".join(concept_data))
    
    with open(os.path.join(tmp_dir, "CONCEPT_RELATIONSHIP.csv"), "w") as f:
        f.write("\n".join(relationship_data))
    
    with open(os.path.join(tmp_dir, "CONCEPT_ANCESTOR.csv"), "w") as f:
        f.write("\n".join(ancestor_data))
    
    return concept_codes

def time_operations(ontology, test_codes: List[str], num_tests: int = 50):
    """Time basic operations for an ontology."""
    sample_codes = random.sample(test_codes, min(num_tests, len(test_codes)))
    
    # Time get_description
    start = time.perf_counter()
    for code in sample_codes:
        ontology.get_description(code)
    description_time = (time.perf_counter() - start) / len(sample_codes) * 1000
    
    # Time get_parents
    start = time.perf_counter()
    for code in sample_codes:
        ontology.get_parents(code)
    parents_time = (time.perf_counter() - start) / len(sample_codes) * 1000
    
    # Time get_children
    start = time.perf_counter()
    for code in sample_codes:
        ontology.get_children(code)
    children_time = (time.perf_counter() - start) / len(sample_codes) * 1000
    
    return {
        'description': description_time,
        'parents': parents_time,
        'children': children_time
    }

def run_quick_comparison():
    """Run a quick performance comparison."""
    print("🚀 Quick Ontology Performance Comparison")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test data
        concept_codes = create_quick_test_data(tmp_dir, 2000)
        
        from src.meds_mcp.server.tools.ontologies import AthenaOntology, LazyAthenaOntology
        
        print(f"\n📊 Loading Performance:")
        
        # Test LazyAthenaOntology loading
        start = time.perf_counter()
        lazy_ontology = LazyAthenaOntology.load_from_athena_snapshot(tmp_dir)
        lazy_load_time = time.perf_counter() - start
        print(f"  LazyAthenaOntology: {lazy_load_time:.3f}s ({len(lazy_ontology):,} concepts)")
        
        # Create equivalent regular ontology
        description_map = {code: f"Description for {code}" for code in concept_codes}
        parents_map = {code: set() for code in concept_codes}
        
        # Add some parent relationships for realism
        for code in concept_codes:
            if random.random() < 0.2:  # 20% have parents
                potential_parents = [c for c in concept_codes if c != code and c.split('/')[0] == code.split('/')[0]]
                if potential_parents:
                    parents_map[code].add(random.choice(potential_parents))
        
        start = time.perf_counter()
        regular_ontology = AthenaOntology(description_map, parents_map)
        regular_load_time = time.perf_counter() - start
        print(f"  Regular AthenaOntology: {regular_load_time:.3f}s ({len(regular_ontology):,} concepts)")
        
        load_speedup = regular_load_time / lazy_load_time if lazy_load_time > 0 else float('inf')
        print(f"  📈 Loading: LazyAthenaOntology is {load_speedup:.1f}x faster")
        
        print(f"\n⚡ Query Performance (average per operation):")
        
        # Test query performance
        lazy_times = time_operations(lazy_ontology, concept_codes, 50)
        regular_times = time_operations(regular_ontology, concept_codes, 50)
        
        operations = [
            ('description', 'Description lookup'),
            ('parents', 'Parent queries'),
            ('children', 'Child queries')
        ]
        
        for op_key, op_name in operations:
            lazy_time = lazy_times[op_key]
            regular_time = regular_times[op_key]
            speedup = lazy_time / regular_time if regular_time > 0 else float('inf')
            faster = "Regular" if speedup > 1 else "Lazy"
            
            print(f"  {op_name}:")
            print(f"    - Regular: {regular_time:.3f}ms")
            print(f"    - Lazy: {lazy_time:.3f}ms")
            print(f"    - 🏆 {faster} is {abs(speedup):.1f}x faster")
        
        print(f"\n💡 Quick Recommendations:")
        
        if load_speedup > 10:  # Lazy loading is significantly faster
            print(f"  ✅ LazyAthenaOntology advantages:")
            print(f"    - {load_speedup:.1f}x faster loading")
            print(f"    - Lower memory usage")
            print(f"    - Better for startup time")
        
        avg_query_slowdown = statistics.mean([
            lazy_times['description'] / regular_times['description'],
            lazy_times['parents'] / regular_times['parents'],
            lazy_times['children'] / regular_times['children']
        ])
        
        if avg_query_slowdown > 2:  # Lazy queries are significantly slower
            print(f"  ✅ Regular AthenaOntology advantages:")
            print(f"    - {avg_query_slowdown:.1f}x faster queries on average")
            print(f"    - Better for frequent repeated operations")
            print(f"    - More predictable performance")
        
        print(f"\n🎯 Use Case Recommendations:")
        print(f"  📱 Choose LazyAthenaOntology for:")
        print(f"    - Batch processing / ETL pipelines")
        print(f"    - Memory-constrained environments")
        print(f"    - Infrequent ontology lookups")
        print(f"    - Applications with startup time requirements")
        
        print(f"  🚀 Choose Regular AthenaOntology for:")
        print(f"    - Interactive web applications")
        print(f"    - Real-time query services")
        print(f"    - High-frequency ontology operations")
        print(f"    - When memory is not a constraint")

if __name__ == "__main__":
    try:
        run_quick_comparison()
        print(f"\n✅ Comparison completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 