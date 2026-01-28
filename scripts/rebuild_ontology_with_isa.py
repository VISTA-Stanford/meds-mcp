#!/usr/bin/env python3
"""
Rebuild ontology parquet files including "Is a" relationships.

This script rebuilds the ontology from the Athena snapshot with the corrected
code that includes "Is a" relationships (not just "Maps to" and CONCEPT_ANCESTOR).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.athena import AthenaOntology


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild ontology parquet files with 'Is a' relationships included"
    )
    parser.add_argument(
        "--athena_path",
        type=str,
        required=True,
        help="Path to Athena snapshot directory or zip archive"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/athena_omop_ontologies",
        help="Path to save rebuilt parquet files (default: data/athena_omop_ontologies)"
    )
    parser.add_argument(
        "--custom_mappings",
        type=str,
        help="Optional path to custom mappings CSV file"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Rebuilding ontology with 'Is a' relationships")
    print("="*80)
    print(f"Athena snapshot: {args.athena_path}")
    print(f"Output path: {args.output_path}")
    print()
    
    # Load custom mappings if provided
    code_metadata = {}
    if args.custom_mappings:
        import pandas as pd
        print(f"Loading custom mappings from {args.custom_mappings}...")
        mappings_df = pd.read_csv(args.custom_mappings, compression="infer", dtype=str)
        for row in mappings_df.itertuples():
            key = f"{row.vocabulary_id}/{row.concept_code}"
            code_metadata[key] = {"description": str(row.concept_name)}
        print(f"Loaded {len(code_metadata)} custom mappings")
    
    # Load ontology with corrected code (includes "Is a" relationships)
    print(f"\nLoading ontology from {args.athena_path}...")
    print("(This includes 'Is a' relationships now)")
    ontology = AthenaOntology.load_from_athena_snapshot(
        args.athena_path,
        code_metadata=code_metadata if code_metadata else None,
        ignore_invalid=True
    )
    
    print(f"Loaded ontology with {len(ontology.description_map):,} concepts")
    
    # Count relationships
    total_parents = sum(len(parents) for parents in ontology.parents_map.values())
    print(f"Total parent relationships: {total_parents:,}")
    
    # Test with C34.02
    test_code = "ICD10CM/C34.02"
    if test_code in ontology.description_map:
        parents = ontology.get_parents(test_code)
        print(f"\nTest: {test_code}")
        print(f"  Description: {ontology.get_description(test_code)}")
        print(f"  Parents found: {len(parents)}")
        for parent in sorted(parents):
            print(f"    - {parent}: {ontology.get_description(parent)}")
    else:
        print(f"\nWarning: Test code {test_code} not found in ontology")
    
    # Save to parquet
    print(f"\nSaving to parquet format at {args.output_path}...")
    ontology.save_to_parquet(args.output_path)
    print(f"✅ Ontology saved successfully!")
    print(f"\nYou can now use:")
    print(f"  python scripts/benchmark_ontology_loading.py --code '{test_code}'")


if __name__ == "__main__":
    main()
