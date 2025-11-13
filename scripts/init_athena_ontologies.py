#!/usr/bin/env python3
"""
Athena Ontologies Initialization Script

See https://athena.ohdsi.org/vocabulary/list

Notes:
You *must* use Athena's internal CPT initialization scripts. This needs to be 
run after downloading the ontologies and before this script. This script will
build a lookup trie for the ontology and save it to a Parquet file. Expected 
runtime is ~60 seconds

Usage:
    python init_athena_ontologies.py \
        --athena_path PATH \
        [--save_parquet PARQUET_PATH] \
        [--sample N] \
        [--custom_mappings CSV_PATH]
        
python scripts/init_athena_ontologies.py \
--athena_path data/athena_ontologies_snapshot.zip \
--custom_mappings data/stanford_custom_concepts.csv.gz \
--save_parquet data/athena_omop_ontologies \
--print_sample 5

"""
import argparse
import random
import time
import pandas as pd
from meds2text.ontology.athena import AthenaOntology
from meds2text.ontology import OntologyDescriptionLookupTable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize Athena OMOP ontologies and optionally save to Parquet."
    )
    parser.add_argument(
        "--athena_path",
        required=True,
        help="Path to Athena OMOP ontologies directory",
    )
    parser.add_argument(
        "--save_parquet",
        help="File path or prefix to save ontology to Parquet format",
    )
    parser.add_argument(
        "--print_sample",
        type=int,
        default=2,
        help="Number of sample description_map items to print",
    )
    parser.add_argument(
        "--custom_mappings",
        help="Path to custom mappings CSV file (optionally gzipped)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.time()
    print("Starting Athena ontology initialization...")

    # Load custom mappings if provided
    custom_mappings = {}
    if args.custom_mappings:
        try:
            print(f"Loading custom mappings from {args.custom_mappings}...")
            mappings_df = pd.read_csv(
                args.custom_mappings, compression="infer", dtype=str
            )
            for row in mappings_df.itertuples():
                key = f"{row.vocabulary_id}/{row.concept_code}"
                custom_mappings[key] = {"description": str(row.concept_name)}
            print(
                f"Loaded {len(custom_mappings)} custom mappings from {args.custom_mappings}"
            )
        except Exception as e:
            print(f"Error loading custom mappings from {args.custom_mappings}: {e}")
            raise

    # Load ontology with optional custom mappings
    print(f"Loading ontology from {args.athena_path}...")
    ontology = AthenaOntology.load_from_athena_snapshot(
        args.athena_path, code_metadata=custom_mappings or None
    )
    print(f"Loaded ontology with {len(ontology.description_map)} concepts")

    # Optionally save to Parquet
    if args.save_parquet:
        print(f"Saving ontology to Parquet format at {args.save_parquet}...")
        ontology.save_to_parquet(args.save_parquet)
        # save trie
        print("Building and saving lookup trie...")
        trie = OntologyDescriptionLookupTable.load_from_parquet(args.save_parquet)
        trie.save(args.save_parquet)
        print(f"Ontology saved to Parquet and trie at: {args.save_parquet}")

    else:
        print("No Parquet save path provided; skipping save.")

    # Print sample items from description_map
    items = list(ontology.description_map.items())
    print(f"Printing first {args.print_sample} description_map entries:")
    for key, desc in items[: args.print_sample]:
        print(f"- {key}: {desc}")

    # Print sample items from custom mappings if available
    if custom_mappings:
        mapping_items = list(custom_mappings.items())
        sample_map_count = min(args.print_sample, len(mapping_items))
        print(f"Printing {sample_map_count} sample custom mapping entries:")
        for key, meta in random.sample(mapping_items, sample_map_count):
            print(f"- {key}: {meta['description']}")

    # Report processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()
