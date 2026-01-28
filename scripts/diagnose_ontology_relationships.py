#!/usr/bin/env python3
"""
Diagnostic script to investigate missing ontology relationships.

Checks what relationships exist in the raw Athena snapshot for a given code.
"""

import sys
import zipfile
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.ontologies import AthenaFileReader


def find_concept_id(code: str, concepts_df: pl.LazyFrame) -> int:
    """Find concept_id for a given code."""
    vocab, concept_code = code.split("/", 1)
    
    result = (
        concepts_df
        .filter(
            (pl.col("vocabulary_id") == vocab)
            & (pl.col("concept_code") == concept_code)
        )
        .select("concept_id")
        .collect()
    )
    
    if result.height > 0:
        return result.row(0)[0]
    return None


def diagnose_code(athena_path: str, code: str):
    """Diagnose relationships for a specific code."""
    print(f"\n{'='*80}")
    print(f"Diagnosing relationships for: {code}")
    print(f"{'='*80}\n")
    
    with AthenaFileReader(athena_path) as reader:
        concepts_df = reader.read_csv("CONCEPT.csv")
        
        # Find concept_id
        concept_id = find_concept_id(code, concepts_df)
        if concept_id is None:
            print(f"ERROR: Code {code} not found in CONCEPT.csv")
            return
        
        print(f"Found concept_id: {concept_id}")
        
        # Get concept details
        concept_details = (
            concepts_df
            .filter(pl.col("concept_id").cast(pl.Int64) == concept_id)
            .select(["concept_id", "concept_code", "concept_name", "vocabulary_id", "standard_concept", "invalid_reason"])
            .collect()
        )
        
        if concept_details.height > 0:
            row = concept_details.row(0)
            print(f"Concept details:")
            print(f"  concept_id: {row[0]}")
            print(f"  concept_code: {row[1]}")
            print(f"  concept_name: {row[2]}")
            print(f"  vocabulary_id: {row[3]}")
            print(f"  standard_concept: {row[4]}")
            print(f"  invalid_reason: {row[5]}")
        
        # Check CONCEPT_RELATIONSHIP for "Is a" relationships
        print(f"\n{'='*80}")
        print("CONCEPT_RELATIONSHIP.csv - 'Is a' relationships:")
        print(f"{'='*80}")
        
        relationships_df = reader.read_csv("CONCEPT_RELATIONSHIP.csv")
        
        # Find where this concept is concept_id_1 (child)
        is_a_as_child = (
            relationships_df
            .filter(
                (pl.col("concept_id_1").cast(pl.Int64) == concept_id)
                & (pl.col("relationship_id") == "Is a")
            )
            .select(["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"])
            .collect()
        )
        
        print(f"\n'{code}' as CHILD (concept_id_1) - 'Is a' relationships:")
        if is_a_as_child.height > 0:
            for row in is_a_as_child.rows():
                parent_id = row[1]
                invalid = row[3]
                # Find parent code
                parent_code = find_code_by_concept_id(parent_id, concepts_df)
                print(f"  -> concept_id_2={parent_id} ({parent_code}) invalid_reason={invalid}")
        else:
            print("  (none found)")
        
        # Find where this concept is concept_id_2 (parent)
        is_a_as_parent = (
            relationships_df
            .filter(
                (pl.col("concept_id_2").cast(pl.Int64) == concept_id)
                & (pl.col("relationship_id") == "Is a")
            )
            .select(["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"])
            .collect()
        )
        
        print(f"\n'{code}' as PARENT (concept_id_2) - 'Is a' relationships:")
        if is_a_as_parent.height > 0:
            for row in is_a_as_parent.rows():
                child_id = row[0]
                invalid = row[3]
                child_code = find_code_by_concept_id(child_id, concepts_df)
                print(f"  <- concept_id_1={child_id} ({child_code}) invalid_reason={invalid}")
        else:
            print("  (none found)")
        
        # Check CONCEPT_ANCESTOR
        print(f"\n{'='*80}")
        print("CONCEPT_ANCESTOR.csv - ancestor relationships:")
        print(f"{'='*80}")
        
        ancestors_df = reader.read_csv("CONCEPT_ANCESTOR.csv")
        
        # Find ancestors (where this is descendant)
        ancestors = (
            ancestors_df
            .filter(pl.col("descendant_concept_id").cast(pl.Int64) == concept_id)
            .select(["ancestor_concept_id", "min_levels_of_separation"])
            .collect()
        )
        
        print(f"\n'{code}' as DESCENDANT - ancestors:")
        if ancestors.height > 0:
            for row in ancestors.rows():
                ancestor_id = row[0]
                levels = row[1]
                ancestor_code = find_code_by_concept_id(ancestor_id, concepts_df)
                print(f"  -> ancestor_concept_id={ancestor_id} ({ancestor_code}) levels={levels}")
        else:
            print("  (none found)")
        
        # Find descendants (where this is ancestor)
        descendants = (
            ancestors_df
            .filter(pl.col("ancestor_concept_id").cast(pl.Int64) == concept_id)
            .select(["descendant_concept_id", "min_levels_of_separation"])
            .collect()
        )
        
        print(f"\n'{code}' as ANCESTOR - descendants:")
        if descendants.height > 0:
            print(f"  Found {descendants.height} descendants (showing first 10):")
            for row in descendants.head(10).rows():
                descendant_id = row[0]
                levels = row[1]
                descendant_code = find_code_by_concept_id(descendant_id, concepts_df)
                print(f"  <- descendant_concept_id={descendant_id} ({descendant_code}) levels={levels}")
            if descendants.height > 10:
                print(f"  ... and {descendants.height - 10} more")
        else:
            print("  (none found)")
        
        # Check "Maps to" relationships
        print(f"\n{'='*80}")
        print("CONCEPT_RELATIONSHIP.csv - 'Maps to' relationships:")
        print(f"{'='*80}")
        
        maps_to = (
            relationships_df
            .filter(
                (pl.col("concept_id_1").cast(pl.Int64) == concept_id)
                & (pl.col("relationship_id") == "Maps to")
            )
            .select(["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"])
            .collect()
        )
        
        print(f"\n'{code}' - 'Maps to' relationships:")
        if maps_to.height > 0:
            for row in maps_to.rows():
                target_id = row[1]
                invalid = row[3]
                target_code = find_code_by_concept_id(target_id, concepts_df)
                print(f"  -> concept_id_2={target_id} ({target_code}) invalid_reason={invalid}")
        else:
            print("  (none found)")


def find_code_by_concept_id(concept_id: int, concepts_df: pl.LazyFrame) -> str:
    """Find code string for a concept_id."""
    result = (
        concepts_df
        .filter(pl.col("concept_id").cast(pl.Int64) == concept_id)
        .select(["vocabulary_id", "concept_code"])
        .collect()
    )
    
    if result.height > 0:
        vocab, code = result.row(0)
        return f"{vocab}/{code}"
    return f"UNKNOWN/{concept_id}"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose ontology relationships for a code")
    parser.add_argument(
        "--athena_path",
        type=str,
        required=True,
        help="Path to Athena snapshot directory or zip archive"
    )
    parser.add_argument(
        "--code",
        type=str,
        default="ICD10CM/C34.02",
        help="Code to diagnose (default: ICD10CM/C34.02)"
    )
    
    args = parser.parse_args()
    
    diagnose_code(args.athena_path, args.code)


if __name__ == "__main__":
    main()
