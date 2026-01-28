#!/usr/bin/env python3
"""
Analyze relationship types in CONCEPT_RELATIONSHIP to see what we might be missing.
"""

import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meds_mcp.server.tools.ontologies import AthenaFileReader


def analyze_relationships(athena_path: str):
    """Analyze all relationship types in the snapshot."""
    print("="*80)
    print("Analyzing CONCEPT_RELATIONSHIP.csv")
    print("="*80)
    
    with AthenaFileReader(athena_path) as reader:
        relationships_df = reader.read_csv("CONCEPT_RELATIONSHIP.csv")
        
        # Count by relationship type
        relationship_counts = (
            relationships_df
            .group_by("relationship_id")
            .agg([
                pl.count().alias("count"),
                pl.col("invalid_reason").is_null().sum().alias("valid_count"),
                (pl.col("invalid_reason").is_not_null()).sum().alias("invalid_count")
            ])
            .sort("count", descending=True)
            .collect()
        )
        
        print("\nRelationship types (sorted by frequency):")
        print(f"{'Relationship ID':<40} {'Total':>12} {'Valid':>12} {'Invalid':>12}")
        print("-" * 80)
        
        total = 0
        valid_total = 0
        for row in relationship_counts.rows():
            rel_id, count, valid, invalid = row
            print(f"{rel_id:<40} {count:>12,} {valid:>12,} {invalid:>12,}")
            total += count
            valid_total += valid
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total:>12,} {valid_total:>12,} {total - valid_total:>12,}")
        
        # Check for hierarchical relationships
        print("\n" + "="*80)
        print("Hierarchical relationship types (likely important for parent-child):")
        print("="*80)
        
        hierarchical_keywords = ["is a", "subsumes", "isa", "parent", "child", "ancestor", "descendant"]
        hierarchical_rels = relationship_counts.filter(
            pl.col("relationship_id").str.to_lowercase().str.contains("|".join(hierarchical_keywords))
        )
        
        if hierarchical_rels.height > 0:
            for row in hierarchical_rels.rows():
                rel_id, count, valid, invalid = row
                print(f"  {rel_id}: {valid:,} valid relationships")
        else:
            print("  (none found with hierarchical keywords)")
        
        # Check bidirectional relationships
        print("\n" + "="*80)
        print("Bidirectional relationship pairs:")
        print("="*80)
        
        # Get reverse relationships
        reverse_pairs = (
            relationships_df
            .select("relationship_id")
            .unique()
            .with_columns([
                pl.col("relationship_id").str.replace(" ", "").str.to_lowercase().alias("normalized")
            ])
            .collect()
        )
        
        # Common bidirectional pairs
        bidirectional_patterns = {
            "isa": "subsumes",
            "subsumes": "isa",
            "maps to": "mapped from",
            "mapped from": "maps to",
        }
        
        rel_ids = [row[0] for row in reverse_pairs.rows()]
        for rel_id in rel_ids[:20]:  # Show first 20
            normalized = rel_id.replace(" ", "").lower()
            if normalized in bidirectional_patterns:
                reverse = bidirectional_patterns[normalized]
                if any(r.replace(" ", "").lower() == reverse for r in rel_ids):
                    print(f"  {rel_id} <-> {reverse}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze relationship types in Athena snapshot")
    parser.add_argument(
        "--athena_path",
        type=str,
        required=True,
        help="Path to Athena snapshot directory or zip archive"
    )
    
    args = parser.parse_args()
    
    analyze_relationships(args.athena_path)


if __name__ == "__main__":
    main()
