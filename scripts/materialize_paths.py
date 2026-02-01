#!/usr/bin/env python3
"""
Materialize codes into sets of ancestor paths.

For each code, finds all paths from the code to root nodes (nodes with no parents)
using configurable relationship types.
"""

import sys
import argparse
import random
import yaml
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from meds_mcp.server.tools.sparse_graph_ontology import SparseGraphOntology
from meds_mcp.server.tools.ontologies import LazyAthenaOntology


def find_root_nodes(ontology, vocabularies: Optional[List[str]] = None) -> Set[str]:
    """Find root nodes (nodes with no parents) in specified vocabularies."""
    # Sample approach: get codes and check if they have parents
    # For efficiency, we'll use a sampling approach
    if vocabularies:
        vocab_filter = "|".join(vocabularies)
        result = (
            ontology.concepts_df
            .filter(pl.col("code").str.starts_with(vocab_filter))
            .select("code")
            .collect()
        )
        all_codes = result["code"].to_list()
    else:
        # Get all codes (expensive but accurate)
        result = ontology.concepts_df.select("code").collect()
        all_codes = result["code"].to_list()
    
    # Check which codes have no parents
    root_nodes = set()
    sample_size = min(10000, len(all_codes))  # Sample for performance
    sampled = random.sample(all_codes, sample_size)
    
    for code in sampled:
        parents = ontology.get_parents(code, vocabularies=vocabularies)
        if not parents:
            root_nodes.add(code)
    
    return root_nodes


def find_all_paths_to_roots(
    ontology,
    code: str,
    vocabularies: Optional[List[str]] = None,
    relationship_types: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    current_path: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Find all paths from code to any root node.
    
    Returns list of paths, where each path is a list of codes from code to root.
    """
    if current_path is None:
        current_path = []
    
    # Check for cycles
    if code in current_path:
        return []  # Cycle detected
    
    if max_depth is not None and len(current_path) >= max_depth:
        return []
    
    current_path = current_path + [code]
    
    # Get parents with relationship filtering
    parents = ontology.get_parents(
        code,
        relationship_types=relationship_types,
        vocabularies=vocabularies,
    )
    
    if not parents:
        # This is a root node
        return [current_path]
    
    # Recursively find paths from each parent
    all_paths = []
    for parent in parents:
        parent_paths = find_all_paths_to_roots(
            ontology,
            parent,
            vocabularies=vocabularies,
            relationship_types=relationship_types,
            max_depth=max_depth,
            current_path=current_path,
        )
        all_paths.extend(parent_paths)
    
    return all_paths


def materialize_icd10cm_paths(ontology, code: str) -> List[List[str]]:
    """Materialize ICD10CM paths using 'Is a' relationships only."""
    return find_all_paths_to_roots(
        ontology,
        code,
        vocabularies=["ICD10CM"],
        relationship_types=["Is a"],
    )


def materialize_snomed_paths(ontology, code: str) -> List[List[str]]:
    """Materialize SNOMED paths using 'Is a' relationships only."""
    return find_all_paths_to_roots(
        ontology,
        code,
        vocabularies=["SNOMED"],
        relationship_types=["Is a"],
    )


def materialize_loinc_paths(ontology, code: str) -> List[List[str]]:
    """Materialize LOINC paths using 'Is a' relationships only."""
    return find_all_paths_to_roots(
        ontology,
        code,
        vocabularies=["LOINC"],
        relationship_types=["Is a"],
    )


def load_rxnorm_config(config_path: str) -> Dict:
    """Load RxNorm configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('rxnorm_min_graph', {})


def get_rxnorm_vocabularies(config: Dict, default: Optional[List[str]] = None, use_config: bool = False) -> List[str]:
    """Get vocabulary allowlist from config or default.
    
    Args:
        config: RxNorm config dictionary
        default: Default vocabularies if not using config (default: ["RxNorm"])
        use_config: If True, use vocabularies_allowlist from config; if False, use default
    """
    if default is None:
        default = ["RxNorm"]  # Default to RxNorm only
    
    if use_config:
        scope = config.get("scope", {})
        vocabularies = scope.get("vocabularies_allowlist", default)
    else:
        vocabularies = default
    
    # Ensure it's a list
    if isinstance(vocabularies, str):
        vocabularies = [vocabularies]
    
    return vocabularies


def get_rxnorm_relationship_types(config: Dict) -> Tuple[List[str], List[str]]:
    """Extract relationship types from RxNorm config.
    
    Returns:
        Tuple of (path_relationship_types, set_relationship_types)
        - path_relationship_types: "Is a" and "Maps to" (for building paths)
        - set_relationship_types: All other relationships (for collecting sets)
    """
    path_relationship_types = []  # "Is a", "Maps to"
    set_relationship_types = []   # Everything else
    
    edges = config.get('edges', [])
    exclude = set(config.get('exclude_relationship_ids', []))
    
    # Path relationships that define hierarchical structure
    path_rels = {"Is a", "Maps to", "RxNorm is a"}
    
    for edge_config in edges:
        if edge_config.get('enabled', True):  # Default to enabled
            for rel_id in edge_config.get('relationship_id_allowlist', []):
                if rel_id not in exclude:
                    if rel_id in path_rels:
                        path_relationship_types.append(rel_id)
                    else:
                        set_relationship_types.append(rel_id)
    
    # Ensure "Is a" and "Maps to" are included for paths
    if "Is a" not in path_relationship_types:
        path_relationship_types.append("Is a")
    if "Maps to" not in path_relationship_types:
        path_relationship_types.append("Maps to")
    
    return path_relationship_types, set_relationship_types


def materialize_rxnorm_paths(
    ontology,
    code: str,
    path_relationship_types: List[str],
    vocabularies: Optional[List[str]] = None,
    require_both_endpoints: bool = True,
) -> List[List[str]]:
    """
    Materialize RxNorm paths using "Is a" and "Maps to" relationships only.
    
    Args:
        ontology: Ontology instance
        code: Code to materialize
        path_relationship_types: List of path relationship types (typically ["Is a", "Maps to"])
        vocabularies: List of vocabularies to include (default: ["RxNorm"])
        require_both_endpoints: If True, only include edges where both endpoints are in vocabularies
    """
    # Default to RxNorm only (not RxNorm Extension)
    if vocabularies is None:
        vocabularies = ["RxNorm"]
    
    # Use only path relationships for building hierarchical paths
    return find_all_paths_to_roots(
        ontology,
        code,
        vocabularies=vocabularies,
        relationship_types=path_relationship_types,
    )


def get_rxnorm_relationship_sets(
    ontology,
    code: str,
    set_relationship_types: List[str],
    vocabularies: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Get sets of related concepts for RxNorm code using non-path relationships.
    
    IMPORTANT: Branded drugs don't have direct ingredient relationships!
    Chain: Branded -> "Tradename of" -> Generic -> "RxNorm has ing" -> Ingredients
    
    This function handles both branded and generic drugs by:
    1. First trying direct relationships on the input code
    2. If no ingredients found and code is branded, following "Tradename of" to generic
    3. Getting ingredients from the generic form
    
    Args:
        ontology: Ontology instance
        code: Code to get relationships for
        set_relationship_types: List of relationship types to collect
        vocabularies: List of vocabularies to filter by (default: ["RxNorm"])
    
    Returns:
        Dictionary mapping relationship type to set of related codes
    """
    if vocabularies is None:
        vocabularies = ["RxNorm"]
    
    relationship_sets: Dict[str, Set[str]] = {}
    
    # Try direct relationships first
    for rel_type in set_relationship_types:
        related = ontology.get_parents(
            code,
            relationship_types=[rel_type],
            vocabularies=vocabularies,
        )
        if related:
            relationship_sets[rel_type] = related
    
    # If no ingredients found, check if this is a branded drug
    # Branded drugs have format "... [BrandName]" and need to go through generic
    ingredient_types = [t for t in set_relationship_types if 'ing' in t.lower()]
    has_ingredients = any(t in relationship_sets for t in ingredient_types)
    
    if not has_ingredients and ingredient_types:
        # Check for branded drug pattern (contains "[" in description)
        desc = ontology.get_description(code) or ""
        if "[" in desc:
            # This is a branded drug - follow "Tradename of" to get generic
            # Load relationship type if needed
            if hasattr(ontology, 'load_relationship_type'):
                ontology.load_relationship_type('Tradename of')
            
            generic_codes = ontology.get_parents(code, relationship_types=['Tradename of'])
            
            # Get ingredients from generic form(s)
            for generic in generic_codes:
                for rel_type in ingredient_types:
                    related = ontology.get_parents(
                        generic,
                        relationship_types=[rel_type],
                        vocabularies=vocabularies,
                    )
                    if related:
                        # Label as "via generic"
                        key = f"{rel_type}"
                        if key not in relationship_sets:
                            relationship_sets[key] = set()
                        relationship_sets[key].update(related)
    
    return relationship_sets


def sample_codes_with_diverse_relationships(
    ontology, 
    vocabulary: str, 
    count: int = 5,
    relationship_types: Optional[List[str]] = None
) -> List[str]:
    """Sample codes that have relationships beyond just 'Is a'.
    
    For RxNorm, this helps find codes that use the configured relationship types.
    """
    # Only works with LazyAthenaOntology (has relationship data)
    if not hasattr(ontology, 'relationships_df'):
        # Fall back to regular sampling
        return sample_codes(ontology, vocabulary, count)
    
    # Get all codes in vocabulary
    concepts_df = ontology.concepts_df
    schema = concepts_df.collect_schema()
    
    if "vocabulary_id" in schema and "concept_code" in schema:
        # LazyAthenaOntology - construct code
        result = (
            concepts_df
            .filter(pl.col("vocabulary_id") == vocabulary)
            .with_columns([
                (pl.col("vocabulary_id") + "/" + pl.col("concept_code")).alias("code"),
                pl.col("concept_id").cast(pl.Int64).alias("concept_id")
            ])
            .select(["code", "concept_id"])
            .collect()
        )
        all_codes = result["code"].to_list()
        code_to_concept_id = dict(zip(result["code"], result["concept_id"]))
    else:
        # Fallback to regular sampling
        return sample_codes(ontology, vocabulary, count)
    
    if len(all_codes) == 0:
        return []
    
    # If relationship_types specified, find codes that have those relationships
    if relationship_types:
        # Find codes that have at least one of the specified relationship types
        # (excluding "Is a" to find more diverse relationships)
        non_isa_types = [rt for rt in relationship_types if rt != "Is a"]
        
        if non_isa_types:
            # Query relationships_df for codes with these relationship types
            codes_with_rels = (
                ontology.relationships_df
                .filter(
                    (pl.col("relationship_id").is_in(non_isa_types))
                    & (pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
                )
                .select(pl.col("concept_id_1").cast(pl.Int64).alias("concept_id"))
                .unique()
                .collect()
            )
            
            # Map concept_ids back to codes
            concept_id_to_code = {v: k for k, v in code_to_concept_id.items()}
            candidate_codes = [
                concept_id_to_code.get(cid) 
                for cid in codes_with_rels["concept_id"].to_list()
                if cid in concept_id_to_code
            ]
            candidate_codes = [c for c in candidate_codes if c is not None]
            
            if len(candidate_codes) > 0:
                # Sample from codes with diverse relationships
                return random.sample(candidate_codes, min(count, len(candidate_codes)))
    
    # Fallback to regular sampling
    return random.sample(all_codes, min(count, len(all_codes)))


def sample_codes(ontology, vocabulary: str, count: int = 5) -> List[str]:
    """Sample random codes from a vocabulary.
    
    For "RxNorm", only samples codes starting with "RxNorm/" (excludes "RxNorm Extension/").
    """
    # Handle different ontology types
    if hasattr(ontology, 'concepts_df'):
        # SparseGraphOntology or LazyAthenaOntology
        concepts_df = ontology.concepts_df
        
        # Check if it's LazyAthenaOntology (has vocabulary_id/concept_code) or SparseGraphOntology (has code)
        try:
            # Try to get schema to check columns (use collect_schema to avoid warning)
            schema = concepts_df.collect_schema()
            if "code" in schema:
                # SparseGraphOntology - has code column
                # For RxNorm, be precise: only "RxNorm/" not "RxNorm Extension/"
                if vocabulary == "RxNorm":
                    result = (
                        concepts_df
                        .filter(
                            (pl.col("code").str.starts_with("RxNorm/"))
                            & (~pl.col("code").str.starts_with("RxNorm Extension/"))
                        )
                        .select("code")
                        .collect()
                    )
                else:
                    result = (
                        concepts_df
                        .filter(pl.col("code").str.starts_with(vocabulary))
                        .select("code")
                        .collect()
                    )
                all_codes = result["code"].to_list()
            elif "vocabulary_id" in schema and "concept_code" in schema:
                # LazyAthenaOntology - need to construct code
                # For RxNorm, filter to exact vocabulary match (excludes RxNorm Extension)
                result = (
                    concepts_df
                    .filter(pl.col("vocabulary_id") == vocabulary)
                    .with_columns([
                        (pl.col("vocabulary_id") + "/" + pl.col("concept_code")).alias("code")
                    ])
                    .select("code")
                    .collect()
                )
                all_codes = result["code"].to_list()
            else:
                raise ValueError(f"Unknown schema: {list(schema.keys())}")
        except Exception:
            # Fallback: try code column first
            try:
                # For RxNorm, be precise: only "RxNorm/" not "RxNorm Extension/"
                if vocabulary == "RxNorm":
                    result = (
                        concepts_df
                        .filter(
                            (pl.col("code").str.starts_with("RxNorm/"))
                            & (~pl.col("code").str.starts_with("RxNorm Extension/"))
                        )
                        .select("code")
                        .collect()
                    )
                else:
                    result = (
                        concepts_df
                        .filter(pl.col("code").str.starts_with(vocabulary))
                        .select("code")
                        .collect()
                    )
                all_codes = result["code"].to_list()
            except Exception:
                # Try vocabulary_id/concept_code
                result = (
                    concepts_df
                    .filter(pl.col("vocabulary_id") == vocabulary)
                    .with_columns([
                        (pl.col("vocabulary_id") + "/" + pl.col("concept_code")).alias("code")
                    ])
                    .select("code")
                    .collect()
                )
                all_codes = result["code"].to_list()
    else:
        raise ValueError("Ontology does not have concepts_df attribute")
    
    if len(all_codes) == 0:
        return []
    
    return random.sample(all_codes, min(count, len(all_codes)))


def get_relationship_type(ontology, child_code: str, parent_code: str) -> Optional[str]:
    """Get the relationship type between two codes, if available.
    
    Returns the relationship type, or None if not available.
    If multiple relationship types exist, returns the first one found.
    """
    # Only LazyAthenaOntology has access to relationship data
    if not hasattr(ontology, 'relationships_df'):
        return None
    
    child_concept_id = ontology.code_to_concept_id_map.get(child_code)
    parent_concept_id = ontology.code_to_concept_id_map.get(parent_code)
    
    if child_concept_id is None or parent_concept_id is None:
        return None
    
    # Query relationships_df for the relationship type
    result = (
        ontology.relationships_df.filter(
            (pl.col("concept_id_1").cast(pl.Int64) == child_concept_id)
            & (pl.col("concept_id_2").cast(pl.Int64) == parent_concept_id)
            & (pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
        )
        .select("relationship_id")
        .collect()
    )
    
    if len(result) > 0:
        # If multiple relationship types exist, return the first one
        # (In practice, there's usually only one valid relationship between two concepts)
        rel_type = result["relationship_id"][0]
        if len(result) > 1:
            # Multiple relationship types - show all in debug mode?
            # For now, just return the first
            pass
        return rel_type
    
    # Also check CONCEPT_ANCESTOR (which uses "Is a" implicitly)
    # This is a fallback - CONCEPT_ANCESTOR represents transitive "Is a" relationships
    if hasattr(ontology, 'ancestors_df'):
        ancestor_result = (
            ontology.ancestors_df.filter(
                (pl.col("descendant_concept_id").cast(pl.Int64) == child_concept_id)
                & (pl.col("ancestor_concept_id").cast(pl.Int64) == parent_concept_id)
                & (pl.col("min_levels_of_separation") == "1")
            )
            .select(pl.lit("Is a").alias("relationship_id"))
            .collect()
        )
        if len(ancestor_result) > 0:
            return "Is a"
    
    return None


def _wrap_text(text: str, width: int, indent: str = "") -> List[str]:
    """Wrap text to specified width, returning list of lines."""
    if not text:
        return [""]
    
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + (1 if current_line else 0) <= width:
            current_line.append(word)
            current_len += len(word) + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines if lines else [""]


def print_sets_table(code: str, paths: List[List[str]], ontology, relationship_sets: Optional[Dict[str, Set[str]]] = None, code_width: int = 30, desc_width: int = 50):
    """Print sets mode as a formatted table.
    
    Args:
        code: Starting code
        paths: List of paths to extract ancestors from
        ontology: Ontology instance
        relationship_sets: Additional relationship sets (e.g., RxNorm ingredients)
        code_width: Width of code column
        desc_width: Width of description column
    """
    # Collect all unique ancestor codes from paths
    ancestor_set = set()
    for path in paths:
        ancestor_set.update(path[1:])  # Exclude starting node
    
    # Combine with relationship sets
    all_sets: Dict[str, Set[str]] = {}
    if ancestor_set:
        all_sets["ancestors"] = ancestor_set
    if relationship_sets:
        all_sets.update(relationship_sets)
    
    # Print header for starting code
    start_desc = ontology.get_description(code) or ""
    print(f"\n{code}")
    print(f"  {start_desc}")
    print()
    
    if not all_sets:
        print("  (No ancestors or relationships found)")
        print()
        return
    
    # Print table header
    print(f"{'Code':<{code_width}} {'Description':<{desc_width}}")
    print(f"{'-'*code_width} {'-'*desc_width}")
    
    # Print each relationship type section
    for rel_type, related_codes in sorted(all_sets.items()):
        if not related_codes:
            continue
        
        # Section header
        print(f"\n  [{rel_type}]")
        
        # Sort codes for consistent output
        for rel_code in sorted(related_codes):
            rel_desc = ontology.get_description(rel_code) or ""
            
            # Wrap description if needed
            desc_lines = _wrap_text(rel_desc, desc_width)
            
            # Print first line with code
            print(f"  {rel_code:<{code_width-2}} {desc_lines[0]}")
            
            # Print continuation lines (indented)
            for line in desc_lines[1:]:
                print(f"  {'':<{code_width-2}} {line}")
    
    print()


def print_paths_compact(code: str, paths: List[List[str]], ontology, debug: bool = False, relationship_sets: Optional[Dict[str, Set[str]]] = None, mode: str = "paths"):
    """Print paths or sets in compact format.
    
    Args:
        code: Code to print
        paths: List of paths (for mode="paths") or used to extract ancestor set (for mode="sets")
        ontology: Ontology instance
        debug: Show relationship types
        relationship_sets: Dictionary of relationship type -> set of codes (for RxNorm mode="sets")
        mode: "paths" or "sets"
    """
    desc = ontology.get_description(code)
    desc_str = f"  # {desc}" if desc else ""
    
    if mode == "sets":
        # Use table format for sets
        print_sets_table(code, paths, ontology, relationship_sets)
        return
    
    # Paths mode (default)
    print(f"{code} -> [")
    
    if not paths:
        print("  # (No paths found - this may be a root node)")
    else:
        # Check if debug mode is available
        has_relationship_data = hasattr(ontology, 'relationships_df')
        if debug and not has_relationship_data:
            print("  # Note: Relationship type debugging requires LazyAthenaOntology (use --athena_path)")
            debug = False  # Fall back to non-debug mode
        
        # Sort by length (longest first)
        paths_sorted = sorted(paths, key=len, reverse=True)
        
        # Print each path as a list
        for i, path in enumerate(paths_sorted):
            if debug:
                # Show relationship types between each step with descriptions
                path_parts = []
                # First node: code # description
                node0_desc = ontology.get_description(path[0]) or ""
                path_parts.append(f'"{path[0]}"  # {node0_desc}')
                for j in range(len(path) - 1):
                    child = path[j]
                    parent = path[j + 1]
                    rel_type = get_relationship_type(ontology, child, parent)
                    rel_str = f" --[{rel_type or '?'}]--> " if rel_type else " -> "
                    parent_desc = ontology.get_description(parent) or ""
                    path_parts.append(f'{rel_str}"{parent}"  # {parent_desc}')
                # Format as: ["code1" # desc1 --[rel]--> "code2" # desc2]
                path_str = "".join(path_parts)
                comma = "," if i < len(paths_sorted) - 1 else ""
                print(f"  [{path_str}]{comma}")
            else:
                # Format path with codes and descriptions: ["code1" # desc1, "code2" # desc2]
                path_parts = []
                for node in path:
                    node_desc = ontology.get_description(node) or ""
                    path_parts.append(f'"{node}"  # {node_desc}')
                path_str = ", ".join(path_parts)
                comma = "," if i < len(paths_sorted) - 1 else ""
                print(f"  [{path_str}]{comma}")
    
    print(f"]{desc_str}")
    print()


def print_paths(code: str, paths: List[List[str]], ontology, max_paths: int = 10, compact: bool = False, debug: bool = False, relationship_sets: Optional[Dict[str, Set[str]]] = None, mode: str = "paths"):
    """Print paths or sets in a readable format.
    
    Args:
        mode: "paths" to show paths, "sets" to show relationship sets
    """
    if compact:
        print_paths_compact(code, paths, ontology, debug=debug, relationship_sets=relationship_sets, mode=mode)
        return
    
    print(f"\n{'='*80}")
    print(f"Code: {code}")
    desc = ontology.get_description(code)
    if desc:
        print(f"Description: {desc}")
    print(f"Total paths to root: {len(paths)}")
    print(f"{'='*80}")
    
    if not paths:
        print("  (No paths found - this may be a root node)")
        return
    
    # Sort by length (longest first)
    paths_sorted = sorted(paths, key=len, reverse=True)
    
    # Show up to max_paths
    for i, path in enumerate(paths_sorted[:max_paths], 1):
        print(f"\nPath {i} ({len(path)-1} steps):")
        for j, node in enumerate(path):
            node_desc = ontology.get_description(node) or "N/A"
            marker = " [ROOT]" if j == len(path) - 1 else " [START]" if j == 0 else ""
            indent = "  " * j
            print(f"{indent}{node:30} {node_desc}{marker}")
    
    if len(paths) > max_paths:
        print(f"\n  ... and {len(paths) - max_paths} more paths")


def main():
    parser = argparse.ArgumentParser(
        description="Materialize codes into sets of ancestor paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample ICD10CM codes and show paths
  python scripts/materialize_paths.py --vocabulary ICD10CM --sample 3
  
  # Sample RxNorm codes with custom relationships
  python scripts/materialize_paths.py --vocabulary RxNorm --sample 2 --rxnorm-rels
  
  # Test specific code
  python scripts/materialize_paths.py --code "ICD10CM/C34.02"
        """
    )
    
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="data/athena_omop_ontologies",
        help="Path to parquet files directory"
    )
    parser.add_argument(
        "--athena_path",
        type=str,
        help="Path to Athena snapshot (required for relationship type filtering)"
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        choices=["ICD10CM", "SNOMED", "LOINC", "RxNorm"],
        help="Vocabulary to sample from"
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Specific code to materialize (overrides --vocabulary)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of codes to sample"
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=5,
        help="Maximum number of paths to display per code (verbose mode)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact output format: code -> [[path1], [path2], ...]"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show relationship types between nodes in paths (requires LazyAthenaOntology)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paths", "sets"],
        default="paths",
        help="Output mode: 'paths' shows all paths to root, 'sets' shows relationship sets (for RxNorm)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth to traverse"
    )
    parser.add_argument(
        "--rxnorm-rels",
        action="store_true",
        help="[DEPRECATED] RxNorm relationship types are now auto-detected from config. This flag is ignored."
    )
    parser.add_argument(
        "--rxnorm-config",
        type=str,
        default="scripts/rxnorm_min_graph.yaml",
        help="Path to RxNorm configuration YAML file"
    )
    parser.add_argument(
        "--vocabularies",
        type=str,
        nargs="+",
        help="Filter paths to specific vocabularies (e.g., --vocabularies RxNorm RxNorm Extension). For RxNorm, defaults to RxNorm only (not RxNorm Extension)"
    )
    parser.add_argument(
        "--all-vocabs",
        action="store_true",
        help="Sample from all vocabularies (ICD10CM, SNOMED, LOINC, RxNorm)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load ontology
    # Default: Use precomputed sparse graph (fast! ~0.3s load)
    # Sparse graph supports FULL relationship type filtering IF it was built with relationship metadata
    # To rebuild sparse graph with relationship types, run:
    #   python scripts/rebuild_sparse_graph.py --athena_path <path> --parquet_path <path>
    # Only use Athena snapshot if explicitly requested with --athena_path
    
    # Use sparse graph by default (fast!)
    use_athena = False
    if args.athena_path:
        # User explicitly provided athena_path - use LazyAthenaOntology
        use_athena = True
        print("Loading LazyAthenaOntology (from Athena snapshot)...")
        print("  Note: This is slower than sparse graph (~5s vs ~0.3s)")
        print("  Tip: Rebuild sparse graph with relationship types to avoid loading snapshot")
    else:
        # Use fast sparse graph
        print("Loading SparseGraphOntology (fast! uses pre-generated sparse graph)...")
        print("  Using pre-computed graph - no Athena snapshot needed")
        print("  Relationship type filtering: Supported if graph was built with metadata")
    
    if use_athena:
        ontology = LazyAthenaOntology.load_from_athena_snapshot(args.athena_path)
    else:
        # Use new directory-based format with per-relationship-type matrices
        graph_path = f"{args.parquet_path}/ontology_graphs"
        ontology = SparseGraphOntology.load_from_parquet(
            args.parquet_path,
            graph_path=graph_path,
            load_graph=True,
        )
    
    print("Ontology loaded.\n")
    
    # Determine codes to process
    if args.code:
        codes_to_process = {args.code: None}  # vocab will be determined from code
    elif args.all_vocabs:
        # Sample from all vocabularies
        print("Sampling from multiple vocabularies...")
        codes_to_process = {}
        for vocab in ["ICD10CM", "SNOMED", "LOINC"]:
            sampled = sample_codes(ontology, vocab, 2)
            for code in sampled:
                codes_to_process[code] = vocab
        # Add RxNorm if requested
        if args.rxnorm_rels or args.athena_path:
            rxnorm_sampled = sample_codes(ontology, "RxNorm", 2)
            for code in rxnorm_sampled:
                codes_to_process[code] = "RxNorm"
            print(f"  Added {len(rxnorm_sampled)} RxNorm codes")
        print(f"Total codes to process: {len(codes_to_process)}")
    elif args.vocabulary:
        # Random sampling (no smart filtering)
        sampled = sample_codes(ontology, args.vocabulary, args.sample)
        codes_to_process = {code: args.vocabulary for code in sampled}
        print(f"Sampled {len(codes_to_process)} codes from {args.vocabulary}")
    else:
        parser.error("Must specify either --code, --vocabulary, or --all-vocabs")
    
    # Load RxNorm config if processing RxNorm codes (auto-detect, no flag needed)
    rxnorm_config = None
    rxnorm_path_relationship_types = None
    rxnorm_set_relationship_types = None
    rxnorm_vocabularies = None
    rxnorm_require_both = False
    
    # Check if we're processing any RxNorm codes
    has_rxnorm = any(
        v in ["RxNorm", "RxNorm Extension"] or 
        (v is None and (code.startswith("RxNorm") or code.startswith("RxNorm Extension")))
        for code, v in codes_to_process.items()
    )
    
    if has_rxnorm or args.rxnorm_rels:
        # Auto-load RxNorm config
        config_path = Path(args.rxnorm_config)
        if config_path.exists():
            rxnorm_config = load_rxnorm_config(str(config_path))
            rxnorm_path_relationship_types, rxnorm_set_relationship_types = get_rxnorm_relationship_types(rxnorm_config)
            rxnorm_require_both = rxnorm_config.get('scope', {}).get('require_both_endpoints_in_allowlist', True)
            
            # Get vocabularies: command line override > default (RxNorm only)
            # Don't use config's vocabularies_allowlist by default - default to RxNorm only
            if args.vocabularies:
                rxnorm_vocabularies = args.vocabularies
            else:
                # Default to RxNorm only (not RxNorm Extension)
                rxnorm_vocabularies = ["RxNorm"]
            
            print(f"Auto-loaded RxNorm config:")
            print(f"  Path relationships (Is a, Maps to): {len(rxnorm_path_relationship_types)} types")
            print(f"  Set relationships (Has ingredient, etc.): {len(rxnorm_set_relationship_types)} types")
            print(f"  Vocabularies: {', '.join(rxnorm_vocabularies)}")
        else:
            # Fallback: use "Is a" and "Maps to" for paths, empty for sets
            rxnorm_path_relationship_types = ["Is a", "Maps to"]
            rxnorm_set_relationship_types = []
            rxnorm_require_both = True
            
            # Use command line vocabularies or default to RxNorm only
            rxnorm_vocabularies = args.vocabularies if args.vocabularies else ["RxNorm"]
            print(f"Using default RxNorm relationships:")
            print(f"  Path relationships: {rxnorm_path_relationship_types}")
            print(f"  Set relationships: (none)")
            print(f"  Vocabularies: {', '.join(rxnorm_vocabularies)}")
    
    # Lazy-load additional relationship types if needed (for SparseGraphOntology)
    if has_rxnorm and not use_athena and hasattr(ontology, 'load_relationship_type'):
        all_rxnorm_types = (rxnorm_path_relationship_types or []) + (rxnorm_set_relationship_types or [])
        if all_rxnorm_types:
            print("\nLazy-loading RxNorm relationship types...")
            for rel_type in all_rxnorm_types:
                ontology.load_relationship_type(rel_type)
            print()
    
    # Process each code
    for code, vocab in codes_to_process.items():
        if vocab is None:
            vocab = code.split("/")[0] if "/" in code else None
        
        # Initialize relationship_sets for all vocabularies
        relationship_sets = None
        
        if vocab == "ICD10CM":
            paths = materialize_icd10cm_paths(ontology, code)
        elif vocab == "SNOMED":
            paths = materialize_snomed_paths(ontology, code)
        elif vocab == "LOINC":
            paths = materialize_loinc_paths(ontology, code)
        elif vocab in ["RxNorm", "RxNorm Extension"]:
            # RxNorm: use "Is a"/"Maps to" for paths, other relationships for sets
            if rxnorm_path_relationship_types:
                # Get paths using hierarchical relationships ("Is a", "Maps to")
                paths = materialize_rxnorm_paths(
                    ontology,
                    code,
                    path_relationship_types=rxnorm_path_relationship_types,
                    vocabularies=rxnorm_vocabularies,
                    require_both_endpoints=rxnorm_require_both if use_athena else False,
                )
                
                # Get relationship sets using other relationship types
                if rxnorm_set_relationship_types:
                    relationship_sets = get_rxnorm_relationship_sets(
                        ontology,
                        code,
                        set_relationship_types=rxnorm_set_relationship_types,
                        vocabularies=rxnorm_vocabularies,
                    )
            else:
                # No config found - use default "Is a" for paths
                vocabularies = args.vocabularies if args.vocabularies else ["RxNorm"]
                paths = find_all_paths_to_roots(
                    ontology,
                    code,
                    vocabularies=vocabularies,
                    relationship_types=["Is a"],
                    max_depth=args.max_depth,
                )
        else:
            # Default: use "Is a" relationships
            paths = find_all_paths_to_roots(
                ontology,
                code,
                relationship_types=["Is a"],
                max_depth=args.max_depth,
            )
        
        # Use the mode specified by the user
        print_paths(code, paths, ontology, max_paths=args.max_paths, compact=args.compact, debug=args.debug, relationship_sets=relationship_sets, mode=args.mode)


if __name__ == "__main__":
    main()
