# Filtered Ontology Queries Guide

## Overview

All ontology classes now support filtering by:
1. **Vocabulary** - Restrict to specific vocabularies (e.g., ICD10CM, SNOMED)
2. **Relationship Type** - Filter by relationship type (e.g., "Is a", "Maps to")
3. **Max Depth** - Limit traversal depth
4. **Combined Filters** - Use multiple filters together

## API Reference

### `get_parents()`

```python
parents = ontology.get_parents(
    code: str,
    relationship_types: Optional[List[str]] = None,  # e.g., ["Is a", "Maps to"]
    vocabularies: Optional[List[str]] = None,        # e.g., ["ICD10CM", "SNOMED"]
) -> Set[str]
```

### `get_ancestor_subgraph()`

```python
subgraph = ontology.get_ancestor_subgraph(
    code: str,
    vocabularies: Optional[List[str]] = None,
    relationship_types: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
) -> nx.DiGraph  # or Dict[str, Set[str]] for sparse matrix
```

## Examples

### 1. Filter by Vocabulary

```python
# Get only ICD10CM parents
parents_icd10cm = ontology.get_parents("ICD10CM/C34.02", vocabularies=["ICD10CM"])

# Get only SNOMED parents
parents_snomed = ontology.get_parents("ICD10CM/C34.02", vocabularies=["SNOMED"])

# Get parents from multiple vocabularies
parents_multi = ontology.get_parents("ICD10CM/C34.02", vocabularies=["ICD10CM", "SNOMED"])
```

### 2. Filter by Relationship Type

```python
# Get only "Is a" relationships (hierarchical)
parents_isa = ontology.get_parents("ICD10CM/C34.02", relationship_types=["Is a"])

# Get only "Maps to" relationships (vocabulary mappings)
parents_maps_to = ontology.get_parents("ICD10CM/C34.02", relationship_types=["Maps to"])

# Get both types
parents_both = ontology.get_parents("ICD10CM/C34.02", relationship_types=["Is a", "Maps to"])
```

### 3. Combined Filters

```python
# ICD10CM vocabulary + "Is a" relationships only
parents_filtered = ontology.get_parents(
    "ICD10CM/C34.02",
    relationship_types=["Is a"],
    vocabularies=["ICD10CM"]
)
```

### 4. Ancestor Subgraph with Filters

```python
# Get ancestor subgraph restricted to ICD10CM, max depth 2
subgraph = ontology.get_ancestor_subgraph(
    "ICD10CM/C34.02",
    vocabularies=["ICD10CM"],
    max_depth=2
)

# Get subgraph with only "Is a" relationships
subgraph_isa = ontology.get_ancestor_subgraph(
    "ICD10CM/C34.02",
    relationship_types=["Is a"],
    vocabularies=["ICD10CM"]
)
```

## Relationship Types

### Standard Types

- **"Is a"** - Primary hierarchical relationship (child IS A parent)
- **"Maps to"** - Vocabulary mapping (non-standard → standard concept)
- **CONCEPT_ANCESTOR** - Pre-computed transitive closure (used internally)

### Other Types (Not Currently Filtered)

- "Subsumes" - Reverse of "Is a" (redundant)
- "Mapped from" - Reverse of "Maps to" (redundant)
- "Concept same as" - Equivalence (not hierarchical)
- "Concept replaced by" - Deprecation (handled by invalid_reason)

## Implementation Notes

### Sparse Matrix Format

- Vocabulary filtering: ✅ Fully supported
- Relationship type filtering: ⚠️ Limited (sparse matrix combines all relationships)
  - For full relationship type filtering, use `LazyAthenaOntology` which queries original dataframes

### LazyAthenaOntology

- Vocabulary filtering: ✅ Fully supported
- Relationship type filtering: ✅ Fully supported (queries original dataframes)

### AthenaOntology

- Vocabulary filtering: ✅ Fully supported
- Relationship type filtering: ⚠️ Limited (all relationships stored together)
  - For full filtering, rebuild parquet files with relationship type metadata

## Usage Examples

See `scripts/demo_filtered_queries.py` for comprehensive examples:

```bash
# Demonstrate all filtering options
python scripts/demo_filtered_queries.py --use_sparse --code "ICD10CM/C34.02"

# Test with different code
python scripts/demo_filtered_queries.py --code "SNOMED/73211009"
```

## Performance

Filtering adds minimal overhead:
- Vocabulary filtering: < 0.1ms (set intersection)
- Relationship type filtering: Varies (queries dataframes for LazyAthenaOntology)
- Combined filters: Same as individual filters
