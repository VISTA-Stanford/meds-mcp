# OMOP Ontology Relationships Guide

## Standard Relationship Types

Based on OMOP CDM specification, here are the key relationship types:

### Hierarchical Relationships (Parent-Child)

1. **"Is a"** - Primary hierarchical relationship
   - Most important for ontology traversal
   - concept_id_1 IS A concept_id_2 (child -> parent)
   - Used for building concept hierarchies
   - **INCLUDED** ✅

2. **"Subsumes"** - Reverse of "Is a"
   - concept_id_1 SUBSUMES concept_id_2 (parent -> child)
   - Usually redundant with "Is a" (can be derived)
   - **NOT INCLUDED** (can be derived from "Is a")

### Mapping Relationships

3. **"Maps to"** - Maps non-standard to standard concepts
   - Used for vocabulary mappings (ICD10CM -> SNOMED)
   - concept_id_1 MAPS TO concept_id_2
   - **INCLUDED** ✅ (for non-standard concepts)

4. **"Maps to value"** - Special mapping for value_as_concept_id
   - Used in MEASUREMENT and OBSERVATION tables
   - **NOT INCLUDED** (not relevant for ontology hierarchy)

5. **"Mapped from"** - Reverse of "Maps to"
   - Usually redundant
   - **NOT INCLUDED**

### Other Relationships

6. **"Concept replaced by"** - Deprecation relationship
   - **NOT INCLUDED** (handled by invalid_reason)

7. **"Concept same as"** - Equivalence relationship
   - **NOT INCLUDED** (not hierarchical)

8. **"Concept was_a"** - Historical relationship
   - **NOT INCLUDED** (historical only)

## Current Implementation

We currently include:
- ✅ **"Is a"** relationships (primary hierarchy)
- ✅ **"Maps to"** relationships (for non-standard concepts)
- ✅ **CONCEPT_ANCESTOR** (pre-computed transitive closure, min_levels_of_separation == 1)

This covers all hierarchical relationships needed for ontology traversal.

## Recommendation

**Keep current approach** - "Is a" + "Maps to" + CONCEPT_ANCESTOR is sufficient for:
- Building concept hierarchies
- Finding parent/child relationships
- Traversing ontology graphs

Other relationship types are either:
- Redundant (can be derived)
- Not hierarchical (equivalence, replacement)
- Historical (was_a)
