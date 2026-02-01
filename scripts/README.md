# Ontology Scripts

## Main Scripts

### `explore_ontology_paths.py`

Explore paths from leaf nodes to ancestors. Samples leaf nodes (nodes with no children) for target vocabularies and generates paths.

**Usage:**
```bash
# Sample 5 leaf nodes from ICD10CM and show longest paths
python scripts/explore_ontology_paths.py --vocabularies ICD10CM --sample 5 --longest

# Show all paths (not just longest) for SNOMED leaf nodes
python scripts/explore_ontology_paths.py --vocabularies SNOMED --sample 3 --all-paths

# Explore specific code with all paths, allow cross-vocabulary
python scripts/explore_ontology_paths.py --code "ICD10CM/C34.02" --all-paths --cross-vocab

# Limit depth
python scripts/explore_ontology_paths.py --code "ICD10CM/C34.02" --longest --max-depth 2
```

**Options:**
- `--vocabularies`: Target vocabularies to sample from (e.g., `ICD10CM SNOMED`)
- `--code`: Specific code to explore (overrides `--vocabularies`)
- `--sample`: Number of leaf nodes to sample (default: 5)
- `--longest`: Show only longest paths (remove redundant edges) - **default**
- `--all-paths`: Show all paths (not just longest)
- `--cross-vocab`: Allow cross-vocabulary paths (default: same vocabulary only)
- `--max-depth`: Maximum depth to traverse

### `summarize_ontology.py`

Summarize ontology statistics: vocabulary distribution, relationships, graph structure.

**Usage:**
```bash
# Full summary
python scripts/summarize_ontology.py

# Skip graph analysis (faster)
python scripts/summarize_ontology.py --skip-graph

# Include relationship type analysis (requires Athena snapshot)
python scripts/summarize_ontology.py --athena_path data/athena_ontologies_snapshot.zip
```

**Output:**
- Vocabulary distribution (top vocabularies by concept count)
- Relationship distribution (parent counts per code)
- Relationship types distribution (if --athena_path provided)
- Graph structure (if graph file exists)

### `materialize_paths.py`

Materialize codes into sets of ancestor paths (all paths from code to root nodes).

**Usage:**
```bash
# Sample ICD10CM codes and show paths
python scripts/materialize_paths.py --vocabulary ICD10CM --sample 3

# Sample from all vocabularies (ICD10CM, SNOMED, LOINC)
python scripts/materialize_paths.py --all-vocabs

# RxNorm with custom relationship types (requires --athena_path)
python scripts/materialize_paths.py --vocabulary RxNorm --sample 2 --rxnorm-rels --athena_path data/athena_ontologies_snapshot.zip

# Test specific code
python scripts/materialize_paths.py --code "ICD10CM/C34.02" --max-paths 5
```

**Options:**
- `--vocabulary`: Sample from specific vocabulary (ICD10CM, SNOMED, LOINC, RxNorm)
- `--all-vocabs`: Sample from ICD10CM, SNOMED, LOINC (and RxNorm if --rxnorm-rels)
- `--code`: Test specific code
- `--sample`: Number of codes to sample
- `--max-paths`: Maximum paths to display per code
- `--rxnorm-rels`: Use RxNorm-specific relationship types (requires --athena_path)
- `--rxnorm-config`: Path to RxNorm config YAML (default: scripts/rxnorm_min_graph.yaml)
- `--athena_path`: Path to Athena snapshot (required for relationship type filtering)

## Other Scripts

- `init_athena_ontologies.py` - Initialize ontology data from Athena snapshot
- `download_data.py` - Download required data files
- `test_mcp_client_sdk.py` - Test MCP client SDK
