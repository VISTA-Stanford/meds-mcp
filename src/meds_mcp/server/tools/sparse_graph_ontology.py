"""
Optimized ontology representation using sparse matrices for graph storage.

Architecture:
- ONE sparse matrix per relationship type (e.g., is_a.npz, maps_to.npz, has_ingredient.npz)
- Shared node index across all matrices (stored in codes.parquet)
- Lazy loading: only load matrices for relationship types you need
- Edge existence is implicit - no need for metadata dictionary

This is MUCH faster than storing all edges in one matrix + huge edge->type dictionary.

Directory structure:
  ontology_graphs/
    codes.parquet        # Shared code index: (idx, code)
    is_a.npz             # Sparse matrix for "Is a" edges
    maps_to.npz          # Sparse matrix for "Maps to" edges  
    has_ingredient.npz   # etc.
"""

import os
import time
from pathlib import Path
from typing import Optional, Set, List, Dict, Any, Tuple
import numpy as np
import scipy.sparse as sp
import polars as pl

# Import AthenaFileReader for building from snapshot
try:
    from meds_mcp.server.tools.ontologies import AthenaFileReader
except ImportError:
    try:
        from meds_mcp.server.tools.athena import AthenaFileReader
    except ImportError:
        AthenaFileReader = None


def _sanitize_relationship_name(rel_id: str) -> str:
    """Convert relationship ID to a valid filename."""
    return rel_id.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _save_matrix(matrix: sp.csr_matrix, path: str):
    """Save a sparse matrix using numpy's native format."""
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.array(matrix.shape, dtype=np.int64),
    )


def _load_matrix(path: str) -> sp.csr_matrix:
    """Load a sparse matrix from numpy format."""
    with np.load(path, allow_pickle=False) as loaded:
        return sp.csr_matrix(
            (loaded['data'], loaded['indices'], loaded['indptr']),
            shape=tuple(loaded['shape']),
        )


class SparseGraphOntology:
    """
    Ultra-fast ontology representation using one sparse matrix per relationship type.
    
    Key insight: Edge existence is implicit in the matrix for each relationship type.
    No need for a huge dictionary mapping edges to types.
    
    Structure:
    - relationship_matrices: Dict[str, sp.csr_matrix] - one matrix per relationship type
    - codes.parquet: Shared code index, lazy-loaded with Polars
    """
    
    def __init__(
        self,
        concepts_df: pl.LazyFrame,
        code_to_idx: Optional[Dict[str, int]] = None,
        idx_to_code: Optional[Dict[int, str]] = None,
        parent_matrix: Optional[sp.csr_matrix] = None,  # Legacy: combined matrix
        graph_path: Optional[str] = None,
        code_trie: Optional[Any] = None,  # Unused, kept for compatibility
        codes_array: Optional[List[str]] = None,
        relationship_types_map: Optional[Dict[Tuple[int, int], Set[str]]] = None,  # Legacy
        relationship_matrices: Optional[Dict[str, sp.csr_matrix]] = None,  # NEW: per-type matrices
    ):
        self.concepts_df = concepts_df
        self._code_to_idx = code_to_idx
        self.idx_to_code = idx_to_code
        self._codes_array = codes_array
        self._parent_matrix = parent_matrix  # Legacy combined matrix
        self.graph_path = graph_path
        self._description_cache: Dict[str, str] = {}
        
        # NEW: One matrix per relationship type (much more efficient!)
        self._relationship_matrices: Dict[str, sp.csr_matrix] = relationship_matrices or {}
        
        # Legacy support - but this is the SLOW path we're deprecating
        self._relationship_types_map = relationship_types_map or {}
    
    @property
    def relationship_types_map(self) -> Dict[Tuple[int, int], Set[str]]:
        """Legacy accessor - prefer using relationship_matrices instead."""
        return self._relationship_types_map
    
    @relationship_types_map.setter  
    def relationship_types_map(self, value):
        self._relationship_types_map = value or {}
    
    @property
    def code_to_idx(self) -> Dict[str, int]:
        """Lazy-build code_to_idx from codes_array if needed."""
        if self._code_to_idx is None and self._codes_array is not None:
            # Build on first access
            self._code_to_idx = {code: idx for idx, code in enumerate(self._codes_array)}
        return self._code_to_idx or {}
    
    def _get_idx_from_code(self, code: str) -> Optional[int]:
        """Get index for code by querying metadata (lazy)."""
        # Query metadata to find the row index for this code
        result = (
            self.concepts_df
            .with_row_index("_idx")
            .filter(pl.col("code") == code)
            .select("_idx")
            .collect()
        )
        
        if result.height > 0:
            return result.row(0)[0]
        return None
    
    
    @classmethod
    def build_from_athena_snapshot(
        cls,
        athena_path: str,
        parquet_path: str,
        graph_path: Optional[str] = None,
    ) -> "SparseGraphOntology":
        """
        Build sparse graphs from Athena snapshot - ONE MATRIX PER RELATIONSHIP TYPE.
        
        This creates a directory with:
        - codes.parquet: Shared code index (idx, code)
        - {relationship_type}.npz: One sparse matrix per relationship type
        
        Args:
            athena_path: Path to Athena snapshot (zip file or directory)
            parquet_path: Directory containing descriptions.parquet (for metadata queries)
            graph_path: Directory to save sparse graphs (defaults to parquet_path/ontology_graphs/)
        """
        if AthenaFileReader is None:
            raise ImportError("AthenaFileReader not available. Cannot build from Athena snapshot.")
        
        print("Building sparse graphs from Athena snapshot (one matrix per relationship type)...")
        
        if graph_path is None:
            graph_path = os.path.join(parquet_path, "ontology_graphs")
        
        os.makedirs(graph_path, exist_ok=True)
        
        # Load concepts to build code_to_idx mapping
        descriptions_path = os.path.join(parquet_path, "descriptions.parquet")
        concepts_df = pl.scan_parquet(descriptions_path)
        codes_df = concepts_df.select("code").collect()
        codes = codes_df["code"].to_list()
        code_to_idx = {code: idx for idx, code in enumerate(codes)}
        n_nodes = len(code_to_idx)
        
        print(f"Building sparse matrices for {n_nodes:,} nodes...")
        
        # Collect edges grouped by relationship type
        edges_by_type: Dict[str, List[Tuple[int, int]]] = {}
        
        with AthenaFileReader(athena_path) as reader:
            # Load concepts to get concept_id -> code mapping
            concepts_df_athena = reader.read_csv("CONCEPT.csv")
            code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
            concept_id_col = pl.col("concept_id").cast(pl.Int64)
            
            mapping_df = (
                concepts_df_athena.select([
                    code_col.alias("code"),
                    concept_id_col.alias("concept_id"),
                ])
                .collect()
            )
            
            concept_id_to_code_map = {}
            for code, concept_id in mapping_df.rows():
                if code and concept_id is not None:
                    concept_id_to_code_map[concept_id] = code
            
            print(f"  Loaded {len(concept_id_to_code_map):,} concept mappings")
            
            # Load ALL relationships
            relationships_df = reader.read_csv("CONCEPT_RELATIONSHIP.csv")
            
            # Filter valid relationships
            relationships_df = relationships_df.filter(
                (pl.col("concept_id_1") != pl.col("concept_id_2"))
                & (pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
            )
            
            print("  Processing relationships...")
            relationships_collected = relationships_df.select([
                pl.col("concept_id_1").cast(pl.Int64).alias("concept_id_1"),
                pl.col("concept_id_2").cast(pl.Int64).alias("concept_id_2"),
                pl.col("relationship_id").alias("relationship_id"),
            ]).collect()
            
            edge_count = 0
            for concept_id_1, concept_id_2, relationship_id in relationships_collected.rows():
                child_code = concept_id_to_code_map.get(concept_id_1)
                parent_code = concept_id_to_code_map.get(concept_id_2)
                
                if child_code and parent_code and child_code in code_to_idx and parent_code in code_to_idx:
                    child_idx = code_to_idx[child_code]
                    parent_idx = code_to_idx[parent_code]
                    
                    # Group by relationship type
                    if relationship_id not in edges_by_type:
                        edges_by_type[relationship_id] = []
                    edges_by_type[relationship_id].append((child_idx, parent_idx))
                    
                    edge_count += 1
                    if edge_count % 1000000 == 0:
                        print(f"    Processed {edge_count:,} edges...")
        
        print(f"Found {edge_count:,} total edges across {len(edges_by_type)} relationship types")
        
        # Save code index as parquet for lazy loading
        codes_path = os.path.join(graph_path, "codes.parquet")
        codes_df = pl.DataFrame({
            "idx": list(range(len(codes))),
            "code": codes,
        })
        codes_df.write_parquet(codes_path)
        print(f"  Saved code index to {codes_path}")
        
        # Build and save one sparse matrix per relationship type
        relationship_matrices: Dict[str, sp.csr_matrix] = {}
        total_size = os.path.getsize(codes_path)
        
        for rel_type, edges in edges_by_type.items():
            rows = [e[0] for e in edges]
            cols = [e[1] for e in edges]
            data = np.ones(len(edges), dtype=np.int8)
            matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            
            # Save to file
            safe_name = _sanitize_relationship_name(rel_type)
            matrix_path = os.path.join(graph_path, f"{safe_name}.npz")
            _save_matrix(matrix, matrix_path)
            
            relationship_matrices[rel_type] = matrix
            total_size += os.path.getsize(matrix_path)
        
        print(f"\nSaved {len(edges_by_type)} relationship matrices")
        print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
        
        # Save manifest listing all relationship types
        manifest_path = os.path.join(graph_path, "manifest.txt")
        with open(manifest_path, "w") as f:
            for rel_type in sorted(edges_by_type.keys()):
                safe_name = _sanitize_relationship_name(rel_type)
                edge_count = len(edges_by_type[rel_type])
                f.write(f"{rel_type}\t{safe_name}.npz\t{edge_count}\n")
        
        print(f"  Saved manifest to {manifest_path}")
        
        # Return ontology instance
        return cls(
            concepts_df=concepts_df,
            code_to_idx=code_to_idx,
            idx_to_code=None,
            parent_matrix=None,  # No combined matrix - use relationship_matrices
            graph_path=graph_path,
            codes_array=codes,
            relationship_matrices=relationship_matrices,
        )
    
    @classmethod
    def load_from_parquet(
        cls,
        parquet_path: str,
        graph_path: Optional[str] = None,
        load_graph: bool = True,
        relationship_types: Optional[List[str]] = None,
    ) -> "SparseGraphOntology":
        """
        Load ontology from parquet files with optional sparse graph.
        
        Args:
            parquet_path: Directory containing descriptions.parquet and parents.parquet
            graph_path: Optional path to sparse graphs directory (new format) or .npz file (legacy)
            load_graph: If True, load graph
            relationship_types: List of relationship types to load. If None, loads common ones.
        """
        # Keep metadata truly lazy
        descriptions_path = os.path.join(parquet_path, "descriptions.parquet")
        concepts_df = pl.scan_parquet(descriptions_path)
        
        # Try new per-relationship-type format first
        graphs_dir = graph_path if graph_path else os.path.join(parquet_path, "ontology_graphs")
        
        if os.path.isdir(graphs_dir) and os.path.exists(os.path.join(graphs_dir, "codes.parquet")):
            # NEW FORMAT: Directory with per-relationship-type matrices
            print(f"Loading sparse graphs from {graphs_dir}...")
            graph_start = time.time()
            
            # Load code index
            codes_df_loaded = pl.read_parquet(os.path.join(graphs_dir, "codes.parquet"))
            codes = codes_df_loaded["code"].to_list()
            code_to_idx = {code: idx for idx, code in enumerate(codes)}
            
            # Load manifest to see available relationship types
            manifest_path = os.path.join(graphs_dir, "manifest.txt")
            available_types: Dict[str, str] = {}  # rel_type -> filename
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            available_types[parts[0]] = parts[1]
            
            # Determine which types to load
            if relationship_types is None:
                # Default: load "Is a" and "Maps to" for basic hierarchy traversal
                types_to_load = ["Is a", "Maps to"]
            else:
                types_to_load = relationship_types
            
            # Load requested matrices
            relationship_matrices: Dict[str, sp.csr_matrix] = {}
            for rel_type in types_to_load:
                if rel_type in available_types:
                    matrix_path = os.path.join(graphs_dir, available_types[rel_type])
                    if os.path.exists(matrix_path):
                        relationship_matrices[rel_type] = _load_matrix(matrix_path)
                else:
                    # Try to find by sanitized name
                    safe_name = _sanitize_relationship_name(rel_type)
                    matrix_path = os.path.join(graphs_dir, f"{safe_name}.npz")
                    if os.path.exists(matrix_path):
                        relationship_matrices[rel_type] = _load_matrix(matrix_path)
            
            print(f"  Loaded {len(relationship_matrices)} relationship matrices in {time.time() - graph_start:.2f}s")
            print(f"  Types: {list(relationship_matrices.keys())}")
            
            return cls(
                concepts_df=concepts_df,
                code_to_idx=code_to_idx,
                codes_array=codes,
                graph_path=graphs_dir,
                relationship_matrices=relationship_matrices,
            )
        
        # LEGACY FORMAT: Single .npz file
        elif graph_path and (graph_path.endswith(".npz") or graph_path.endswith(".pkl.gz")):
            matrix_path = graph_path if graph_path.endswith(".npz") else graph_path.replace(".pkl.gz", ".npz")
            if os.path.exists(matrix_path):
                print(f"Loading legacy sparse graph from {matrix_path}...")
                graph_start = time.time()
                result = cls._load_sparse_graph(matrix_path)
                parent_matrix = result[0]
                loaded_code_to_idx = result[1] or {}
                loaded_codes_array = result[3]
                loaded_rel_types = result[4] if len(result) >= 5 else None
                print(f"Sparse graph loaded in {time.time() - graph_start:.2f}s")
                
                return cls(
                    concepts_df=concepts_df,
                    code_to_idx=loaded_code_to_idx,
                    codes_array=loaded_codes_array,
                    parent_matrix=parent_matrix,
                    graph_path=graph_path,
                    relationship_types_map=loaded_rel_types,
                )
        
        # No pre-built graph - build from parquet (slow, legacy path)
        if load_graph:
            print("Building sparse graph from parquet (this may take a minute)...")
            graph_start = time.time()
            codes_df_built = pl.scan_parquet(descriptions_path).select("code").collect()
            codes = codes_df_built["code"].to_list()
            code_to_idx = {code: idx for idx, code in enumerate(codes)}
            parent_matrix = cls._build_sparse_graph_from_parquet(parquet_path, code_to_idx)
            print(f"Sparse graph built in {time.time() - graph_start:.2f}s")
            
            return cls(
                concepts_df=concepts_df,
                code_to_idx=code_to_idx,
                codes_array=codes,
                parent_matrix=parent_matrix,
                graph_path=graph_path,
            )
        
        # No graph
        return cls(concepts_df=concepts_df)
    
    @property
    def parent_matrix(self) -> sp.csr_matrix:
        """Lazy-load sparse graph if not already loaded."""
        if self._parent_matrix is None:
            if self.graph_path:
                print(f"Lazy-loading sparse graph from {self.graph_path}...")
                result = self._load_sparse_graph(self.graph_path)
                matrix = result[0]
                self._parent_matrix = matrix
                if len(result) == 5 and result[4] and not self.relationship_types_map:
                    self.relationship_types_map = result[4]
            else:
                raise RuntimeError("Graph not available and no graph_path provided")
        return self._parent_matrix
    
    def get_description(self, code: str) -> Optional[str]:
        """Get description for a code using Polars (lazy, queries on-demand)."""
        if code in self._description_cache:
            return self._description_cache[code]
        
        # Lazy query - Polars will scan parquet and filter efficiently
        result = (
            self.concepts_df
            .filter(pl.col("code") == code)
            .select("description")
            .collect()
        )
        
        if result.height > 0:
            desc = result.row(0)[0]
            self._description_cache[code] = desc
            return desc
        return None
    
    def get_descriptions_batch(self, codes: List[str]) -> Dict[str, str]:
        """Get descriptions for multiple codes efficiently using Polars join."""
        # Filter out cached
        uncached = [c for c in codes if c not in self._description_cache]
        if not uncached:
            return {c: self._description_cache[c] for c in codes}
        
        # Batch query using Polars filter with is_in
        result = (
            self.concepts_df
            .filter(pl.col("code").is_in(uncached))
            .select(["code", "description"])
            .collect()
        )
        
        # Update cache and return
        descriptions = {}
        for row in result.rows():
            code, desc = row
            self._description_cache[code] = desc
            descriptions[code] = desc
        
        # Add cached results
        for code in codes:
            if code in self._description_cache:
                descriptions[code] = self._description_cache[code]
        
        return descriptions
    
    def _get_parent_indices_from_matrix(self, idx: int, matrix: sp.csr_matrix) -> Set[int]:
        """Get parent indices from a specific matrix."""
        return set(matrix[idx].indices)
    
    def _get_parent_indices(self, idx: int, relationship_types: Optional[List[str]] = None) -> Set[int]:
        """Get parent indices, optionally filtered by relationship type."""
        # NEW: Use per-relationship-type matrices if available
        if self._relationship_matrices:
            parent_indices: Set[int] = set()
            if relationship_types is None:
                # Use all loaded matrices
                for matrix in self._relationship_matrices.values():
                    parent_indices.update(self._get_parent_indices_from_matrix(idx, matrix))
            else:
                # Use only specified relationship types
                for rel_type in relationship_types:
                    if rel_type in self._relationship_matrices:
                        matrix = self._relationship_matrices[rel_type]
                        parent_indices.update(self._get_parent_indices_from_matrix(idx, matrix))
            return parent_indices
        
        # LEGACY: Use combined matrix
        if self._parent_matrix is not None:
            return set(self._parent_matrix[idx].indices)
        
        return set()
    
    def _materialize_codes(self, indices: Set[int]) -> Set[str]:
        """Materialize code strings from indices by querying metadata."""
        idx_to_code = self._materialize_codes_batch(indices)
        return set(idx_to_code.values())
    
    def get_parents(
        self,
        code: str,
        relationship_types: Optional[List[str]] = None,
        vocabularies: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        Get immediate parents with optional filtering.
        
        Uses per-relationship-type matrices for efficient, direct filtering.
        No slow dictionary lookup needed!
        
        Args:
            code: Code to get parents for
            relationship_types: Optional list of relationship types to include
                              (e.g., ["Is a", "Maps to"]). If None, includes all loaded types.
            vocabularies: Optional list of vocabulary prefixes to filter by
                         (e.g., ["ICD10CM", "SNOMED"]). If None, includes all.
        
        Returns:
            Set of parent codes matching the filters
        """
        # Get index for code
        idx = self._get_idx_from_code(code)
        if idx is None:
            return set()
        
        # Get parent indices - filtering by relationship type is built-in and FAST
        parent_indices = self._get_parent_indices(idx, relationship_types)
        
        # Materialize codes from indices
        parents = self._materialize_codes(parent_indices)
        
        # Apply vocabulary filter if specified
        if vocabularies:
            def _get_vocab(c: str) -> str:
                return c.split("/")[0] if "/" in c else c
            parents = {p for p in parents if _get_vocab(p) in vocabularies}
        
        return parents
    
    def _get_child_indices(self, idx: int, relationship_types: Optional[List[str]] = None) -> Set[int]:
        """Get child indices (transpose of parent indices)."""
        child_indices: Set[int] = set()
        
        # NEW: Use per-relationship-type matrices
        if self._relationship_matrices:
            matrices_to_use = {}
            if relationship_types is None:
                matrices_to_use = self._relationship_matrices
            else:
                matrices_to_use = {k: v for k, v in self._relationship_matrices.items() if k in relationship_types}
            
            for matrix in matrices_to_use.values():
                csc_matrix = matrix.tocsc()
                child_indices.update(csc_matrix[:, idx].indices)
            return child_indices
        
        # LEGACY: Use combined matrix
        if self._parent_matrix is not None:
            csc_matrix = self._parent_matrix.tocsc()
            return set(csc_matrix[:, idx].indices)
        
        return set()
    
    def get_children(self, code: str, relationship_types: Optional[List[str]] = None) -> Set[str]:
        """Get immediate children using sparse matrix transpose."""
        idx = self._get_idx_from_code(code)
        if idx is None:
            return set()
        
        child_indices = self._get_child_indices(idx, relationship_types)
        return self._materialize_codes(child_indices)
    
    def load_relationship_type(self, rel_type: str) -> bool:
        """
        Lazy-load a specific relationship type matrix.
        
        Returns True if successfully loaded, False if not available.
        """
        if rel_type in self._relationship_matrices:
            return True  # Already loaded
        
        if not self.graph_path or not os.path.isdir(self.graph_path):
            return False
        
        safe_name = _sanitize_relationship_name(rel_type)
        matrix_path = os.path.join(self.graph_path, f"{safe_name}.npz")
        
        if os.path.exists(matrix_path):
            print(f"  Lazy-loading relationship type: {rel_type}")
            self._relationship_matrices[rel_type] = _load_matrix(matrix_path)
            return True
        
        return False
    
    def get_available_relationship_types(self) -> List[str]:
        """Get list of available relationship types from the manifest."""
        if not self.graph_path or not os.path.isdir(self.graph_path):
            return list(self._relationship_matrices.keys())
        
        manifest_path = os.path.join(self.graph_path, "manifest.txt")
        if os.path.exists(manifest_path):
            types = []
            with open(manifest_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        types.append(parts[0])
            return types
        
        return list(self._relationship_matrices.keys())
    
    def get_ancestor_subgraph(
        self,
        code: str,
        vocabularies: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Set[str]]:
        """
        Get ancestor subgraph using sparse matrix traversal (works in index space).
        
        Uses per-relationship-type matrices for direct filtering - no slow dictionary lookup!
        
        Args:
            code: Starting code
            vocabularies: Optional list of vocabulary prefixes to include
            relationship_types: Optional list of relationship types to include
                              (e.g., ["Is a", "Maps to"]). If None, includes all loaded types.
            max_depth: Optional maximum depth to traverse
        
        Returns:
            Dictionary mapping codes to their parent sets (subgraph)
        """
        start_idx = self._get_idx_from_code(code)
        if start_idx is None:
            return {}
        
        # Build subgraph using BFS traversal (entirely in index space)
        subgraph_indices: Dict[int, Set[int]] = {}
        visited = set()
        queue = [(start_idx, 0)]  # (index, depth)
        
        while queue:
            current_idx, depth = queue.pop(0)
            
            if current_idx in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            
            visited.add(current_idx)
            
            # Get parent indices - uses per-type matrices directly (FAST!)
            parent_indices = self._get_parent_indices(current_idx, relationship_types)
            subgraph_indices[current_idx] = parent_indices
            
            # Add parents to queue
            for parent_idx in parent_indices:
                if parent_idx not in visited:
                    queue.append((parent_idx, depth + 1))
        
        # Materialize all codes at once (batch query)
        all_indices = set(subgraph_indices.keys()) | set().union(*subgraph_indices.values())
        idx_to_code_map = self._materialize_codes_batch(all_indices)
        
        # Convert subgraph from indices to codes
        subgraph: Dict[str, Set[str]] = {}
        for node_idx, parent_indices_set in subgraph_indices.items():
            node_code = idx_to_code_map.get(node_idx)
            if node_code is None:
                continue
            
            # Apply vocabulary filter if specified
            if vocabularies:
                def _get_vocab(c: str) -> str:
                    return c.split("/")[0] if "/" in c else c
                vocab = _get_vocab(node_code)
                if vocab not in vocabularies and node_code != code:
                    continue
            
            parent_codes = {idx_to_code_map.get(p) for p in parent_indices_set if idx_to_code_map.get(p)}
            
            # Filter parents by vocabulary
            if vocabularies:
                def _get_vocab(c: str) -> str:
                    return c.split("/")[0] if "/" in c else c
                parent_codes = {p for p in parent_codes if _get_vocab(p) in vocabularies or p == code}
            
            subgraph[node_code] = parent_codes
        
        return subgraph
    
    def _materialize_codes_batch(self, indices: Set[int]) -> Dict[int, str]:
        """Materialize code strings from indices by batch querying metadata."""
        if not indices:
            return {}
        
        # Batch query metadata for all indices at once
        result = (
            self.concepts_df
            .with_row_index("_idx")
            .filter(pl.col("_idx").is_in(list(indices)))
            .select(["_idx", "code"])
            .collect()
        )
        
        return {row[0]: row[1] for row in result.rows()}
    
    def _find_all_paths(self, start_idx: int, end_idx: int, max_depth: Optional[int] = None) -> List[List[int]]:
        """Find all paths from start_idx to end_idx in the DAG."""
        if start_idx == end_idx:
            return [[start_idx]]
        
        all_paths = []
        
        def dfs(current_idx: int, path: List[int], visited: set, depth: int = 0):
            if max_depth is not None and depth > max_depth:
                return
            if current_idx in visited:
                return
            
            visited.add(current_idx)
            path.append(current_idx)
            
            if current_idx == end_idx:
                all_paths.append(path.copy())
            else:
                # Get parent indices
                parent_indices = self._get_parent_indices(current_idx)
                for parent_idx in parent_indices:
                    if parent_idx not in path:  # Avoid cycles
                        dfs(parent_idx, path, visited, depth + 1)
            
            path.pop()
            visited.remove(current_idx)
        
        dfs(start_idx, [], set())
        return all_paths
    
    def _is_redundant_edge(self, child_idx: int, parent_idx: int) -> bool:
        """Check if edge (child_idx -> parent_idx) is redundant (has longer path)."""
        # Find all paths from child to parent
        paths = self._find_all_paths(child_idx, parent_idx)
        
        # If there's a path longer than 1 edge, this direct edge is redundant
        return any(len(path) > 2 for path in paths)
    
    def get_ancestor_subgraph_filtered(
        self,
        code: str,
        vocabularies: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        remove_redundant_edges: bool = True,
    ) -> Dict[str, Set[str]]:
        """
        Get ancestor subgraph with option to remove redundant "skip" edges.
        
        Args:
            code: Starting code
            vocabularies: Optional list of vocabulary prefixes to include
            relationship_types: Optional list of relationship types to include
            max_depth: Optional maximum depth to traverse
            remove_redundant_edges: If True, remove edges that have longer paths
        
        Returns:
            Dictionary mapping codes to their parent sets (subgraph)
        """
        start_idx = self._get_idx_from_code(code)
        if start_idx is None:
            return {}
        
        matrix = self.parent_matrix
        
        # Build subgraph using BFS traversal (entirely in index space)
        subgraph_indices: Dict[int, Set[int]] = {}
        visited = set()
        queue = [(start_idx, 0)]  # (index, depth)
        
        while queue:
            current_idx, depth = queue.pop(0)
            
            if current_idx in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            
            visited.add(current_idx)
            
            # Get parent indices
            parent_indices = set(matrix[current_idx].indices)
            
            # Filter out redundant edges if requested
            if remove_redundant_edges:
                filtered_parents = set()
                for parent_idx in parent_indices:
                    # Check if this edge is redundant (has a longer path)
                    if not self._is_redundant_edge(current_idx, parent_idx):
                        filtered_parents.add(parent_idx)
                parent_indices = filtered_parents
            
            subgraph_indices[current_idx] = parent_indices
            
            # Add parents to queue
            for parent_idx in parent_indices:
                if parent_idx not in visited:
                    queue.append((parent_idx, depth + 1))
        
        # Materialize all codes at once (batch query)
        all_indices = set(subgraph_indices.keys()) | set().union(*subgraph_indices.values())
        idx_to_code_map = self._materialize_codes_batch(all_indices)
        
        # Convert subgraph from indices to codes
        subgraph: Dict[str, Set[str]] = {}
        for node_idx, parent_indices_set in subgraph_indices.items():
            node_code = idx_to_code_map.get(node_idx)
            if node_code is None:
                continue
            
            # Apply vocabulary filter if specified
            if vocabularies:
                def _get_vocab(c: str) -> str:
                    return c.split("/")[0] if "/" in c else c
                vocab = _get_vocab(node_code)
                if vocab not in vocabularies and node_code != code:
                    continue
            
            parent_codes = {idx_to_code_map.get(p) for p in parent_indices_set if idx_to_code_map.get(p)}
            
            # Filter parents by vocabulary
            if vocabularies:
                def _get_vocab(c: str) -> str:
                    return c.split("/")[0] if "/" in c else c
                parent_codes = {p for p in parent_codes if _get_vocab(p) in vocabularies or p == code}
            
            subgraph[node_code] = parent_codes
        
        return subgraph
    
    def get_all_paths_to_ancestor(
        self,
        code: str,
        ancestor_code: str,
        vocabularies: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Get all paths from code to ancestor_code.
        
        Args:
            code: Starting code
            ancestor_code: Target ancestor code
            vocabularies: Optional list of vocabulary prefixes to filter by
            max_depth: Optional maximum depth
        
        Returns:
            List of paths, each path is a list of codes from code to ancestor_code
            Sorted by length (longest first)
        """
        start_idx = self._get_idx_from_code(code)
        ancestor_idx = self._get_idx_from_code(ancestor_code)
        
        if start_idx is None or ancestor_idx is None:
            return []
        
        # Find all paths in index space
        paths_indices = self._find_all_paths(start_idx, ancestor_idx, max_depth)
        
        # Materialize codes for all paths
        all_indices = set()
        for path in paths_indices:
            all_indices.update(path)
        idx_to_code_map = self._materialize_codes_batch(all_indices)
        
        # Convert paths to codes
        paths_codes = []
        for path_indices in paths_indices:
            path_codes = [idx_to_code_map.get(idx) for idx in path_indices if idx_to_code_map.get(idx)]
            if len(path_codes) == len(path_indices):  # All codes found
                # Apply vocabulary filter if specified
                if vocabularies:
                    def _get_vocab(c: str) -> str:
                        return c.split("/")[0] if "/" in c else c
                    if all(_get_vocab(c) in vocabularies for c in path_codes):
                        paths_codes.append(path_codes)
                else:
                    paths_codes.append(path_codes)
        
        # Sort by length (longest first)
        paths_codes.sort(key=len, reverse=True)
        
        return paths_codes
    
    def get_longest_path_to_ancestor(
        self,
        code: str,
        ancestor_code: str,
        vocabularies: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Get the longest path from code to ancestor_code.
        
        Args:
            code: Starting code
            ancestor_code: Target ancestor code
            vocabularies: Optional list of vocabulary prefixes to filter by
        
        Returns:
            Longest path as list of codes, or None if no path exists
        """
        all_paths = self.get_all_paths_to_ancestor(code, ancestor_code, vocabularies)
        return all_paths[0] if all_paths else None
    
    def __len__(self) -> int:
        """Return number of concepts."""
        return len(self.code_to_idx) if self.code_to_idx else 0
    
    @staticmethod
    def _build_sparse_graph_from_parquet(
        parquet_path: str,
        code_to_idx: Dict[str, int],
    ) -> sp.csr_matrix:
        """Build sparse matrix from parquet files."""
        print("Reading parents parquet...")
        parents_table = pq.read_table(os.path.join(parquet_path, "parents.parquet"))
        
        n_nodes = len(code_to_idx)
        print(f"Building sparse matrix for {n_nodes:,} nodes...")
        
        # Build COO (Coordinate) format first (efficient for building)
        rows = []
        cols = []
        
        parent_codes = parents_table["code"].to_pylist()
        parent_lists = parents_table["parents"].to_pylist()
        
        edge_count = 0
        for code, parents in zip(parent_codes, parent_lists):
            if code not in code_to_idx:
                continue
            
            child_idx = code_to_idx[code]
            
            if parents:
                # Handle numpy arrays/pandas Series
                if hasattr(parents, 'tolist'):
                    parents = parents.tolist()
                elif not isinstance(parents, list):
                    parents = list(parents) if parents else []
                
                for parent in parents:
                    if parent in code_to_idx:
                        parent_idx = code_to_idx[parent]
                        rows.append(child_idx)
                        cols.append(parent_idx)
                        edge_count += 1
        
        print(f"Found {edge_count:,} edges")
        
        # Convert to CSR format (efficient for row access)
        # Use int8 instead of bool_ to avoid pickling issues when saving compressed
        data = np.ones(edge_count, dtype=np.int8)
        matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        print(f"Sparse matrix built: {matrix.nnz:,} non-zero entries")
        print(f"Matrix size: {matrix.shape}, density: {matrix.nnz / (n_nodes * n_nodes) * 100:.4f}%")
        
        return matrix
    
    @staticmethod
    def _save_sparse_graph(
        matrix: sp.csr_matrix,
        code_to_idx: Dict[str, int],
        path: str,
        relationship_types_map: Optional[Dict[Tuple[int, int], Set[str]]] = None,
    ):
        """Save sparse graph to disk using numpy's native format.
        
        Args:
            matrix: Sparse matrix
            code_to_idx: Code to index mapping
            path: Base path for saving files
            relationship_types_map: Optional mapping of (child_idx, parent_idx) -> set of relationship types
        """
        import numpy as np
        
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save matrix using scipy's native format (much faster than pickle)
        # Normalize path: if .npz is passed directly, use it; otherwise convert .pkl.gz to .npz
        if path.endswith(".npz"):
            matrix_path = path
        else:
            matrix_path = path.replace(".pkl.gz", ".npz")
        
        print(f"Saving sparse matrix to {matrix_path}...")
        # Save CSR matrix components directly using numpy (bypasses scipy's save_npz
        # which has pickle compatibility issues with numpy 2.x)
        np.savez_compressed(
            matrix_path,
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=np.array(matrix.shape, dtype=np.int64),
        )
        
        # Save relationship type metadata if provided
        if relationship_types_map:
            # Derive relationship types path from matrix path (not original path)
            base_path = matrix_path.replace(".npz", "")
            rel_types_path = f"{base_path}_relationship_types.pkl.gz"
            print(f"Saving relationship type metadata to {rel_types_path}...")
            with gzip.open(rel_types_path, "wb") as f:
                pickle.dump(relationship_types_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Don't save codes_array or code_to_idx - we can query metadata on-demand
        # Graph operations work entirely in index space (0, 1, 2, ...)
        # Codes are materialized only when needed by querying the parquet metadata
        
        # Calculate total size
        total_size = 0
        if os.path.exists(matrix_path):
            total_size += os.path.getsize(matrix_path)
        if relationship_types_map and os.path.exists(rel_types_path):
            total_size += os.path.getsize(rel_types_path)
        
        print(f"Sparse graph saved ({total_size / 1024 / 1024:.1f} MB total)")
    
    @staticmethod
    def _load_sparse_graph(path: str):
        """Load sparse graph from disk using numpy's native format.
        
        Args:
            path: Base path (can be .pkl.gz or .npz - will be normalized)
        
        Returns:
            Tuple of (matrix, code_to_idx, code_trie, codes_array, relationship_types_map)
        """
        # Normalize path: if .npz is passed directly, use it; otherwise convert .pkl.gz to .npz
        if path.endswith(".npz"):
            matrix_path = path
            base_path = path.replace(".npz", ".pkl.gz")  # For relationship types path
        else:
            matrix_path = path.replace(".pkl.gz", ".npz")
            base_path = path
        
        if not os.path.exists(matrix_path):
            # Fallback: try old pickle format
            if os.path.exists(path) and path.endswith(".pkl.gz"):
                raise ValueError(
                    f"Old pickle format detected at {path}. "
                    f"Please rebuild the graph to use the new numpy format."
                )
            raise FileNotFoundError(
                f"Sparse matrix file not found: {matrix_path}\n"
                f"Expected path: {matrix_path}\n"
                f"Base path provided: {path}\n"
                f"Please rebuild the sparse graph by running:\n"
                f"  python scripts/rebuild_sparse_graph.py --athena_path data/athena_ontologies_snapshot.zip --parquet_path data/athena_omop_ontologies --graph_path {matrix_path}"
            )
        
        matrix_start = time.time()
        try:
            # Load CSR matrix components directly using numpy
            # allow_pickle=True is safe here because we're loading files we created ourselves
            with np.load(matrix_path, allow_pickle=True) as loaded:
                # Check if this is our new format (has 'data', 'indices', 'indptr', 'shape')
                if 'data' in loaded and 'indices' in loaded and 'indptr' in loaded:
                    matrix = sp.csr_matrix(
                        (loaded['data'], loaded['indices'], loaded['indptr']),
                        shape=tuple(loaded['shape']),
                    )
                else:
                    # Old scipy format - try scipy's loader
                    raise ValueError("Old format detected")
        except ValueError as e:
            if "pickled" in str(e).lower() or "pickle" in str(e).lower() or "Old format" in str(e):
                # Try scipy's loader with allow_pickle (for old files)
                try:
                    # scipy.sparse.load_npz doesn't support allow_pickle, so we can't load old files
                    raise ValueError(
                        f"The sparse graph file {matrix_path} is in an old or incompatible format.\n"
                        f"File size: {os.path.getsize(matrix_path) if os.path.exists(matrix_path) else 0} bytes\n\n"
                        f"Please delete and rebuild:\n"
                        f"  rm {matrix_path}\n"
                        f"  python scripts/rebuild_sparse_graph.py --athena_path data/athena_ontologies_snapshot.zip --parquet_path data/athena_omop_ontologies --graph_path {matrix_path}"
                    ) from e
                except Exception:
                    raise
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load sparse graph from {matrix_path}: {e}\n"
                f"Please rebuild:\n"
                f"  rm {matrix_path}\n"
                f"  python scripts/rebuild_sparse_graph.py --athena_path data/athena_ontologies_snapshot.zip --parquet_path data/athena_omop_ontologies --graph_path {matrix_path}"
            ) from e
        matrix_time = time.time() - matrix_start
        
        # Load relationship type metadata if available
        # Derive relationship types path from matrix path
        base_name = matrix_path.replace(".npz", "")
        rel_types_path = f"{base_name}_relationship_types.pkl.gz"
        relationship_types_map = None
        if os.path.exists(rel_types_path):
            try:
                rel_start = time.time()
                with gzip.open(rel_types_path, "rb") as f:
                    relationship_types_map = pickle.load(f)
                rel_time = time.time() - rel_start
                print(f"  Breakdown: Matrix={matrix_time:.2f}s, RelationshipTypes={rel_time:.2f}s")
            except Exception:
                # Silently ignore - relationship types metadata not available
                print(f"  Breakdown: Matrix={matrix_time:.2f}s")
        else:
            print(f"  Breakdown: Matrix={matrix_time:.2f}s")
        
        # Don't load codes_array - we'll query metadata on-demand
        # Graph operations work entirely in index space
        codes_array = None
        code_to_idx = None
        code_trie = None  # Not using trie anymore
        
        return matrix, code_to_idx, code_trie, codes_array, relationship_types_map
