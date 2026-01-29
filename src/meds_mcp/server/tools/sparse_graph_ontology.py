"""
Optimized ontology representation using sparse matrices for graph storage.

Uses:
- scipy.sparse.csr_matrix for efficient graph storage (much faster than NetworkX pickle)
- Code->index mapping for O(1) lookups
- Memory-mapped or compressed sparse matrix for fast loading
"""

import os
import pickle
import gzip
import time
from pathlib import Path
from typing import Optional, Set, List, Dict, Any
import numpy as np
import scipy.sparse as sp
import polars as pl
import pyarrow.parquet as pq
try:
    import marisa_trie
except ImportError:
    marisa_trie = None


class SparseGraphOntology:
    """
    Ultra-fast ontology representation using sparse matrices.
    
    Graph is stored as:
    - CSR (Compressed Sparse Row) matrix: rows = children, cols = parents
    - code_to_idx: maps code string -> matrix index
    - idx_to_code: maps matrix index -> code string
    
    This is much faster to load than NetworkX graphs (seconds vs minutes).
    """
    
    def __init__(
        self,
        concepts_df: pl.LazyFrame,
        code_to_idx: Dict[str, int],
        idx_to_code: Dict[int, str],
        parent_matrix: Optional[sp.csr_matrix] = None,
        graph_path: Optional[str] = None,
        code_trie: Optional[Any] = None,
        codes_array: Optional[List[str]] = None,
    ):
        self.concepts_df = concepts_df
        self.code_to_idx = code_to_idx  # Fallback dict
        self.idx_to_code = idx_to_code  # Fallback dict
        self._code_trie = code_trie  # Optimized trie for code->idx
        self._codes_array = codes_array  # Optimized array for idx->code
        self._parent_matrix = parent_matrix
        self.graph_path = graph_path
        self._description_cache: Dict[str, str] = {}
    
    def _get_idx(self, code: str) -> Optional[int]:
        """Get index for code using dict (fast enough)."""
        return self.code_to_idx.get(code)
    
    def _get_code(self, idx: int) -> Optional[str]:
        """Get code for index using array if available, else dict."""
        if self._codes_array is not None:
            if 0 <= idx < len(self._codes_array):
                return self._codes_array[idx]
            return None
        if self.idx_to_code is not None:
            return self.idx_to_code.get(idx)
        return None
    
    @classmethod
    def load_from_parquet(
        cls,
        parquet_path: str,
        graph_path: Optional[str] = None,
        load_graph: bool = True,
    ) -> "SparseGraphOntology":
        """
        Load ontology from parquet files with optional sparse graph.
        
        Args:
            parquet_path: Directory containing descriptions.parquet and parents.parquet
            graph_path: Optional path to pre-computed sparse graph file
            load_graph: If True, load graph (lazy if graph_path provided)
        """
        # Keep metadata truly lazy - just store path, scan on demand
        descriptions_path = os.path.join(parquet_path, "descriptions.parquet")
        concepts_df = pl.scan_parquet(descriptions_path)  # Truly lazy, no I/O yet
        
        # Load or build sparse graph
        parent_matrix = None
        loaded_code_to_idx = {}
        loaded_codes_array = None
        
        if load_graph:
            matrix_path = graph_path.replace(".pkl.gz", ".npz") if graph_path else None
            if matrix_path and os.path.exists(matrix_path):
                print(f"Loading sparse graph from {matrix_path}...")
                graph_start = time.time()
                parent_matrix, loaded_code_to_idx, _, loaded_codes_array = cls._load_sparse_graph(graph_path)
                print(f"Sparse graph loaded in {time.time() - graph_start:.2f}s")
            else:
                print("Building sparse graph from parquet (this may take a minute)...")
                graph_start = time.time()
                # Build code_to_idx from parquet (minimal - just codes)
                codes_df = pl.scan_parquet(descriptions_path).select("code").collect()
                codes = codes_df["code"].to_list()
                code_to_idx = {code: idx for idx, code in enumerate(codes)}
                # Build codes_array for fast idx->code lookup
                codes_array = codes  # Already in order
                parent_matrix = cls._build_sparse_graph_from_parquet(parquet_path, code_to_idx)
                if graph_path:
                    cls._save_sparse_graph(parent_matrix, code_to_idx, graph_path)
                    print(f"Sparse graph saved")
                print(f"Sparse graph built in {time.time() - graph_start:.2f}s")
                loaded_code_to_idx = code_to_idx
                loaded_codes_array = codes_array
        
        # Use loaded mappings from graph file
        final_code_to_idx = loaded_code_to_idx
        final_idx_to_code = None  # Not needed if we have codes_array
        
        return cls(
            concepts_df=concepts_df,
            code_to_idx=final_code_to_idx,
            idx_to_code=final_idx_to_code,
            parent_matrix=parent_matrix,
            graph_path=graph_path,
            code_trie=None,
            codes_array=loaded_codes_array,
        )
    
    @property
    def parent_matrix(self) -> sp.csr_matrix:
        """Lazy-load sparse graph if not already loaded."""
        if self._parent_matrix is None:
            if self.graph_path:
                print(f"Lazy-loading sparse graph from {self.graph_path}...")
                matrix, _, _, _ = self._load_sparse_graph(self.graph_path)
                self._parent_matrix = matrix
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
    
    def get_parents(
        self,
        code: str,
        relationship_types: Optional[List[str]] = None,
        vocabularies: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        Get immediate parents with optional filtering.
        
        Args:
            code: Code to get parents for
            relationship_types: Optional list of relationship types to include
                              (e.g., ["Is a", "Maps to"]). If None, includes all.
            vocabularies: Optional list of vocabulary prefixes to filter by
                         (e.g., ["ICD10CM", "SNOMED"]). If None, includes all.
        
        Returns:
            Set of parent codes matching the filters
        """
        idx = self._get_idx(code)
        if idx is None:
            return set()
        
        matrix = self.parent_matrix
        
        # Get parent indices (non-zero entries in this row)
        parent_indices = matrix[idx].indices
        
        # Convert indices back to codes
        parents = set()
        for i in parent_indices:
            parent_code = self._get_code(i)
            if parent_code:
                parents.add(parent_code)
        
        # Apply vocabulary filter if specified
        if vocabularies:
            def _get_vocab(c: str) -> str:
                return c.split("/")[0] if "/" in c else c
            parents = {p for p in parents if _get_vocab(p) in vocabularies}
        
        # Note: Relationship type filtering requires querying original data
        # For now, sparse matrix includes all relationships
        # To filter by relationship type, you'd need to query the original
        # relationship dataframes or store relationship type metadata
        
        return parents
    
    def get_children(self, code: str) -> Set[str]:
        """Get immediate children using sparse matrix transpose."""
        idx = self._get_idx(code)
        if idx is None:
            return set()
        
        matrix = self.parent_matrix
        
        # Transpose to get children (non-zero entries in this column)
        # CSR format: transpose means we need to look at columns
        # Use CSC format for efficient column access
        csc_matrix = matrix.tocsc()
        child_indices = csc_matrix[:, idx].indices
        
        children = set()
        for i in child_indices:
            child_code = self._get_code(i)
            if child_code:
                children.add(child_code)
        return children
    
    def get_ancestor_subgraph(
        self,
        code: str,
        vocabularies: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Set[str]]:
        """
        Get ancestor subgraph using sparse matrix traversal.
        
        Args:
            code: Starting code
            vocabularies: Optional list of vocabulary prefixes to include
            relationship_types: Optional list of relationship types to include
                              (e.g., ["Is a", "Maps to"]). If None, includes all.
            max_depth: Optional maximum depth to traverse
        
        Returns:
            Dictionary mapping codes to their parent sets (subgraph)
        """
        if code not in self.code_to_idx:
            return {}
        
        import networkx as nx
        
        matrix = self.parent_matrix
        start_idx = self.code_to_idx[code]
        
        # Build subgraph using BFS traversal
        subgraph = {}
        visited = set()
        queue = [(start_idx, 0)]  # (index, depth)
        
        def _get_vocabulary(c: str) -> str:
            return c.split("/")[0] if "/" in c else c
        
        while queue:
            current_idx, depth = queue.pop(0)
            
            if current_idx in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            
            visited.add(current_idx)
            current_code = self._get_code(current_idx)
            if current_code is None:
                continue
            
            # Check vocabulary filter
            if vocabularies and current_code != code:
                vocab = _get_vocabulary(current_code)
                if vocab not in vocabularies:
                    continue
            
            # Get parents
            parent_indices = matrix[current_idx].indices
            parent_codes = set()
            for i in parent_indices:
                parent_code = self._get_code(i)
                if parent_code:
                    parent_codes.add(parent_code)
            
            # Filter parents by vocabulary
            if vocabularies:
                parent_codes = {p for p in parent_codes if _get_vocabulary(p) in vocabularies or p == code}
            
            subgraph[current_code] = parent_codes
            
            # Add parents to queue
            for parent_idx in parent_indices:
                if parent_idx not in visited:
                    queue.append((parent_idx, depth + 1))
        
        return subgraph
    
    def __len__(self) -> int:
        """Return number of concepts."""
        return len(self.code_to_idx)
    
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
        data = np.ones(edge_count, dtype=np.bool_)
        matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        print(f"Sparse matrix built: {matrix.nnz:,} non-zero entries")
        print(f"Matrix size: {matrix.shape}, density: {matrix.nnz / (n_nodes * n_nodes) * 100:.4f}%")
        
        return matrix
    
    @staticmethod
    def _save_sparse_graph(
        matrix: sp.csr_matrix,
        code_to_idx: Dict[str, int],
        path: str,
    ):
        """Save sparse graph to disk using numpy's native format and marisa-trie."""
        import numpy as np
        
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save matrix using scipy's native format (much faster than pickle)
        matrix_path = path.replace(".pkl.gz", ".npz")
        print(f"Saving sparse matrix to {matrix_path}...")
        sp.save_npz(matrix_path, matrix, compressed=True)
        
        # Save code_to_idx as simple JSON (Python dicts are already fast)
        idx_path = path.replace(".pkl.gz", "_index.json")
        import json
        print(f"Saving code index to {idx_path}...")
        with open(idx_path, "w") as f:
            json.dump(code_to_idx, f)
        
        # Build codes array for idx->code - use compressed format
        # Store as newline-separated text file (much smaller than numpy object array)
        n_codes = len(code_to_idx)
        codes_array = [""] * n_codes
        for code, idx in code_to_idx.items():
            if 0 <= idx < n_codes:
                codes_array[idx] = code
        
        # Save codes as binary pickle (faster than JSON, smaller than numpy object array)
        # Use protocol 5 (binary) for speed
        codes_path = path.replace(".pkl.gz", "_codes.pkl")
        print(f"Saving codes array to {codes_path}...")
        with open(codes_path, "wb") as f:
            pickle.dump(codes_array, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Calculate total size
        total_size = 0
        if os.path.exists(matrix_path):
            total_size += os.path.getsize(matrix_path)
        if os.path.exists(codes_path):
            total_size += os.path.getsize(codes_path)
        if os.path.exists(idx_path):
            total_size += os.path.getsize(idx_path)
        
        print(f"Sparse graph saved ({total_size / 1024 / 1024:.1f} MB total)")
    
    @staticmethod
    def _load_sparse_graph(path: str) -> tuple[sp.csr_matrix, Dict[str, int], Any, List[str]]:
        """Load sparse graph from disk using numpy's native format and marisa-trie."""
        # Load matrix using scipy's native format
        matrix_path = path.replace(".pkl.gz", ".npz")
        if not os.path.exists(matrix_path):
            # Fallback: try old pickle format
            if os.path.exists(path):
                raise ValueError(
                    f"Old pickle format detected at {path}. "
                    f"Please rebuild the graph to use the new numpy format."
                )
            raise FileNotFoundError(f"Sparse matrix file not found: {matrix_path}")
        
        matrix = sp.load_npz(matrix_path)
        
        # Load code_to_idx from JSON (simple and fast)
        idx_path = path.replace(".pkl.gz", "_index.json")
        code_to_idx = {}
        if os.path.exists(idx_path):
            import json
            with open(idx_path, "r") as f:
                code_to_idx = json.load(f)
        
        # Load codes array (for idx->code) - use binary pickle for speed
        codes_array = None
        codes_path_pkl = path.replace(".pkl.gz", "_codes.pkl")
        codes_path_npy = path.replace(".pkl.gz", "_codes.npy")  # Old format fallback
        codes_path_gz = path.replace(".pkl.gz", "_codes.txt.gz")  # Older format fallback
        
        if os.path.exists(codes_path_pkl):
            # Fast binary pickle format
            with open(codes_path_pkl, "rb") as f:
                codes_array = pickle.load(f)
        elif os.path.exists(codes_path_npy):
            # Fallback to old numpy format
            codes_array = np.load(codes_path_npy, allow_pickle=True).tolist()
        elif os.path.exists(codes_path_gz):
            # Fallback to old compressed text format
            print(f"Loading codes array from {codes_path_gz} (old format)...")
            with gzip.open(codes_path_gz, "rt", encoding="utf-8") as f:
                codes_array = [line.rstrip("\n") for line in f]
        
        code_trie = None  # Not using trie anymore
        
        return matrix, code_to_idx, code_trie, codes_array
