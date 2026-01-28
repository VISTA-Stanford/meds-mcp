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
    ):
        self.concepts_df = concepts_df
        self.code_to_idx = code_to_idx
        self.idx_to_code = idx_to_code
        self._parent_matrix = parent_matrix
        self.graph_path = graph_path
        self._description_cache: Dict[str, str] = {}
    
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
        start_time = time.time()
        
        # Load concepts metadata as LazyFrame (not materialized)
        concepts_table = pq.read_table(os.path.join(parquet_path, "descriptions.parquet"))
        concepts_df = pl.from_arrow(concepts_table).lazy()
        
        # Build code index (small - just mappings)
        codes = concepts_table["code"].to_pylist()
        code_to_idx = {code: idx for idx, code in enumerate(codes)}
        idx_to_code = {idx: code for idx, code in enumerate(codes)}
        
        print(f"Loaded metadata in {time.time() - start_time:.2f}s")
        
        # Load or build sparse graph
        parent_matrix = None
        if load_graph:
            if graph_path and os.path.exists(graph_path):
                print(f"Loading sparse graph from {graph_path}...")
                graph_start = time.time()
                parent_matrix = cls._load_sparse_graph(graph_path)
                print(f"Sparse graph loaded in {time.time() - graph_start:.2f}s")
            else:
                print("Building sparse graph from parquet (this may take a minute)...")
                graph_start = time.time()
                parent_matrix = cls._build_sparse_graph_from_parquet(parquet_path, code_to_idx)
                if graph_path:
                    cls._save_sparse_graph(parent_matrix, code_to_idx, graph_path)
                    print(f"Sparse graph saved to {graph_path}")
                print(f"Sparse graph built in {time.time() - graph_start:.2f}s")
        
        return cls(
            concepts_df=concepts_df,
            code_to_idx=code_to_idx,
            idx_to_code=idx_to_code,
            parent_matrix=parent_matrix,
            graph_path=graph_path,
        )
    
    @property
    def parent_matrix(self) -> sp.csr_matrix:
        """Lazy-load sparse graph if not already loaded."""
        if self._parent_matrix is None:
            if self.graph_path and os.path.exists(self.graph_path):
                print(f"Lazy-loading sparse graph from {self.graph_path}...")
                self._parent_matrix = self._load_sparse_graph(self.graph_path)
            else:
                raise RuntimeError("Graph not available and no graph_path provided")
        return self._parent_matrix
    
    def get_description(self, code: str) -> Optional[str]:
        """Get description for a code using Polars."""
        if code in self._description_cache:
            return self._description_cache[code]
        
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
        if code not in self.code_to_idx:
            return set()
        
        idx = self.code_to_idx[code]
        matrix = self.parent_matrix
        
        # Get parent indices (non-zero entries in this row)
        parent_indices = matrix[idx].indices
        
        # Convert indices back to codes
        parents = {self.idx_to_code[i] for i in parent_indices if i in self.idx_to_code}
        
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
        if code not in self.code_to_idx:
            return set()
        
        idx = self.code_to_idx[code]
        matrix = self.parent_matrix
        
        # Transpose to get children (non-zero entries in this column)
        # CSR format: transpose means we need to look at columns
        # Use CSC format for efficient column access
        csc_matrix = matrix.tocsc()
        child_indices = csc_matrix[:, idx].indices
        
        return {self.idx_to_code[i] for i in child_indices if i in self.idx_to_code}
    
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
            current_code = self.idx_to_code.get(current_idx)
            if current_code is None:
                continue
            
            # Check vocabulary filter
            if vocabularies and current_code != code:
                vocab = _get_vocabulary(current_code)
                if vocab not in vocabularies:
                    continue
            
            # Get parents
            parent_indices = matrix[current_idx].indices
            parent_codes = {self.idx_to_code[i] for i in parent_indices if i in self.idx_to_code}
            
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
        """Save sparse graph to disk."""
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"Saving sparse graph to {path}...")
        
        # Save as compressed numpy format (much faster than pickle)
        save_dict = {
            "matrix_data": matrix.data,
            "matrix_indices": matrix.indices,
            "matrix_indptr": matrix.indptr,
            "matrix_shape": matrix.shape,
            "code_to_idx": code_to_idx,
        }
        
        with gzip.open(path, "wb") as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.exists(path):
            file_size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"Sparse graph saved ({file_size_mb:.1f} MB)")
    
    @staticmethod
    def _load_sparse_graph(path: str) -> sp.csr_matrix:
        """Load sparse graph from disk."""
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        
        # Reconstruct CSR matrix
        matrix = sp.csr_matrix(
            (data["matrix_data"], data["matrix_indices"], data["matrix_indptr"]),
            shape=data["matrix_shape"]
        )
        
        return matrix
