"""
Fast ontology representation using hybrid approach:
- Polars for metadata and filtering
- Pre-computed graph structure for fast traversal
"""

import os
import gzip
import pickle
import time
from pathlib import Path
from typing import Optional, Set, List, Dict, Any
import networkx as nx
import polars as pl
import pyarrow.parquet as pq


class FastHybridOntology:
    """
    Hybrid ontology representation optimized for fast loading and traversal.
    
    Uses:
    - Polars DataFrames for metadata (descriptions, vocabulary filtering)
    - Pre-computed NetworkX graph for fast parent/child traversal
    - Lazy graph loading (only loads when needed)
    """
    
    def __init__(
        self,
        concepts_df: pl.LazyFrame,
        code_to_idx: Dict[str, int],
        idx_to_code: Dict[int, str],
        graph: Optional[nx.DiGraph] = None,
        graph_path: Optional[str] = None,
    ):
        self.concepts_df = concepts_df
        self.code_to_idx = code_to_idx
        self.idx_to_code = idx_to_code
        self._graph = graph
        self.graph_path = graph_path
        self._description_cache: Dict[str, str] = {}
    
    @classmethod
    def load_from_parquet(
        cls,
        parquet_path: str,
        graph_path: Optional[str] = None,
        load_graph: bool = True,
    ) -> "FastHybridOntology":
        """
        Load ontology from parquet files with optional graph.
        
        Args:
            parquet_path: Directory containing descriptions.parquet and parents.parquet
            graph_path: Optional path to pre-computed graph file
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
        
        # Load or build graph
        graph = None
        if load_graph:
            if graph_path and os.path.exists(graph_path):
                print(f"Loading pre-computed graph from {graph_path}...")
                graph_start = time.time()
                try:
                    graph = cls._load_graph(graph_path)
                    print(f"Graph loaded in {time.time() - graph_start:.2f}s")
                except Exception as e:
                    print(f"Warning: Failed to load graph from {graph_path}: {e}")
                    print("Building graph from parquet instead...")
                    graph_start = time.time()
                    graph = cls._build_graph_from_parquet(parquet_path, code_to_idx)
                    if graph_path:
                        cls._save_graph(graph, graph_path)
                    print(f"Graph built in {time.time() - graph_start:.2f}s")
            else:
                print("Building graph from parquet (this may take a minute)...")
                graph_start = time.time()
                graph = cls._build_graph_from_parquet(parquet_path, code_to_idx)
                graph_build_time = time.time() - graph_start
                print(f"Graph built in {graph_build_time:.2f}s")
                if graph_path:
                    save_start = time.time()
                    cls._save_graph(graph, graph_path)
                    print(f"Graph saved in {time.time() - save_start:.2f}s")
        
        return cls(
            concepts_df=concepts_df,
            code_to_idx=code_to_idx,
            idx_to_code=idx_to_code,
            graph=graph,
            graph_path=graph_path,
        )
    
    @property
    def graph(self) -> nx.DiGraph:
        """Lazy-load graph if not already loaded."""
        if self._graph is None:
            if self.graph_path and os.path.exists(self.graph_path):
                try:
                    print(f"Lazy-loading graph from {self.graph_path}...")
                    self._graph = self._load_graph(self.graph_path)
                except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                    # Graph file corrupted, rebuild it
                    print("Graph file corrupted, rebuilding...")
                    parquet_path = os.path.dirname(self.graph_path) if self.graph_path else None
                    if parquet_path:
                        self._graph = self._build_graph_from_parquet(parquet_path, self.code_to_idx)
                        if self.graph_path:
                            self._save_graph(self._graph, self.graph_path)
                    else:
                        raise RuntimeError("Cannot rebuild graph: no parquet path available")
            else:
                raise RuntimeError("Graph not available and no graph_path provided")
        return self._graph
    
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
    
    def get_parents(self, code: str) -> Set[str]:
        """Get immediate parents using graph (very fast)."""
        if code not in self.code_to_idx:
            return set()
        
        G = self.graph
        if code not in G:
            return set()
        
        return set(G.predecessors(code))
    
    def get_children(self, code: str) -> Set[str]:
        """Get immediate children using graph (very fast)."""
        if code not in self.code_to_idx:
            return set()
        
        G = self.graph
        if code not in G:
            return set()
        
        return set(G.successors(code))
    
    def get_ancestor_subgraph(
        self,
        code: str,
        vocabularies: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Get ancestor subgraph using graph traversal.
        
        Args:
            code: Starting code
            vocabularies: Optional list of vocabulary prefixes to include
            max_depth: Optional maximum depth to traverse
        
        Returns:
            NetworkX DiGraph containing ancestor subgraph
        """
        if code not in self.code_to_idx:
            return nx.DiGraph()
        
        G = self.graph
        if code not in G:
            return nx.DiGraph()
        
        result = nx.DiGraph()
        visited = set()
        
        def _get_vocabulary(c: str) -> str:
            return c.split("/")[0] if "/" in c else c
        
        def traverse(node: str, depth: int = 0):
            if node in visited:
                return
            if max_depth is not None and depth > max_depth:
                return
            
            visited.add(node)
            
            # Check vocabulary filter (but always include starting node)
            if vocabularies and node != code:
                vocab = _get_vocabulary(node)
                if vocab not in vocabularies:
                    return
            
            # Add node with metadata
            desc = self.get_description(node)
            result.add_node(node, description=desc, is_starting_node=(node == code))
            
            # Traverse parents
            for parent in G.predecessors(node):
                if vocabularies:
                    parent_vocab = _get_vocabulary(parent)
                    if parent_vocab not in vocabularies and parent != code:
                        continue
                
                result.add_edge(node, parent)
                traverse(parent, depth + 1)
        
        traverse(code)
        return result
    
    def get_descendant_subgraph(
        self,
        code: str,
        vocabularies: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> nx.DiGraph:
        """Get descendant subgraph using graph traversal."""
        if code not in self.code_to_idx:
            return nx.DiGraph()
        
        G = self.graph
        if code not in G:
            return nx.DiGraph()
        
        result = nx.DiGraph()
        visited = set()
        
        def _get_vocabulary(c: str) -> str:
            return c.split("/")[0] if "/" in c else c
        
        def traverse(node: str, depth: int = 0):
            if node in visited:
                return
            if max_depth is not None and depth > max_depth:
                return
            
            visited.add(node)
            
            if vocabularies and node != code:
                vocab = _get_vocabulary(node)
                if vocab not in vocabularies:
                    return
            
            desc = self.get_description(node)
            result.add_node(node, description=desc, is_starting_node=(node == code))
            
            for child in G.successors(node):
                if vocabularies:
                    child_vocab = _get_vocabulary(child)
                    if child_vocab not in vocabularies and child != code:
                        continue
                
                result.add_edge(child, node)  # child -> parent direction
                traverse(child, depth + 1)
        
        traverse(code)
        return result
    
    def filter_by_vocabulary(self, vocabularies: List[str]) -> pl.DataFrame:
        """Filter concepts by vocabulary using Polars (fast)."""
        return (
            self.concepts_df
            .filter(pl.col("code").str.starts_with(pl.lit(vocabularies[0])))
            .collect()
        )
    
    def __len__(self) -> int:
        """Return number of concepts."""
        return len(self.code_to_idx)
    
    @staticmethod
    def _build_graph_from_parquet(
        parquet_path: str,
        code_to_idx: Dict[str, int],
    ) -> nx.DiGraph:
        """Build NetworkX graph from parquet files."""
        print("Reading parents parquet...")
        parents_table = pq.read_table(os.path.join(parquet_path, "parents.parquet"))
        
        G = nx.DiGraph()
        
        # Add all nodes first
        print("Adding nodes to graph...")
        for code in code_to_idx.keys():
            G.add_node(code)
        
        # Add edges
        print("Adding edges to graph...")
        parent_codes = parents_table["code"].to_pylist()
        parent_lists = parents_table["parents"].to_pylist()
        
        edge_count = 0
        for code, parents in zip(parent_codes, parent_lists):
            if parents:
                for parent in parents:
                    if parent in code_to_idx:  # Only add if parent exists
                        G.add_edge(code, parent)
                        edge_count += 1
        
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    @staticmethod
    def _save_graph(graph: nx.DiGraph, path: str):
        """Save graph to disk with compression."""
        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save graph
        with gzip.open(path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size after saving
        if os.path.exists(path):
            file_size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"Graph saved to {path} ({file_size_mb:.1f} MB)")
        else:
            print(f"Graph saved to {path}")
    
    @staticmethod
    def _load_graph(path: str) -> nx.DiGraph:
        """Load graph from disk."""
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            # Graph file might be corrupted or incomplete
            print(f"Warning: Could not load graph from {path}: {e}")
            print("Graph file may be corrupted. Will rebuild if needed.")
            raise FileNotFoundError(f"Graph file corrupted or incomplete: {path}")
