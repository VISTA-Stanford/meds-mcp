# Alternative Ontology Representation Options

## Current Problem

Loading 9.2M concepts and 3.3M parent relationships into Python dictionaries takes ~20-30 seconds and uses significant memory. The current `LazyAthenaOntology` uses Polars but still does individual queries with `.collect()` for each operation.

## Option 1: Optimized Polars-Based Representation (Recommended)

### Concept
Keep everything in Polars DataFrames/LazyFrames, use fast joins, and cache frequently accessed patterns.

### Implementation Strategy

```python
class FastPolarsOntology:
    def __init__(self, parquet_path: str):
        # Load as LazyFrames - no materialization until needed
        self.concepts = pl.scan_parquet(f"{parquet_path}/concepts.parquet")
        self.parents = pl.scan_parquet(f"{parquet_path}/parents.parquet")
        
        # Pre-build join indices for fast lookups
        # Store as Arrow tables or memory-mapped parquet
        self._parent_index = None  # Lazy - build on first use
        
    def get_parents(self, code: str) -> Set[str]:
        # Single join operation - no intermediate Python objects
        result = (
            self.concepts
            .filter(pl.col("code") == code)
            .join(
                self.parents,
                left_on="concept_id",
                right_on="child_concept_id",
                how="inner"
            )
            .select("parent_code")
            .collect()
        )
        return set(result["parent_code"].to_list())
    
    def get_ancestor_subgraph(self, code: str, vocabularies: List[str] = None):
        # Use recursive CTE or iterative joins
        # Polars doesn't support recursive CTEs, but we can:
        # 1. Use iterative joins with a depth limit
        # 2. Pre-compute transitive closure for common queries
        # 3. Use a graph library that works with Polars (like NetworkX but backed by Polars)
        pass
```

### Advantages
- ✅ Fast joins (Polars is optimized for this)
- ✅ Memory efficient (lazy evaluation)
- ✅ Can leverage Polars' query optimization
- ✅ Easy to filter by vocabulary, date ranges, etc.
- ✅ Can cache frequently accessed subgraphs

### Disadvantages
- ❌ Polars doesn't support recursive queries natively
- ❌ Still need to materialize results for graph traversal
- ❌ Multiple `.collect()` calls for complex traversals

### Performance Estimate
- Load time: **< 1 second** (just metadata)
- Query time: **10-100ms** per query (depends on join complexity)

---

## Option 2: Pre-Generated Graph in Binary Format

### Concept
Pre-compute the full graph structure and serialize it in a format optimized for fast loading and traversal.

### Implementation Options

#### 2A: NetworkX Graph Serialization

```python
import networkx as nx
import pickle
import gzip

# Pre-generation (one-time)
def save_graph_to_disk(ontology: AthenaOntology, output_path: str):
    G = nx.DiGraph()
    
    # Add all nodes with metadata
    for code, desc in ontology.description_map.items():
        G.add_node(code, description=desc)
    
    # Add all edges
    for code, parents in ontology.parents_map.items():
        for parent in parents:
            G.add_edge(code, parent)
    
    # Serialize with compression
    with gzip.open(f"{output_path}/ontology_graph.pkl.gz", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

# Loading (fast)
def load_graph_from_disk(path: str) -> nx.DiGraph:
    with gzip.open(f"{path}/ontology_graph.pkl.gz", "rb") as f:
        return pickle.load(f)
```

**Performance**: Load time ~5-10 seconds, but then all queries are instant.

#### 2B: Custom Binary Format (Most Optimized)

```python
import struct
import mmap

class BinaryGraphFormat:
    """
    Custom binary format optimized for graph traversal:
    
    Format:
    - Header: magic number, version, node count, edge count
    - Node index: sorted array of (code_hash, offset) for O(log n) lookup
    - Node data: (code_len, code, desc_len, desc, parent_count, parent_offsets...)
    - Edge data: (child_offset, parent_offset)
    
    Uses memory-mapped files for zero-copy access.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.mmap = None
        self.node_index = None
        
    def load(self):
        """Memory-map the file - instant access"""
        with open(self.file_path, "rb") as f:
            self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse header
        magic, version, node_count, edge_count = struct.unpack(">IIQQ", self.mmap[:24])
        
        # Build in-memory index (small - just offsets)
        self.node_index = self._build_index()
    
    def get_parents(self, code: str) -> List[str]:
        """O(log n) lookup + direct memory access"""
        node_offset = self._find_node(code)
        if node_offset is None:
            return []
        
        # Read node data directly from memory-mapped file
        parent_offsets = self._read_parents(node_offset)
        return [self._read_code(offset) for offset in parent_offsets]
```

**Performance**: Load time **< 1 second** (just memory mapping), query time **< 1ms**.

#### 2C: Graph Database Format (e.g., DGL, PyTorch Geometric)

```python
import dgl
import torch

# Pre-generation
def create_dgl_graph(ontology: AthenaOntology) -> dgl.DGLGraph:
    # Create graph with node features (descriptions)
    src_nodes = []
    dst_nodes = []
    
    for code, parents in ontology.parents_map.items():
        code_idx = code_to_idx[code]
        for parent in parents:
            parent_idx = code_to_idx[parent]
            src_nodes.append(code_idx)
            dst_nodes.append(parent_idx)
    
    G = dgl.graph((src_nodes, dst_nodes))
    G.ndata['code'] = torch.tensor([code_to_idx[c] for c in codes])
    G.ndata['description'] = torch.tensor([encode_desc(d) for d in descriptions])
    
    return G

# Save/load
dgl.save_graphs("ontology_graph.bin", [G])
G = dgl.load_graphs("ontology_graph.bin")[0][0]
```

**Performance**: Load time **2-5 seconds**, optimized for GPU acceleration if needed.

---

## Option 3: Hybrid Approach (Best of Both Worlds)

### Concept
Combine Polars for filtering/querying with a pre-computed graph structure for traversal.

```python
class HybridOntology:
    def __init__(self, parquet_path: str, graph_path: str = None):
        # Polars for metadata and filtering
        self.concepts = pl.scan_parquet(f"{parquet_path}/concepts.parquet")
        
        # Pre-computed graph for fast traversal
        if graph_path:
            self.graph = load_graph_from_disk(graph_path)
        else:
            # Build graph on first use, cache it
            self.graph = self._build_graph_from_parquet(parquet_path)
            save_graph_to_disk(self.graph, graph_path)
    
    def get_parents(self, code: str) -> Set[str]:
        # Fast graph lookup
        return set(self.graph.predecessors(code))
    
    def filter_by_vocabulary(self, vocabularies: List[str]) -> pl.DataFrame:
        # Use Polars for efficient filtering
        return (
            self.concepts
            .filter(pl.col("vocabulary_id").is_in(vocabularies))
            .collect()
        )
    
    def get_ancestor_subgraph(self, code: str, vocabularies: List[str] = None):
        # Use graph for traversal
        G = nx.DiGraph()
        visited = set()
        
        def traverse(node):
            if node in visited:
                return
            visited.add(node)
            
            # Filter by vocabulary if needed
            if vocabularies:
                vocab = node.split("/")[0]
                if vocab not in vocabularies and node != code:
                    return
            
            G.add_node(node, description=self.get_description(node))
            for parent in self.graph.predecessors(node):
                G.add_edge(node, parent)
                traverse(parent)
        
        traverse(code)
        return G
```

### Advantages
- ✅ Fast graph traversal (pre-computed structure)
- ✅ Efficient filtering (Polars)
- ✅ Can update Polars data without rebuilding graph
- ✅ Best performance for both query types

---

## Option 4: Graph Database (Overkill but Scalable)

### Concept
Use a proper graph database like Neo4j, ArangoDB, or even SQLite with graph extensions.

```python
# Using SQLite with graph support
import sqlite3

def create_graph_db(ontology: AthenaOntology, db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE nodes (
            code TEXT PRIMARY KEY,
            description TEXT,
            vocabulary_id TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE edges (
            child_code TEXT,
            parent_code TEXT,
            FOREIGN KEY (child_code) REFERENCES nodes(code),
            FOREIGN KEY (parent_code) REFERENCES nodes(code)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_edges_child ON edges(child_code)")
    cursor.execute("CREATE INDEX idx_edges_parent ON edges(parent_code)")
    
    # Insert data
    # ... bulk insert ...
    
    conn.commit()
    conn.close()

# Query
def get_parents(code: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT parent_code FROM edges WHERE child_code = ?
    """, (code,))
    return [row[0] for row in cursor.fetchall()]
```

### Advantages
- ✅ Handles billions of nodes
- ✅ ACID transactions
- ✅ Can query with SQL
- ✅ Persistent storage

### Disadvantages
- ❌ Overkill for 9M nodes
- ❌ Additional dependency
- ❌ Slower than in-memory for this scale

---

## Recommended Approach: Option 3 (Hybrid)

### Implementation Plan

1. **Keep Polars for metadata** - Fast filtering, vocabulary queries
2. **Pre-compute graph structure** - Save as compressed NetworkX graph or custom binary format
3. **Lazy load graph** - Only load when needed, cache in memory
4. **Use graph for traversal** - Fast ancestor/descendant queries

### Expected Performance

- **Initial load**: < 1 second (just metadata)
- **Graph load** (first use): 5-10 seconds (one-time, then cached)
- **Query time**: < 1ms (graph traversal)
- **Memory**: ~500MB-1GB (graph + metadata)

### Migration Path

1. Add `save_graph()` method to `AthenaOntology`
2. Create `FastHybridOntology` class
3. Update `load_from_parquet()` to optionally load graph
4. Benchmark and compare

Would you like me to implement Option 3 (Hybrid Approach)?
