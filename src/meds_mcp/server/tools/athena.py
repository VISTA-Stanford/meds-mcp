import os
import collections
import zipfile
import io
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
from typing import Dict, Optional, Iterable, Set, Type, Any, Union, List, Tuple
import re
import pickle
import hashlib

try:
    import bm25s
    import Stemmer  # PyStemmer for stemming
    HAS_BM25S = True
    print("✅ BM25S libraries imported successfully")
except ImportError:
    HAS_BM25S = False


class AthenaFileReader:
    """Helper class to read files from either a zip archive or directory."""

    def __init__(self, athena_path: str):
        self.athena_path = athena_path
        self.is_zip = athena_path.lower().endswith(".zip")
        self.zip_file = None
        self.parent_dir = None

        if self.is_zip:
            self.zip_file = zipfile.ZipFile(athena_path, "r")
            self._detect_parent_directory()
        else:
            self._detect_parent_directory()

    def _detect_parent_directory(self):
        """Detect if files are in a parent directory and find the correct path."""
        required_files = [
            "CONCEPT.csv",
            "CONCEPT_RELATIONSHIP.csv",
            "CONCEPT_ANCESTOR.csv",
        ]

        if self.is_zip:
            # Check zip contents for parent directory
            zip_contents = self.zip_file.namelist()

            # Look for files directly in root or in a parent directory
            for file_path in zip_contents:
                filename = os.path.basename(file_path)
                if filename in required_files:
                    # Found a required file, check if it's in a parent directory
                    dir_path = os.path.dirname(file_path)
                    if dir_path and not self.parent_dir:
                        # Check if all required files are in this directory
                        all_files_present = all(
                            os.path.join(dir_path, req_file) in zip_contents
                            for req_file in required_files
                        )
                        if all_files_present:
                            self.parent_dir = dir_path
                            break
        else:
            # Check directory structure
            if os.path.isdir(self.athena_path):
                # Check if files are directly in the directory
                files_in_root = [
                    f for f in os.listdir(self.athena_path) if f in required_files
                ]

                if len(files_in_root) == len(required_files):
                    # Files are directly in the root directory
                    self.parent_dir = ""
                else:
                    # Look for a subdirectory containing all required files
                    for item in os.listdir(self.athena_path):
                        item_path = os.path.join(self.athena_path, item)
                        if os.path.isdir(item_path):
                            subdir_files = [
                                f for f in os.listdir(item_path) if f in required_files
                            ]
                            if len(subdir_files) == len(required_files):
                                self.parent_dir = item
                                break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.zip_file:
            self.zip_file.close()

    def preprocess_csv(self, file_path: str) -> str:
        """
        Preprocess Athena CSV files.

        Args:
            file_path: Path to the CSV file to preprocess

        Returns:
            Path to the preprocessed file (or original if no preprocessing needed)
        """
        # Stub for preprocessing, replace with actual logic if needed.
        return file_path

    def read_csv(self, filename: str) -> pl.LazyFrame:
        """Read a CSV file from either zip archive or directory."""
        if self.is_zip:
            # Read from zip archive
            file_path = filename
            if self.parent_dir:
                file_path = os.path.join(self.parent_dir, filename)

            try:
                with self.zip_file.open(file_path) as file:
                    # Read the content as bytes and decode to string
                    content = file.read().decode("utf-8")
                    # Create a StringIO object for Polars to read from
                    return pl.scan_csv(
                        io.StringIO(content),
                        separator="\t",
                        infer_schema_length=0,
                        quote_char=None,
                    )
            except KeyError:
                raise FileNotFoundError(f"File {file_path} not found in zip archive")
        else:
            # Read from directory
            if self.parent_dir:
                file_path = os.path.join(self.athena_path, self.parent_dir, filename)
            else:
                file_path = os.path.join(self.athena_path, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")

            cleaned_path = self.preprocess_csv(file_path)
            return pl.scan_csv(
                cleaned_path,
                separator="\t",
                infer_schema_length=0,
                quote_char=None,
            )


class AthenaOntology:
    """
    Full OMOP Ontology with parent-child relationships.
    TODO: Optimize for speed and memory usage.
    """

    def __init__(
        self,
        description_map: Dict[str, str],
        parents_map: Dict[str, Set[str]],
        children_map: Optional[Dict[str, Set[str]]] = None,
    ):
        self.description_map = description_map
        self.parents_map = parents_map
        self.children_map = children_map or self._build_children_map(parents_map)
        
        # Initialize search index (built lazily)
        self._bm25_index = None
        self._corpus_codes = None  # Maps corpus index to code
        self._stemmer = None
        self._cache_dir = None

    def _build_children_map(
        self, parents_map: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        children_map = collections.defaultdict(set)
        for code, parents in parents_map.items():
            for parent in parents:
                children_map[parent].add(code)
        return children_map

    def __len__(self) -> int:
        """Return the number of concepts in the ontology."""
        return len(self.description_map)

    def save_to_parquet(self, file_path: str, compression: str = "zstd"):
        """Save the ontology as Parquet files in the specified directory."""
        # Ensure the directory exists
        os.makedirs(file_path, exist_ok=True)

        # Prepare the Parquet tables
        description_table = pa.Table.from_pydict(
            {
                "code": list(self.description_map.keys()),
                "description": list(self.description_map.values()),
            }
        )
        parents_table = pa.Table.from_pydict(
            {
                "code": list(self.parents_map.keys()),
                "parents": [list(parents) for parents in self.parents_map.values()],
            }
        )

        # Write the tables to files in the specified directory
        pq.write_table(
            description_table,
            os.path.join(file_path, "descriptions.parquet"),
            compression=compression,
        )
        pq.write_table(
            parents_table,
            os.path.join(file_path, "parents.parquet"),
            compression=compression,
        )

    @classmethod
    def load_from_parquet(cls, file_path: str):
        """Load the ontology from Parquet files in the specified directory.
        
        Optimized for large datasets (millions of rows). Uses pandas for efficient
        column access, which is faster than converting entire Arrow tables to Python lists.
        """
        import pandas as pd
        import numpy as np
        
        description_table = pq.read_table(
            os.path.join(file_path, "descriptions.parquet")
        )
        parents_table = pq.read_table(os.path.join(file_path, "parents.parquet"))

        # Convert to pandas - Arrow->Pandas conversion is optimized
        # Then extract as numpy arrays for fast dict creation
        desc_df = description_table.to_pandas()
        description_map = dict(zip(desc_df["code"], desc_df["description"]))
        
        # For parents, convert to pandas and build dict efficiently
        par_df = parents_table.to_pandas()
        parent_map = collections.defaultdict(set)
        # Use iterrows is slow, but we need to convert lists to sets
        # Optimize by using vectorized operations where possible
        for code, parents in zip(par_df["code"], par_df["parents"]):
            # Handle None/NaN and empty lists properly
            if parents is None or (isinstance(parents, (list, np.ndarray)) and len(parents) == 0):
                parent_map[code] = set()
            else:
                # Convert to list first if it's a numpy array or pandas Series
                if isinstance(parents, (np.ndarray, pd.Series)):
                    parents = parents.tolist() if hasattr(parents, 'tolist') else list(parents)
                parent_map[code] = set(parents)
        
        # faster to rebuilt children_map from parent_map
        return cls(description_map, parent_map, children_map=None)

    @classmethod
    def load_from_athena_snapshot(
        cls,
        athena_path: str,
        code_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        ignore_invalid: bool = True,
    ) -> "AthenaOntology":
        """
        Load ontology from Athena snapshot and code metadata.

        Args:
            athena_path: Path to Athena snapshot directory or zip archive (.zip file)
            code_metadata: Optional dictionary mapping codes to metadata including parent codes
            ignore_invalid: If True, skip concepts where invalid_reason is not empty
        """
        print("Load from Athena Vocabulary Snapshot...")
        description_map: Dict[str, str] = {}
        parents_map: Dict[str, Set[str]] = collections.defaultdict(set)
        code_metadata = code_metadata or {}

        try:
            with AthenaFileReader(athena_path) as reader:
                concept_file = reader.read_csv("CONCEPT.csv")

                # Filter out invalid concepts if requested
                if ignore_invalid:
                    concept_file = concept_file.filter(
                        (pl.col("invalid_reason").is_null())
                        | (pl.col("invalid_reason") == "")
                    )

                code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
                description_col = pl.col("concept_name")
                concept_id_col = pl.col("concept_id").cast(pl.Int64)

                processed_concepts = (
                    concept_file.select(
                        [
                            code_col.alias("code"),
                            concept_id_col.alias("concept_id"),
                            description_col.alias("description"),
                            pl.col("standard_concept")
                            .is_null()
                            .alias("is_non_standard"),
                        ]
                    )
                    .collect()
                    .rows()
                )

                concept_id_to_code_map = {}
                non_standard_concepts = set()

                for (
                    code,
                    concept_id,
                    description,
                    is_non_standard,
                ) in processed_concepts:
                    if code and concept_id is not None:
                        concept_id_to_code_map[concept_id] = code
                        if code not in description_map:
                            description_map[code] = description
                        if is_non_standard:
                            non_standard_concepts.add(concept_id)

                # Add OMOP concept_id to description map as OMOP_CONCEPT_ID/concept_id -> concept_name
                df = concept_file.select([concept_id_col, description_col]).collect()

                # Iterate over rows; each row is a tuple (concept_id, description)
                for concept_id, description in df.rows():
                    description_map[f"OMOP_CONCEPT_ID/{concept_id}"] = description

                # Process CONCEPT_RELATIONSHIP.csv
                relationship_file = reader.read_csv("CONCEPT_RELATIONSHIP.csv")
                
                # Filter for "Is a" relationships (hierarchical parent-child)
                is_a_relationships = relationship_file.filter(
                    (pl.col("relationship_id") == "Is a")
                    & (pl.col("concept_id_1") != pl.col("concept_id_2"))
                    & (pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
                )

                for concept_id_1, concept_id_2 in (
                    is_a_relationships.select(
                        pl.col("concept_id_1").cast(pl.Int64),
                        pl.col("concept_id_2").cast(pl.Int64),
                    )
                    .collect()
                    .rows()
                ):
                    # Add "Is a" relationships: concept_id_1 is a child of concept_id_2
                    if (
                        concept_id_1 in concept_id_to_code_map
                        and concept_id_2 in concept_id_to_code_map
                    ):
                        parents_map[concept_id_to_code_map[concept_id_1]].add(
                            concept_id_to_code_map[concept_id_2]
                        )
                
                # Also process "Maps to" relationships for non-standard concepts
                maps_to_relationships = relationship_file.filter(
                    (pl.col("relationship_id") == "Maps to")
                    & (pl.col("concept_id_1") != pl.col("concept_id_2"))
                    & (pl.col("invalid_reason").is_null() | (pl.col("invalid_reason") == ""))
                )

                for concept_id_1, concept_id_2 in (
                    maps_to_relationships.select(
                        pl.col("concept_id_1").cast(pl.Int64),
                        pl.col("concept_id_2").cast(pl.Int64),
                    )
                    .collect()
                    .rows()
                ):
                    # Only add "Maps to" for non-standard concepts
                    if concept_id_1 in non_standard_concepts:
                        if (
                            concept_id_1 in concept_id_to_code_map
                            and concept_id_2 in concept_id_to_code_map
                        ):
                            parents_map[concept_id_to_code_map[concept_id_1]].add(
                                concept_id_to_code_map[concept_id_2]
                            )

                # Process CONCEPT_ANCESTOR.csv
                ancestor_file = reader.read_csv("CONCEPT_ANCESTOR.csv")
                ancestor_file = ancestor_file.filter(
                    pl.col("min_levels_of_separation") == "1"
                )

                for concept_id, parent_concept_id in (
                    ancestor_file.select(
                        pl.col("descendant_concept_id").cast(pl.Int64),
                        pl.col("ancestor_concept_id").cast(pl.Int64),
                    )
                    .collect()
                    .rows()
                ):
                    if (
                        concept_id in concept_id_to_code_map
                        and parent_concept_id in concept_id_to_code_map
                    ):
                        parents_map[concept_id_to_code_map[concept_id]].add(
                            concept_id_to_code_map[parent_concept_id]
                        )

                # Optional code metadata
                for code, code_info in code_metadata.items():
                    if code_info.get("description"):
                        description_map[code] = code_info["description"]
                    if code_info.get("parent_codes"):
                        parents_map[code] = set(code_info["parent_codes"])

            return cls(description_map, parents_map)

        except Exception as e:
            raise RuntimeError(f"Error processing Athena files: {e}")

    def get_description(self, code: str) -> Optional[str]:
        """Get description for a code."""
        return self.description_map.get(code)

    def get_children(self, code: str) -> Set[str]:
        """Get immediate children of a code."""
        return self.children_map.get(code, set())

    def get_parents(
        self,
        code: str,
        relationship_types: Optional[List[str]] = None,
        vocabularies: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        Get immediate parents of a code with optional filtering.
        
        Args:
            code: Code to get parents for
            relationship_types: Optional list of relationship types to include
                              (e.g., ["Is a", "Maps to"]). If None, includes all.
                              Note: Current implementation includes all relationships.
            vocabularies: Optional list of vocabulary prefixes to filter by
                         (e.g., ["ICD10CM", "SNOMED"]). If None, includes all.
        
        Returns:
            Set of parent codes matching the filters
        """
        parents = self.parents_map.get(code, set())
        
        # Apply vocabulary filter if specified
        if vocabularies:
            def _get_vocab(c: str) -> str:
                return c.split("/")[0] if "/" in c else c
            parents = {p for p in parents if _get_vocab(p) in vocabularies}
        
        # Note: Relationship type filtering would require storing relationship
        # type metadata separately. For now, all relationships are included.
        
        return parents

    def get_ancestor_subgraph(
        self,
        code: str,
        vocabularies: Optional[list[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Get ancestor subgraph (parents and their parents) for a given code.

        Args:
            code: The starting code
            vocabularies: Optional list of vocabulary prefixes to include (e.g., ["RxNorm", "ATC"]).
                        Use ["*"] to include all vocabularies (default behavior).
            relationship_types: Optional list of relationship types to include
                              (e.g., ["Is a", "Maps to"]). If None, includes all.
            max_depth: Optional maximum depth to traverse

        Returns:
            NetworkX DiGraph containing the ancestor subgraph
        """
        if code not in self.description_map:
            return nx.DiGraph()

        # Handle vocabulary filtering
        if vocabularies is None or vocabularies == ["*"] or (isinstance(vocabularies, list) and "*" in vocabularies):
            allowed_vocabularies = None  # Allow all vocabularies
        else:
            allowed_vocabularies = set(vocabularies)

        def _get_vocabulary(code: str) -> str:
            """Extract vocabulary from code (e.g., 'RxNorm/123' -> 'RxNorm')."""
            return code.split("/")[0] if "/" in code else code

        def _get_filtered_subgraph(current_code: str, visited: set, G: nx.DiGraph, depth: int = 0):
            """Recursively build filtered subgraph."""
            if current_code in visited:
                return
            if max_depth is not None and depth > max_depth:
                return

            visited.add(current_code)

            # Check vocabulary filtering (but always include the starting node)
            if allowed_vocabularies is not None and current_code != code:
                vocab = _get_vocabulary(current_code)
                if vocab not in allowed_vocabularies:
                    return

            # Add node with metadata
            G.add_node(
                current_code,
                description=self.description_map.get(current_code, ""),
                is_starting_node=(current_code == code),
            )

            # Get parents with vocabulary filtering
            parents = self.get_parents(current_code, vocabularies=vocabularies)
            
            # Add parents
            for parent in parents:
                if parent in self.description_map:  # Ensure parent exists
                    G.add_edge(current_code, parent)
                    _get_filtered_subgraph(parent, visited, G, depth + 1)

        G = nx.DiGraph()
        visited = set()

        _get_filtered_subgraph(code, visited, G)

        return G

    def get_descendant_subgraph(
        self, code: str, vocabularies: Optional[list[str]] = None
    ) -> nx.DiGraph:
        """
        Get descendant subgraph (children and their children) for a given code.

        Args:
            code: The starting code
            vocabularies: Optional list of vocabulary prefixes to include (e.g., ["RxNorm", "ATC"]).
                        Use ["*"] to include all vocabularies (default behavior).

        Returns:
            NetworkX DiGraph containing the descendant subgraph
        """
        if code not in self.description_map:
            return nx.DiGraph()

        # Handle vocabulary filtering
        if vocabularies is None or vocabularies == ["*"] or (isinstance(vocabularies, list) and "*" in vocabularies):
            allowed_vocabularies = None  # Allow all vocabularies
        else:
            allowed_vocabularies = set(vocabularies)

        def _get_vocabulary(code: str) -> str:
            """Extract vocabulary from code (e.g., 'RxNorm/123' -> 'RxNorm')."""
            return code.split("/")[0] if "/" in code else code

        def _get_filtered_subgraph(current_code: str, visited: set, G: nx.DiGraph):
            """Recursively build filtered subgraph."""
            if current_code in visited:
                return

            visited.add(current_code)

            # Check vocabulary filtering (but always include the starting node)
            if allowed_vocabularies is not None and current_code != code:
                vocab = _get_vocabulary(current_code)
                if vocab not in allowed_vocabularies:
                    return

            # Add node with metadata
            G.add_node(
                current_code,
                description=self.description_map.get(current_code, ""),
                is_starting_node=(current_code == code),
            )

            # Add children
            for child in self.get_children(current_code):
                if child in self.description_map:  # Ensure child exists
                    # Check if child should be included based on vocabulary filtering
                    if (
                        allowed_vocabularies is None
                        or _get_vocabulary(child) in allowed_vocabularies
                    ):
                        G.add_edge(
                            child, current_code
                        )  # Note: edge direction is child -> parent
                        _get_filtered_subgraph(child, visited, G)

        G = nx.DiGraph()
        visited = set()

        _get_filtered_subgraph(code, visited, G)

        return G

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search using medical-friendly tokenization.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of stemmed lowercase tokens
        """
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Apply stemming if available
        if self._stemmer and tokens:
            tokens = [self._stemmer.stemWord(token) for token in tokens]
        
        return tokens

    def _get_corpus_hash(self) -> str:
        """
        Generate a hash of the corpus for cache invalidation.
        
        Returns:
            SHA256 hash of the sorted description map
        """
        # Create a deterministic hash based on the corpus content
        sorted_items = sorted(self.description_map.items())
        corpus_str = "\n".join(f"{code}:{desc}" for code, desc in sorted_items)
        return hashlib.sha256(corpus_str.encode('utf-8')).hexdigest()[:16]

    def _get_cache_path(self, cache_dir: str = "data/cache") -> str:
        """
        Get the cache file path for the BM25 index.
        
        Args:
            cache_dir: Directory to store cache files
            
        Returns:
            Path to the cache file
        """
        corpus_hash = self._get_corpus_hash()
        cache_filename = f"bm25_index_{corpus_hash}"
        return os.path.join(cache_dir, cache_filename)

    def _save_index_to_cache(self, cache_dir: str = "data/cache") -> bool:
        """
        Save the BM25 index to cache.
        
        Args:
            cache_dir: Directory to store cache files
            
        Returns:
            True if successfully saved, False otherwise
        """
        if not self._bm25_index or not self._corpus_codes:
            return False
        
        try:
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_path = self._get_cache_path(cache_dir)
            
            # Save BM25 index using bm25s built-in save
            self._bm25_index.save(cache_path)
            
            # Save additional metadata
            metadata = {
                'corpus_codes': self._corpus_codes,
                'corpus_hash': self._get_corpus_hash(),
                'stemmer_available': self._stemmer is not None
            }
            
            with open(f"{cache_path}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"BM25 index cached to: {cache_path}")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save BM25 index to cache: {e}")
            return False

    def _load_index_from_cache(self, cache_dir: str = "data/cache") -> bool:
        """
        Load the BM25 index from cache.
        
        Args:
            cache_dir: Directory to load cache files from
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            cache_path = self._get_cache_path(cache_dir)
            metadata_path = f"{cache_path}_metadata.pkl"
            
            # Check if cache files exist
            if not os.path.exists(metadata_path):
                return False
            
            # Load and verify metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verify corpus hasn't changed
            if metadata.get('corpus_hash') != self._get_corpus_hash():
                print("Cache invalidated: corpus has changed")
                return False
            
            # Load BM25 index
            self._bm25_index = bm25s.BM25.load(cache_path)
            self._corpus_codes = metadata['corpus_codes']
            
            # Initialize stemmer if it was available when cached
            if metadata.get('stemmer_available', False):
                try:
                    self._stemmer = Stemmer.Stemmer('english')
                except:
                    print("Warning: Stemmer was available when cached but not now")
                    self._stemmer = None
            
            print(f"BM25 index loaded from cache: {cache_path}")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load BM25 index from cache: {e}")
            return False

    def _build_search_index(self, cache_dir: str = "data/cache"):
        """Build BM25 search index using bm25s library with caching."""
        if not HAS_BM25S:
            raise ImportError(
                "bm25s library not found. Install with: pip install bm25s PyStemmer"
            )
        
        if self._bm25_index is not None:
            return  # Already built
        
        self._cache_dir = cache_dir
        
        # Try to load from cache first
        if self._load_index_from_cache(cache_dir):
            return
            
        print("Building BM25 search index using bm25s...")
        
        # Initialize stemmer
        try:
            self._stemmer = Stemmer.Stemmer('english')
        except:
            print("Warning: Could not initialize stemmer, using basic tokenization")
            self._stemmer = None
        
        # Prepare corpus
        codes = list(self.description_map.keys())
        descriptions = [self.description_map[code] for code in codes]
        
        # Store mapping from corpus index to code
        self._corpus_codes = codes
        
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(desc) for desc in descriptions]
        
        print(f"Tokenizing {len(descriptions)} descriptions...")
        
        # Build BM25 index
        self._bm25_index = bm25s.BM25()
        self._bm25_index.index(tokenized_corpus)
        
        print(f"BM25 search index built for {len(codes)} descriptions")
        
        # Save to cache for future use
        self._save_index_to_cache(cache_dir)

    def search_descriptions(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Search descriptions using BM25 ranking via bm25s library.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (code, description, score) sorted by score descending
        """
        if not HAS_BM25S:
            print("Warning: bm25s library not available. Install with: pip install bm25s PyStemmer")
            return []
        
        # Build index if not already built
        self._build_search_index()
        
        if not query.strip():
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Use bm25s get_scores method for proper BM25 scoring
        query_scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        import numpy as np
        top_indices = np.argsort(query_scores)[-top_k:][::-1]  # Sort descending
        top_scores = query_scores[top_indices]
        
        # Convert results to our format
        results = []
        
        for idx, score in zip(top_indices, top_scores):
            idx = int(idx)
            if 0 <= idx < len(self._corpus_codes) and score > 0:
                code = self._corpus_codes[idx]
                description = self.description_map[code]
                results.append((code, description, float(score)))
        
        return results

    def set_cache_directory(self, cache_dir: str):
        """
        Set the cache directory for BM25 index storage.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self._cache_dir = cache_dir

    def clear_search_cache(self, cache_dir: Optional[str] = None):
        """
        Clear cached BM25 index files.
        
        Args:
            cache_dir: Directory containing cache files (uses default if None)
        """
        if cache_dir is None:
            cache_dir = self._cache_dir or "data/cache"
        
        try:
            cache_path = self._get_cache_path(cache_dir)
            metadata_path = f"{cache_path}_metadata.pkl"
            
            # Remove cache files if they exist
            for file_path in [metadata_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed cache file: {file_path}")
            
            # Remove bm25s index files (they may be stored as multiple files)
            import glob
            cache_pattern = f"{cache_path}*"
            for file_path in glob.glob(cache_pattern):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed cache file: {file_path}")
            
            print(f"Search cache cleared from: {cache_dir}")
            
        except Exception as e:
            print(f"Warning: Failed to clear search cache: {e}")

    def get_code_metadata(self, code: str) -> Dict[str, Any]:
        """
        Get metadata for a given code.

        Args:
            code: The code to get metadata for

        Returns:
            Dictionary containing code metadata
        """
        if code not in self.description_map:
            return {"error": f"Code {code} not found in ontology"}

        return {
            "code": code,
            "description": self.description_map.get(code, ""),
            "vocabulary": code.split("/")[0],
        }

    def get_graph_metadata(self, G: nx.DiGraph) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all nodes in a graph.

        TODO: Add relationship metadata to edges

        Args:
            G: NetworkX DiGraph containing the ontology subgraph

        Returns:
            Dictionary mapping node codes to their metadata dictionaries
        """
        return {node: self.get_code_metadata(node) for node in G.nodes()}


if __name__ == "__main__":

    ontology = AthenaOntology.load_from_parquet("data/athena_omop_ontologies")
    print("Total concepts:", len(ontology))

    # Test with RxNorm code
    code = "SNOMED/363358000"
    # code = "LOINC/8480-6"
    descendants = ontology.get_descendant_subgraph(
        code, vocabularies=[code.split("/")[0]]
    )
    print(f"Descendant subgraph nodes: {len(descendants.nodes())}")
    print(f"Descendant subgraph edges: {len(descendants.edges())}")

    # Test metadata
    metadata = ontology.get_code_metadata(code)
    print(f"RxNorm code metadata: {metadata}")

    # Test the helper function
    print("\n=== Testing get_graph_nodes_metadata helper ===")
    nodes_metadata = ontology.get_graph_metadata(descendants)
    print(f"Number of nodes with metadata: {len(nodes_metadata)}")

    # Show a few examples
    print("Sample node metadata:")
    for i, (code, metadata) in enumerate(list(nodes_metadata.items())[:3]):
        print(f"  {code}: {metadata}")
        if i >= 2:  # Show only first 3
            break
    
    # Test search functionality with caching
    print("\n=== Testing Search Functionality with Caching ===")
    if HAS_BM25S:
        search_queries = ["blood pressure", "malignant neoplasm", "diabetes"]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            results = ontology.search_descriptions(query, top_k=5)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, (code, description, score) in enumerate(results):
                    print(f"  {i+1}. {code}: {description[:80]}... (score: {score:.3f})")
            else:
                print("  No results found")
    else:
        print("bm25s library not installed. Install with: pip install bm25s PyStemmer")
