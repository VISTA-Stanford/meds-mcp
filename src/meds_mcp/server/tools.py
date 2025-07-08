"""
All MCP tools defined in a single module with shared FastMCP instance.
"""

import os
import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import re

from llama_index.core.schema import TextNode, Document, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext
from mcp.server.fastmcp import FastMCP

# Create a single FastMCP instance for all tools
mcp = FastMCP("meds-mcp")

# Import the module to access the global document store
import meds_mcp.server.rag.simple_storage as storage_module

# Initialize Athena ontology at startup
print("Loading Athena ontology from data/athena_omop_ontologies/...")
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
    from athena import AthenaOntology
    parquet_path = os.getenv("ATHENA_VOCABULARIES_PATH", "data/athena_omop_ontologies/")
    _global_ontology = AthenaOntology.load_from_parquet(parquet_path)
    print("Athena ontology loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load Athena ontology: {e}")
    _global_ontology = None

def set_document_store(store):
    """Set the global document store for search tools (for backward compatibility)."""
    pass

def get_document_store():
    """Get the current document store from the storage module."""
    return storage_module._document_store

def parse_timestamp_from_metadata(timestamp_value) -> Optional[datetime.datetime]:
    """Parse timestamp from metadata, handling both string and datetime types."""
    if timestamp_value is None:
        return None
    
    if isinstance(timestamp_value, datetime.datetime):
        return timestamp_value
    
    if isinstance(timestamp_value, str):
        # Try ISO format first
        try:
            return datetime.datetime.fromisoformat(timestamp_value)
        except ValueError:
            pass
        
        # Try simple format
        try:
            return datetime.datetime.strptime(timestamp_value, "%Y-%m-%d %H:%M")
        except ValueError:
            pass
    
    return None

class SortOrder(Enum):
    RELEVANCE = "relevance"
    DATE_ASC = "date_asc"
    DATE_DESC = "date_desc"

@dataclass
class SearchFilters:
    """Filters for patient search operations."""
    start: Optional[datetime.datetime] = None
    end: Optional[datetime.datetime] = None
    max_encounters: Optional[int] = None
    event_types: Optional[List[str]] = None
    exclude_empty_events: bool = False
    min_text_length: Optional[int] = None
    max_results: Optional[int] = None
    sort_by: SortOrder = SortOrder.RELEVANCE
    # Generic attribute filters - allows filtering on any XML attributes
    attribute_filters: Optional[Dict[str, Any]] = None

    @property
    def time_range(self) -> Optional[tuple[datetime.datetime, datetime.datetime]]:
        """Get time_range as tuple if both start and end are set."""
        if self.start is not None and self.end is not None:
            return (self.start, self.end)
        return None

class PatientTimelineRetriever:
    """Patient timeline retriever using document store and BM25 retriever."""
    
    def __init__(self, document_store):
        self.document_store = document_store
        self.global_nodes = document_store.global_nodes
        self.bm25_retrievers = document_store.bm25_retrievers
        self.patients = document_store.patients
    
    def search(
        self,
        query: str,
        person_id: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search patient events with optional filters.

        Args:
            query: Search query string
            person_id: Optional person_id to search within specific patient
            filters: Optional SearchFilters object to constrain the search

        Returns:
            List of search results with event data
        """
        if filters is None:
            filters = SearchFilters()
        
        # Get retriever to use
        if person_id:
            # Search within specific patient
            if person_id not in self.bm25_retrievers:
                return []
            retriever = self.bm25_retrievers[person_id]
        else:
            # Search across all patients using global nodes
            if not self.global_nodes:
                return []
            retriever = BM25Retriever(nodes=self.global_nodes, similarity_top_k=len(self.global_nodes))
        
        # Get initial search results
        initial_results = retriever.retrieve(query)
        
        # Apply filters
        filtered_results = self._apply_filters(initial_results, filters)
        
        # Sort results
        sorted_results = self._sort_results(filtered_results, filters.sort_by)
        
        # Limit results
        if filters.max_results:
            sorted_results = sorted_results[:filters.max_results]
        
        # Convert to dictionary format
        return [
            {
                "id": result.node.node_id,
                "content": result.node.text,
                "metadata": result.node.metadata,
                "timestamp": result.node.metadata.get('timestamp'),
                "event_type": result.node.metadata.get('event_type'),
                "code": result.node.metadata.get('code'),
                "name": result.node.metadata.get('name'),
                "person_id": result.node.metadata.get('person_id'),
                "score": result.score,
            }
            for result in sorted_results
        ]
    
    def _apply_filters(self, results, filters: SearchFilters):
        """Apply all filters to search results."""
        filtered = results
        
        # 1. Apply time range filter
        if filters.time_range:
            start, end = filters.time_range
            filtered = [
                result for result in filtered
                if result.node.metadata.get('timestamp') and 
                parse_timestamp_from_metadata(result.node.metadata['timestamp']) and
                start <= parse_timestamp_from_metadata(result.node.metadata['timestamp']) <= end
            ]
        
        # 2. Apply encounter filtering
        if filters.max_encounters:
            filtered = self._limit_by_encounters(filtered, filters.max_encounters)
        
        # 3. Apply event type filtering
        if filters.event_types:
            filtered = [
                result for result in filtered
                if result.node.metadata.get('event_type') in filters.event_types
            ]
        
        # 4. Apply attribute filters (for XML attributes like code, name, unit, etc.)
        if filters.attribute_filters:
            filtered = self._apply_attribute_filters(filtered, filters.attribute_filters)
        
        # 5. Apply other filters
        if filters.exclude_empty_events:
            filtered = [
                result for result in filtered
                if result.node.text.strip()
            ]
        
        if filters.min_text_length:
            filtered = [
                result for result in filtered
                if len(result.node.text) >= filters.min_text_length
            ]
        
        return filtered
    
    def _apply_attribute_filters(self, results, attribute_filters: Dict[str, Any]):
        """Apply filters on XML attributes like code, name, unit, etc."""
        import re
        
        filtered = results
        for attr_name, attr_value in attribute_filters.items():
            if isinstance(attr_value, str):
                # Case-insensitive exact match
                filtered = [
                    result for result in filtered
                    if result.node.metadata.get(attr_name) and 
                    result.node.metadata[attr_name].lower() == attr_value.lower()
                ]
            elif isinstance(attr_value, list):
                # Case-insensitive match against any value in the list
                filtered = [
                    result for result in filtered
                    if result.node.metadata.get(attr_name) and 
                    result.node.metadata[attr_name].lower() in [v.lower() for v in attr_value]
                ]
            elif hasattr(attr_value, 'pattern'):  # Regex pattern
                # Regex match
                filtered = [
                    result for result in filtered
                    if result.node.metadata.get(attr_name) and 
                    re.search(attr_value, result.node.metadata[attr_name])
                ]
        
        return filtered
    
    def _limit_by_encounters(self, results, max_encounters: int):
        """Limit results to a maximum number of encounters."""
        if not results:
            return results
        
        # Group by encounter (assuming encounter_id in metadata)
        encounter_groups = {}
        for result in results:
            encounter_id = result.node.metadata.get('encounter_id', 'unknown')
            if encounter_id not in encounter_groups:
                encounter_groups[encounter_id] = []
            encounter_groups[encounter_id].append(result)
        
        # Take top encounters
        sorted_encounters = sorted(encounter_groups.items(), 
                                 key=lambda x: max(r.score for r in x[1]), 
                                 reverse=True)
        
        limited_results = []
        for encounter_id, encounter_results in sorted_encounters[:max_encounters]:
            limited_results.extend(encounter_results)
        
        return limited_results
    
    def _sort_results(self, results, sort_by: SortOrder):
        """Sort results based on the specified sort order."""
        if sort_by == SortOrder.RELEVANCE:
            # Already sorted by BM25 score
            return results
        elif sort_by == SortOrder.DATE_ASC:
            return sorted(results, key=lambda r: parse_timestamp_from_metadata(r.node.metadata.get('timestamp')) or datetime.datetime.min)
        elif sort_by == SortOrder.DATE_DESC:
            return sorted(results, key=lambda r: parse_timestamp_from_metadata(r.node.metadata.get('timestamp')) or datetime.datetime.min, reverse=True)
        else:
            return results
    
    def get_events_by_type(self, event_type: str, person_id: Optional[str] = None, filters: Optional[SearchFilters] = None) -> List[Dict[str, Any]]:
        """Get events by type with optional filtering."""
        if filters is None:
            filters = SearchFilters()
        
        # Get all events for the specified type
        all_events = []
        
        if person_id:
            # Get events for specific patient
            if person_id not in self.patients:
                return []
            patient = self.patients[person_id]
            for node in patient.nodes:
                if node.metadata.get('event_type') == event_type:
                    all_events.append(NodeWithScore(node=node, score=1.0))
        else:
            # Get events across all patients
            for node in self.global_nodes:
                if node.metadata.get('event_type') == event_type:
                    all_events.append(NodeWithScore(node=node, score=1.0))
        
        # Apply filters
        filtered_results = self._apply_filters(all_events, filters)
        
        # Sort results
        sorted_results = self._sort_results(filtered_results, filters.sort_by)
        
        # Limit results
        if filters.max_results:
            sorted_results = sorted_results[:filters.max_results]
        
        # Convert to dictionary format
        return [
            {
                "id": result.node.node_id,
                "content": result.node.text,
                "metadata": result.node.metadata,
                "timestamp": result.node.metadata.get('timestamp'),
                "event_type": result.node.metadata.get('event_type'),
                "code": result.node.metadata.get('code'),
                "name": result.node.metadata.get('name'),
                "person_id": result.node.metadata.get('person_id'),
                "score": result.score,
            }
            for result in sorted_results
        ]
    
    def get_historical_values(self, attribute_filters: Dict[str, Any], 
                            person_id: Optional[str] = None,
                            filters: Optional[SearchFilters] = None) -> List[Dict[str, Any]]:
        """Get historical values using attribute matching."""
        if filters is None:
            filters = SearchFilters()
        
        # Get all measurement events
        measurement_events = self.get_events_by_type("measurement", person_id, filters)
        
        # Filter by attribute filters
        filtered_events = []
        for event in measurement_events:
            matches = True
            for attr_name, attr_value in attribute_filters.items():
                if isinstance(attr_value, str):
                    if event['metadata'].get(attr_name, '').lower() != attr_value.lower():
                        matches = False
                        break
                elif isinstance(attr_value, list):
                    if event['metadata'].get(attr_name, '').lower() not in [v.lower() for v in attr_value]:
                        matches = False
                        break
            
            if matches:
                filtered_events.append(event)
        
        # Convert to historical values format
        historical_values = []
        for event in filtered_events:
            timestamp = parse_timestamp_from_metadata(event['metadata'].get('timestamp'))
            if timestamp:
                historical_values.append({
                    'timestamp': timestamp,
                    'value': event['content'],
                    'unit': event['metadata'].get('unit'),
                    'code': event['metadata'].get('code'),
                    'name': event['metadata'].get('name'),
                    'person_id': event['person_id'],
                    'node_id': event['id']
                })
        
        # Sort by timestamp
        historical_values.sort(key=lambda x: x['timestamp'])
        
        return historical_values

# MCP Tools

@mcp.tool()
async def search_patient_events(
    query: str,
    person_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search patient events with optional filters.
    
    Args:
        query: Search query string
        person_id: Optional person_id to search within specific patient
        filters: Optional dict with filter parameters:
            - start_date: Start date (ISO format: YYYY-MM-DD or YYYY-MM-DD HH:MM)
            - end_date: End date (ISO format: YYYY-MM-DD or YYYY-MM-DD HH:MM)
            - max_encounters: Maximum number of encounters to include
            - event_types: List of event types to include
            - exclude_empty_events: Whether to exclude events with empty content
            - min_text_length: Minimum text length for events
            - max_results: Maximum number of results to return
            - sort_by: Sort order ("relevance", "date_asc", "date_desc")
            - attribute_filters: Dict of XML attribute filters
        
    Returns:
        List of search results with event data
    """
    document_store = get_document_store()
    if document_store is None:
        raise ValueError("Document store not initialized. Call set_document_store() first.")
    
    # Parse filters dict into SearchFilters object
    search_filters = None
    if filters:
        # Parse dates
        start = None
        end = None
        if filters.get('start_date'):
            start = parse_timestamp_from_metadata(filters['start_date'])
        if filters.get('end_date'):
            end = parse_timestamp_from_metadata(filters['end_date'])
        
        # Parse sort order
        try:
            sort_order = SortOrder(filters.get('sort_by', 'relevance'))
        except ValueError:
            sort_order = SortOrder.RELEVANCE
        
        search_filters = SearchFilters(
            start=start,
            end=end,
            max_encounters=filters.get('max_encounters'),
            event_types=filters.get('event_types'),
            exclude_empty_events=filters.get('exclude_empty_events', False),
            min_text_length=filters.get('min_text_length'),
            max_results=filters.get('max_results'),
            sort_by=sort_order,
            attribute_filters=filters.get('attribute_filters')
        )
    
    # Create retriever and search
    retriever = PatientTimelineRetriever(document_store)
    return retriever.search(query, person_id, search_filters)

@mcp.tool()
async def get_events_by_type(
    event_type: str,
    person_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get events by type with optional filtering.
    
    Args:
        event_type: Type of events to retrieve
        person_id: Optional person_id to search within specific patient
        filters: Optional dict with filter parameters (same as search_patient_events)
        
    Returns:
        List of events of the specified type
    """
    document_store = get_document_store()
    if document_store is None:
        raise ValueError("Document store not initialized. Call set_document_store() first.")
    
    # Parse filters dict into SearchFilters object
    search_filters = None
    if filters:
        # Parse dates
        start = None
        end = None
        if filters.get('start_date'):
            start = parse_timestamp_from_metadata(filters['start_date'])
        if filters.get('end_date'):
            end = parse_timestamp_from_metadata(filters['end_date'])
        
        # Parse sort order
        try:
            sort_order = SortOrder(filters.get('sort_by', 'relevance'))
        except ValueError:
            sort_order = SortOrder.RELEVANCE
        
        search_filters = SearchFilters(
            start=start,
            end=end,
            max_encounters=filters.get('max_encounters'),
            event_types=filters.get('event_types'),
            exclude_empty_events=filters.get('exclude_empty_events', False),
            min_text_length=filters.get('min_text_length'),
            max_results=filters.get('max_results'),
            sort_by=sort_order,
            attribute_filters=filters.get('attribute_filters')
        )
    
    # Create retriever and get events
    retriever = PatientTimelineRetriever(document_store)
    return retriever.get_events_by_type(event_type, person_id, search_filters)

@mcp.tool()
async def get_historical_values(
    attribute_filters: Dict[str, Any],
    person_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get historical values using attribute matching.
    
    Args:
        attribute_filters: Dict for attribute matching (e.g., {"code": "LOINC/2160-0", "name": "Creatinine"})
        person_id: Optional person_id to search within specific patient
        filters: Optional dict with filter parameters (same as search_patient_events)
        
    Returns:
        List of observations with timestamps, sorted chronologically
    """
    document_store = get_document_store()
    if document_store is None:
        raise ValueError("Document store not initialized. Call set_document_store() first.")
    
    # Parse filters dict into SearchFilters object
    search_filters = None
    if filters:
        # Parse dates
        start = None
        end = None
        if filters.get('start_date'):
            start = parse_timestamp_from_metadata(filters['start_date'])
        if filters.get('end_date'):
            end = parse_timestamp_from_metadata(filters['end_date'])
        
        # Parse sort order
        try:
            sort_order = SortOrder(filters.get('sort_by', 'date_asc'))
        except ValueError:
            sort_order = SortOrder.DATE_ASC
        
        search_filters = SearchFilters(
            start=start,
            end=end,
            max_encounters=filters.get('max_encounters'),
            event_types=filters.get('event_types'),
            exclude_empty_events=filters.get('exclude_empty_events', False),
            min_text_length=filters.get('min_text_length'),
            max_results=filters.get('max_results'),
            sort_by=sort_order
        )
    
    # Create retriever and get historical values
    retriever = PatientTimelineRetriever(document_store)
    return retriever.get_historical_values(attribute_filters, person_id, search_filters)

# Document store tools
@mcp.tool()
async def load_patient_xml(filepath: str, chunk_element: str = "event") -> str:
    """
    Load a patient document from XML file.
    
    Args:
        filepath: Path to the XML file
        chunk_element: XML element to chunk on (default: "event")
        
    Returns:
        Person ID of the loaded document
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return storage_module._document_store.load_patient_document(filepath, chunk_element)

@mcp.tool()
async def load_patient_timeline(person_id: str, chunk_element: str = "event") -> str:
    """
    Load a patient timeline by person_id from the configured data directory.
    
    Args:
        person_id: Patient identifier (filename without extension)
        chunk_element: XML element to chunk on (default: "event")
        
    Returns:
        Person ID of the loaded document
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    if storage_module._data_dir is None:
        raise RuntimeError("Data directory not configured. Call initialize_document_store with data_dir parameter.")
    
    return storage_module._document_store.load_patient_timeline(person_id, storage_module._data_dir, chunk_element)

@mcp.tool()
async def get_patient_event(person_id: str, node_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific event by node_id for a patient.
    
    Args:
        person_id: Patient identifier
        node_id: Event node identifier
        
    Returns:
        Event data and metadata, or None if not found
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    event = storage_module._document_store.get_event_by_node_id(person_id, node_id)
    if not event:
        return None
    
    return {
        'node_id': event.node_id,
        'text': event.text,
        'metadata': event.metadata,
        'person_id': person_id
    }

@mcp.tool()
async def list_patients() -> List[Dict[str, Any]]:
    """
    List all loaded patients.
    
    Returns:
        List of patient metadata
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return storage_module._document_store.list_patients()

@mcp.tool()
async def get_document_store_stats() -> Dict[str, Any]:
    """
    Get document store statistics.
    
    Returns:
        Store statistics
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return storage_module._document_store.get_stats()

@mcp.tool()
async def list_all_node_ids() -> List[Dict[str, Any]]:
    """
    List all node_ids in the document store.
    
    Returns:
        List of all node_ids with metadata including person_id and text preview
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return storage_module._document_store.list_all_node_ids()

@mcp.tool()
async def list_patient_node_ids(person_id: str) -> List[Dict[str, Any]]:
    """
    List all node_ids for a specific patient.
    
    Args:
        person_id: Patient identifier
        
    Returns:
        List of node_ids for the specified patient with metadata
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return storage_module._document_store.list_patient_node_ids(person_id)

@mcp.tool()
async def get_patient_events(person_id: str) -> List[Dict[str, Any]]:
    """
    Get all events for a specific patient.
    
    Args:
        person_id: Patient identifier
        
    Returns:
        List of all events for the patient with full text and metadata
    """
    if storage_module._document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return storage_module._document_store.get_patient_events(person_id)

# Ontology tools
@mcp.tool()
def get_code_metadata(code: str) -> dict:
    """
    Get the metadata for a code.
    """
    if _global_ontology is None:
        return {"error": "Athena ontology not available"}
    
    return {"text": _global_ontology.get_description(code)}

@mcp.tool()
def get_ancestor_subgraph(code: str, vocabularies: list[str] = None) -> dict:
    """
    Get the ancestor subgraph of a code, optionally restricted to specific vocabularies.
    
    Args:
        code: The starting medical code
        vocabularies: List of allowed vocabularies (e.g., ['RxNorm', 'ATC']). 
                     Use '*' to allow all vocabularies. Default is None (all vocabularies).
    """
    if _global_ontology is None:
        return {"error": "Athena ontology not available"}
    
    G = _global_ontology.get_ancestor_subgraph(code, vocabularies)
    return _global_ontology.get_graph_metadata(G)

@mcp.tool()
def get_descendant_subgraph(code: str, vocabularies: list[str] = None) -> dict:
    """
    Get the descendant subgraph of a code.
    """
    if _global_ontology is None:
        return {"error": "Athena ontology not available"}
    
    G = _global_ontology.get_descendant_subgraph(code, vocabularies)
    return _global_ontology.get_graph_metadata(G) 