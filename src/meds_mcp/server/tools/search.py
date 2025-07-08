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

# Global MCP instance
mcp = FastMCP("search")

# Import the module to access the global document store
import meds_mcp.server.rag.simple_storage as storage_module


def set_document_store(store):
    """Set the global document store for search tools (for backward compatibility)."""
    # This function is kept for backward compatibility but not needed anymore
    # since we access _document_store through the module
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
        """Apply filters on XML attributes like code, name, unit, etc.
        
        Args:
            results: List of search results
            attribute_filters: Dict where keys are attribute names and values can be:
                - str: Exact match (case-insensitive for string attributes)
                - List[str]: Any of the values in the list (case-insensitive for string attributes)
                - Pattern: Regex pattern (import re first)
        """
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
                # Case-insensitive match for any value in the list
                attr_value_lower = [str(v).lower() for v in attr_value]
                filtered = [
                    result for result in filtered
                    if result.node.metadata.get(attr_name) and 
                    str(result.node.metadata[attr_name]).lower() in attr_value_lower
                ]
            elif hasattr(attr_value, 'pattern'):  # Regex pattern
                # Regex match (case-insensitive by default)
                filtered = [
                    result for result in filtered
                    if result.node.metadata.get(attr_name) and 
                    attr_value.search(str(result.node.metadata[attr_name]))
                ]
            else:
                # Try exact match for other types (case-sensitive)
                filtered = [
                    result for result in filtered
                    if result.node.metadata.get(attr_name) == attr_value
                ]
        
        return filtered
    
    def _limit_by_encounters(self, results, max_encounters: int):
        """Limit results to the most recent encounters."""
        if not results:
            return results
        
        # Group by encounter and sort encounters by most recent document
        encounter_groups = {}
        for result in results:
            encounter_id = result.node.metadata.get('encounter_id') or "unknown"
            if encounter_id not in encounter_groups:
                encounter_groups[encounter_id] = []
            encounter_groups[encounter_id].append(result)
        
        # Sort encounters by their most recent document timestamp
        sorted_encounters = sorted(
            encounter_groups.items(),
            key=lambda x: max(
                parse_timestamp_from_metadata(doc.node.metadata.get('timestamp')) or datetime.datetime.min
                for doc in x[1] 
                if doc.node.metadata.get('timestamp')
            ),
            reverse=True
        )
        
        # Take only the most recent encounters
        limited_encounters = sorted_encounters[:max_encounters]
        
        # Flatten results from selected encounters
        limited_results = []
        for encounter_id, encounter_results in limited_encounters:
            limited_results.extend(encounter_results)
        
        return limited_results
    
    def _sort_results(self, results, sort_by: SortOrder):
        """Sort results according to specified order."""
        if sort_by == SortOrder.RELEVANCE:
            # Results are already sorted by relevance from BM25 retriever
            return results
        elif sort_by == SortOrder.DATE_ASC:
            return sorted(
                results, 
                key=lambda x: parse_timestamp_from_metadata(x.node.metadata.get('timestamp')) or datetime.datetime.min
            )
        elif sort_by == SortOrder.DATE_DESC:
            return sorted(
                results, 
                key=lambda x: parse_timestamp_from_metadata(x.node.metadata.get('timestamp')) or datetime.datetime.min,
                reverse=True
            )
        else:
            return results
    
    def get_events_by_type(self, event_type: str, person_id: Optional[str] = None, filters: Optional[SearchFilters] = None) -> List[Dict[str, Any]]:
        """Get events by type with optional filtering."""
        if filters is None:
            filters = SearchFilters()
        
        # Get nodes to search
        if person_id:
            # Search within specific patient
            if person_id not in self.patients:
                return []
            nodes = self.patients[person_id].nodes
        else:
            # Search across all patients
            nodes = self.global_nodes
        
        # Filter nodes by event type
        type_nodes = [
            node for node in nodes 
            if node.metadata.get('event_type') == event_type
        ]
        
        # Convert to search results for consistent filtering
        results = [NodeWithScore(node=node, score=1.0) for node in type_nodes]
        
        # Apply filters
        filtered_results = self._apply_filters(results, filters)
        
        # Convert to dictionary format
        return [
            {
                "id": result.node.node_id,
                "content": result.node.text,
                "metadata": result.node.metadata,
                "timestamp": result.node.metadata.get('timestamp'),
                "event_type": result.node.metadata.get('event_type'),
                "encounter_id": result.node.metadata.get('encounter_id'),
                "person_id": result.node.metadata.get('person_id'),
            }
            for result in filtered_results
        ]

    def get_historical_values(self, attribute_filters: Dict[str, Any], 
                            person_id: Optional[str] = None,
                            filters: Optional[SearchFilters] = None) -> List[Dict[str, Any]]:
        """
        Get historical values using attribute matching.
        
        Args:
            attribute_filters: Dict for attribute matching (e.g., {"code": "LOINC/2160-0", "name": "Creatinine"})
            person_id: Optional person_id to search within specific patient
            filters: Optional SearchFilters object to constrain the search (time range, etc.)
            
        Returns:
            List of observations with timestamps, sorted chronologically
        """
        if filters is None:
            filters = SearchFilters()
        
        # Get nodes to search
        if person_id:
            # Search within specific patient
            if person_id not in self.patients:
                return []
            matching_nodes = self.patients[person_id].nodes
        else:
            # Search across all patients
            matching_nodes = self.global_nodes
        
        # Apply attribute filters
        if attribute_filters:
            # Convert to search results for consistent filtering
            temp_results = [NodeWithScore(node=node, score=1.0) for node in matching_nodes]
            
            # Apply attribute filtering
            filtered_results = self._apply_attribute_filters(temp_results, attribute_filters)
            matching_nodes = [result.node for result in filtered_results]
        
        # Convert to search results for consistent filtering
        results = [NodeWithScore(node=node, score=1.0) for node in matching_nodes]
        
        # Apply additional filters
        filtered_results = self._apply_filters(results, filters)
        
        # Convert to dictionary format and extract values
        observations = []
        for result in filtered_results:
            # Try to extract the actual value from the content or metadata
            value = None
            
            # Check if there's a 'value' field in metadata
            if 'value' in result.node.metadata:
                value = result.node.metadata['value']
            else:
                # Try to extract from the XML content
                content = result.node.text
                # Look for common value patterns in XML
                import re
                # Try to find numeric values
                numeric_match = re.search(r'value="([^"]*)"', content)
                if numeric_match:
                    value = numeric_match.group(1)
                else:
                    # Try to find any text content that might be a value
                    # Remove XML tags and get text content
                    clean_text = re.sub(r'<[^>]+>', '', content).strip()
                    if clean_text and len(clean_text) < 100:  # Reasonable length for a value
                        value = clean_text
            
            observation = {
                "id": result.node.node_id,
                "timestamp": parse_timestamp_from_metadata(result.node.metadata.get('timestamp')),
                "value": value,
                "code": result.node.metadata.get('code'),
                "name": result.node.metadata.get('name'),
                "unit": result.node.metadata.get('unit'),
                "metadata": result.node.metadata,
                "content": result.node.text,
                "person_id": result.node.metadata.get('person_id'),
            }
            observations.append(observation)
        
        # Sort by timestamp (ascending - oldest first)
        observations.sort(key=lambda x: x['timestamp'] or datetime.datetime.min)
        
        return observations


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


# Helper function to load patient document (for compatibility)
def load_patient_doc(filepath: str, chunk_element: str = "event", id_metadata_key: str = "person_id"):
    """Load a patient document and return a list of nodes."""
    from llama_index.core.schema import Document
    from meds_mcp.utils.xml_loader import SimpleXMLNodeParser
  
    with open(filepath, "r") as f:
        text = f.read()
    doc_id = os.path.splitext(os.path.basename(filepath))[0]
    doc = Document(
        text=text, doc_id=doc_id, metadata={"person_id": doc_id, "modality": "ehr"}
    )
    parser = SimpleXMLNodeParser(chunk_element="event", id_metadata_key="person_id")
    return parser.get_nodes_from_documents([doc])


async def main():
    """Example usage with document store."""
    # Example usage with document store
    from meds_mcp.server.rag.simple_storage import XMLDocumentStore
    
    # Initialize document store
    doc_store = XMLDocumentStore()
    
    # Load a patient document
    person_id = doc_store.load_patient_document("/Users/jfries/code/lumia/data/collections/debug/135917824.xml")
    
    # Set the global document store
    set_document_store(doc_store)
    
    # Test basic search
    results = await search_patient_events("cancer")
    print(f"Basic search results: {len(results)}")

    # Test filtering by XML attributes
    print("\n" + "="*100)
    print("Filtering by XML attributes:")
    
    # Filter by specific code
    code_results = await search_patient_events("cancer", filters={"attribute_filters": {"name": "progress note"}})
    print(f"Results with name=progress note: {len(code_results)}")

    # Time-bounded search using start/end parameters
    results = await search_patient_events("cancer", filters={"start_date": "2008-01-01", "end_date": "2009-07-01"})
    print(f"Results: {len(results)}")
    
    # Get historical values for a specific code (e.g., systolic blood pressure)
    print("\n" + "="*100)
    print("Historical values for systolic blood pressure:")
    bp_history = await get_historical_values({"code": "SNOMED/271649006"})
    print(f"Found {len(bp_history)} blood pressure readings")
    
    # Show first few readings
    for i, reading in enumerate(bp_history[:5]):
        print(f"  {reading['timestamp']}: {reading['value']} {reading.get('unit', '')}")
    
    # Get historical values with time filter
    print("\n" + "="*100)
    print("Blood pressure readings in 2008:")
    bp_2008 = await get_historical_values(
        {"code": "SNOMED/271649006"}, 
        filters={"start_date": "2008-01-01", "end_date": "2008-12-31"}
    )
    print(f"Found {len(bp_2008)} readings in 2008")
    for reading in bp_2008:
        print(f"  {reading['timestamp']}: {reading['value']} {reading.get('unit', '')}")

    bp = await get_historical_values({"code": "LOINC/2160-0"})
    print(f"Found {len(bp)} blood pressure readings")
    for reading in bp:
        print(f"  {reading['timestamp']}: {reading['value']} {reading.get('unit', '')}")
    
    # Complex attribute matching examples
    print("\n" + "="*100)
    print("Complex attribute matching examples:")
    
    # Match by code AND name
    creatinine_readings = await get_historical_values({
        "code": "LOINC/2160-0",
        "name": "Creatinine"
    })
    print(f"Found {len(creatinine_readings)} creatinine readings (code AND name match)")
    
    # Match by code OR name (using list)
    bp_variants = await get_historical_values({
        "code": ["STANFORD_MEAS"],
        "name": ["BUN/CREATININE"]
    })
    print(f"Found {len(bp_variants)} readings matching multiple codes/names")
    print(bp_variants)
    
    # Match by code with time filter
    recent_creatinine = await get_historical_values(
        {"code": "LOINC/2160-0"},
        filters={"start_date": "2008-06-01", "end_date": "2008-12-31"}
    )
    print(f"Found {len(recent_creatinine)} recent creatinine readings")
    
    # Match by name pattern (case-insensitive)
    bp_by_name = await get_historical_values({"name": "blood pressure"})
    print(f"Found {len(bp_by_name)} readings with 'blood pressure' in name")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
