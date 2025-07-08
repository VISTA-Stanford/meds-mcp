import os
import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import re

from llama_index.core.schema import TextNode, Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext


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
    """Patient timeline retriever using llama-index docstore and BM25 retriever."""
    
    def __init__(self):
        self.docstore = SimpleDocumentStore()
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore)
        self.retriever = None
        self.nodes = []
    
    def add_documents(self, nodes: List[TextNode]) -> None:
        """Add nodes from a loader to the store and build retriever."""
        # Nodes are already in the right format, just use them directly
        self.nodes = nodes
        # Add them to docstore
        self.docstore.add_documents(nodes)
        
        # Build BM25 retriever
        # HACK 
        # This is a hack to ensure that we get all documents in the search results
        # TODO: We should use a more efficient retriever that can handle this
        self.retriever = BM25Retriever(nodes=self.nodes, similarity_top_k=len(nodes))
    
    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search patient events with optional filters.

        Args:
            query: Search query string
            filters: Optional SearchFilters object to constrain the search

        Returns:
            List of search results with event data
        """
        if filters is None:
            filters = SearchFilters()
        
        if not self.retriever:
            return []
        
        # Get initial search results
        initial_results = self.retriever.retrieve(query)
        
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
                #"encounter_id": result.node.metadata.get('encounter_id'),
                #"patient_id": result.node.metadata.get('patient_id'),
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
    
    def get_events_by_type(self, event_type: str, filters: Optional[SearchFilters] = None) -> List[Dict[str, Any]]:
        """Get events by type with optional filtering."""
        if filters is None:
            filters = SearchFilters()
        
        # Filter nodes by event type
        type_nodes = [
            node for node in self.nodes 
            if node.metadata.get('event_type') == event_type
        ]
        
        # Convert to search results for consistent filtering
        from llama_index.core.schema import NodeWithScore
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
                "patient_id": result.node.metadata.get('patient_id'),
            }
            for result in filtered_results
        ]

    def get_historical_values(self, attribute_filters: Dict[str, Any], 
                            filters: Optional[SearchFilters] = None) -> List[Dict[str, Any]]:
        """
        TODO: This is a hack to get historical values using attribute matching.
        We should move to a actual queries over MEDS dataframes for these types of queries.
        
        Get historical values using attribute matching.
        
        Args:
            attribute_filters: Dict for attribute matching (e.g., {"code": "LOINC/2160-0", "name": "Creatinine"})
            filters: Optional SearchFilters object to constrain the search (time range, etc.)
            
        Returns:
            List of observations with timestamps, sorted chronologically
        """
        if filters is None:
            filters = SearchFilters()
        
        # Start with all nodes
        matching_nodes = self.nodes
        
        # Apply attribute filters
        if attribute_filters:
            # Convert to search results for consistent filtering
            from llama_index.core.schema import NodeWithScore
            temp_results = [NodeWithScore(node=node, score=1.0) for node in matching_nodes]
            
            # Apply attribute filtering
            filtered_results = self._apply_attribute_filters(temp_results, attribute_filters)
            matching_nodes = [result.node for result in filtered_results]
        
        # Convert to search results for consistent filtering
        from llama_index.core.schema import NodeWithScore
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
            }
            observations.append(observation)
        
        # Sort by timestamp (ascending - oldest first)
        observations.sort(key=lambda x: x['timestamp'] or datetime.datetime.min)
        
        return observations


def load_patient_doc(filepath: str, chunk_element: str = "event", id_metadata_key: str = "person_id"):
    """Load a patient document and return a list of nodes."""
    from llama_index.core.schema import Document
    from lumia.loaders.xml_loader import SimpleXMLNodeParser
  
    with open(filepath, "r") as f:
        text = f.read()
    doc_id = os.path.splitext(os.path.basename(filepath))[0]
    doc = Document(
        text=text, doc_id=doc_id, metadata={"person_id": doc_id, "modality": "ehr"}
    )
    parser = SimpleXMLNodeParser(chunk_element="event", id_metadata_key="person_id")
    return parser.get_nodes_from_documents([doc])

if __name__ == "__main__":

    docs = load_patient_doc("/Users/jfries/code/lumia/data/collections/debug/135909143.xml")
    retriever = PatientTimelineRetriever()
    retriever.add_documents(docs)

    # Test basic search
    results = retriever.search("cancer")
    print(f"Basic search results: {len(results)}")

    # Test filtering by XML attributes (now available when chunk_element="event")
    print("\n" + "="*100)
    print("Filtering by XML attributes:")
    
    # Filter by specific code
    code_results = retriever.search("cancer", filters=SearchFilters(
        attribute_filters={"name": "progress note"}
    ))
    print(f"Results with name=progress note: {len(code_results)}")

    # Time-bounded search using start/end parameters
    results = retriever.search("cancer", filters=SearchFilters(
        start=datetime.datetime(2008, 1, 1),
        end=datetime.datetime(2009, 7, 1)
    ))
    print(f"Results: {len(results)}")
    
    # Get historical values for a specific code (e.g., systolic blood pressure)
    print("\n" + "="*100)
    print("Historical values for systolic blood pressure:")
    bp_history = retriever.get_historical_values({"code": "SNOMED/271649006"})
    print(f"Found {len(bp_history)} blood pressure readings")
    
    # Show first few readings
    for i, reading in enumerate(bp_history[:5]):
        print(f"  {reading['timestamp']}: {reading['value']} {reading.get('unit', '')}")
    
    # Get historical values with time filter
    print("\n" + "="*100)
    print("Blood pressure readings in 2008:")
    bp_2008 = retriever.get_historical_values(
        {"code": "SNOMED/271649006"}, 
        filters=SearchFilters(
            start=datetime.datetime(2008, 1, 1),
            end=datetime.datetime(2008, 12, 31)
        )
    )
    print(f"Found {len(bp_2008)} readings in 2008")
    for reading in bp_2008:
        print(f"  {reading['timestamp']}: {reading['value']} {reading.get('unit', '')}")

    bp = retriever.get_historical_values({"code": "LOINC/2160-0"})
    print(f"Found {len(bp)} blood pressure readings")
    for reading in bp:
        print(f"  {reading['timestamp']}: {reading['value']} {reading.get('unit', '')}")
    
    # Complex attribute matching examples
    print("\n" + "="*100)
    print("Complex attribute matching examples:")
    
    # Match by code AND name
    creatinine_readings = retriever.get_historical_values({
        "code": "LOINC/2160-0",
        "name": "Creatinine"
    })
    print(f"Found {len(creatinine_readings)} creatinine readings (code AND name match)")
    
    # Match by code OR name (using list)
    bp_variants = retriever.get_historical_values({
        "code": ["STANFORD_MEAS"],
        "name": ["BUN/CREATININE"]
    })
    print(f"Found {len(bp_variants)} readings matching multiple codes/names")
    print(bp_variants)
    
    # Match by code with time filter
    recent_creatinine = retriever.get_historical_values(
        {"code": "LOINC/2160-0"},
        filters=SearchFilters(
            start=datetime.datetime(2008, 6, 1),
            end=datetime.datetime(2008, 12, 31)
        )
    )
    print(f"Found {len(recent_creatinine)} recent creatinine readings")
    
    # Match by name pattern (case-insensitive)
    bp_by_name = retriever.get_historical_values({"name": "blood pressure"})
    print(f"Found {len(bp_by_name)} readings with 'blood pressure' in name")

# Example usage:
#
# # Create retriever
# retriever = PatientTimelineRetriever()
#
# # Add documents (from your XML loader)
# retriever.add_documents(documents)
#
# # Search all events for "surgery"
# retriever.search("surgery")
#
# # Limit to notes only (inclusion filter)
# retriever.search("surgery", filters=SearchFilters(event_types=["note"]))
#
# # Time-bounded search using start/end parameters
# retriever.search("surgery", filters=SearchFilters(
#     start=datetime(2020, 1, 1),
#     end=datetime(2021, 1, 1)
# ))
#
# # Last 5 encounters only
# retriever.search("surgery", filters=SearchFilters(max_encounters=5))
#
# # Notes in last 5 encounters after 2020 (inclusion filter)
# retriever.search("surgery", filters=SearchFilters(
#     start=datetime(2020, 1, 1),
#     max_encounters=5,
#     event_types=["note"]  # Only include note events
# ))
#
# # Filter by XML attributes (code, name, unit, etc.)
# retriever.search("surgery", filters=SearchFilters(
#     attribute_filters={
#         "code": "12345",  # Exact match
#         "name": ["procedure", "surgery"],  # Case-insensitive match for any of these values
#         "unit": re.compile(r"ICU|ER"),  # Regex pattern
#     }
# ))
#
# # Case-insensitive name matching examples:
# retriever.search("surgery", filters=SearchFilters(
#     attribute_filters={
#         "name": "SURGERY",  # Will match "surgery", "Surgery", "SURGERY", etc.
#         "provider_name": ["Dr. Smith", "DR. JONES"],  # Will match "dr. smith", "Dr. Jones", etc.
#     }
# ))
#
# # Combine multiple filter types
# retriever.search("surgery", filters=SearchFilters(
#     start=datetime(2020, 1, 1),
#     event_types=["note"],
#     attribute_filters={
#         "code": ["12345", "67890"],
#         "unit": "ICU",  # Case-insensitive
#         "name": "emergency procedure"  # Case-insensitive
#     },
#     max_results=10
# ))
