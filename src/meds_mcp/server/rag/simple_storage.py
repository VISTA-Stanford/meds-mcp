import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from meds_mcp.utils.xml_loader import SimpleXMLNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
import pandas as pd

mcp = FastMCP("demo")

@dataclass
class PatientDocument:
    """Represents a patient document with events."""
    person_id: str
    filepath: str
    content: str
    nodes: List[TextNode]
    latest_timestamp: Optional[pd.Timestamp] = None
    token_df: Optional[pd.DataFrame] = None


class XMLDocumentStore:
    """In-memory document store for XML patient data with BM25 indexing."""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the XML document store.
        
        Args:
            cache_dir: Directory to store BM25 index cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory storage
        self.patients: Dict[str, PatientDocument] = {}
        self.bm25_retrievers: Dict[str, BM25Retriever] = {}
        self.storage_contexts: Dict[str, StorageContext] = {}
        
        # Global nodes for cross-patient access
        self.global_nodes: List[TextNode] = []
    
    def _get_cache_path(self, person_id: str) -> Path:
        """Get cache file path for a person."""
        return self.cache_dir / f"bm25_index_{person_id}.pkl"
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get file hash for cache invalidation."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_cached_index(self, person_id: str) -> Optional[BM25Retriever]:
        """Load cached BM25 index if available and valid."""
        cache_path = self._get_cache_path(person_id)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is still valid
            if cached_data.get('file_hash') == self._get_file_hash(cached_data['filepath']):
                return cached_data['retriever']
        except Exception as e:
            print(f"Error loading cached index for {person_id}: {e}")
        
        return None
    
    def _save_cached_index(self, person_id: str, retriever: BM25Retriever, filepath: str):
        """Save BM25 index to cache."""
        cache_path = self._get_cache_path(person_id)
        try:
            cached_data = {
                'retriever': retriever,
                'filepath': filepath,
                'file_hash': self._get_file_hash(filepath)
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"Error saving cached index for {person_id}: {e}")
    
    def compute_entry_token_lengths(self, xml_content: str) -> pd.DataFrame:
        """Compute token lengths for timeline analysis."""
        # This is a placeholder - implement based on your specific needs
        # You might want to parse XML and extract timestamps with token counts
        return pd.DataFrame()
    
    def load_patient_document(self, filepath: str, chunk_element: str = "event") -> str:
        """
        Load a patient document from XML file.
        
        Args:
            filepath: Path to XML file
            chunk_element: XML element to chunk on (default: "event")
            
        Returns:
            Person ID of the loaded document
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Extract person_id from filename
        person_id = filepath.stem
        
        # Check if already loaded
        if person_id in self.patients:
            print(f"Patient {person_id} already loaded")
            return person_id
        
        print(f"Loading patient document: {filepath}")
        
        # Read XML content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create llama_index Document
        doc = Document(
            text=content,
            doc_id=person_id,
            metadata={"person_id": person_id, "modality": "EHR"}
        )
        
        # Compute token timeline
        token_df = self.compute_entry_token_lengths(content)
        
        # Find latest timestamp
        latest_timestamp = None
        if not token_df.empty:
            latest_timestamp = token_df.index.max()
        
        # Parse XML into nodes
        parser = SimpleXMLNodeParser(chunk_element=chunk_element, id_metadata_key="person_id")
        nodes = parser.get_nodes_from_documents([doc])
        
        # Check for cached BM25 index
        retriever = self._load_cached_index(person_id)
        
        if retriever is None:
            # Create new BM25 retriever
            try:
                retriever = BM25Retriever(nodes=nodes, similarity_top_k=len(nodes))
                print(f"Successfully indexed {len(nodes)} events for patient {person_id}")
                
                # Cache the index
                self._save_cached_index(person_id, retriever, str(filepath))
            except Exception as e:
                print(f"Error creating BM25 retriever for {person_id}: {str(e)}")
                retriever = None
        
        # Create storage context
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)
        
        # Store everything
        self.patients[person_id] = PatientDocument(
            person_id=person_id,
            filepath=str(filepath),
            content=content,
            nodes=nodes,
            latest_timestamp=latest_timestamp,
            token_df=token_df
        )
        
        self.bm25_retrievers[person_id] = retriever
        self.storage_contexts[person_id] = storage_context
        
        # Add to global nodes
        self.global_nodes.extend(nodes)
        
        return person_id
    
    def get_patient_document(self, person_id: str) -> Optional[PatientDocument]:
        """Get a patient document by person_id."""
        return self.patients.get(person_id)
    
    def get_event_by_node_id(self, person_id: str, node_id: str) -> Optional[TextNode]:
        """Get a specific event by node_id for a patient."""
        patient = self.patients.get(person_id)
        if not patient:
            return None
        
        for node in patient.nodes:
            if node.node_id == node_id:
                return node
        
        return None
    

    
    def list_patients(self) -> List[Dict[str, Any]]:
        """List all loaded patients with metadata."""
        return [
            {
                'person_id': patient.person_id,
                'filepath': patient.filepath,
                'event_count': len(patient.nodes),
                'latest_timestamp': patient.latest_timestamp,
                'has_retriever': person_id in self.bm25_retrievers
            }
            for person_id, patient in self.patients.items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total_events = sum(len(patient.nodes) for patient in self.patients.values())
        
        return {
            'total_patients': len(self.patients),
            'total_events': total_events,
            'cache_directory': str(self.cache_dir),
            'patients_with_retrievers': len(self.bm25_retrievers),
            'total_global_nodes': len(self.global_nodes)
        }
    
    def list_all_node_ids(self) -> List[Dict[str, Any]]:
        """List all node_ids in the document store with metadata."""
        all_nodes = []
        
        for person_id, patient in self.patients.items():
            for node in patient.nodes:
                all_nodes.append({
                    'node_id': node.node_id,
                    'person_id': person_id,
                    'text_preview': node.text[:100] + "..." if len(node.text) > 100 else node.text,
                    'metadata': node.metadata
                })
        
        return all_nodes
    
    def list_patient_node_ids(self, person_id: str) -> List[Dict[str, Any]]:
        """List all node_ids for a specific patient."""
        patient = self.patients.get(person_id)
        if not patient:
            return []
        
        return [
            {
                'node_id': node.node_id,
                'person_id': person_id,
                'text_preview': node.text[:100] + "..." if len(node.text) > 100 else node.text,
                'metadata': node.metadata
            }
            for node in patient.nodes
        ]
    
    def get_patient_events(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific patient."""
        patient = self.patients.get(person_id)
        if not patient:
            return []
        
        return [
            {
                'node_id': node.node_id,
                'person_id': person_id,
                'text': node.text,
                'metadata': node.metadata
            }
            for node in patient.nodes
        ]


# Global document store instance
_document_store: Optional[XMLDocumentStore] = None


def initialize_document_store(cache_dir: str = "cache"):
    """Initialize the global document store."""
    global _document_store
    _document_store = XMLDocumentStore(cache_dir)


@mcp.resource("docstore://{person_id}/events/{node_id}")
async def get_event_resource(person_id: str, node_id: str):
    """Get a specific event as an MCP resource."""
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    event = _document_store.get_event_by_node_id(person_id, node_id)
    if not event:
        raise ValueError(f"Event {node_id} not found for patient {person_id}")
    
    return {
        "uri": f"docstore://{person_id}/events/{node_id}",
        "name": f"Event {node_id}",
        "description": f"Medical event for patient {person_id}",
        "mimeType": "application/json"
    }


@mcp.resource("docstore://{person_id}/source")
async def get_source_xml_resource(person_id: str):
    """Get the full source XML document as a string."""
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    # Get the patient document to find the filepath
    patient = _document_store.get_patient_document(person_id)
    if not patient:
        raise ValueError(f"Patient {person_id} not found in document store")
    
    # Read the source file directly
    try:
        with open(patient.filepath, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        return {
            "uri": f"docstore://{person_id}/source",
            "name": f"Source XML for {person_id}",
            "description": f"Full source XML document for patient {person_id}",
            "mimeType": "application/xml",
            "content": xml_content
        }
    except FileNotFoundError:
        raise ValueError(f"Source file not found: {patient.filepath}")
    except Exception as e:
        raise RuntimeError(f"Error reading source file: {e}")


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
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return _document_store.load_patient_document(filepath, chunk_element)


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
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    event = _document_store.get_event_by_node_id(person_id, node_id)
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
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return _document_store.list_patients()


@mcp.tool()
async def get_document_store_stats() -> Dict[str, Any]:
    """
    Get document store statistics.
    
    Returns:
        Store statistics
    """
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return _document_store.get_stats()


@mcp.tool()
async def list_all_node_ids() -> List[Dict[str, Any]]:
    """
    List all node_ids in the document store.
    
    Returns:
        List of all node_ids with metadata including person_id and text preview
    """
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return _document_store.list_all_node_ids()


@mcp.tool()
async def list_patient_node_ids(person_id: str) -> List[Dict[str, Any]]:
    """
    List all node_ids for a specific patient.
    
    Args:
        person_id: Patient identifier
        
    Returns:
        List of node_ids for the specified patient with metadata
    """
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return _document_store.list_patient_node_ids(person_id)


@mcp.tool()
async def get_patient_events(person_id: str) -> List[Dict[str, Any]]:
    """
    Get all events for a specific patient.
    
    Args:
        person_id: Patient identifier
        
    Returns:
        List of all events for the patient with full text and metadata
    """
    if _document_store is None:
        raise RuntimeError("Document store not initialized")
    
    return _document_store.get_patient_events(person_id)


# Example usage
if __name__ == "__main__":
    # Initialize the document store
    initialize_document_store("/users/jfries/code/meds-mcp/data/scratch/cache")
    
    # Example: Load a patient document
    try:
        person_id = _document_store.load_patient_document("data/collections/dev-corpus/135917824.xml")
        print(f"Loaded patient: {person_id}")
        
        # List patients
        patients = _document_store.list_patients()
        print(f"Loaded patients: {patients}")
        
        # Get statistics
        stats = _document_store.get_stats()
        print(f"Store statistics: {stats}")
        
        # List all node_ids
        all_node_ids = _document_store.list_all_node_ids()
        print(f"All node_ids: {all_node_ids}")
        
        # List node_ids for a specific patient
        patient_node_ids = _document_store.list_patient_node_ids(person_id)
        print(f"Node_ids for patient {person_id}: {patient_node_ids}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the XML file exists and is valid.")


# Example MCP Tool and Resource Calls
async def example_mcp_calls():
    """
    Example function calls for all MCP tools and resources in this script.
    These can be called from an MCP client or server.
    """
    
    # ============================================================================
    # MCP TOOLS
    # ============================================================================
    
    # 1. Load a patient XML document and build BM25 index
    person_id = await load_patient_xml("data/collections/dev-corpus/135917824.xml")
    print(f"Loaded patient: {person_id}")
    
    # 2. Get a specific event by node_id
    event = await get_patient_event("135917824", "135917824_event_0")
    if event:
        print(f"Event: {event['node_id']}")
        print(f"Text preview: {event['text'][:100]}...")
    
    # 3. List all patients in the store
    patients = await list_patients()
    print(f"Loaded patients: {patients}")
    
    # 4. List all node_ids across all patients
    all_nodes = await list_all_node_ids()
    print(f"Total nodes: {len(all_nodes)}")
    for node in all_nodes[:3]:  # Show first 3
        print(f"  {node['node_id']} - {node['text_preview']}")
    
    # 5. List node_ids for a specific patient
    patient_nodes = await list_patient_node_ids("135917824")
    print(f"Patient nodes: {len(patient_nodes)}")
    for node in patient_nodes[:3]:  # Show first 3
        print(f"  {node['node_id']} - {node['text_preview']}")
    
    # 6. Get all events for a specific patient
    patient_events = await get_patient_events("135917824")
    print(f"Patient events: {len(patient_events)}")
    for event in patient_events[:3]:  # Show first 3
        print(f"  {event['node_id']} - {event['text'][:100]}...")
    
    # 7. Get document store statistics
    stats = await get_document_store_stats()
    print(f"Store stats: {stats}")
    
    # ============================================================================
    # MCP RESOURCES
    # ============================================================================
    
    # 8. Get a specific event as a resource
    # Resource URI: docstore://135917824/events/135917824_event_0
    event_resource = await get_event_resource("135917824", "135917824_event_0")
    print(f"Event resource: {event_resource['uri']}")
    print(f"Event name: {event_resource['name']}")
    
    # 9. Get the full source XML document as a resource
    # Resource URI: docstore://135917824/source
    source_resource = await get_source_xml_resource("135917824")
    print(f"Source resource: {source_resource['uri']}")
    print(f"Source name: {source_resource['name']}")
    print(f"XML content length: {len(source_resource['content'])} characters")
    print(f"First 200 chars: {source_resource['content'][:200]}...")


# Example of how to use the MCP tools in a client
def example_client_usage():
    """
    Example of how a client might use these MCP tools.
    """
    print("=== MCP Client Usage Examples ===")
    
    # Tool calls (these would be made to the MCP server)
    tools_to_call = [
        {
            "name": "load_patient_xml",
            "arguments": {"filepath": "data/collections/dev-corpus/135917824.xml"}
        },
        {
            "name": "list_patients",
            "arguments": {}
        },
        {
            "name": "get_patient_events", 
            "arguments": {"person_id": "135917824"}
        },
        {
            "name": "list_all_node_ids",
            "arguments": {}
        },
        {
            "name": "get_patient_event",
            "arguments": {"person_id": "135917824", "node_id": "135917824_event_0"}
        },
        {
            "name": "get_document_store_stats",
            "arguments": {}
        }
    ]
    
    # Resource URIs to access
    resource_uris = [
        "docstore://135917824/source",
        "docstore://135917824/events/135917824_event_0",
        "docstore://135917824/events/135917824_event_1"
    ]
    
    print("Tools to call:")
    for tool in tools_to_call:
        print(f"  {tool['name']}({tool['arguments']})")
    
    print("\nResource URIs to access:")
    for uri in resource_uris:
        print(f"  {uri}")


if __name__ == "__main__":
    # Run the example MCP calls
    import asyncio
    asyncio.run(example_mcp_calls())
    
    # Show client usage examples
    example_client_usage()

