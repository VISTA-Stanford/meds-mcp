import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

from llama_index.core.schema import TextNode, Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext

from meds_mcp.utils.xml_loader import SimpleXMLNodeParser

# Global document store instance
_document_store: Optional["XMLDocumentStore"] = None


class XMLDocumentLoader:
    """Simple document loader for XML files."""

    def __init__(self):
        self.parser = SimpleXMLNodeParser(
            chunk_element="event", id_metadata_key="person_id"
        )

    def load_data(self, filepath: str) -> List[Document]:
        """Load XML file and return Document objects."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Extract person_id from filename
        person_id = filepath.stem

        # Read XML content
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Create Document
        doc = Document(
            text=content,
            doc_id=person_id,
            metadata={
                "person_id": person_id,
                "modality": "EHR",
                "filepath": str(filepath),
            },
        )

        return [doc]


class Patient:
    """Represents a patient with their timeline nodes."""

    def __init__(self, person_id: str):
        self.person_id = person_id
        self.nodes: List[TextNode] = []
        self.documents: List[Document] = []

    def add_node(self, node: TextNode):
        """Add a node to the patient's timeline."""
        self.nodes.append(node)

    def add_document(self, document: Document):
        """Add a document to the patient's collection."""
        self.documents.append(document)

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events for this patient."""
        events = []
        for node in self.nodes:
            events.append(
                {
                    "id": node.node_id,
                    "content": node.text,
                    "metadata": node.metadata,
                    "timestamp": node.metadata.get("timestamp"),
                    "event_type": node.metadata.get("event_type"),
                    "code": node.metadata.get("code"),
                    "name": node.metadata.get("name"),
                    "person_id": node.metadata.get("person_id"),
                }
            )
        return events


class XMLDocumentStore:
    """Document store for patient timelines using XML files."""

    def __init__(
        self, data_dir: str, cache_dir: str = "cache", load_all_patients: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Storage components
        self.docstore = SimpleDocumentStore()
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore)

        # Patient management
        self.patients: Dict[str, Patient] = {}
        self.global_nodes: List[TextNode] = []
        self.bm25_retrievers: Dict[str, BM25Retriever] = {}

        # Node index for O(1) lookups by node_id
        self.node_index: Dict[str, TextNode] = {}

        # XML loader and parser
        self.xml_loader = XMLDocumentLoader()
        self.xml_parser = SimpleXMLNodeParser(chunk_element="event", id_metadata_key="person_id")
        
        # Load all patients on initialization
        self.load_all_patients()
    
    def load_patient_xml(self, filepath: str) -> Dict[str, Any]:
        """Load a patient XML file and add it to the store."""
        try:
            # Load XML document
            documents = self.xml_loader.load_data(filepath)

            if not documents:
                return {"error": "No documents loaded from XML file"}

            # Process each document
            results = []
            for doc in documents:
                person_id = doc.metadata.get("person_id")
                if not person_id:
                    continue

                # Create or get patient
                if person_id not in self.patients:
                    self.patients[person_id] = Patient(person_id)

                patient = self.patients[person_id]

                # Add document to patient
                patient.add_document(doc)

                # Create nodes from document using XML parser
                nodes = self.xml_parser.get_nodes_from_documents([doc])

                # Add nodes to patient and global collection
                for node in nodes:
                    patient.add_node(node)
                    self.global_nodes.append(node)
                    self.node_index[node.node_id] = node  # Add to index

                # Store document in docstore
                self.docstore.add_documents([doc])

                results.append(
                    {
                        "person_id": person_id,
                        "document_id": doc.doc_id,
                        "nodes_created": len(nodes),
                    }
                )

            # Create BM25 retrievers for affected patients
            for result in results:
                person_id = result["person_id"]
                self._create_patient_retriever(person_id)

            return {"success": True, "results": results}

        except Exception as e:
            return {"error": f"Failed to load XML file: {str(e)}"}

    def load_patient_timeline(self, person_id: str) -> Dict[str, Any]:
        """Load patient timeline by person_id from the data directory."""
        try:
            # Check if patient is already loaded
            if person_id in self.patients:
                patient = self.patients[person_id]
                print(
                    f"Patient {person_id} already loaded ({len(patient.nodes)} events)"
                )
                return {
                    "success": True,
                    "person_id": person_id,
                    "events_loaded": len(patient.nodes),
                    "documents_loaded": len(patient.documents),
                    "already_loaded": True,
                }

            # Look for XML file with matching person_id
            xml_file = self.data_dir / f"{person_id}.xml"

            if not xml_file.exists():
                return {"error": f"No XML file found for person_id {person_id}"}

            print(f"Loading patient document: {xml_file}")

            # Load the XML file
            result = self.load_patient_xml(str(xml_file))

            if result.get("error"):
                return result

            # Create and cache BM25 retriever for this patient
            self._create_patient_retriever(person_id)

            # Get patient stats
            patient = self.patients.get(person_id)
            if patient:
                return {
                    "success": True,
                    "person_id": person_id,
                    "events_loaded": len(patient.nodes),
                    "documents_loaded": len(patient.documents),
                    "already_loaded": False,
                }
            else:
                return {"error": f"Patient {person_id} not found after loading"}

        except Exception as e:
            return {"error": f"Failed to load patient timeline: {str(e)}"}

    def get_patient_timeline(self, person_id: str) -> Dict[str, Any]:
        """Get the entire original XML file content for a patient."""
        try:
            # Look for XML file with matching person_id
            xml_file = self.data_dir / f"{person_id}.xml"

            if not xml_file.exists():
                return {"error": f"No XML file found for person_id {person_id}"}

            print(f"Reading original XML file: {xml_file}")

            # Read the entire XML file content
            with open(xml_file, "r", encoding="utf-8") as f:
                xml_content = f.read()

            return {
                "success": True,
                "person_id": person_id,
                "xml_content": xml_content,
                "file_path": str(xml_file),
                "file_size": len(xml_content),
            }

        except Exception as e:
            return {"error": f"Failed to read patient timeline XML: {str(e)}"}

    def _create_patient_retriever(self, person_id: str):
        """Create BM25 retriever for a specific patient."""
        if person_id not in self.patients:
            return

        patient = self.patients[person_id]
        if not patient.nodes:
            return

        num_nodes = len(patient.nodes)
        
        # Use LlamaIndex's official persistence methods
        persist_dir = self.cache_dir / f"bm25_index_{person_id}"

        try:
            if persist_dir.exists():
                print(f"  📦 Loading cached BM25 index for {person_id} ({num_nodes} events)...", flush=True, end="")
                retriever = BM25Retriever.from_persist_dir(str(persist_dir))
                self.bm25_retrievers[person_id] = retriever
                print(f" ✅", flush=True)
                return
        except Exception as e:
            print(f" ❌ Error: {e}", flush=True)
            print(f"  🔄 Recreating index for {person_id}...", flush=True)

        # Create new retriever - this can be slow for large patients
        print(f"  🔨 Building BM25 index for {person_id} ({num_nodes} events)...", flush=True)
        print(f"     (This may take a while for large patients - please wait)", flush=True)
        
        import time
        start_time = time.time()
        
        retriever = BM25Retriever(
            nodes=patient.nodes, similarity_top_k=len(patient.nodes)
        )
        self.bm25_retrievers[person_id] = retriever

        elapsed = time.time() - start_time
        print(f"  ✅ Indexed {num_nodes} events in {elapsed:.1f}s", flush=True)

        # Persist the retriever using LlamaIndex's official method
        try:
            print(f"  💾 Caching index...", flush=True, end="")
            retriever.persist(str(persist_dir))
            print(f" ✅", flush=True)
        except Exception as e:
            print(f" ⚠️  Warning: Could not cache: {e}", flush=True)

    def get_patient_event(self, node_id: str) -> Dict[str, Any]:
        """Get a specific patient event by node ID."""
        if node_id in self.node_index:
            node = self.node_index[node_id]
            return {
                "id": node.node_id,
                "content": node.text,
                "metadata": node.metadata,
                "person_id": node.metadata.get("person_id"),
            }
        return {"error": f"Node {node_id} not found"}

    def list_patients(self) -> List[str]:
        """List all patient IDs in the store."""
        return list(self.patients.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the document store."""
        total_nodes = sum(len(patient.nodes) for patient in self.patients.values())
        total_docs = sum(len(patient.documents) for patient in self.patients.values())

        return {
            "total_patients": len(self.patients),
            "total_documents": total_docs,
            "total_nodes": total_nodes,
            "patients_with_retrievers": len(self.bm25_retrievers),
            "global_nodes": len(self.global_nodes),
            "indexed_nodes": len(self.node_index),
        }

    def list_all_node_ids(self) -> List[str]:
        """List all node IDs in the store."""
        node_ids = []
        for patient in self.patients.values():
            for node in patient.nodes:
                node_ids.append(node.node_id)
        return node_ids

    def list_patient_node_ids(self, person_id: str) -> List[str]:
        """List all node IDs for a specific patient."""
        if person_id not in self.patients:
            return []

        patient = self.patients[person_id]
        return [node.node_id for node in patient.nodes]

    def get_all_patient_events(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific patient."""
        if person_id not in self.patients:
            return []

        patient = self.patients[person_id]
        return patient.get_events()

    def load_all_patients(self):
        """Scan data_dir for XML files and load each patient."""
        xml_files = list(self.data_dir.glob("*.xml"))
        total_files = len(xml_files)
        print(f"\n📁 Found {total_files} XML files to load", flush=True)
        
        for idx, filepath in enumerate(xml_files, 1):
            print(f"\n[{idx}/{total_files}] Loading patient from {filepath.name}...", flush=True)
            result = self.load_patient_xml(str(filepath))
            if result.get("error"):
                print(f"❌ Error loading {filepath.name}: {result['error']}", flush=True)
            else:
                patient_id = result['results'][0]['person_id']
                print(f"✅ Loaded patient {patient_id}", flush=True)
        
        print(f"\n✅ Finished loading {len(self.patients)} patients", flush=True)

def initialize_document_store(data_dir: str, cache_dir: str = "cache", load_all_patients: bool = False) -> XMLDocumentStore:
    """Initialize the global document store."""
    global _document_store

    _document_store = XMLDocumentStore(data_dir, cache_dir, load_all_patients)
    return _document_store


def get_document_store() -> Optional[XMLDocumentStore]:
    """Get the global document store instance."""
    return _document_store


# Plain function implementations (no decorators)
async def load_patient_xml(filepath: str) -> Dict[str, Any]:
    """Load a patient XML file and add it to the document store."""
    if _document_store is None:
        return {"error": "Document store not initialized"}

    try:
        result = _document_store.load_patient_xml(filepath)
        return result
    except Exception as e:
        return {"error": f"Failed to load XML file: {str(e)}"}


async def load_patient_timeline(person_id: str) -> Dict[str, Any]:
    """Load patient timeline by person_id from the data directory."""
    if _document_store is None:
        return {"error": "Document store not initialized"}

    try:
        result = _document_store.load_patient_timeline(person_id)
        return result
    except Exception as e:
        return {"error": f"Failed to load patient timeline: {str(e)}"}


async def get_patient_timeline(person_id: str) -> Dict[str, Any]:
    """Get the entire original XML file content for a patient."""
    if _document_store is None:
        return {"error": "Document store not initialized"}

    try:
        result = _document_store.get_patient_timeline(person_id)
        return result
    except Exception as e:
        return {"error": f"Failed to get patient timeline: {str(e)}"}


async def get_patient_event(node_id: str) -> Dict[str, Any]:
    """Get a specific patient event by node ID."""
    if _document_store is None:
        return {"error": "Document store not initialized"}

    try:
        result = _document_store.get_patient_event(node_id)
        return result
    except Exception as e:
        return {"error": f"Failed to get patient event: {str(e)}"}


async def list_patients() -> List[str]:
    """List all patients in the document store."""
    if _document_store is None:
        return []

    try:
        return _document_store.list_patients()
    except Exception as e:
        return []


async def get_document_store_stats() -> Dict[str, Any]:
    """Get statistics about the document store."""
    if _document_store is None:
        return {"error": "Document store not initialized"}

    try:
        return _document_store.get_stats()
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}"}


async def list_all_node_ids() -> List[str]:
    """List all node IDs in the document store."""
    if _document_store is None:
        return []

    try:
        return _document_store.list_all_node_ids()
    except Exception as e:
        return []


async def list_patient_node_ids(person_id: str) -> List[str]:
    """List all node IDs for a specific patient."""
    if _document_store is None:
        return []

    try:
        return _document_store.list_patient_node_ids(person_id)
    except Exception as e:
        return []


async def get_all_patient_events(person_id: str) -> List[Dict[str, Any]]:
    """Get all events for a specific patient."""
    if _document_store is None:
        return []

    try:
        return _document_store.get_all_patient_events(person_id)
    except Exception as e:
        return []
