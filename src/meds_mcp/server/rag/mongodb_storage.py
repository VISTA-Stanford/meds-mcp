from pathlib import Path
from typing import Dict, List, Optional, Any
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import Document
import xml.etree.ElementTree as ET


class MongoTimelineStore:
    """
    Simple document store that maps person_id to XML documents using llama_index.
    Initializes from existing MongoDB database and provides methods to check indexing status.

    DEPRICATED 
    
    NOTE: MongoDB has a hard limit of 16MB for documents. This is a problem for
    a percentage of long patient timelines. Unfortunatley MongoDB does not intend
    to support this use case. 

    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        database_name: str = "vista-dev",
        collection_name: str = "starr",
    ):
        """
        Initialize the document store from existing MongoDB database.

        Args:
            mongo_uri: MongoDB connection URI
            database_name: Name of the MongoDB database
            collection_name: Name of the collection within the database
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name

        # Initialize llama_index MongoDocumentStore from existing database
        # Note: collection_name is handled internally by the URI or database name
        self.doc_store = MongoDocumentStore.from_uri(
            uri=mongo_uri, db_name=database_name
        )

    def is_person_indexed(self, person_id: str) -> bool:
        """
        Check if a person_id is already indexed in the database.

        Args:
            person_id: The person ID to check

        Returns:
            True if person_id exists, False otherwise
        """
        try:
            doc = self.doc_store.get_document(person_id)
            return doc is not None
        except KeyError:
            return False

    def get_indexed_person_ids(self) -> List[str]:
        """
        Get all person IDs that are currently indexed in the database.

        Returns:
            List of indexed person IDs
        """
        all_docs = self.doc_store.docs
        person_ids = []
        for doc_id, doc in all_docs.items():
            if doc.metadata and "person_id" in doc.metadata:
                person_ids.append(doc.metadata["person_id"])
        return person_ids

    def extract_person_id(self, file_path: Path) -> Optional[str]:
        """
        Extract person_id from XML file.
        TO BE IMPLEMENTED: Add your person_id extraction logic here.

        Args:
            file_path: Path to the XML file

        Returns:
            person_id or None if not found
        """
        # Extract person_id from filename (remove extension)
        return file_path.stem

    def extract_metadata(self, file_path: Path, person_id: str) -> Dict[str, Any]:
        """
        Extract metadata from XML file.
        TO BE IMPLEMENTED: Add your metadata extraction logic here.

        Args:
            file_path: Path to the XML file
            person_id: The extracted person_id

        Returns:
            Dictionary of metadata
        """
        # TODO: Implement metadata extraction
        # Example: extract start_date, end_date, etc.
        return {
            "person_id": person_id,
            "file_path": str(file_path),
            "source_file": file_path.name,
            # Add your metadata fields here
        }

    def index_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Index a single document into the database.

        Args:
            file_path: Path to the XML file to index

        Returns:
            Dictionary with result information
        """
        if not file_path.exists():
            return {"success": False, "error": f"File {file_path} does not exist"}

        try:
            # Extract person_id from filename
            person_id = file_path.stem

            # Extract metadata
            metadata = self.extract_metadata(file_path, person_id)

            # Read XML content
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = f.read()

            # Check document size and compress if too large
            content_size_mb = len(xml_content.encode("utf-8")) / (1024 * 1024)
            if content_size_mb > 15:  # Leave some buffer below 16MB limit
                return self._compress_and_store_document(
                    person_id, xml_content, metadata, file_path
                )

            # Create llama_index Document
            doc = Document(
                text=xml_content,
                metadata=metadata,
                id_=person_id,  # Use person_id as the document ID
            )

            # Store in llama_index document store
            self.doc_store.add_documents([doc])

            return {
                "success": True,
                "person_id": person_id,
                "file_path": str(file_path),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing {file_path}: {str(e)}",
            }

    def _compress_and_store_document(
        self,
        person_id: str,
        xml_content: str,
        metadata: Dict[str, Any],
        file_path: Path,
    ) -> Dict[str, Any]:
        """
        Compress document content to fit within MongoDB size limits.
        """
        import gzip
        import base64

        try:
            # Compress the XML content
            compressed_content = gzip.compress(xml_content.encode("utf-8"))
            compressed_b64 = base64.b64encode(compressed_content).decode("utf-8")

            # Update metadata to indicate compression
            metadata.update(
                {
                    "compressed": True,
                    "original_size_mb": len(xml_content.encode("utf-8"))
                    / (1024 * 1024),
                    "compressed_size_mb": len(compressed_b64.encode("utf-8"))
                    / (1024 * 1024),
                }
            )

            # Store compressed content
            doc = Document(
                text=compressed_b64,
                metadata=metadata,
                id_=person_id,
            )

            self.doc_store.add_documents([doc])

            return {
                "success": True,
                "person_id": person_id,
                "file_path": str(file_path),
                "compressed": True,
                "original_size_mb": metadata["original_size_mb"],
                "compressed_size_mb": metadata["compressed_size_mb"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error compressing document {person_id}: {str(e)}",
            }

    def index_document_collection(self, directory_path: str) -> Dict[str, Any]:
        """
        Index all XML files from a directory into the database.

        Args:
            directory_path: Path to directory containing XML files

        Returns:
            Dictionary with indexing results
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return {
                "success": False,
                "error": f"Directory {directory_path} does not exist",
            }

        xml_files = list(dir_path.glob("*.xml"))
        if not xml_files:
            return {
                "success": False,
                "error": f"No XML files found in {directory_path}",
            }

        results = {
            "total_files": len(xml_files),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "details": [],
        }

        for xml_file in xml_files:
            result = self.index_document(xml_file)
            results["details"].append(result)

            if result["success"]:
                results["successful"] += 1
                print(f"Indexed {xml_file.name} for person_id: {result['person_id']}")
            else:
                results["failed"] += 1
                results["errors"].append(result["error"])
                print(f"Failed to index {xml_file.name}: {result['error']}")

        return results

    def get_document(self, person_id: str) -> Optional[Document]:
        """
        Retrieve a document by person_id.

        Args:
            person_id: The person ID to retrieve

        Returns:
            Document or None if not found
        """
        try:
            doc = self.doc_store.get_document(person_id)
            if doc and doc.metadata.get("compressed"):
                return self._decompress_document(doc)
            return doc
        except KeyError:
            return None

    def get_large_document(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a large document that may have been chunked or compressed.

        Args:
            person_id: The person ID to retrieve

        Returns:
            Dictionary with reconstructed document or None if not found
        """
        try:
            # First try to get the main document
            doc = self.doc_store.get_document(person_id)
            if doc:
                if doc.metadata.get("compressed"):
                    decompressed_doc = self._decompress_document(doc)
                    return {
                        "document": decompressed_doc,
                        "type": "compressed",
                        "original_size_mb": doc.metadata.get("original_size_mb"),
                    }
                elif doc.metadata.get("chunked"):
                    # Get all chunks for this person
                    chunks = self._get_document_chunks(person_id)
                    if chunks:
                        reconstructed_content = self._reconstruct_chunked_document(
                            chunks
                        )
                        return {
                            "document": Document(
                                text=reconstructed_content,
                                metadata=doc.metadata.copy(),
                                id_=person_id,
                            ),
                            "type": "chunked",
                            "chunks": len(chunks),
                            "original_size_mb": doc.metadata.get("size_mb"),
                        }
                else:
                    return {"document": doc, "type": "normal"}

            # Check if this is a chunk
            if "_chunk_" in person_id:
                base_person_id = person_id.split("_chunk_")[0]
                return self.get_large_document(base_person_id)

        except KeyError:
            pass

        return None

    def _decompress_document(self, doc: Document) -> Document:
        """
        Decompress a compressed document.
        """
        import gzip
        import base64

        try:
            # Decode base64 and decompress
            compressed_content = base64.b64decode(doc.text.encode("utf-8"))
            decompressed_content = gzip.decompress(compressed_content).decode("utf-8")

            # Create new document with decompressed content
            metadata = doc.metadata.copy()
            metadata.pop("compressed", None)
            metadata.pop("compressed_size_mb", None)
            metadata.pop("compression_ratio", None)

            return Document(
                text=decompressed_content, metadata=metadata, id_=doc.doc_id
            )
        except Exception as e:
            print(f"Error decompressing document {doc.doc_id}: {e}")
            return doc

    def _get_document_chunks(self, person_id: str) -> List[Document]:
        """
        Get all chunks for a chunked document.
        """
        chunks = []
        all_docs = self.doc_store.docs

        for doc_id, doc in all_docs.items():
            if doc.metadata.get("original_person_id") == person_id and doc.metadata.get(
                "is_chunk"
            ):
                chunks.append(doc)

        # Sort by chunk index
        chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        return chunks

    def _reconstruct_chunked_document(self, chunks: List[Document]) -> str:
        """
        Reconstruct a document from its chunks.
        """
        if not chunks:
            return ""

        # For logical chunks, we need to reconstruct the XML structure
        chunk_type = chunks[0].metadata.get("chunk_type")

        if chunk_type == "logical":
            # Reconstruct XML by combining logical chunks
            reconstructed_parts = []
            for chunk in chunks:
                reconstructed_parts.append(chunk.text)

            # You might need to add XML wrapper here depending on your structure
            return "\n".join(reconstructed_parts)
        else:
            # For size-based chunks, just concatenate
            return "".join(chunk.text for chunk in chunks)

    def delete_document(self, person_id: str) -> bool:
        """
        Delete a document by person_id.

        Args:
            person_id: The person ID to delete

        Returns:
            True if document was deleted, False if not found
        """
        try:
            self.doc_store.delete_document(person_id)
            return True
        except KeyError:
            return False

    def clear_all(self):
        """Clear all documents from the database"""
        # Get all document IDs and delete them
        all_docs = list(self.doc_store.docs.keys())
        for doc_id in all_docs:
            try:
                self.doc_store.delete_document(doc_id)
            except KeyError:
                continue

    def close(self):
        """Close the document store"""
        # llama_index MongoDocumentStore handles connection cleanup
        pass


# Keep the original import for compatibility
MongoDocumentStore = MongoDocumentStore
