from lxml import etree
from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
    Document,
)
from typing import List, Optional
import uuid
import datetime


def parse_timestamp(timestamp_str: str) -> Optional[datetime.datetime]:
    """Parse timestamp string to datetime object, trying multiple formats."""
    if not timestamp_str:
        return None
    
    # Parse ISO format: "2008-09-24T11:20:00"
    try:
        return datetime.datetime.fromisoformat(timestamp_str)
    except ValueError:
        pass
    
    # Parse format: "2009-06-24 23:59"
    try:
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
    except ValueError:
        # If parsing fails, return None
        return None


class SimpleXMLNodeParser:
    """Simple chunking of documents based on a specified XML element tag, using lxml,
    and build node IDs from a metadata key (e.g. person_id)."""

    def __init__(
        self,
        chunk_element: str,
        id_metadata_key: Optional[str] = None,
    ):
        """
        :param chunk_element: the XML tag to chunk on (e.g. "entry" or "event")
        :param id_metadata_key: if set, look in doc.metadata[id_metadata_key] for your base_id
        """
        self.chunk_element = chunk_element
        self.id_metadata_key = id_metadata_key
        # strip out blank text nodes
        self._parser = etree.XMLParser(remove_blank_text=True)

    def get_nodes_from_documents(self, documents: List[Document]) -> List[TextNode]:
        all_nodes: List[TextNode] = []

        for doc in documents:
            # pick base_id: metadata first, then doc.doc_id, else random UUID
            if self.id_metadata_key and self.id_metadata_key in (doc.metadata or {}):
                base_id = doc.metadata[self.id_metadata_key]
            else:
                base_id = doc.doc_id or str(uuid.uuid4())
                print(
                    f"Warning: no metadata key {self.id_metadata_key} found in document {doc.doc_id}, using doc_id as base_id"
                )

            root = etree.fromstring(doc.text.encode("utf-8"), parser=self._parser)
            elems = root.xpath(f".//{self.chunk_element}")
            prev_node = None

            for idx, elem in enumerate(elems):
                node_id = f"{base_id}_{self.chunk_element}_{idx}"

                # Add uid attribute to the element
                elem.set("uid", node_id)

                # Get all child elements and add uid attributes
                for child in elem.xpath(".//*"):
                    child_id = f"{node_id}_{child.tag}_{len(child.getparent().xpath(f'.//{child.tag}'))}"
                    child.set("uid", child_id)

                chunk_text = etree.tostring(elem, encoding="unicode", with_tail=False)

                # Get timestamp from parent entry tag if it exists
                timestamp = None
                if self.chunk_element == "event":
                    parent_entry = elem.getparent()
                    if parent_entry is not None and parent_entry.tag == "entry":
                        timestamp_str = parent_entry.get("timestamp")
                        if timestamp_str:
                            timestamp = parse_timestamp(timestamp_str)
                elif self.chunk_element == "entry":
                    timestamp_str = elem.get("timestamp")
                    if timestamp_str:
                        timestamp = parse_timestamp(timestamp_str)

                # Extract XML attributes for events
                event_metadata = {
                    "source_doc": doc.doc_id,
                    "timestamp": timestamp,
                    "uid": node_id,
                }
                
                # If chunking on events, extract common XML attributes
                if self.chunk_element == "event":
                    # Extract all attributes from the XML element
                    for attr_name, attr_value in elem.attrib.items():
                        event_metadata[attr_name] = attr_value
                    
                    # Extract text content of the event element
                    event_text = elem.text.strip() if elem.text else ""
                    event_metadata["value"] = event_text
                    
                    # Also extract event_type if available
                    event_type = elem.get("type") or elem.get("category") or elem.tag
                    event_metadata["event_type"] = event_type

                node = TextNode(
                    text=chunk_text,
                    id_=node_id,
                    metadata=event_metadata,
                )

                # link to previous/next
                if prev_node is not None:
                    prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                        node_id=node.node_id
                    )
                    node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                        node_id=prev_node.node_id
                    )

                # always link back to source document
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=doc.doc_id
                )

                all_nodes.append(node)
                prev_node = node

        return all_nodes
