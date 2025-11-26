"""
Utilities for indexing patient data from XML files into MeiliSearch
from within the meds-mcp server.

This module no longer creates a MeiliSearch client by itself or runs
as a standalone script. Instead, it expects to be called with an
already-configured MCPMeiliSearch instance.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from lxml import etree
from llama_index.core.schema import Document

from meds_mcp.utils.xml_loader import SimpleXMLNodeParser
from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch


def normalize_gender(g: Optional[str]) -> Optional[str]:
    """Normalize gender values from XML to standard format."""
    if not g:
        return None
    g_upper = g.upper()
    if g_upper == "FEMALE":
        return "Female"
    if g_upper == "MALE":
        return "Male"
    return g.title()


def normalize_ethnicity(e: Optional[str]) -> Optional[str]:
    """Normalize ethnicity values from XML to standard format."""
    if not e:
        return None
    e_lower = e.lower()
    if "not hispanic" in e_lower:
        return "Non-Hispanic"
    if "hispanic" in e_lower:
        return "Hispanic"
    return e.title()


def normalize_insurance_type(raw: Optional[str]) -> Optional[str]:
    """Normalize insurance type, keeping only the first value if multiple are present."""
    if not raw:
        return None
    return raw.split("|")[0].strip()


def extract_patient_events(xml_text: str, person_id: str):
    """
    Use SimpleXMLNodeParser to extract all <event> elements and their metadata from XML text.
    Returns a list of nodes with metadata for further aggregation.
    """
    parser = SimpleXMLNodeParser(chunk_element="event", id_metadata_key="person_id")
    doc = Document(text=xml_text, metadata={"person_id": person_id})
    nodes = parser.get_nodes_from_documents([doc])
    return nodes


def parse_patient_xml(filepath: Path) -> Dict[str, Any]:
    """
    Parse a patient XML file and extract demographic info, diagnosis codes, medication codes,
    departments, and encounter count. Uses both direct XML parsing and event extraction.
    Returns a dictionary suitable for indexing in MeiliSearch.
    """
    tree = etree.parse(str(filepath))
    root = tree.getroot()

    person_id = root.attrib.get("person_id")
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    age: Optional[int] = None
    race: Optional[str] = None
    insurance_type: Optional[str] = None
    diagnosis_codes = set()
    medication_codes = set()
    departments = set()
    encounter_count = 0

    # Use the node parser to collect condition / drug_exposure codes from events
    with open(filepath, "r", encoding="utf-8") as f:
        xml_text = f.read()
    events = extract_patient_events(xml_text, person_id)
    for node in events:
        meta = node.metadata or {}
        if meta.get("type") == "condition" and meta.get("code"):
            diagnosis_codes.add(str(meta["code"]).split("/")[-1])
        if meta.get("type") == "drug_exposure" and meta.get("code"):
            medication_codes.add(str(meta["code"]).split("/")[-1])

    # Walk encounters for demographics, departments, and encounter count
    for encounter in root.findall("encounter"):
        person = encounter.find("person")
        if person is not None:
            demo = person.find("demographics")
            if demo is not None:
                gender = normalize_gender(demo.findtext("gender"))
                ethnicity = normalize_ethnicity(demo.findtext("ethnicity"))
                race = demo.findtext("race") or "Unknown"

            age_elem = person.find("age/years")
            if age_elem is not None and age_elem.text:
                try:
                    age = int(age_elem.text)
                except Exception:
                    age = None

            insurance_type = normalize_insurance_type(person.findtext("payerplan"))

        events_elem = encounter.find("events")
        if events_elem is not None:
            for entry in events_elem.findall("entry"):
                for event in entry.findall("event"):
                    etype = event.attrib.get("type")
                    code = event.attrib.get("code")
                    if etype == "condition" and code:
                        diagnosis_codes.add(code.split("/")[-1])
                    if etype == "drug_exposure" and code:
                        medication_codes.add(code.split("/")[-1])
                    if etype == "visit_detail":
                        dept = event.attrib.get("name")
                        if dept:
                            departments.add(dept)
            encounter_count += 1

    # Age range for faceting
    age_range: Optional[str] = None
    if age is not None:
        if age < 18:
            age_range = "0-17"
        elif age < 35:
            age_range = "18-34"
        elif age < 50:
            age_range = "35-49"
        elif age < 65:
            age_range = "50-64"
        else:
            age_range = "65+"

    return {
        "patient_id": person_id,
        "gender": gender,
        "age": age,
        "race": race,
        "ethnicity": ethnicity,
        "insurance_type": insurance_type,
        "age_range": age_range,
        "diagnosis_codes": list(diagnosis_codes) if diagnosis_codes else [],
        "medication_codes": list(medication_codes) if medication_codes else [],
        "departments": list(departments) if departments else [],
        "encounter_count": encounter_count,
    }


def build_patient_index_from_corpus(
    data_dir: str,
    meili: MCPMeiliSearch,
    index_name: str = "patients",
    reset_index: bool = False,
    batch_size: int = 50,
    max_patients: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Index all patient XML files in the given directory into MeiliSearch.

    This function is meant to be called from the server (e.g., during startup)
    with an already-configured MCPMeiliSearch instance.

    Returns a small stats dict:
        {
            "indexed": <number_of_docs>,
            "data_dir": <data_dir>,
            "index_name": <index_name>,
        }
    """
    data_path = Path(data_dir)
    xml_files: List[Path] = list(data_path.glob("*.xml"))
    if not xml_files:
        print(f"[patient_index] No XML files found in {data_dir}")
        return {"indexed": 0, "data_dir": data_dir, "index_name": index_name}

    if max_patients:
        xml_files = xml_files[:max_patients]

    print(f"[patient_index] Found {len(xml_files)} XML files in {data_dir}")

    # Access the underlying index from MCPMeiliSearch
    index = meili.index

    # Optionally reset the index
    if reset_index:
        print(f"[patient_index] Resetting index '{index.uid}'")
        try:
            meili.client.index(index.uid).delete()
        except Exception:
            pass
        meili.client.create_index(index.uid, {"primaryKey": "patient_id"})
        index = meili.client.index(index.uid)

    # Configure index settings (facets / sorting)
    index.update_settings(
        {
            "filterableAttributes": [
                "encounter_count",
                "gender",
                "age",
                "age_range",
                "race",
                "ethnicity",
                "diagnosis_codes",
                "medication_codes",
                "insurance_type",
                "departments",
            ],
            "sortableAttributes": ["age", "encounter_count"],
        }
    )

    documents: List[Dict[str, Any]] = []
    for file_path in xml_files:
        try:
            patient_doc = parse_patient_xml(file_path)
            # skip completely empty docs
            if patient_doc.get("patient_id"):
                documents.append(patient_doc)
        except Exception as e:
            print(f"[patient_index] Error processing {file_path}: {e}")

    if not documents:
        print("[patient_index] No documents extracted from XML files")
        return {"indexed": 0, "data_dir": data_dir, "index_name": index_name}

    # Index in batches
    total = len(documents)
    num_batches = (total + batch_size - 1) // batch_size
    print(f"[patient_index] Indexing {total} patients into '{index.uid}' in {num_batches} batches")

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        task_info = index.add_documents(batch)
        print(
            f"[patient_index] Indexed batch {i // batch_size + 1}/{num_batches} "
            f"(Task: {getattr(task_info, 'task_uid', getattr(task_info, 'taskUid', 'unknown'))})"
        )

    print(f"[patient_index] Indexed {total} patients from XML files successfully")
    return {"indexed": total, "data_dir": data_dir, "index_name": index_name}
