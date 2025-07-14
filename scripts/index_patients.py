"""
Script for indexing patient data from XML files into MeiliSearch.
"""

from pathlib import Path
from lxml import etree
from meilisearch import Client
from meds_mcp.utils.xml_loader import parse_timestamp, SimpleXMLNodeParser

def normalize_gender(g):
    if not g:
        return None
    if g.upper() == "FEMALE":
        return "Female"
    if g.upper() == "MALE":
        return "Male"
    return g.title()

def normalize_ethnicity(e):
    if not e:
        return None
    if "not hispanic" in e.lower():
        return "Non-Hispanic"
    if "hispanic" in e.lower():
        return "Hispanic"
    return e.title()

def normalize_insurance_type(raw):
    if not raw:
        return None
    # If multiple values, keep the first
    return raw.split("|")[0].strip()

def extract_patient_events(xml_text, person_id):
    parser = SimpleXMLNodeParser(chunk_element="event", id_metadata_key="person_id")
    # Wrap xml_text in a Document object as expected by the parser
    from llama_index.core.schema import Document
    doc = Document(text=xml_text, metadata={"person_id": person_id})
    nodes = parser.get_nodes_from_documents([doc])
    return nodes

def parse_patient_xml(filepath):
    tree = etree.parse(str(filepath))
    root = tree.getroot()
    person_id = root.attrib.get("person_id")
    gender = None
    ethnicity = None
    age = None
    race = None
    insurance_type = None
    diagnosis_codes = set()
    medication_codes = set()
    departments = set()
    encounter_count = 0

    with open(filepath, "r") as f:
        xml_text = f.read()
    events = extract_patient_events(xml_text, person_id)
    for node in events:
        meta = node.metadata
        if meta.get("type") == "condition" and meta.get("code"):
            diagnosis_codes.add(meta["code"].split("/")[-1])
        if meta.get("type") == "drug_exposure" and meta.get("code"):
            medication_codes.add(meta["code"].split("/")[-1])

    for encounter in root.findall("encounter"):
        person = encounter.find("person")
        if person is not None:
            demo = person.find("demographics")
            if demo is not None:
                gender = normalize_gender(demo.findtext("gender"))
                ethnicity = normalize_ethnicity(demo.findtext("ethnicity"))
                race = demo.findtext("race") or "Unknown"
            age_elem = person.find("age/years")
            if age_elem is not None:
                try:
                    age = int(age_elem.text)
                except Exception:
                    age = None
            insurance_type = normalize_insurance_type(person.findtext("payerplan"))
        events = encounter.find("events")
        if events is not None:
            for entry in events.findall("entry"):
                for event in entry.findall("event"):
                    if event.attrib.get("type") == "condition":
                        code = event.attrib.get("code")
                        if code:
                            diagnosis_codes.add(code.split("/")[-1])
                    if event.attrib.get("type") == "drug_exposure":
                        code = event.attrib.get("code")
                        if code:
                            medication_codes.add(code.split("/")[-1])
                    if event.attrib.get("type") == "visit_detail":
                        dept = event.attrib.get("name")
                        if dept:
                            departments.add(dept)
            encounter_count += 1

    # Age range for faceting
    age_range = None
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

def index_patients_from_xml(data_dir="data/collections/dev-corpus", index_name="patients"):
    print(f"Indexing patients from XML files in {data_dir}...")

    # Initialize MeiliSearch client
    client = Client("http://localhost:7700")
    try:
        client.get_index(index_name)
        print(f"Index '{index_name}' already exists")
    except Exception:
        print(f"Creating index '{index_name}'")
        client.create_index(index_name, {"primaryKey": "patient_id"})
    index = client.index(index_name)

    # Configure index
    index.update_settings({
        "filterableAttributes": [
            "encounter_count",
            "gender", "age", "age_range", "race", "ethnicity",
            "diagnosis_codes", "medication_codes", "insurance_type", "departments"
        ],
        "sortableAttributes": ["age", "encounter_count"],
    })

    # Find XML files
    data_path = Path(data_dir)
    xml_files = list(data_path.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {data_dir}")
        return

    print(f"Found {len(xml_files)} XML files")

    documents = []
    for file_path in xml_files:
        try:
            patient_doc = parse_patient_xml(file_path)
            documents.append(patient_doc)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Index in batches
    if documents:
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            task_info = index.add_documents(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} (Task ID: {task_info.task_uid})")

    print(f"Indexed {len(documents)} patients from XML files successfully")

if __name__ == "__main__":
    index_patients_from_xml(data_dir="data/collections/dev-corpus")