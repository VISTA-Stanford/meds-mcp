"""
Script for indexing patient data into MeiliSearch from files or the document store.
"""

import os
from pathlib import Path
import polars as pl
import json
from meilisearch import Client
from meds_mcp.server.rag.simple_storage import initialize_document_store, get_document_store

def index_patients(data_dir="data/collections/dev-corpus", index_name="patients"):
    """Index patient data into MeiliSearch"""
    print(f"Indexing patients from {data_dir}...")
    
    # Initialize MeiliSearch client
    client = Client("http://localhost:7700")
    
    # Create index if it doesn't exist
    try:
        # Check if index exists by trying to get it
        client.get_index(index_name)
        print(f"Index '{index_name}' already exists")
    except Exception:
        # Create index if it doesn't exist
        print(f"Creating index '{index_name}'")
        client.create_index(index_name, {"primaryKey": "patient_id"})
    
    index = client.index(index_name)
    
    # Configure index
    index.update_settings({
        "filterableAttributes": [
            "gender", 
            "age", 
            "age_range",
            "race", 
            "ethnicity",
            "diagnosis_codes",
            "medication_codes",
            "insurance_type",
            "departments"
        ],
        "sortableAttributes": ["age", "encounter_count"],
    })
    
    # Load and prepare patient data
    data_path = Path(data_dir)
    patient_files = list(data_path.glob("*.parquet")) or list(data_path.glob("*.json"))
    
    if not patient_files:
        print(f"No patient files found in {data_dir}")
        return
    
    print(f"Found {len(patient_files)} patient files")
    
    documents = []
    for file_path in patient_files:
        try:
            # Read patient data
            if file_path.suffix == '.parquet':
                patient = pl.read_parquet(file_path).to_dicts()[0]
            else:
                with open(file_path, 'r') as f:
                    patient = json.load(f)
            
            # Prepare document for indexing
            document = {
                "patient_id": patient.get("patient_id", "unknown"),
                "gender": patient.get("gender"),
                "age": patient.get("age"),
                "race": patient.get("race"),
                "ethnicity": patient.get("ethnicity"),
                "insurance_type": patient.get("insurance_type"),
            }
            
            # Add age range for faceting
            age = patient.get("age")
            if age:
                if age < 18:
                    document["age_range"] = "0-17"
                elif age < 35:
                    document["age_range"] = "18-34"
                elif age < 50:
                    document["age_range"] = "35-49"
                elif age < 65:
                    document["age_range"] = "50-64"
                else:
                    document["age_range"] = "65+"
            
            # Extract diagnosis codes
            diagnoses = patient.get("diagnoses", [])
            document["diagnosis_codes"] = [d.get("code") for d in diagnoses if d.get("code")]
            
            # Extract medications
            medications = patient.get("medications", [])
            document["medication_codes"] = [m.get("code") for m in medications if m.get("code")]
            
            # Extract encounters
            encounters = patient.get("encounters", [])
            document["encounter_count"] = len(encounters)
            document["departments"] = list(set(e.get("department") for e in encounters if e.get("department")))
            
            # Add the document
            documents.append(document)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if documents:
        # Index in batches of 50
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            task_info = index.add_documents(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} (Task ID: {task_info.task_uid})")
    
    print(f"Indexed {len(documents)} patients successfully")

def index_patients_from_document_store(data_dir="data/collections/dev-corpus", index_name="patients"):
    # Initialize the document store first
    initialize_document_store(data_dir)
    doc_store = get_document_store()
    if not doc_store:
        print("Document store not initialized")
        return

    # Initialize MeiliSearch client
    client = Client("http://localhost:7700")
    try:
        client.get_index(index_name)
        print(f"Index '{index_name}' already exists")
    except Exception:
        print(f"Creating index '{index_name}'")
        client.create_index(index_name, {"primaryKey": "patient_id"})
    index = client.index(index_name)

    # Configure index (same as before)
    index.update_settings({
        "filterableAttributes": [
            "gender", "age", "age_range", "race", "ethnicity",
            "diagnosis_codes", "medication_codes", "insurance_type", "departments"
        ],
        "sortableAttributes": ["age", "encounter_count"],
    })

    # Get all patients from document store
    patient_ids = doc_store.list_patients()
    print(f"Found {len(patient_ids)} patients in document store")

    documents = []
    for patient_id in patient_ids:
        try:
            # Get patient events (assuming doc_store.get_patient_events returns a list of dicts)
            patient_events = doc_store.get_patient_events(patient_id)
            # Aggregate patient-level info from events
            # For simplicity, use the first event for demographic info
            if not patient_events:
                continue
            first_event = patient_events[0]
            document = {
                "patient_id": patient_id,
                "gender": first_event.get("gender"),
                "age": first_event.get("age"),
                "race": first_event.get("race"),
                "ethnicity": first_event.get("ethnicity"),
                "insurance_type": first_event.get("insurance_type"),
                "age_range": None,
                "diagnosis_codes": [],
                "medication_codes": [],
                "encounter_count": len(patient_events),
                "departments": list(set(e.get("department") for e in patient_events if e.get("department")))
            }
            # Age range
            age = document["age"]
            if age:
                if age < 18:
                    document["age_range"] = "0-17"
                elif age < 35:
                    document["age_range"] = "18-34"
                elif age < 50:
                    document["age_range"] = "35-49"
                elif age < 65:
                    document["age_range"] = "50-64"
                else:
                    document["age_range"] = "65+"
            # Diagnosis and medication codes
            document["diagnosis_codes"] = list({e.get("diagnosis_code") for e in patient_events if e.get("diagnosis_code")})
            document["medication_codes"] = list({e.get("medication_code") for e in patient_events if e.get("medication_code")})

            documents.append(document)
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")

    # Index in batches
    if documents:
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            task_info = index.add_documents(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} (Task ID: {task_info.task_uid})")

    print(f"Indexed {len(documents)} patients from document store successfully")

if __name__ == "__main__":
    index_patients_from_document_store(data_dir="data/collections/dev-corpus")