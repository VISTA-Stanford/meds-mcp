"""
Faceted search API for patient data using MeiliSearch.
"""
import json
from typing import List, Optional
from fastapi import APIRouter, Query
from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch

router = APIRouter()
searcher = MCPMeiliSearch()

searcher.index.update_settings({
    "filterableAttributes": [
        "encounter_count",
        "gender", "age", "age_range", "race", "ethnicity",
        "diagnosis_codes", "medication_codes", "insurance_type", "departments"
    ],
    "sortableAttributes": ["age", "encounter_count"],
})

@router.get("/search")
def search_patients(
    query: str = "",
    gender: Optional[List[str]] = Query(None),
    age_range: Optional[List[str]] = Query(None),
    race: Optional[List[str]] = Query(None),
    ethnicity: Optional[List[str]] = Query(None),
    insurance_type: Optional[List[str]] = Query(None),
    department: Optional[List[str]] = Query(None),
    diagnosis_code: Optional[List[str]] = Query(None),
    medication_code: Optional[List[str]] = Query(None),
    min_encounters: int = Query(None),
    max_encounters: int = Query(None),
    min_age: int = Query(None),
    max_age: int = Query(None),
    sort_by: str = Query(
        "relevance",
        enum=["relevance", "age_asc", "age_desc", "encounter_asc", "encounter_desc"]),
    limit: int = 10
):
    """
    Search for patients with faceted search capabilities.
    """
    filters = []

    # Multi-select facets
    def multi_filter(field, values):
        if values:
            return f'{field} IN [{", ".join([json.dumps(v) for v in values])}]'
        return None

    for field, values in [
        ("gender", gender),
        ("age_range", age_range),
        ("race", race),
        ("ethnicity", ethnicity),
        ("insurance_type", insurance_type),
        ("departments", department),
        ("diagnosis_codes", diagnosis_code),
        ("medication_codes", medication_code),
    ]:
        f = multi_filter(field, values)
        if f:
            filters.append(f)

    # Range facets (only add if not None and > 0)
    if min_encounters is not None and min_encounters > 0:
        filters.append(f"encounter_count >= {min_encounters}")
    if max_encounters is not None and max_encounters > 0:
        filters.append(f"encounter_count <= {max_encounters}")
    if min_age is not None and min_age > 0:
        filters.append(f"age >= {min_age}")
    if max_age is not None and max_age > 0:
        filters.append(f"age <= {max_age}")

    filter_str = " AND ".join(filters) if filters else None

    # Debug: print the filter string
    print("Filter string:", filter_str)

    # Facets for UI
    facets = [
        "gender", "age_range", "race", "ethnicity", "insurance_type",
        "departments", "diagnosis_codes", "medication_codes"
    ]

    # Sorting
    sort_map = {
        "age_asc": "age:asc",
        "age_desc": "age:desc",
        "encounter_asc": "encounter_count:asc",
        "encounter_desc": "encounter_count:desc"
    }
    sort = sort_map.get(sort_by) if sort_by != "relevance" else None

    params = {
        "filter": filter_str,
        "facets": facets,
        "limit": limit
    }
    if sort:
        params["sort"] = [sort]

    return searcher.index.search(query, params)
