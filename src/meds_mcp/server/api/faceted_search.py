from fastapi import APIRouter, Query
from typing import List, Optional
from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch

router = APIRouter()
searcher = MCPMeiliSearch()

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
    sort_by: str = Query("relevance", enum=["relevance", "age_asc", "age_desc", "encounter_asc", "encounter_desc"]),
    limit: int = 10
):
    filters = []

    # Multi-select facets
    def multi_filter(field, values):
        if values:
            return f"{field} IN [{', '.join([f'\"{v}\"' for v in values])}]"
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

    # Range facets
    if min_encounters is not None:
        filters.append(f"encounter_count >= {min_encounters}")
    if max_encounters is not None:
        filters.append(f"encounter_count <= {max_encounters}")
    if min_age is not None:
        filters.append(f"age >= {min_age}")
    if max_age is not None:
        filters.append(f"age <= {max_age}")

    filter_str = " AND ".join(filters) if filters else None

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