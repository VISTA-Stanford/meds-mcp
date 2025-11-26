"""
MCP-facing MeiliSearch tools and initialization utilities.

This module:

- Initializes a global MCPMeiliSearch client from the server config.
- Exposes MCP tools for:
    - searching patients (`search_patients`)
    - (re)building the patient index (`reindex_patients`)

These tools are intended to be registered in `main.py` using FastMCP, e.g.:

    from meds_mcp.server.tools.meilisearch_tools import search_patients, reindex_patients

    mcp = FastMCP(name="meds-mcp-server")
    mcp.tool("search_patients")(search_patients)
    mcp.tool("reindex_patients")(reindex_patients)
"""

from typing import Optional, Dict, Any, List

from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch
from meds_mcp.server.indexing.index_patients import build_patient_index_from_corpus

# Globals used by the tools
_meili_client: Optional[MCPMeiliSearch] = None
_config: Optional[Dict[str, Any]] = None


def initialize_meilisearch_from_config(config: Dict[str, Any]) -> Optional[MCPMeiliSearch]:
    """
    Initialize the global MCPMeiliSearch client from the server config.

    Expected config structure:

        meilisearch:
          enabled: true
          host: "http://localhost:7700"
          index_name: "patients"
          reset_on_startup: false
          auto_index: true

    This should be called once in `initialize_server(config)`.
    """
    global _meili_client, _config
    _config = config

    meili_cfg = config.get("meilisearch", {})
    if not meili_cfg.get("enabled", False):
        # Meili is optional; return None if disabled
        return None

    host = meili_cfg.get("host", "http://localhost:7700")
    index_name = meili_cfg.get("index_name", "patients")
    reset_on_startup = bool(meili_cfg.get("reset_on_startup", False))

    _meili_client = MCPMeiliSearch(
        host=host,
        index_name=index_name,
        reset=reset_on_startup,
    )
    return _meili_client


def get_meili() -> MCPMeiliSearch:
    """
    Return the global MCPMeiliSearch client, or raise if not initialized.
    """
    if _meili_client is None:
        raise RuntimeError(
            "MeiliSearch is not initialized. "
            "Call initialize_meilisearch_from_config(config) in initialize_server()."
        )
    return _meili_client


def _dict_to_meili_filter(filters: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Convert a simple dict of filters into a MeiliSearch filter string.

    Example input:
        {
            "gender": "Female",
            "race": ["White", "Asian"],
            "age_range": "50-64",
            "diagnosis_codes": ["I10", "E11"],
        }

    Output:
        "(race = 'White' OR race = 'Asian') AND gender = 'Female' AND age_range = '50-64' AND (diagnosis_codes = 'I10' OR diagnosis_codes = 'E11')"
    """
    if not filters:
        return None

    clauses: List[str] = []

    for key, value in filters.items():
        if value is None:
            continue

        # List -> OR clause
        if isinstance(value, list):
            if not value:
                continue
            ors = " OR ".join([f"{key} = '{v}'" for v in value])
            clauses.append(f"({ors})")
        else:
            # Scalar -> simple equality
            clauses.append(f"{key} = '{value}'")

    if not clauses:
        return None

    return " AND ".join(clauses)


async def search_patients(
    query: str = "",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    MCP tool: search patients using the MeiliSearch 'patients' index.

    Arguments
    ---------
    query : str
        Free-text query string. Can be empty "" for pure faceted search.
    filters : dict, optional
        Simple filter dict that will be converted into Meili filter syntax.
        Example:
            {
                "gender": "Female",
                "race": ["White", "Asian"],
                "age_range": "50-64",
                "diagnosis_codes": ["I10", "E11"],
            }
        Valid keys correspond to the fields indexed by `build_patient_index_from_corpus`,
        such as:
            - "gender"
            - "age"
            - "age_range"
            - "race"
            - "ethnicity"
            - "insurance_type"
            - "diagnosis_codes"
            - "medication_codes"
            - "departments"
            - "encounter_count"
    limit : int
        Maximum number of hits to return.

    Returns
    -------
    dict
        {
            "hits": [...],
            "total": <estimatedTotalHits>,
            "query": <query>,
            "filters": <filters>,
        }
    """
    meili = get_meili()
    filter_str = _dict_to_meili_filter(filters)

    result = meili.search(
        query=query,
        filters=filter_str,
        facets=None,  # You can expose facets later if desired
        limit=limit,
    )

    hits = result.get("hits", [])
    total = result.get("estimatedTotalHits", len(hits))

    return {
        "hits": hits,
        "total": total,
        "query": query,
        "filters": filters or {},
    }


async def reindex_patients(
    reset: bool = True,
) -> Dict[str, Any]:
    """
    MCP tool: trigger a full re-index of the patient corpus into MeiliSearch.

    This uses the `data.corpus_dir` from the server config and rebuilds
    the Meili index using `build_patient_index_from_corpus`.

    Arguments
    ---------
    reset : bool
        If True, deletes/recreates the index before indexing.
        If False, upserts into the existing index.

    Returns
    -------
    dict
        A small stats dict from `build_patient_index_from_corpus`, e.g.:
            {
                "indexed": 123,
                "data_dir": "data/collections/dev-corpus",
                "index_name": "patients",
            }
    """
    if _config is None:
        raise RuntimeError(
            "Server config not registered in meilisearch_tools. "
            "Call initialize_meilisearch_from_config(config) during server initialization."
        )

    data_cfg = _config.get("data", {})
    corpus_dir = data_cfg.get("corpus_dir", "data/collections/dev-corpus")

    meili = get_meili()
    stats = build_patient_index_from_corpus(
        data_dir=corpus_dir,
        meili=meili,
        index_name=meili.index.uid,
        reset_index=reset,
    )
    return stats
