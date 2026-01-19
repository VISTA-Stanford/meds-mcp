# src/meds_mcp/similarity/candidates.py

"""
Meilisearch candidate retrieval (using the existing wrapper)

This module provides functionality to retrieve candidate similar patients
from a MeiliSearch index using an already-configured MCPMeiliSearch instance.
It is designed to be used within the similarity retrieval pipeline,
specifically for fetching candidate patients based on vignette text.
"""


from meds_mcp.server.tools.meilisearch_client import MCPMeiliSearch


class MeiliCandidateRetriever:
    def __init__(self, meili: MCPMeiliSearch):
        self.meili = meili

    def retrieve(
        self,
        query_text: str,
        limit: int = 500,
        filters: str | None = None,
    ) -> list[dict]:
        res = self.meili.search(
            query=query_text,
            filters=filters,
            limit=limit,
        )
        return [
            {
                "patient_id": h["patient_id"],
                "meili_score": h.get("_score", 0.0),
            }
            for h in res.get("hits", [])
        ]
