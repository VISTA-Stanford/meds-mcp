"""
Provides a wrapper class for MeiliSearch client operations, including index management and faceted search for patient data in the MCP server.
"""

from meilisearch import Client

class MCPMeiliSearch:
    def __init__(self, host="http://localhost:7700", index_name="patients", reset=False):
        self.client = Client(host)
        self.index = self.client.index(index_name)
        if reset:
            self.index.delete()
            self.client.create_index(index_name, {"primaryKey": "patient_id"})
        else:
            try:
                self.client.get_index(index_name)
            except Exception:
                self.client.create_index(index_name, {"primaryKey": "patient_id"})

    def search(self, query="", filters=None, facets=None, limit=10):
        params = {}
        if filters:
            params["filter"] = filters
        if facets:
            params["facets"] = facets
        params["limit"] = limit
        return self.index.search(query, params)