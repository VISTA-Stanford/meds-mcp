# MEDS MCP
MEDS Model Context Protocol (MCP) Server and Client



### Development Roadmap

- [ ] Implment a more flexible document store than MongoDB
- [ ] Implement retriever backend supporting [faceted search](https://en.wikipedia.org/wiki/Faceted_search) (e.g., [elasticsearch](https://github.com/elastic/elasticsearch), [meilisearch](https://github.com/meilisearch/meilisearch)) for creating a single index over the STARR patient population. 


## Launch MCP Server & Test Client

### Launch server
```
python src/meds_mcp/server/main.py
```

### Launch client tests
```
python scripts/test_mcp_client_sdk.py
```