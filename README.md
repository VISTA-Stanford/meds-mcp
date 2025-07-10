# MEDS MCP
MEDS Model Context Protocol (MCP) Server and Client


### Development Roadmap

- [ ] \[TOOL\] Implement retriever backend supporting [faceted search](https://en.wikipedia.org/wiki/Faceted_search) (e.g., [elasticsearch](https://github.com/elastic/elasticsearch), [meilisearch](https://github.com/meilisearch/meilisearch)) for creating a single index over the STARR patient population. 
= [ ] foo


## Launch MCP Server

### Launch server
```
python src/meds_mcp/server/main.py \
--config configs/local.py
```

### Test client
```
python scripts/test_mcp_client_sdk.py
```

### Server Configuration YAML

Lightweight configuration for launching the MCP server

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8000

# Data directories
data:
  # Ontology data directory
  ontology_dir: "data/athena_omop_ontologies"
  # Corpus/collections directory
  corpus_dir: "data/collections/dev-corpus"
  # Use lazy loading for ontology (true/false)
  use_lazy_ontology: false

# Logging settings
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
```