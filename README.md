# MEDS MCP
MEDS Model Context Protocol (MCP) Server and Client



### Development Roadmap

- [ ] Implment a more flexible document store than MongoDB
- [ ] Implement retriever backend supporting [faceted search](https://en.wikipedia.org/wiki/Faceted_search) (e.g., [elasticsearch](https://github.com/elastic/elasticsearch), [meilisearch](https://github.com/meilisearch/meilisearch)) for creating a single index over the STARR patient population. 



MongoDB Configuration

For MacOS

`brew services start mongodb-community@8.0`

`brew services stop mongodb-community@8.0`


config file at `/opt/homebrew/etc/mongod.conf`

```
# Add or modify this section
operationProfiling:
  mode: slowOp
  slowOpThresholdMs: 100

# Add this line to increase document size limit
setParameter:
  maxBSONObjectSize: 31457280  # 30MB in bytes
  ```

  Verify the new settings

  mongosh

// Check the current setting
db.adminCommand({getParameter: 1, maxBSONObjectSize: 1})