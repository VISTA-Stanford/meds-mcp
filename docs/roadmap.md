# Development Roadmap

- [ ] Add support for general (non-Stanford) secure LLM client wrapper, e.g., [aisuite](https://github.com/andrewyng/aisuite)
- [ ] Implement retriever backend supporting [faceted search](https://en.wikipedia.org/wiki/faceted_search) (e.g., [elasticsearch](https://github.com/elastic/elasticsearch), [meilisearch](https://github.com/meilisearch/meilisearch)) for creating a single index over the STARR patient population.
- [ ] Bake-off vector store retrievers for event/document embeddings vs. BM25 baseline
- [ ] Support native OMOP queries via BigQuery
