# MEDS MCP
MEDS Model Context Protocol (MCP) Server and Client


### Development Roadmap

- [ ] **\[TOOL\]** Implement retriever backend supporting [faceted search](https://en.wikipedia.org/wiki/Faceted_search) (e.g., [elasticsearch](https://github.com/elastic/elasticsearch), [meilisearch](https://github.com/meilisearch/meilisearch)) for creating a single index over the STARR patient population. 
- [ ] **\[TOOL\]** Incorporate and bake-off vector store retrievers for event/document embedding infrastructure
- [ ] **\[TOOL\]** Support native OMOP queries via BigQuery

## I. Launch MCP Server

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


## II. MCP Chat Demo 

### 1. Apply for Access to MedAlign 

> [!WARNING] The Stanford Dataset DUA prohibts sharing data with third parties including LLM API providers. We follow the [guidelines for responsible](https://physionet.org/news/post/gpt-responsible-use) use as originally outlined by PhysioNet:
> 
> *If you are interested in using the GPT family of models, we suggest using one of the following services:*
>
> - *Azure OpenAI service. You'll need to opt out of human review of the data via this form. Reasons for opting out are: 1) you are processing sensitive data where the likelihood of harmful outputs and/or misuse is low, and 2) you do not have the right to permit Microsoft to process the data for abuse detection due to the data use agreement you have signed.*
> - *Amazon Bedrock. Bedrock provides options for fine-tuning foundation models using private labeled data. After creating a copy of a base foundation model for exclusive use, data is not shared back to the base model for training.*
> - *Google's Gemini via Vertex AI on Google Cloud Platform. Gemini doesn't use your prompts or its responses as data to train its models. If making use of additional features offered through the Gemini for Google Cloud Trusted Tester Program, you should obtain the appropriate opt-outs for data sharing, or otherwise not perform tasks that require the sharing of data.*
> - *Anthropic Claude. Claude does not use your prompts or its responses as data to train its models by default, and routine human review of data is not performed.*

```bash
export REDIVIS_ACCESS_TOKEN="your_token_here"
python scripts/download_data.py medalign --files
```

### 2. Initalize and Launch the MCP Server

```bash
python src/meds_mcp/server/main.py \
--config configs/medalign.yaml
```

### 3. Launch Gradio Chat Demo

```bash
python examples/mcp_chat_demo/demo.py \
--model "apim:gpt-4o-mini" \
--mcp_url "http://localhost:8000/mcp"
```