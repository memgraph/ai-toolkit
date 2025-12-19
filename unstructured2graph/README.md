# unstructured2graph

Convert unstructured documents into knowledge graphs within [Memgraph](https://memgraph.com/).

## Overview

**unstructured2graph** enables you to transform any unstructured data (PDFs, URLs, documents) into a graph database, powering Graph Retrieval-Augmented Generation (GraphRAG) applications. It combines:

- **[Unstructured](https://github.com/Unstructured-IO/unstructured)** - Parse and chunk diverse document formats
- **[LightRAG](https://github.com/HKUDS/LightRAG)** - Extract entities and relationships using LLMs
- **[Memgraph](https://memgraph.com/)** - Store and query your knowledge graph

## Installation

Install from source:

```bash
git clone https://github.com/memgraph/ai-toolkit.git
cd ai-toolkit/unstructured2graph
pip install -e .
```

For full document support (PDF, DOCX, etc.):

```bash
pip install -e ".[all-docs]"
```

## Quick Start

```python
import asyncio
from memgraph_toolbox.api.memgraph import Memgraph
from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import from_unstructured, create_index

async def main():
    memgraph = Memgraph()
    create_index(memgraph, "Chunk", "hash")

    lightrag = MemgraphLightRAGWrapper()
    await lightrag.initialize(working_dir="./lightrag_storage")

    # Ingest documents from URLs or local files
    await from_unstructured(
        sources=["https://example.com/doc.pdf", "./local_file.md"],
        memgraph=memgraph,
        lightrag_wrapper=lightrag,
        link_chunks=True,  # Create NEXT relationships between chunks
    )
    await lightrag.afinalize()

asyncio.run(main())
```

## Key Features

| Feature                  | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| **Multi-format parsing** | PDFs, URLs, HTML, Markdown, DOCX, and more via Unstructured       |
| **Automatic chunking**   | Smart document chunking with configurable options                 |
| **Entity extraction**    | LLM-powered entity and relationship extraction via LightRAG       |
| **Vector search**        | Built-in support for embedding generation and vector indices      |
| **GraphRAG queries**     | Combine vector search with graph traversal for enhanced retrieval |

## API Reference

### Document Processing

- `parse_source(source, partition_kwargs)` - Parse a single file or URL into chunks
- `make_chunks(sources, partition_kwargs)` - Process multiple sources into `ChunkedDocument` objects
- `from_unstructured(sources, memgraph, lightrag_wrapper, ...)` - Full ingestion pipeline

### Graph Operations

- `create_nodes_from_list(memgraph, nodes, label, batch_size)` - Batch insert nodes
- `connect_chunks_to_entities(memgraph, chunk_label, entity_label)` - Link entities to source chunks
- `link_nodes_in_order(memgraph, ...)` - Create sequential relationships between chunks
- `create_vector_search_index(memgraph, label, property)` - Create vector index for similarity search
- `compute_embeddings(memgraph, label)` - Generate embeddings for nodes

## Documentation

For detailed usage examples and getting started guides, check out the official documentation:

👉 **[unstructured2graph Documentation](https://memgraph.com/docs/ai-ecosystem/unstructured2graph)**

## Requirements

- Python 3.10+
- Memgraph database instance

### LLM API Key

This library uses LightRAG for entity and relationship extraction, which requires an LLM API key. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```
