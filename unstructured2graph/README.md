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
from unstructured2graph import from_unstructured


async def main():
    memgraph = Memgraph(user_agent="unstructured2graph")

    lightrag = MemgraphLightRAGWrapper()
    await lightrag.initialize(working_dir="./lightrag_storage")

    # Ingest documents from URLs or local files
    await from_unstructured(
        sources=["https://example.com/doc.pdf", "./local_file.md"],
        memgraph=memgraph,
        lightrag_wrapper=lightrag,
        link_chunks=True,  # create NEXT relationships between chunks
        enforce_ontology=True,  # promote entity_type to real labels (:Person, :Organization, ...)
    )
    await lightrag.afinalize()


asyncio.run(main())
```

The `Chunk.hash` uniqueness constraint is created for you inside `from_unstructured()` / `from_texts()` — no manual index step is needed.

### Ingesting raw text

For in-memory strings (no file or URL), use `from_texts`. It returns one `Chunk` group per input string, so you can trace an output chunk back to the text that produced it:

```python
from unstructured2graph import from_texts

grouped = await from_texts(
    texts=["Ada Lovelace collaborated with Charles Babbage in London."],
    memgraph=memgraph,
    lightrag_wrapper=lightrag,
    enforce_ontology=True,
)
```

> **Persistence:** `MemgraphLightRAGWrapper` now persists LightRAG's *full*
> working state into Memgraph by default — the entity/relationship graph plus
> the key/value store, vector store, and document-status store. The
> `working_dir` argument is still accepted (and used as a fallback location for
> any store not backed by Memgraph), but with the default settings the JSON
> stores are no longer written there. See the
> [lightrag-memgraph README](../integrations/lightrag-memgraph/README.md#storage--persistence)
> for the label/index schema and opt-out flags.

## Entity typing / ontology

LightRAG writes every extracted entity under a single **workspace** label (default `base`) with its type only as an `entity_type` *property* — so out of the box you get `(:base {entity_type: "person"})`, not `(:Person)`. unstructured2graph can promote that type into a real Memgraph label. Two independent, opt-in flags on `from_unstructured()` / `from_texts()` (both default `False`):

| Flag | Behavior |
|---|---|
| `promote_labels=True` | Promote **every** `entity_type` to a PascalCase label (`"natural object"` → `:NaturalObject`). No fixed vocabulary, no conformance flagging. |
| `enforce_ontology=True` | Promote only types in an **ontology**; entities outside it are kept but flagged `ontology_conformant = false`. Takes precedence over `promote_labels`. |

Neither flag ever deletes or rejects a node — the workspace label and raw `entity_type` are always preserved. Re-running after growing the ontology clears the flag on entities that now conform.

The ontology is a YAML file. `ontology_path` defaults to a bundled `default_ontology.yaml` that mirrors LightRAG's built-in vocabulary (Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject). A custom one looks like:

```yaml
entity_types:
  - label: Person
    description: Human individuals, real or fictional
  - label: Organization
    description: Companies, institutions, government bodies, groups
```

```python
await from_unstructured(
    sources=["./local_file.pdf"],
    memgraph=memgraph,
    lightrag_wrapper=lightrag,
    enforce_ontology=True,
    ontology_path="my_ontology.yaml",  # omit to use the bundled default
)
```

**Steering extraction with the same vocabulary (optional).** The flags above gate *promotion* after extraction. To also steer what LightRAG *extracts*, load the same YAML and pass its `addon_params()` into the wrapper — using one path at both sites keeps them in sync:

```python
from unstructured2graph import load_ontology

ontology = load_ontology("my_ontology.yaml")
await lightrag.initialize(
    working_dir="./lightrag_storage",
    addon_params=ontology.addon_params(),  # {"entity_types_guidance": "..."}
)
await from_unstructured(..., enforce_ontology=True, ontology_path="my_ontology.yaml")
```

## Key Features

| Feature                  | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| **Multi-format parsing** | PDFs, URLs, HTML, Markdown, DOCX, and more via Unstructured       |
| **Automatic chunking**   | Smart document chunking with configurable options                 |
| **Entity extraction**    | LLM-powered entity and relationship extraction via LightRAG       |
| **Typed entities**       | Promote `entity_type` to real labels (`:Person`, ...), optionally gated by an ontology |
| **Vector search**        | Built-in support for embedding generation and vector indices      |
| **GraphRAG queries**     | Combine vector search with graph traversal for enhanced retrieval |

## API Reference

### Document Processing

- `parse_source(source, partition_kwargs=None)` — parse a single file or URL into a list of `Chunk`s
- `parse_text(text, partition_kwargs=None)` — chunk a raw in-memory string (no file/URL involved)
- `make_chunks(sources, partition_kwargs=None)` — process multiple sources into `ChunkedDocument` objects
- `from_unstructured(sources, memgraph, lightrag_wrapper=None, only_chunks=False, link_chunks=False, entity_workspace=None, partition_kwargs=None, promote_labels=False, enforce_ontology=False, ontology_path=None)` — full ingestion for files/URLs; returns `list[list[Chunk]]`, one group per source
- `from_texts(texts, memgraph, lightrag_wrapper=None, only_chunks=False, entity_workspace=None, promote_labels=False, enforce_ontology=False, ontology_path=None)` — full ingestion for raw strings; returns `list[list[Chunk]]`, one group per input text (no `link_chunks`/`partition_kwargs`)

### Ontology

- `load_ontology(path)` → `Ontology` — parse an ontology YAML file
- `Ontology`, `EntityType` — the vocabulary types; `Ontology.addon_params()` renders LightRAG extraction guidance
- `DEFAULT_ONTOLOGY`, `DEFAULT_ONTOLOGY_PATH` — the bundled default vocabulary
- `promote_entity_types_to_labels(memgraph, workspace_label, ontology)` — the ontology-gated promotion (what `enforce_ontology` calls)
- `promote_all_entity_types_to_labels(memgraph, workspace_label)` — unrestricted promotion (what `promote_labels` calls)

### Graph Operations

- `create_nodes_from_list(memgraph, nodes, label, batch_size, merge_key=None)` — batch insert; pass `merge_key` to upsert (`MERGE`) instead of `CREATE`
- `connect_chunks_to_entities(memgraph, chunk_label, entity_label)` — link entities to source chunks (`entity_label` is the LightRAG workspace label, e.g. `base`)
- `link_nodes_in_order(memgraph, find_label, find_property, from_to_dicts, create_edge_type)` — create sequential relationships between nodes
- `create_vector_search_index(memgraph, label, property, dimension=384, index_name="vs_name")` — create a vector index for similarity search
- `compute_embeddings(memgraph, label)` — generate embeddings for nodes

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
