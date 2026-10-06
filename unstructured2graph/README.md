# unstructured2graph

Convert unstructured documents into knowledge graphs within [Memgraph](https://memgraph.com/).

## Overview

**unstructured2graph** enables you to transform any unstructured data (PDFs, URLs, documents) into a graph database, powering Graph Retrieval-Augmented Generation (GraphRAG) applications. It combines:

- **[Unstructured](https://github.com/Unstructured-IO/unstructured)** - Parse and chunk diverse document formats
- A pluggable extraction backend - **[LightRAG](https://github.com/HKUDS/LightRAG)** (LLM-based) or **[GLiNER2](https://github.com/fastino-ai/GLiNER2)** (local, LLM-free) - to extract entities and relationships
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

GLiNER2, the default extraction backend, comes with the package. Its
`gliner2[local]` dependency pins `transformers<5`, below the CVE-2026-1839
floor the workspace used to hold; restoring it is tracked in #443.

## Choosing an extraction backend

Entity/relation extraction is pluggable behind an `ExtractionBackend`. Two are provided:

| | `GLiNER2Backend` (default) | `LightRAGBackend` |
|---|---|---|
| **Extraction** | Local model, no LLM | LLM-based (via LightRAG) |
| **Cost / network** | Free, fully offline after the model download | Per-call LLM cost, needs `OPENAI_API_KEY` (or another configured LLM) |
| **Cross-chunk coreference** | By type identity — a `global` type (a name) merges across chunks by normalized text, a `chunk` type stays per chunk, a value is one node per mention; no LLM normalization of aliases | Yes — LightRAG's LLM normalizes mentions, so "Apple" in two chunks can merge into one entity |
| **Relation edges** | One Cypher edge type per relation label (e.g. `:works_for`), drawn from the ontology's `relation_types` | One generic `:DIRECTED` edge type (LightRAG's own convention) |
| **Relation vocabulary** | Closed — only extracts relations named in `Ontology.relation_types` | Open-ended (whatever the LLM extracts) |

Both write entities under a **workspace** label with an `entity_type` property and a `file_path` property equal to the source chunk's hash — the rest of the pipeline (chunk-to-entity linking, ontology-gated label promotion) works identically regardless of backend.

## Quick Start

```python
import asyncio
from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import from_unstructured
from unstructured2graph.gliner2_backend import GLiNER2Backend


async def main():
    memgraph = Memgraph(user_agent="unstructured2graph")
    backend = GLiNER2Backend()  # downloads fastino/gliner2.5-base-v1 on first use

    # Ingest documents from URLs or local files
    await from_unstructured(
        sources=["https://example.com/doc.pdf", "./local_file.md"],
        memgraph=memgraph,
        extraction_backend=backend,
        link_chunks=True,  # create NEXT relationships between chunks
        enforce_ontology=True,  # promote entity_type to real labels (:Person, :Organization, ...)
    )


asyncio.run(main())
```

The `Chunk.hash` uniqueness constraint is created for you inside `from_unstructured()` / `from_texts()` — no manual index step is needed.

### Using LightRAG instead (LLM-based)

```python
from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import LightRAGBackend, from_unstructured

lightrag = MemgraphLightRAGWrapper()
await lightrag.initialize(working_dir="./lightrag_storage")

await from_unstructured(
    sources=["./local_file.md"],
    memgraph=memgraph,
    extraction_backend=LightRAGBackend(lightrag),
    enforce_ontology=True,
)
await lightrag.afinalize()
```

`GLiNER2Backend` is not imported by `unstructured2graph`'s top-level package, so `import unstructured2graph` doesn't load torch — import it from `unstructured2graph.gliner2_backend` directly.

It extracts on gliner2's joint path, where each relation type's `start_labels`/`end_labels` constrain decoding. It extracts one window per segment, splitting only a segment longer than `chunk_size` words. It writes:

- entity nodes, merged per the type's `identity` (see the ontology section), each linked to every chunk that mentions it by `MENTIONED_IN`, whose `sources` lists the `source_id` of each segment that mentions it;
- relationships keyed by source chunk and, when the segment has one, its `source_id`, so a fact said in two turns is two relationships. Each carries `chunk`, `confidence`, `valid_at` (a Memgraph `datetime`, from the segment's timestamp), `source_id`, `role` (the speaker) and `text` (the sentence or sentences covering both endpoints, at most 300 characters);
- the chunk's user's own mentions onto `(:User {user_id})`, decided by the `mention_resolver` (default `resolve_user_mentions`). The caller must create that node.

Self-loops left after identity resolution are dropped. `backend.stats` counts windows, re-typed and dropped mentions, and self-loops.

### Ingesting documents verbatim, with segments

`from_documents` stores each `Document`'s text exactly as given, as one `Chunk`, instead of re-chunking it through `unstructured` (whose partitioner rewrites text). Its `segments` travel with the chunk: for a conversation, one per turn, with `role`, `valid_at` and an opaque `source_id` naming the turn's node. The GLiNER2 backend uses them; LightRAG reads the text alone.

```python
from unstructured2graph import Document, Segment, from_documents

text = "user: I ran my best 5K today: 25:50.\n\nassistant: Congratulations!"
turns = (
    Segment(0, 36, "user", "2023-05-30T17:27:00+00:00"),
    Segment(38, len(text), "assistant", "2023-05-30T17:27:01+00:00"),
)
await from_documents([Document(text, turns, user_id="u1")], memgraph, backend, enforce_ontology=True)
```

### Ingesting raw text

For in-memory strings (no file or URL), use `from_texts`. It returns one `Chunk` group per input string, so you can trace an output chunk back to the text that produced it:

```python
from unstructured2graph import from_texts

grouped = await from_texts(
    texts=["Ada Lovelace collaborated with Charles Babbage in London."],
    memgraph=memgraph,
    extraction_backend=LightRAGBackend(lightrag),
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

An extraction backend writes every extracted entity under a single **workspace** label (default `base` for LightRAG, `gliner2` for GLiNER2) with its type only as an `entity_type` *property* — so out of the box you get `(:base {entity_type: "person"})`, not `(:Person)`. unstructured2graph can promote that type into a real Memgraph label. Two independent, opt-in flags on `from_unstructured()` / `from_texts()` (both default `False`):

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
    identity: global      # optional: global | chunk (default) | span
  - label: Organization
    description: Companies, institutions, government bodies, groups
relation_types:  # optional -- read directly by GLiNER2Backend; no LightRAG equivalent
  - label: works_for
    description: Employment relationship between a person and an organization
    start_labels: [Person]        # optional; omitted = any declared entity type
    end_labels: [Organization]
```

Ontologies are parsed and validated by [`hygm`](../hygm/README.md)'s `ManualStrategy`. An endpoint naming an undeclared type is rejected, and so is a `User` endpoint without `Person` beside it. `identity` sets how a GLiNER2 mention becomes a node:
- `global`: one node per normalized text, across every chunk (names);
- `chunk`: one node per chunk (generic nouns);
- `span`: one node per mention (values).

A relation's `description` is documentation only for GLiNER2, whose joint compiler drops it. The label is what steers extraction.

Under `enforce_ontology`, relationships are checked post hoc against `start_labels`/`end_labels` (`enforce_relation_domain_range`). A mismatch is kept and flagged `ontology_conformant = false`, never deleted (ADR 0004). `ontology_report()` reads the counts back: flagged entities and relationships, relationships of undeclared types, and declared types with no instances.

```python
await from_unstructured(
    sources=["./local_file.pdf"],
    memgraph=memgraph,
    extraction_backend=LightRAGBackend(lightrag),
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

`GLiNER2Backend` doesn't need this step — passing it the same `Ontology` (via its `ontology=` constructor argument) already steers extraction directly, entity types and relation types both.

## Key Features

| Feature                  | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| **Multi-format parsing** | PDFs, URLs, HTML, Markdown, DOCX, and more via Unstructured       |
| **Automatic chunking**   | Smart document chunking with configurable options                 |
| **Pluggable extraction** | LLM-powered extraction via LightRAG, or local/offline via GLiNER2 |
| **Typed entities**       | Promote `entity_type` to real labels (`:Person`, ...), optionally gated by an ontology |
| **Typed relations**      | GLiNER2Backend writes relations as per-label edges (e.g. `:works_for`) whose endpoints the ontology constrains at extraction and checks post hoc |
| **Vector search**        | Built-in support for embedding generation and vector indices      |
| **GraphRAG queries**     | Combine vector search with graph traversal for enhanced retrieval |

## API Reference

### Document Processing

- `parse_source(source, partition_kwargs=None)` — parse a single file or URL into a list of `Chunk`s
- `parse_text(text, partition_kwargs=None)` — chunk a raw in-memory string (no file/URL involved)
- `make_chunks(sources, partition_kwargs=None)` — process multiple sources into `ChunkedDocument` objects
- `from_unstructured(sources, memgraph, extraction_backend=None, only_chunks=False, link_chunks=False, entity_workspace=None, partition_kwargs=None, promote_labels=False, enforce_ontology=False, ontology_path=None)` — full ingestion for files/URLs; returns `list[list[Chunk]]`, one group per source
- `from_texts(texts, memgraph, extraction_backend=None, only_chunks=False, entity_workspace=None, promote_labels=False, enforce_ontology=False, ontology_path=None)` — full ingestion for raw strings; returns `list[list[Chunk]]`, one group per input text (no `link_chunks`/`partition_kwargs`)
- `from_documents(documents, memgraph, extraction_backend, entity_workspace=None, promote_labels=False, enforce_ontology=False, ontology_path=None)` — ingest `Document`s verbatim, one `Chunk` each, with their `Segment`s

### Extraction backends

- `ExtractionBackend` — the protocol both backends below satisfy; `workspace_label` (the Memgraph label entities are written under) and `async aingest_chunk(memgraph, chunk)`
- `LightRAGBackend(wrapper)` — wraps an initialized `MemgraphLightRAGWrapper`
- `unstructured2graph.gliner2_backend.GLiNER2Backend(model_name=..., ontology=None, workspace="gliner2", model=None, entity_confidence_threshold=None, relation_confidence_threshold=None, chunk_size=384, chunk_overlap=64, mention_resolver=resolve_user_mentions, candidate_cap=4096)` — local GLiNER2 model, the default backend; not exported from the top-level package, so `import unstructured2graph` stays light

### Ontology

- `load_ontology(path)` → `Ontology` — parse an ontology YAML file
- `Ontology`, `EntityType`, `RelationType` — the vocabulary types; `Ontology.addon_params()` renders LightRAG extraction guidance (entity types only)
- `DEFAULT_ONTOLOGY`, `DEFAULT_ONTOLOGY_PATH` — the bundled default vocabulary (entity-only)
- `promote_entity_types_to_labels(memgraph, workspace_label, ontology)` — the ontology-gated promotion (what `enforce_ontology` calls)
- `promote_all_entity_types_to_labels(memgraph, workspace_label)` — unrestricted promotion (what `promote_labels` calls)
- `enforce_relation_domain_range(memgraph, workspace_label, ontology, chunk_hashes=None)` → `list[ValidationIssue]` — flag relationships outside their domain/range (what `enforce_ontology` also calls)
- `ontology_report(memgraph, workspace_label, ontology, chunk_hashes=None)` → `OntologyReport` — integrity and coverage counts

### Graph Operations

- `create_nodes_from_list(memgraph, nodes, label, batch_size, merge_key=None)` — batch insert; pass `merge_key` to upsert (`MERGE`) instead of `CREATE`
- `connect_chunks_to_entities(memgraph, chunk_label, entity_label)` — link entities to source chunks (`entity_label` is the extraction backend's workspace label, e.g. `base` or `gliner2`)
- `link_nodes_in_order(memgraph, find_label, find_property, from_to_dicts, create_edge_type)` — create sequential relationships between nodes
- `upsert_typed_relationships(memgraph, node_label, match_key, relationships_by_type)` — upsert relationships under distinct Cypher relationship types
- `ensure_lookup(memgraph, label, property)` — unique constraint plus index (in Memgraph a uniqueness constraint alone does not index)
- `create_vector_search_index(memgraph, label, property, dimension=384, index_name="vs_name")` — create a vector index for similarity search
- `compute_embeddings(memgraph, label)` — generate embeddings for nodes

## Documentation

For detailed usage examples and getting started guides, check out the official documentation:

👉 **[unstructured2graph Documentation](https://memgraph.com/docs/ai-ecosystem/unstructured2graph)**

## Requirements

- Python 3.10+
- Memgraph database instance

### LLM API Key (LightRAGBackend only)

`LightRAGBackend` uses LightRAG for entity and relationship extraction, which requires an LLM API key. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

`GLiNER2Backend` needs no API key — it runs entirely locally.
