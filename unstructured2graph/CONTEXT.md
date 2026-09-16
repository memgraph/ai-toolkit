# unstructured2graph

Converts files, URLs, raw text into `Chunk` nodes in Memgraph. Full ingestion also sends each Chunk to LightRAG for entity/relationship extraction.

## Language

**Chunk**:
Smallest persisted unit of input: text + SHA-256 hash. One hash -> one `Chunk` node, acts as dedup key. LightRAG also uses hash to link extracted entities back to Chunk.
_Avoid_: segment, passage, excerpt

**Source**:
File path/URL used to load + group content. Stays on in-memory ChunkedDocument during ingestion; no Source node persisted.
_Avoid_: document (for raw input)

**ChunkedDocument**:
Chunks from one Source, paired with that Source. Raw Text never creates one — no path/URL to preserve.
_Avoid_: "document" for raw Text

**Text**:
In-memory string, no stable source identity. Chunks from Text never linked via `NEXT` — separate strings don't form one ordered document.
_Avoid_: document, source

## Ingestion modes

**Full Ingestion**:
Persists Chunks, sends to LightRAG, links extracted entities back via `MENTIONED_IN`. Requires `MemgraphLightRAGWrapper`.

**Chunk-Only Ingestion**:
Persists Chunks, no LightRAG/entity extraction. Use to defer costly extraction step.
_Avoid_: `only_chunks` mode (parameter name, not domain term)

**Sequential Linking**:
Links one Source's Chunks in reading order via `NEXT`. Optional, off by default. Applies to Source input only, not raw Text.
_Avoid_: `link_chunks` (parameter name, not domain term)

## Entity typing

**Workspace**:
Memgraph label LightRAG adds to extracted entity nodes. Separates one LightRAG workspace from another; stays on node after label promotion. Default fallback `base`.
_Avoid_: entity label

**Entity Type**:
LightRAG's raw classification string on an entity (e.g. `person`, `organization`). Stays a property even after promotion to a Memgraph label.

**Ontology**:
YAML-defined list of allowed entity types (`label` + `description`). Can restrict Label Promotion. Caller may also render same Ontology as LightRAG prompt guidance, but must pass it separately — guidance doesn't validate LightRAG output.
_Avoid_: schema (Memgraph doesn't enforce it as DB schema)

**Label Promotion**:
Adds Memgraph label derived from Entity Type, without removing Workspace label. Unrestricted promotion: accepts every sanitized Entity Type. Ontology-gated promotion: only types in Ontology. Both requested -> gated wins.

**Ontology Conformance**:
Whether Entity Type appears in Ontology, during gated Label Promotion. Non-conforming entities stay stored with raw Entity Type + get `ontology_conformant: false` — only label promotion withheld.

## Flagged ambiguities

- Ontology prompt guidance and Ontology-gated Label Promotion share one file, enforce different things. Guidance steers LLM; promotion checks its output in application code.
- Ontology Conformance never rejects an entity. Only controls label promotion.
- Label Promotion doesn't require an Ontology. Unrestricted promotion has no conformance check.
