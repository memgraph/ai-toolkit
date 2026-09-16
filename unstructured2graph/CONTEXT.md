# unstructured2graph

Converts files, URLs, or raw text into `Chunk` nodes in Memgraph. Full ingestion
also sends each Chunk to LightRAG for entity and relationship extraction.

## Language

**Chunk**:
The smallest persisted unit of input: text plus its SHA-256 hash. One hash maps
to one `Chunk` node and acts as its deduplication key. LightRAG also uses this
hash to link extracted entities back to the Chunk.
_Avoid_: segment, passage, excerpt

**Source**:
A file path or URL used to load and group content. It remains on the in-memory
ChunkedDocument during ingestion; no Source node is persisted.
_Avoid_: document (when referring to raw input)

**ChunkedDocument**:
Chunks created from one Source, paired with that Source. Raw Text does not create
a ChunkedDocument because it has no path or URL to preserve.
_Avoid_: using "document" for raw Text

**Text**:
An in-memory string with no stable source identity. Chunks from Text are not
linked with `NEXT`; separate strings do not form one ordered document.
_Avoid_: document, source

## Ingestion modes

**Full Ingestion**:
Persists Chunks, sends them to LightRAG, then links extracted entities back with
`MENTIONED_IN`. Requires `MemgraphLightRAGWrapper`.

**Chunk-Only Ingestion**:
Persists Chunks without LightRAG or entity extraction. Use this to defer the
costly extraction step.
_Avoid_: `only_chunks` mode (parameter name, not domain term)

**Sequential Linking**:
Links one Source's Chunks in reading order with `NEXT`. It is optional and off by
default. It applies to Source input, not raw Text.
_Avoid_: `link_chunks` (parameter name, not domain term)

## Entity typing

**Workspace**:
The Memgraph label LightRAG adds to extracted entity nodes. It separates one
LightRAG workspace from another and remains on the node after label promotion.
The default fallback is `base`.
_Avoid_: entity label

**Entity Type**:
LightRAG's raw classification string on an entity, such as `person` or
`organization`. It remains a property even when promoted to a Memgraph label.

**Ontology**:
A YAML-defined list of allowed entity types (`label` and `description`). It can
restrict Label Promotion. A caller may also render the same Ontology as LightRAG
prompt guidance, but must pass it separately. Prompt guidance does not validate
LightRAG output.
_Avoid_: schema (Memgraph does not enforce it as a DB schema)

**Label Promotion**:
Adds a Memgraph label derived from Entity Type without removing the Workspace
label. Unrestricted promotion accepts every sanitized Entity Type. Ontology-gated
promotion accepts only types in the Ontology. If both modes are requested, the
gated mode wins.

**Ontology Conformance**:
Whether an Entity Type appears in the Ontology during gated Label Promotion.
Non-conforming entities remain stored with their raw Entity Type and receive
`ontology_conformant: false`; only label promotion is withheld.

## Flagged ambiguities

- Ontology prompt guidance and Ontology-gated Label Promotion use the same file
  but enforce different things. Guidance steers the LLM; promotion checks its
  output in application code.
- Ontology Conformance never rejects an entity. It only controls label promotion.
- Label Promotion does not require an Ontology. Unrestricted promotion has no
  conformance check.
