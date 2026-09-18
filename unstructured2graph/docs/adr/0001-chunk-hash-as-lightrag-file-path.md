# Use Chunk.hash as LightRAG's file_path for entity linking

LightRAG's `ainsert(file_paths=...)` only exposes a per-input identifier meant for actual file paths, no purpose-built back-reference field. Pass each `Chunk.hash` as that identifier so `connect_chunks_to_entities` can later link extracted entities to source chunks via simple property match (`n.file_path = m.hash`) — no separate hash-to-entity mapping outside LightRAG.

**Generalized by the `ExtractionBackend` protocol**: `file_path=chunk.hash` on every extracted entity node is now the contract every backend must satisfy (see `extraction_backend.py`), not a LightRAG-specific quirk. `GLiNER2Backend` has no equivalent API to repurpose — it sets `file_path` directly — but follows the same convention so `connect_chunks_to_entities` works unmodified regardless of backend.
