# Use Chunk.hash as LightRAG's file_path for entity linking

LightRAG's `ainsert(file_paths=...)` only exposes a per-input identifier meant for actual file paths, no purpose-built back-reference field. Pass each `Chunk.hash` as that identifier so `connect_chunks_to_entities` can later link extracted entities to source chunks via simple property match (`n.file_path = m.hash`) — no separate hash-to-entity mapping outside LightRAG.
