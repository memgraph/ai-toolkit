# Use Chunk.hash as LightRAG's file_path for entity linking

LightRAG's `ainsert(file_paths=...)` only exposes a per-input identifier meant for actual file paths, with no purpose-built back-reference field. We pass each `Chunk.hash` as that identifier so `connect_chunks_to_entities` can later link extracted entities back to their source chunks via a simple property match (`n.file_path = m.hash`), instead of maintaining a separate hash-to-entity mapping outside LightRAG.
