"""Memgraph-backed vector storage for LightRAG.

Persists LightRAG's vector namespaces (``entities``, ``relationships``,
``chunks``) as nodes searched via Memgraph's native vector index
(``CREATE VECTOR INDEX`` + ``CALL vector_search.search``). Each record is a
node labelled ``LightRAGVector_<workspace>_<namespace>``, keyed by ``id``,
with the embedding on an ``embedding`` property.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, final

import numpy as np
from lightrag.base import BaseVectorStorage
from lightrag.utils import compute_mdhash_id, logger

from ._connection import (
    close_driver,
    get_database,
    get_driver,
    sanitize_index_name,
    sanitize_label,
)

# Memgraph requires a capacity at index-creation time.
DEFAULT_VECTOR_INDEX_CAPACITY = 1_000_000


@final
@dataclass
class MemgraphVectorStorage(BaseVectorStorage):
    """Vector storage backend that persists LightRAG embeddings in Memgraph."""

    def __post_init__(self):
        self._validate_embedding_func()
        workspace = os.environ.get("MEMGRAPH_WORKSPACE") or self.workspace or "base"
        self.workspace = workspace
        self._label = sanitize_label(f"LightRAGVector_{workspace}_{self.namespace}")
        self._index_name = sanitize_index_name(f"lightrag_vec_{workspace}_{self.namespace}")
        self._index_ready = False

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is not None:
            self.cosine_better_than_threshold = cosine_threshold
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

    @property
    def _dimension(self) -> int:
        return self.embedding_func.embedding_dim

    async def initialize(self):
        """Create the id index and the native vector index (both idempotent)."""
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            try:
                await (await session.run(f"CREATE INDEX ON :`{self._label}`(id)")).consume()
            except Exception as e:
                logger.warning(
                    f"[{self.workspace}] Index creation on :`{self._label}`(id) may have failed or already exists: {e}"
                )
            await self._ensure_vector_index(session)
            await (await session.run("RETURN 1")).consume()
        logger.info(
            f"[{self.workspace}] Initialized Memgraph vector storage for {self.namespace} "
            f"(dim={self._dimension}, index={self._index_name})"
        )

    async def finalize(self):
        await close_driver()

    async def index_done_callback(self) -> None:
        # Memgraph persists automatically; nothing to flush.
        pass

    async def _ensure_vector_index(self, session) -> None:
        """Create the vector index once; cached since Memgraph doesn't raise if it already exists."""
        if self._index_ready:
            return
        query = (
            f"CREATE VECTOR INDEX {self._index_name} ON :`{self._label}`(embedding) "
            f'WITH CONFIG {{"dimension": {int(self._dimension)}, '
            f'"capacity": {DEFAULT_VECTOR_INDEX_CAPACITY}, "metric": "cos"}}'
        )
        try:
            await (await session.run(query)).consume()
            logger.info(f"[{self.workspace}] Created vector index {self._index_name} on :`{self._label}`(embedding)")
        except Exception as e:
            # Index may already exist, which is not an error.
            logger.debug(f"[{self.workspace}] Vector index {self._index_name} creation skipped or already exists: {e}")
        self._index_ready = True

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, returning plain float lists suitable as Cypher params."""
        batches = [texts[i : i + self._max_batch_size] for i in range(0, len(texts), self._max_batch_size)]
        results = await asyncio.gather(*[self.embedding_func(batch) for batch in batches])
        embeddings = np.concatenate(results)
        return [np.asarray(vec, dtype=float).tolist() for vec in embeddings]

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        current_time = int(time.time())
        ids = list(data.keys())
        contents = [data[i]["content"] for i in ids]
        embeddings = await self._embed(contents)
        if len(embeddings) != len(ids):
            logger.error(
                f"[{self.workspace}] Embedding count {len(embeddings)} != data count {len(ids)} for {self.namespace}"
            )
            return

        entries = []
        for i, key in enumerate(ids):
            value = data[key]
            props = {k: v for k, v in value.items() if k in self.meta_fields}
            props["content"] = value.get("content")
            entries.append(
                {
                    "id": key,
                    "props": props,
                    "embedding": embeddings[i],
                    "ts": current_time,
                }
            )

        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await self._ensure_vector_index(session)
            await (
                await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{self._label}` {{id: e.id}})
                    ON CREATE SET n.created_at = e.ts
                    SET n += e.props, n.embedding = e.embedding, n.updated_at = e.ts
                    """,
                    entries=entries,
                )
            ).consume()
        logger.debug(f"[{self.workspace}] Upserted {len(entries)} vectors to {self.namespace}")

    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = np.asarray(query_embedding, dtype=float).tolist()
        else:
            embedded = await self.embedding_func([query], _priority=5)
            embedding = np.asarray(embedded[0], dtype=float).tolist()

        driver = await get_driver()
        # default_access_mode="READ" is just a routing hint; it doesn't affect the vector index's isolation.
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            try:
                # Memgraph's vector index runs at READ_UNCOMMITTED with deferred GC, so
                # vector_search.search can hand back a handle to an already-deleted node --
                # reading a property off it raises 50N42. We never touch that raw `node`
                # handle: re-MATCH by identity and read properties only off live nodes, so
                # a dangling candidate is silently excluded instead of erroring.
                # Index name is a sanitized string literal (not accepted as a query param).
                result = await session.run(
                    f"""
                    CALL vector_search.search("{self._index_name}", $top_k, $embedding)
                    YIELD node, similarity
                    WITH collect(node) AS cand_nodes,
                         collect({{node: node, similarity: similarity}}) AS cand_pairs
                    MATCH (live:`{self._label}`)
                    WHERE live IN cand_nodes
                    WITH live, [p IN cand_pairs WHERE p.node = live | p.similarity][0] AS similarity
                    WHERE similarity >= $threshold
                    RETURN live.id AS id, similarity AS similarity, properties(live) AS props
                    ORDER BY similarity DESC
                    """,
                    top_k=int(top_k),
                    embedding=embedding,
                    threshold=self.cosine_better_than_threshold,
                )
                records = [record async for record in result]
                await result.consume()
            except Exception as e:
                # Real failure, not a dangling index entry (those are excluded above) -- surface it.
                logger.error(f"[{self.workspace}] Vector search failed for {self.namespace}: {e}")
                raise

        output: list[dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            props.pop("embedding", None)
            created_at = props.pop("created_at", None)
            props.pop("updated_at", None)
            props["id"] = record["id"]
            props["distance"] = record["similarity"]
            props["created_at"] = created_at
            output.append(props)
        return output

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"MATCH (n:`{self._label}` {{id: $id}}) RETURN properties(n) AS props",
                id=id,
            )
            record = await result.single()
            await result.consume()
        if not record:
            return None
        props = dict(record["props"])
        props.pop("embedding", None)
        props["id"] = id
        return props

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                MATCH (n:`{self._label}`)
                WHERE n.id IN $ids
                RETURN n.id AS id, properties(n) AS props
                """,
                ids=list(ids),
            )
            records = [record async for record in result]
            await result.consume()
        output = []
        for record in records:
            props = dict(record["props"])
            props.pop("embedding", None)
            props["id"] = record["id"]
            output.append(props)
        return output

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                MATCH (n:`{self._label}`)
                WHERE n.id IN $ids AND n.embedding IS NOT NULL
                RETURN n.id AS id, n.embedding AS embedding
                """,
                ids=list(ids),
            )
            vectors = {record["id"]: list(record["embedding"]) async for record in result}
            await result.consume()
        return vectors

    async def delete(self, ids: list[str]):
        if not ids:
            return
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            # Null the embedding before DETACH DELETE so the entry leaves the vector index
            # (a bare delete would leave a dangling entry -- see query()'s 50N42 note).
            await (
                await session.run(
                    f"""
                    UNWIND $ids AS target_id
                    MATCH (n:`{self._label}` {{id: target_id}})
                    SET n.embedding = NULL
                    DETACH DELETE n
                    """,
                    ids=list(ids),
                )
            ).consume()

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            # See delete() -- null the embedding first so these leave the vector index.
            await (
                await session.run(
                    f"""
                    MATCH (n:`{self._label}`)
                    WHERE n.src_id = $entity_name OR n.tgt_id = $entity_name
                    SET n.embedding = NULL
                    DETACH DELETE n
                    """,
                    entity_name=entity_name,
                )
            ).consume()

    async def drop(self) -> dict[str, str]:
        try:
            driver = await get_driver()
            async with driver.session(database=get_database()) as session:
                # Null embeddings first in case DROP VECTOR INDEX below fails (see delete()).
                await (await session.run(f"MATCH (n:`{self._label}`) SET n.embedding = NULL DETACH DELETE n")).consume()
                try:
                    await (await session.run(f"DROP VECTOR INDEX {self._index_name}")).consume()
                except Exception as e:
                    logger.debug(f"[{self.workspace}] Dropping vector index {self._index_name} skipped: {e}")
            self._index_ready = False
            logger.info(f"[{self.workspace}] Dropped Memgraph vector storage for {self.namespace}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping vector storage {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
