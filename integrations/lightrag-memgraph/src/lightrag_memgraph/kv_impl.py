"""Memgraph-backed key/value storage for LightRAG.

Each record is a node labelled ``LightRAGKV_<workspace>_<namespace>``, keyed
by ``id``, with the value dict stored as JSON in a ``data`` property.
Namespacing the label keeps it from colliding with the graph backend's
entity nodes (which use the bare workspace label).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import BaseKVStorage
from lightrag.utils import logger

from ._connection import (
    close_driver,
    get_database,
    get_driver,
    sanitize_label,
)


@final
@dataclass
class MemgraphKVStorage(BaseKVStorage):
    """Key/value storage backend that persists LightRAG KV namespaces in Memgraph."""

    def __post_init__(self):
        workspace = os.environ.get("MEMGRAPH_WORKSPACE") or self.workspace or "base"
        self.workspace = workspace
        self._label = sanitize_label(f"LightRAGKV_{workspace}_{self.namespace}")

    async def initialize(self):
        """Create the id index for this namespace (idempotent)."""
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            try:
                await (await session.run(f"CREATE INDEX ON :`{self._label}`(id)")).consume()
            except Exception as e:  # index may already exist, which is not an error
                logger.warning(
                    f"[{self.workspace}] Index creation on :`{self._label}`(id) may have failed or already exists: {e}"
                )
            await (await session.run("RETURN 1")).consume()
        logger.info(f"[{self.workspace}] Initialized Memgraph KV storage for {self.namespace}")

    async def finalize(self):
        await close_driver()

    async def index_done_callback(self) -> None:
        # Memgraph persists automatically; nothing to flush.
        pass

    def _decode(self, id: str, raw: str | None) -> dict[str, Any] | None:
        """Decode a stored JSON value into the dict LightRAG expects."""
        if raw is None:
            return None
        value = json.loads(raw)
        value.setdefault("create_time", 0)
        value.setdefault("update_time", 0)
        value["_id"] = id
        return value

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"MATCH (n:`{self._label}` {{id: $id}}) RETURN n.data AS data",
                id=id,
            )
            record = await result.single()
            await result.consume()
            return self._decode(id, record["data"] if record else None)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        # Aligned with `ids`; None in the slot of any id that doesn't exist (mirrors JsonKVStorage).
        if not ids:
            return []
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                UNWIND $ids AS target_id
                OPTIONAL MATCH (n:`{self._label}` {{id: target_id}})
                RETURN target_id AS id, n.data AS data
                """,
                ids=list(ids),
            )
            found: dict[str, str] = {}
            async for record in result:
                if record["data"] is not None:
                    found[record["id"]] = record["data"]
            await result.consume()
        return [self._decode(i, found.get(i)) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        # Returns the subset of `keys` that do NOT exist in storage.
        if not keys:
            return set()
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                UNWIND $keys AS k
                OPTIONAL MATCH (n:`{self._label}` {{id: k}})
                WITH k, n
                WHERE n IS NULL
                RETURN k
                """,
                keys=list(keys),
            )
            missing = {record["k"] async for record in result}
            await result.consume()
        return missing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        current_time = int(time.time())
        # MERGE picks new vs. existing payload via ON CREATE/ON MATCH -- no separate filter_keys() round trip.
        entries = []
        for k, v in data.items():
            value = dict(v)
            if self.namespace.endswith("text_chunks") and "llm_cache_list" not in value:
                value["llm_cache_list"] = []
            value["_id"] = k

            new_value = dict(value)
            new_value["create_time"] = current_time
            new_value["update_time"] = current_time

            update_value = dict(value)
            update_value["update_time"] = current_time

            entries.append(
                {
                    "id": k,
                    "new_data": json.dumps(new_value, ensure_ascii=False),
                    "update_data": json.dumps(update_value, ensure_ascii=False),
                    "ts": current_time,
                }
            )

        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await (
                await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{self._label}` {{id: e.id}})
                    ON CREATE SET n.data = e.new_data, n.created_at = e.ts
                    ON MATCH SET n.data = e.update_data
                    SET n.updated_at = e.ts
                    """,
                    entries=entries,
                )
            ).consume()
        logger.debug(f"[{self.workspace}] Upserted {len(entries)} records to {self.namespace}")

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await (
                await session.run(
                    f"""
                    UNWIND $ids AS target_id
                    MATCH (n:`{self._label}` {{id: target_id}})
                    DETACH DELETE n
                    """,
                    ids=list(ids),
                )
            ).consume()

    async def is_empty(self) -> bool:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(f"MATCH (n:`{self._label}`) RETURN n LIMIT 1")
            record = await result.single()
            await result.consume()
            return record is None

    async def drop(self) -> dict[str, str]:
        try:
            driver = await get_driver()
            async with driver.session(database=get_database()) as session:
                await (await session.run(f"MATCH (n:`{self._label}`) DETACH DELETE n")).consume()
            logger.info(f"[{self.workspace}] Dropped Memgraph KV storage for {self.namespace}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping KV storage {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
