"""Memgraph-backed key/value storage for LightRAG.

Each record is a node labelled ``LightRAGKV_<workspace>_<namespace>``, keyed
by ``id``, with the value dict stored as JSON in a ``data`` property.
Namespacing the label keeps it from colliding with the graph backend's
entity nodes (which use the bare workspace label).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import BaseKVStorage
from lightrag.utils import logger

from ._connection import (
    MemgraphCrudMixin,
    close_driver,
    create_index,
    get_database,
    get_driver,
    resolve_workspace,
    sanitize_label,
)


@final
@dataclass
class MemgraphKVStorage(MemgraphCrudMixin, BaseKVStorage):
    """Key/value storage backend that persists LightRAG KV namespaces in Memgraph."""

    def __post_init__(self):
        workspace = resolve_workspace(self.workspace)
        self.workspace = workspace
        self._label = sanitize_label(f"LightRAGKV_{workspace}_{self.namespace}")

    async def initialize(self):
        """Create the id index for this namespace (idempotent)."""
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await create_index(session, label=self._label, prop="id", workspace=self.workspace)
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

    # is_empty/filter_keys/delete/drop are provided by MemgraphCrudMixin.
