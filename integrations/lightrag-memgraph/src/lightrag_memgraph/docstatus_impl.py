"""Memgraph-backed document-status storage for LightRAG.

Persists LightRAG's ``doc_status`` namespace as nodes in Memgraph instead of a
local JSON file. Each document is one node labelled
``LightRAGDocStatus_<workspace>`` keyed by an ``id`` property. The full status
dict is stored as a JSON string in a ``data`` property, while the fields needed
for filtering / sorting (``status``, ``track_id``, ``file_path``,
``created_at``, ``updated_at``) are also stored as top-level node properties and
indexed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger

from ._connection import (
    close_driver,
    get_database,
    get_driver,
    sanitize_label,
)

# Node properties promoted out of the JSON blob so they can be indexed / queried.
_INDEXED_FIELDS = ("status", "track_id", "file_path", "created_at", "updated_at")
_SORT_FIELDS = {"created_at", "updated_at", "id", "file_path"}


@final
@dataclass
class MemgraphDocStatusStorage(DocStatusStorage):
    """Document-status storage backend that persists LightRAG doc status in Memgraph."""

    def __post_init__(self):
        workspace = os.environ.get("MEMGRAPH_WORKSPACE") or self.workspace or "base"
        self.workspace = workspace
        self._label = sanitize_label(f"LightRAGDocStatus_{workspace}")

    async def initialize(self):
        """Create indexes on id + the queryable fields (idempotent)."""
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            for prop in ("id", *_INDEXED_FIELDS):
                try:
                    await (await session.run(f"CREATE INDEX ON :`{self._label}`({prop})")).consume()
                except Exception as e:
                    logger.warning(
                        f"[{self.workspace}] Index creation on :`{self._label}`({prop}) may have failed or already exists: {e}"
                    )
            await (await session.run("RETURN 1")).consume()
        logger.info(f"[{self.workspace}] Initialized Memgraph doc-status storage")

    async def finalize(self):
        await close_driver()

    async def index_done_callback(self) -> None:
        # Memgraph persists automatically; nothing to flush.
        pass

    @staticmethod
    def _to_status(data: dict[str, Any]) -> DocProcessingStatus:
        """Reconstruct a DocProcessingStatus from a stored dict (mirrors JsonDocStatusStorage)."""
        data = dict(data)
        data.pop("content", None)
        if not data.get("file_path"):
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        return DocProcessingStatus(**data)

    # --- BaseKVStorage interface -------------------------------------------------

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"MATCH (n:`{self._label}` {{id: $id}}) RETURN n.data AS data",
                id=id,
            )
            record = await result.single()
            await result.consume()
        if not record or record["data"] is None:
            return None
        return json.loads(record["data"])

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        # Aligned with `ids`; None in the slot of any id that does not exist.
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
        return [json.loads(found[i]) if i in found else None for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
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
        entries = []
        for doc_id, doc_data in data.items():
            value = dict(doc_data)
            value.setdefault("chunks_list", [])
            indexed = {field: value.get(field) for field in _INDEXED_FIELDS}
            entries.append(
                {
                    "id": doc_id,
                    "data": json.dumps(value, ensure_ascii=False),
                    "indexed": indexed,
                }
            )
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await (
                await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{self._label}` {{id: e.id}})
                    SET n.data = e.data, n += e.indexed
                    """,
                    entries=entries,
                )
            ).consume()
        logger.debug(f"[{self.workspace}] Upserted {len(entries)} doc-status records")

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

    # --- DocStatusStorage interface ---------------------------------------------

    async def get_status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocStatus}
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(f"MATCH (n:`{self._label}`) RETURN n.status AS status, count(n) AS cnt")
            async for record in result:
                if record["status"] in counts:
                    counts[record["status"]] = record["cnt"]
            await result.consume()
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        # Mirror the reference impls (Json/Redis/Mongo/OpenSearch): include an
        # "all" key with the grand total, which the status-counts API relies on.
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(self, statuses: list[DocStatus]) -> dict[str, DocProcessingStatus]:
        if not statuses:
            return {}
        status_values = [s.value for s in statuses]
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                MATCH (n:`{self._label}`)
                WHERE n.status IN $statuses
                RETURN n.id AS id, n.data AS data
                """,
                statuses=status_values,
            )
            records = [record async for record in result]
            await result.consume()
        return self._records_to_status_map(records)

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                MATCH (n:`{self._label}` {{track_id: $track_id}})
                RETURN n.id AS id, n.data AS data
                """,
                track_id=track_id,
            )
            records = [record async for record in result]
            await result.consume()
        return self._records_to_status_map(records)

    def _records_to_status_map(self, records) -> dict[str, DocProcessingStatus]:
        result: dict[str, DocProcessingStatus] = {}
        for record in records:
            if record["data"] is None:
                continue
            try:
                result[record["id"]] = self._to_status(json.loads(record["data"]))
            except (KeyError, TypeError) as e:
                logger.error(f"[{self.workspace}] Missing required field for document {record['id']}: {e}")
        return result

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200
        if sort_field not in _SORT_FIELDS:
            sort_field = "updated_at"
        order = "DESC" if sort_direction.lower() != "asc" else "ASC"
        skip = (page - 1) * page_size

        where = "" if status_filter is None else "WHERE n.status = $status"
        sort_expr = "n.id" if sort_field == "id" else f"n.{sort_field}"
        params: dict[str, Any] = {"skip": skip, "limit": page_size}
        if status_filter is not None:
            params["status"] = status_filter.value

        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            count_result = await session.run(
                f"MATCH (n:`{self._label}`) {where} RETURN count(n) AS total",
                **({"status": status_filter.value} if status_filter is not None else {}),
            )
            count_record = await count_result.single()
            await count_result.consume()
            total = count_record["total"] if count_record else 0

            page_result = await session.run(
                f"""
                MATCH (n:`{self._label}`) {where}
                RETURN n.id AS id, n.data AS data
                ORDER BY {sort_expr} {order}
                SKIP $skip LIMIT $limit
                """,
                **params,
            )
            records = [record async for record in page_result]
            await page_result.consume()

        docs: list[tuple[str, DocProcessingStatus]] = []
        for record in records:
            if record["data"] is None:
                continue
            try:
                docs.append((record["id"], self._to_status(json.loads(record["data"]))))
            except (KeyError, TypeError) as e:
                logger.error(f"[{self.workspace}] Error processing document {record['id']}: {e}")
        return docs, total

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"MATCH (n:`{self._label}` {{file_path: $file_path}}) RETURN n.data AS data LIMIT 1",
                file_path=file_path,
            )
            record = await result.single()
            await result.consume()
        if not record or record["data"] is None:
            return None
        return json.loads(record["data"])

    async def drop(self) -> dict[str, str]:
        try:
            driver = await get_driver()
            async with driver.session(database=get_database()) as session:
                await (await session.run(f"MATCH (n:`{self._label}`) DETACH DELETE n")).consume()
            logger.info(f"[{self.workspace}] Dropped Memgraph doc-status storage")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping doc-status storage: {e}")
            return {"status": "error", "message": str(e)}
