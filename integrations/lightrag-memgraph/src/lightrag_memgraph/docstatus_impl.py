"""Memgraph-backed document-status storage for LightRAG.

Each document is a node labelled ``LightRAGDocStatus_<workspace>``, keyed by
``id``. The full status dict is JSON in a ``data`` property; filter/sort
fields (``status``, ``track_id``, ``file_path``, ``created_at``,
``updated_at``) are also promoted to indexed top-level properties.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, final

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceConflict,
    SourceResolution,
    SourceUnique,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import StorageControlPlaneError
from lightrag.utils import logger

from ._connection import (
    MemgraphCrudMixin,
    as_literal_query,
    close_driver,
    create_index,
    get_database,
    get_driver,
    resolve_workspace,
    sanitize_label,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Node properties promoted out of the JSON blob so they can be indexed / queried.
_INDEXED_FIELDS = ("status", "track_id", "file_path", "content_hash", "created_at", "updated_at")
_SORT_FIELDS = {"created_at", "updated_at", "id", "file_path"}


@final
@dataclass
class MemgraphDocStatusStorage(MemgraphCrudMixin, DocStatusStorage):
    """Document-status storage backend that persists LightRAG doc status in Memgraph."""

    def __post_init__(self):
        workspace = resolve_workspace(self.workspace)
        self.workspace = workspace
        self._label = sanitize_label(f"LightRAGDocStatus_{workspace}")

    async def initialize(self):
        """Create indexes on id + the queryable fields (idempotent)."""
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            for prop in ("id", *_INDEXED_FIELDS):
                await create_index(session, label=self._label, prop=prop, workspace=self.workspace)
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
                as_literal_query(f"MATCH (n:`{self._label}` {{id: $id}}) RETURN n.data AS data"),
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
                as_literal_query(
                    f"""
                    UNWIND $ids AS target_id
                    OPTIONAL MATCH (n:`{self._label}` {{id: target_id}})
                    RETURN target_id AS id, n.data AS data
                    """
                ),
                ids=list(ids),
            )
            found: dict[str, str] = {}
            async for record in result:
                if record["data"] is not None:
                    found[record["id"]] = record["data"]
            await result.consume()
        # LightRAG's own BaseKVStorage.get_by_ids declares -> list[dict[str, Any]], but its
        # reference JsonKVStorage impl (and this one, to match) legitimately puts None in the
        # slot of any id that does not exist -- the upstream annotation itself is imprecise.
        return cast("list[dict[str, Any]]", [json.loads(found[i]) if i in found else None for i in ids])

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
                    as_literal_query(
                        f"""
                        UNWIND $entries AS e
                        MERGE (n:`{self._label}` {{id: e.id}})
                        SET n.data = e.data, n += e.indexed
                        """
                    ),
                    entries=entries,
                )
            ).consume()
        logger.debug(f"[{self.workspace}] Upserted {len(entries)} doc-status records")

    # is_empty/filter_keys/delete are provided by MemgraphCrudMixin.

    # --- DocStatusStorage interface ---------------------------------------------

    async def get_status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocStatus}
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(f"MATCH (n:`{self._label}`) RETURN n.status AS status, count(n) AS cnt")
            )
            async for record in result:
                if record["status"] in counts:
                    counts[record["status"]] = record["cnt"]
            await result.consume()
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        # "all" grand total: other DocStatusStorage impls include it and the status-counts API relies on it.
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        """Documents in any of ``statuses``.

        ``strict=False`` logs and skips a row that cannot be deserialized.
        ``strict=True`` raises instead, so a scheduling caller never observes
        a partial map.
        """
        if not statuses:
            return {}
        status_values = [s.value for s in statuses]
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"""
                    MATCH (n:`{self._label}`)
                    WHERE n.status IN $statuses
                    RETURN n.id AS id, n.data AS data
                    """
                ),
                statuses=status_values,
            )
            records = [record async for record in result]
            await result.consume()
        return self._records_to_status_map(records, strict=strict)

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"""
                    MATCH (n:`{self._label}` {{track_id: $track_id}})
                    RETURN n.id AS id, n.data AS data
                    """
                ),
                track_id=track_id,
            )
            records = [record async for record in result]
            await result.consume()
        return self._records_to_status_map(records)

    def _records_to_status_map(self, records, *, strict: bool = False) -> dict[str, DocProcessingStatus]:
        result: dict[str, DocProcessingStatus] = {}
        for record in records:
            doc_id = record["id"]
            raw = record["data"]
            if raw is None:
                if strict:
                    raise TypeError(f"doc_status record {doc_id} has no data")
                continue
            try:
                result[doc_id] = self._to_status(json.loads(raw))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.error(f"[{self.workspace}] Could not read document {doc_id}: {e}")
                if strict:
                    raise
        return result

    def _scheduling_record_from_row(
        self, doc_id: str, row: dict[str, Any], *, strict: bool
    ) -> DocSchedulingRecord | None:
        """Project one stored row. Strict raises; relaxed returns None.

        The caller still counts a skipped row as consumed, so a relaxed page
        advances past it instead of reading it again.
        """
        try:
            raw_status = row["status"]
            status = raw_status if isinstance(raw_status, DocStatus) else DocStatus(raw_status)
            created_at = row["created_at"]
            updated_at = row.get("updated_at", created_at)
            if not isinstance(created_at, str) or not isinstance(updated_at, str):
                raise TypeError("created_at/updated_at must be strings")
            metadata = row.get("metadata")
            return DocSchedulingRecord(
                id=doc_id,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                file_path=row.get("file_path") or "no-file-path",
                track_id=row.get("track_id"),
                has_custom_chunk_journal=isinstance(metadata, dict)
                and isinstance(metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"[{self.workspace}] Unusable scheduling row {doc_id}: {e}")
            if strict:
                raise
            return None

    @staticmethod
    def _encode_scheduling_cursor(key: tuple[str, str]) -> str:
        return json.dumps(list(key), ensure_ascii=False)

    @staticmethod
    def _decode_scheduling_cursor(opaque: str) -> tuple[str, str]:
        try:
            decoded = json.loads(opaque)
            created, doc_id = decoded
            if not isinstance(created, str) or not isinstance(doc_id, str):
                raise ValueError("cursor fields must be strings")
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            raise StorageControlPlaneError(f"Malformed scheduling cursor for MemgraphDocStatusStorage: {e}") from e
        return (created, doc_id)

    @staticmethod
    def _is_duplicate_row(row: Any) -> bool:
        """True when the row is a duplicate marker (``metadata.is_duplicate``)."""
        if not isinstance(row, dict):
            return False
        metadata = row.get("metadata")
        return bool(isinstance(metadata, dict) and metadata.get("is_duplicate"))

    def _project_scheduling_row(self, doc_id: str, raw: str | None, *, strict: bool) -> DocSchedulingRecord | None:
        """Decode ``raw`` and project it. ``None`` raw is an existing-but-empty node."""
        if raw is None:
            if strict:
                raise TypeError(f"doc_status record {doc_id} has no data")
            logger.error(f"[{self.workspace}] Scheduling row {doc_id} has no data")
            return None
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"[{self.workspace}] Scheduling row {doc_id} is not valid JSON: {e}")
            if strict:
                raise
            return None
        if not isinstance(row, dict):
            if strict:
                raise TypeError(f"doc_status record {doc_id} is not a mapping")
            logger.error(f"[{self.workspace}] Scheduling row {doc_id} is not a mapping")
            return None
        return self._scheduling_record_from_row(doc_id, row, strict=strict)

    async def _fetch_existing_rows(self, doc_ids: Sequence[str]) -> list[tuple[str, str | None]]:
        """``(id, data)`` for ids confirmed to exist. Missing ids are omitted.

        ``data`` is None when the node exists but has no stored JSON. Order
        follows ``doc_ids``.
        """
        if not doc_ids:
            return []
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"""
                    UNWIND $ids AS target_id
                    OPTIONAL MATCH (n:`{self._label}` {{id: target_id}})
                    RETURN target_id AS id, n.id IS NOT NULL AS exists, n.data AS data
                    """
                ),
                ids=list(doc_ids),
            )
            found = [(record["id"], record["data"]) async for record in result if record["exists"]]
            await result.consume()
        return found

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        status_values = self.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )
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

        where = "" if status_values is None else "WHERE n.status IN $statuses"
        sort_expr = "n.id" if sort_field == "id" else f"n.{sort_field}"
        order_by = f"{sort_expr} {order}" if sort_field == "id" else f"n.{sort_field} {order}, n.id {order}"
        params: dict[str, Any] = {"skip": skip, "limit": page_size}
        filter_params: dict[str, Any] = {}
        if status_values is not None:
            filter_params["statuses"] = list(status_values)
        params.update(filter_params)

        driver = await get_driver()

        async def _count() -> int:
            async with driver.session(database=get_database(), default_access_mode="READ") as session:
                result = await session.run(
                    as_literal_query(f"MATCH (n:`{self._label}`) {where} RETURN count(n) AS total"),
                    **filter_params,
                )
                record = await result.single()
                await result.consume()
                return record["total"] if record else 0

        async def _page() -> list:
            async with driver.session(database=get_database(), default_access_mode="READ") as session:
                result = await session.run(
                    as_literal_query(
                        f"""
                        MATCH (n:`{self._label}`) {where}
                        RETURN n.id AS id, n.data AS data
                        ORDER BY {order_by}
                        SKIP $skip LIMIT $limit
                        """
                    ),
                    **params,
                )
                records = [record async for record in result]
                await result.consume()
                return records

        # No data dependency between the count and the page -- run them concurrently
        # instead of two sequential round trips.
        total, records = await asyncio.gather(_count(), _page())

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
                as_literal_query(f"MATCH (n:`{self._label}` {{file_path: $file_path}}) RETURN n.data AS data LIMIT 1"),
                file_path=file_path,
            )
            record = await result.single()
            await result.consume()
        if not record or record["data"] is None:
            return None
        return json.loads(record["data"])

    async def get_doc_by_file_basename(self, basename: str) -> tuple[str, dict[str, Any]] | None:
        """Exact match on file_path. Mirrors JsonDocStatusStorage: empty input / "unknown_source" return None."""
        if not basename or basename == "unknown_source":
            return None
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"MATCH (n:`{self._label}` {{file_path: $basename}}) RETURN n.id AS id, n.data AS data LIMIT 1"
                ),
                basename=basename,
            )
            record = await result.single()
            await result.consume()
        if not record or record["data"] is None:
            return None
        return record["id"], json.loads(record["data"])

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """Earliest ``(created_at, id)`` holder of ``content_hash``.

        ``None`` means confirmed absence. An empty hash returns ``None``
        without querying. ``exclude_doc_id`` skips that row and any duplicate
        marker that only points at it (``metadata.original_doc_id``); the
        scan continues so a pointer cannot hide an earlier independent
        holder. A matching row that cannot be decoded raises.

        Raises:
            json.JSONDecodeError: a matching row is not valid JSON.
            TypeError: a matching row has no data or is not a JSON object.
        """
        if not content_hash:
            return None
        driver = await get_driver()
        winner: tuple[str, dict[str, Any]] | None = None
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"""
                    MATCH (n:`{self._label}`)
                    WHERE n.content_hash = $content_hash
                    RETURN n.id AS id, n.data AS data, coalesce(n.created_at, '') AS created_at
                    ORDER BY coalesce(n.created_at, '') ASC, n.id ASC
                    """
                ),
                content_hash=content_hash,
            )
            async for record in result:
                doc_id = record["id"]
                if doc_id == exclude_doc_id:
                    continue
                raw = record["data"]
                if raw is None:
                    raise TypeError(f"doc_status record {doc_id} has no data")
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise TypeError(f"doc_status record {doc_id} is not a mapping")
                if self._row_points_at_as_duplicate(row, exclude_doc_id):
                    continue
                winner = (doc_id, row)
                break
            await result.consume()
        return winner

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        """One keyset page ordered by ``(created_at, id)`` across ``statuses``.

        A missing ``created_at`` sorts as ``''`` via ``coalesce``, so that
        bucket stays reachable on the next page. Rows that cannot be
        projected are still consumed: ``strict=True`` raises before the page
        is returned, and a relaxed page skips them while advancing
        ``next_position`` past the last underlying row. The sweep is over
        only when ``next_position is CURSOR_END`` (fewer than ``limit`` rows
        matched), not when ``docs`` is empty.

        Raises:
            ValueError: ``limit`` is not positive.
            StorageControlPlaneError: ``position`` is not a known cursor.
            KeyError, TypeError, ValueError, json.JSONDecodeError: a row on
                the page cannot be projected and ``strict`` is true.
        """
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        if position is CURSOR_START:
            last_key: tuple[str, str] | None = None
        elif isinstance(position, CursorAfter):
            last_key = self._decode_scheduling_cursor(position.opaque)
        else:
            raise StorageControlPlaneError(f"Unsupported scheduling cursor for MemgraphDocStatusStorage: {position!r}")

        params: dict[str, Any] = {"statuses": [status.value for status in statuses], "limit": limit}
        keyset = ""
        if last_key is not None:
            # Resume strictly after the last consumed (created_at, id). Both
            # cursor values are parameters; only the sanitized label is interpolated.
            keyset = """
                AND (
                  coalesce(n.created_at, '') > $cursor_created
                  OR (
                    coalesce(n.created_at, '') = $cursor_created
                    AND n.id > $cursor_id
                  )
                )
            """
            params["cursor_created"] = last_key[0]
            params["cursor_id"] = last_key[1]

        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"""
                    MATCH (n:`{self._label}`)
                    WHERE n.status IN $statuses
                    {keyset}
                    RETURN n.id AS id, n.data AS data, coalesce(n.created_at, '') AS created_at
                    ORDER BY coalesce(n.created_at, '') ASC, n.id ASC
                    LIMIT $limit
                    """
                ),
                **params,
            )
            records = [record async for record in result]
            await result.consume()

        docs: dict[str, DocSchedulingRecord] = {}
        for record in records:
            projected = self._project_scheduling_row(record["id"], record["data"], strict=strict)
            if projected is not None:
                docs[record["id"]] = projected
        if len(records) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            last = records[-1]
            next_position = CursorAfter(self._encode_scheduling_cursor((last["created_at"] or "", last["id"])))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def get_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocSchedulingRecord]:
        """Scheduling records for ids confirmed to exist. Missing ids are omitted.

        ``strict=True`` raises when an existing row cannot be projected,
        rather than omitting it and looking like a miss.

        Raises:
            KeyError, TypeError, ValueError, json.JSONDecodeError: an existing
                row cannot be projected and ``strict`` is true.
        """
        result: dict[str, DocSchedulingRecord] = {}
        for doc_id, raw in await self._fetch_existing_rows(doc_ids):
            projected = self._project_scheduling_row(doc_id, raw, strict=strict)
            if projected is not None:
                result[doc_id] = projected
        return result

    async def get_full_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocProcessingStatus]:
        """Full ``DocProcessingStatus`` rows for ids confirmed to exist.

        Uses the same raw-to-status normalisation as ``get_docs_by_statuses``.
        Missing ids are omitted. ``strict=True`` raises on a row that exists
        but cannot be decoded, so a short map is never a silent partial read.

        Raises:
            KeyError, TypeError, ValueError, json.JSONDecodeError: an existing
                row cannot be decoded and ``strict`` is true.
        """
        result: dict[str, DocProcessingStatus] = {}
        for doc_id, raw in await self._fetch_existing_rows(doc_ids):
            if raw is None:
                if strict:
                    raise TypeError(f"doc_status record {doc_id} has no data")
                logger.error(f"[{self.workspace}] doc_status row {doc_id} has no data")
                continue
            try:
                result[doc_id] = self._to_status(json.loads(raw))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.error(f"[{self.workspace}] Unusable doc_status row hydrating {doc_id}: {e}")
                if strict:
                    raise
        return result

    async def resolve_doc_source_strict(self, canonical_source_key: str) -> SourceResolution:
        """Fail-closed resolution of a canonical ``file_path``.

        Only rows with ``metadata.is_duplicate`` unset are primary candidates.
        Zero candidates is :class:`SourceAbsent`, one is :class:`SourceUnique`,
        and two or more is :class:`SourceConflict` proven from the first two
        primaries (the rest of the set is not loaded). Empty input and the
        ``unknown_source`` sentinel are confirmed absent and do not query.
        A stored row that cannot be decoded raises instead of looking absent.

        Raises:
            json.JSONDecodeError: a candidate row is not valid JSON.
            TypeError: a candidate row has no data or is not a JSON object.
            ValueError: a primary row's status is not a ``DocStatus``.
        """
        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()
        driver = await get_driver()
        candidates: list[tuple[str, dict[str, Any]]] = []
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                as_literal_query(
                    f"""
                    MATCH (n:`{self._label}` {{file_path: $file_path}})
                    RETURN n.id AS id, n.data AS data
                    """
                ),
                file_path=canonical_source_key,
            )
            async for record in result:
                doc_id = record["id"]
                raw = record["data"]
                if raw is None:
                    raise TypeError(f"doc_status record {doc_id} has no data")
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise TypeError(f"doc_status record {doc_id} is not a mapping")
                if self._is_duplicate_row(row):
                    continue
                candidates.append((doc_id, row))
                if len(candidates) >= 2:
                    break
            await result.consume()
        if not candidates:
            return SourceAbsent()
        if len(candidates) == 1:
            doc_id, row = candidates[0]
            return SourceUnique(doc_id=doc_id, doc=DocStatusStorage._scheduling_record_from_raw(doc_id, row))
        return SourceConflict(
            candidate_count=None,
            sample_doc_ids=tuple(sorted(doc_id for doc_id, _ in candidates)),
        )

    # drop() is provided by MemgraphCrudMixin.
