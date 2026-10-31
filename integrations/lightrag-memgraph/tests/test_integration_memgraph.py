"""Live-Memgraph integration tests for the LightRAG storage backends.

These tests exercise the KV, vector and doc-status backends against a *real*
Memgraph server (unlike ``test_storages.py``, which only checks the
registration / instantiation paths without any Cypher execution).

One Memgraph instance is reused across the whole suite: a single async Bolt
driver is opened once (session-scoped, via the backends' own
``_connection.get_driver`` which keys a driver per event loop) and shared by
every test. Tests are isolated from one another by giving each test a unique
``workspace`` so their node labels never collide, and every test drops its own
data at the end. This keeps the shared instance consistent "across the changes".

Connection settings are read exactly the way the storage backends read them:
through the toolbox's ``memgraph_env`` (the canonical ``MEMGRAPH_URL`` /
``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD`` / ``MEMGRAPH_DATABASE`` names), with
``MEMGRAPH_HOST`` + ``MEMGRAPH_PORT`` accepted as a convenience.

If no Memgraph server is reachable the whole module is skipped (not failed), so
a unit-only run still passes. CI provides a Memgraph service container so these
tests actually run there.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import numpy as np
import pytest
import pytest_asyncio
from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocProcessingStatus,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.utils import EmbeddingFunc

from lightrag_memgraph import MemgraphLightRAGWrapper
from lightrag_memgraph._connection import (
    as_literal_query,
    close_driver,
    get_database,
    get_driver,
)
from lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from lightrag_memgraph.embeddings import DEFAULT_EMBEDDING_DIM, memgraph_sentence_embed
from lightrag_memgraph.kv_impl import MemgraphKVStorage
from lightrag_memgraph.vector_impl import MemgraphVectorStorage
from memgraph_toolbox.api.memgraph import memgraph_env


def _connection_settings() -> tuple[str, str, str, str]:
    """Resolve (url, username, password, database) via the toolbox's memgraph_env.

    This is exactly how the storage backends resolve their connection, so the
    reachability probe below targets the same instance the backends will use.
    """
    env = memgraph_env()
    return (
        env["MEMGRAPH_URL"],
        env["MEMGRAPH_USER"],
        env["MEMGRAPH_PASSWORD"],
        env["MEMGRAPH_DATABASE"],
    )


# All async tests / fixtures share ONE event loop for the whole session so the
# per-loop shared driver in `_connection` is created once and reused everywhere.
pytestmark = pytest.mark.asyncio(loop_scope="session")


# --- deterministic fake embeddings ------------------------------------------
#
# A tiny fixed-dimension embedding function so no external embedding API is
# needed. Known contents map to fixed unit-ish vectors whose pairwise cosine
# similarities are predictable, which lets the vector test assert nearest
# neighbour ordering and threshold behaviour exactly:
#   cos(cat, cat)    = 1.0
#   cos(cat, kitten) = 0.8    (above the 0.2 threshold -> kept)
#   cos(cat, dog)    = 0.0    (below the threshold      -> dropped)
#   cos(cat, car)    = 0.0    (below the threshold      -> dropped)
_EMBED_DIM = 4
_KNOWN_VECTORS: dict[str, list[float]] = {
    "cat": [1.0, 0.0, 0.0, 0.0],
    "kitten": [0.8, 0.6, 0.0, 0.0],
    "dog": [0.0, 0.0, 1.0, 0.0],
    "car": [0.0, 0.0, 0.0, 1.0],
}


def _vector_for(text: str) -> list[float]:
    if text in _KNOWN_VECTORS:
        return list(_KNOWN_VECTORS[text])
    # Deterministic fallback for any other text so the func never fails.
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.random(_EMBED_DIM)
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).tolist()


def _fake_embedding_func(dim: int = _EMBED_DIM) -> EmbeddingFunc:
    async def _embed(texts, **_kwargs):
        return np.array([_vector_for(t) for t in texts], dtype=float)

    return EmbeddingFunc(embedding_dim=dim, func=_embed)


def _global_config() -> dict:
    return {
        "working_dir": "/tmp/lightrag_memgraph_integration",
        "embedding_batch_num": 4,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


# --- connection / reachability ----------------------------------------------


def _resolve_uri() -> str:
    """Resolve the Memgraph bolt URI the same way the backends will.

    The storage backends resolve their connection through the toolbox's
    ``memgraph_env`` (canonical ``MEMGRAPH_URL`` name). Honour ``MEMGRAPH_URL``
    directly, or build one from ``MEMGRAPH_HOST`` / ``MEMGRAPH_PORT`` as a
    convenience, then export ``MEMGRAPH_URL`` so the storage backends see the
    same target.
    """
    uri = os.environ.get("MEMGRAPH_URL")
    if not uri:
        host = os.environ.get("MEMGRAPH_HOST")
        port = os.environ.get("MEMGRAPH_PORT")
        if host:
            uri = f"bolt://{host}:{port or 7687}"
    if uri:
        os.environ["MEMGRAPH_URL"] = uri
    else:
        uri = _connection_settings()[0]
    return uri


@pytest.fixture(scope="session")
def memgraph_uri() -> str:
    """Skip the whole module unless a live Memgraph is reachable (session scope)."""
    uri = _resolve_uri()
    _uri, username, password, database = _connection_settings()

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - neo4j is a hard dependency
        pytest.skip(f"neo4j driver not importable: {exc}")

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password), connection_timeout=5)
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("RETURN 1").consume()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"No live Memgraph reachable at {uri}: {exc}")
    finally:
        if driver is not None:
            driver.close()
    return uri


@pytest.fixture(scope="session")
def vector_search_supported(memgraph_uri: str) -> bool:
    """Skip vector tests if the server has no ``vector_search`` module."""
    from neo4j import GraphDatabase

    _uri, username, password, database = _connection_settings()
    driver = GraphDatabase.driver(memgraph_uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            names = {record["name"] for record in session.run("CALL mg.procedures() YIELD name RETURN name")}
        if not any(n.startswith("vector_search.") for n in names):
            pytest.skip("Memgraph server has no vector_search module")
    finally:
        driver.close()
    return True


@pytest.fixture(scope="session")
def embeddings_module_supported(memgraph_uri: str) -> bool:
    """Skip if the server has no ``embeddings`` module (needs the memgraph-mage image)."""
    from neo4j import GraphDatabase

    _uri, username, password, database = _connection_settings()
    driver = GraphDatabase.driver(memgraph_uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            names = {record["name"] for record in session.run("CALL mg.procedures() YIELD name RETURN name")}
        if not any(n.startswith("embeddings.") for n in names):
            pytest.skip("Memgraph server has no embeddings module (needs the memgraph-mage image)")
    finally:
        driver.close()
    return True


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def shared_driver(memgraph_uri: str):
    """Open the shared async driver once and reuse it for the whole session."""
    driver = await get_driver()
    yield driver
    await close_driver()


@pytest.fixture(autouse=True)
def _no_workspace_override(monkeypatch):
    """Ensure MEMGRAPH_WORKSPACE never overrides each test's unique workspace."""
    monkeypatch.delenv("MEMGRAPH_WORKSPACE", raising=False)


@pytest.fixture
def workspace() -> str:
    """A unique workspace per test so node labels never collide."""
    return "it_" + uuid.uuid4().hex[:10]


# --- KV storage --------------------------------------------------------------


async def test_kv_roundtrip(shared_driver, workspace):
    store = MemgraphKVStorage(
        namespace="full_docs",
        workspace=workspace,
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )
    await store.initialize()
    try:
        assert await store.is_empty() is True

        await store.upsert(
            {
                "doc-1": {"content": "hello world", "source": "a"},
                "doc-2": {"content": "second doc", "source": "b"},
            }
        )

        assert await store.is_empty() is False

        one = await store.get_by_id("doc-1")
        assert one is not None
        assert one["content"] == "hello world"
        assert one["source"] == "a"
        assert one["_id"] == "doc-1"
        # upsert stamps create_time / update_time for new keys.
        assert one["create_time"] > 0
        assert one["update_time"] > 0

        assert await store.get_by_id("missing") is None

        # get_by_ids is aligned with the input order, None in missing slots.
        many = await store.get_by_ids(["doc-1", "missing", "doc-2"])
        assert len(many) == 3
        assert many[0]["_id"] == "doc-1"
        assert many[1] is None
        assert many[2]["_id"] == "doc-2"

        # filter_keys returns the keys that do NOT exist.
        missing = await store.filter_keys({"doc-1", "doc-2", "doc-3"})
        assert missing == {"doc-3"}

        await store.delete(["doc-1"])
        assert await store.get_by_id("doc-1") is None
        assert await store.get_by_id("doc-2") is not None

        result = await store.drop()
        assert result["status"] == "success"
        assert await store.is_empty() is True
    finally:
        await store.drop()


# --- doc-status storage ------------------------------------------------------


def _doc_store(workspace: str) -> MemgraphDocStatusStorage:
    return MemgraphDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )


def _doc(status: DocStatus, summary: str, file_path: str, content_hash: str | None = None) -> dict:
    doc = {
        "content": "full content that should be dropped from status",
        "content_summary": summary,
        "content_length": len(summary),
        "status": status.value,
        "file_path": file_path,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "chunks_list": [],
    }
    if content_hash is not None:
        doc["content_hash"] = content_hash
    return doc


async def test_doc_status_roundtrip(shared_driver, workspace):
    store = MemgraphDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )
    await store.initialize()
    try:
        await store.upsert(
            {
                "d1": _doc(DocStatus.PROCESSED, "one", "f1.txt", content_hash="hash-1"),
                "d2": _doc(DocStatus.PROCESSED, "two", "f2.txt", content_hash="hash-2"),
                "d3": _doc(DocStatus.PENDING, "three", "f3.txt"),
                "d4": _doc(DocStatus.FAILED, "four", "f4.txt"),
            }
        )

        raw = await store.get_by_id("d1")
        assert raw is not None
        assert raw["status"] == DocStatus.PROCESSED.value

        # get_doc_by_file_basename / get_doc_by_content_hash return (doc_id, data).
        by_name = await store.get_doc_by_file_basename("f1.txt")
        assert by_name is not None
        assert by_name[0] == "d1"
        assert by_name[1]["file_path"] == "f1.txt"
        assert await store.get_doc_by_file_basename("does-not-exist.txt") is None
        assert await store.get_doc_by_file_basename("unknown_source") is None

        by_hash = await store.get_doc_by_content_hash("hash-2")
        assert by_hash is not None
        assert by_hash[0] == "d2"
        assert by_hash[1]["content_hash"] == "hash-2"
        assert await store.get_doc_by_content_hash("missing-hash") is None

        # get_doc_by_file_path returns the full stored dict (get_by_ids format).
        by_path = await store.get_doc_by_file_path("f3.txt")
        assert by_path is not None
        assert by_path["status"] == DocStatus.PENDING.value

        # get_docs_paginated supports the new multi-status filter.
        page, total = await store.get_docs_paginated(status_filters=[DocStatus.PROCESSED], page=1, page_size=10)
        assert total == 2
        assert {doc_id for doc_id, _ in page} == {"d1", "d2"}

        processed = await store.get_docs_by_status(DocStatus.PROCESSED)
        assert set(processed.keys()) == {"d1", "d2"}
        # Values are reconstructed DocProcessingStatus objects with content dropped.
        d1 = processed["d1"]
        assert d1.status == DocStatus.PROCESSED
        assert d1.file_path == "f1.txt"
        assert not hasattr(d1, "content")

        pending = await store.get_docs_by_status(DocStatus.PENDING)
        assert set(pending.keys()) == {"d3"}

        counts = await store.get_status_counts()
        assert counts[DocStatus.PROCESSED.value] == 2
        assert counts[DocStatus.PENDING.value] == 1
        assert counts[DocStatus.FAILED.value] == 1

        all_counts = await store.get_all_status_counts()
        # The "all" grand total must be present and equal the sum.
        assert all_counts["all"] == 4
        assert all_counts[DocStatus.PROCESSED.value] == 2

        await store.delete(["d3"])
        assert await store.get_by_id("d3") is None
        assert (await store.get_all_status_counts())["all"] == 3

        result = await store.drop()
        assert result["status"] == "success"
        assert await store.is_empty() is True
    finally:
        await store.drop()


@pytest.mark.parametrize("sort_direction", ["asc", "desc"])
async def test_doc_status_pagination_breaks_ties_by_id(shared_driver, workspace, sort_direction):
    store = MemgraphDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )
    await store.initialize()
    try:
        doc_count = 25  # page_size is clamped to >= 10, so this spans three pages
        doc_ids = [f"doc-{i:02d}" for i in range(doc_count)]
        # Written out of id order, so storage order can't pass for the tiebreaker.
        scrambled = [doc_ids[(i * 7) % doc_count] for i in range(doc_count)]
        # _doc() gives every document the same updated_at: the whole set is one tie.
        await store.upsert({doc_id: _doc(DocStatus.PROCESSED, doc_id, f"{doc_id}.txt") for doc_id in scrambled})

        async def page_through() -> list[str]:
            seen: list[str] = []
            for page_number in (1, 2, 3):
                page, total = await store.get_docs_paginated(
                    page=page_number, page_size=10, sort_direction=sort_direction
                )
                assert total == doc_count
                seen.extend(doc_id for doc_id, _ in page)
            return seen

        first_pass = await page_through()
        assert len(first_pass) == len(set(first_pass)), "a document appeared on more than one page"
        assert set(first_pass) == set(doc_ids), "a document was missing from every page"
        assert first_pass == sorted(doc_ids, reverse=sort_direction == "desc")
        assert await page_through() == first_pass
    finally:
        await store.drop()


def _doc_at(
    status: DocStatus,
    summary: str,
    file_path: str,
    created_at: str,
    *,
    content_hash: str | None = None,
    metadata: dict | None = None,
    updated_at: str | None = None,
) -> dict:
    doc = _doc(status, summary, file_path, content_hash=content_hash)
    doc["created_at"] = created_at
    doc["updated_at"] = updated_at or created_at
    if metadata is not None:
        doc["metadata"] = metadata
    return doc


async def test_doc_status_scheduling_page_orders_by_created_at_then_id(shared_driver, workspace):
    """Keyset pages follow (created_at, id) and stop only on CURSOR_END."""
    store = _doc_store(workspace)
    await store.initialize()
    try:
        await store.upsert(
            {
                # Same timestamp: id is the tiebreak, and they are written out of order.
                "m": _doc_at(DocStatus.PROCESSED, "m", "m.txt", "2026-02-01T00:00:00"),
                "a": _doc_at(DocStatus.PENDING, "a", "a.txt", "2026-02-01T00:00:00"),
                "z": _doc_at(DocStatus.FAILED, "z", "z.txt", "2026-03-01T00:00:00"),
                "early": _doc_at(DocStatus.PROCESSED, "early", "early.txt", "2026-01-01T00:00:00"),
            }
        )

        seen: list[str] = []
        position = CURSOR_START
        for _ in range(4):
            page_result = await store.get_docs_by_statuses_page(
                [DocStatus.PROCESSED, DocStatus.PENDING, DocStatus.FAILED],
                limit=2,
                position=position,
            )
            seen.extend(page_result.docs)
            if page_result.next_position is CURSOR_END:
                break
            position = page_result.next_position
        else:
            raise AssertionError("scheduling sweep did not terminate")

        assert seen == ["early", "a", "m", "z"]
        # A page that filled must be resumable, and the resume must not repeat.
        first = await store.get_docs_by_statuses_page(
            [DocStatus.PROCESSED, DocStatus.PENDING],
            limit=2,
        )
        assert list(first.docs) == ["early", "a"]
        assert isinstance(first.next_position, CursorAfter)
        second = await store.get_docs_by_statuses_page(
            [DocStatus.PROCESSED, DocStatus.PENDING],
            limit=2,
            position=first.next_position,
        )
        assert list(second.docs) == ["m"]
        assert second.next_position is CURSOR_END

        journal = await store.get_docs_by_ids(["early"])
        assert journal["early"].has_custom_chunk_journal is False
    finally:
        await store.drop()


async def test_doc_status_batch_reads_and_source_resolution(shared_driver, workspace):
    store = _doc_store(workspace)
    await store.initialize()
    try:
        await store.upsert(
            {
                "primary": _doc_at(
                    DocStatus.PROCESSED,
                    "primary",
                    "shared.txt",
                    "2026-01-01T00:00:00",
                    metadata={CUSTOM_CHUNK_PATCH_METADATA_KEY: {"ops": [1]}},
                ),
                "other": _doc_at(DocStatus.PENDING, "other", "shared.txt", "2026-01-02T00:00:00"),
                "marker": _doc_at(
                    DocStatus.PROCESSED,
                    "marker",
                    "only-marker.txt",
                    "2026-01-03T00:00:00",
                    metadata={"is_duplicate": True, "original_doc_id": "primary"},
                ),
                "solo": _doc_at(DocStatus.FAILED, "solo", "solo.txt", "2026-01-04T00:00:00"),
            }
        )

        scheduling = await store.get_docs_by_ids(["solo", "missing", "primary"], strict=True)
        assert set(scheduling) == {"solo", "primary"}
        assert scheduling["primary"].has_custom_chunk_journal is True
        assert scheduling["primary"].status == DocStatus.PROCESSED
        assert scheduling["solo"].file_path == "solo.txt"

        full = await store.get_full_docs_by_ids(["solo", "missing"], strict=True)
        assert set(full) == {"solo"}
        assert isinstance(full["solo"], DocProcessingStatus)
        assert full["solo"].content_summary == "solo"
        assert not hasattr(full["solo"], "content")

        absent = await store.resolve_doc_source_strict("does-not-exist.txt")
        assert isinstance(absent, SourceAbsent)
        # A file path that only has a duplicate marker is not a primary.
        marker_only = await store.resolve_doc_source_strict("only-marker.txt")
        assert isinstance(marker_only, SourceAbsent)

        unique = await store.resolve_doc_source_strict("solo.txt")
        assert isinstance(unique, SourceUnique)
        assert unique.doc_id == "solo"
        assert unique.doc.status == DocStatus.FAILED

        conflict = await store.resolve_doc_source_strict("shared.txt")
        assert isinstance(conflict, SourceConflict)
        assert conflict.candidate_count is None
        assert conflict.sample_doc_ids == ("other", "primary")
    finally:
        await store.drop()


async def test_doc_status_content_hash_is_earliest_and_excludes_pointers(shared_driver, workspace):
    store = _doc_store(workspace)
    await store.initialize()
    try:
        await store.upsert(
            {
                # Later created_at but smaller id: must NOT win.
                "a": _doc_at(DocStatus.PROCESSED, "a", "a.txt", "2026-02-01T00:00:00", content_hash="H"),
                # Earliest independent holder.
                "z": _doc_at(DocStatus.PROCESSED, "z", "z.txt", "2026-01-02T00:00:00", content_hash="H"),
                # Sorts before z, but only points at z.
                "p": _doc_at(
                    DocStatus.PROCESSED,
                    "p",
                    "p.txt",
                    "2026-01-01T00:00:00",
                    content_hash="H",
                    metadata={"is_duplicate": True, "original_doc_id": "z"},
                ),
            }
        )

        found = await store.get_doc_by_content_hash("H")
        assert found is not None
        assert found[0] == "p"  # earliest row overall is the pointer

        excluded = await store.get_doc_by_content_hash("H", exclude_doc_id="z")
        assert excluded is not None
        # Pointer at z is skipped, and z itself is skipped, so "a" remains.
        assert excluded[0] == "a"
        assert excluded[1]["content_hash"] == "H"

        untouched = await store.get_doc_by_content_hash("H", exclude_doc_id="not-a-holder")
        assert untouched is not None
        assert untouched[0] == "p"
        only_self = await store.get_doc_by_content_hash("missing", exclude_doc_id="z")
        assert only_self is None
    finally:
        await store.drop()


async def test_doc_status_strict_reads_fail_on_corrupt_rows(shared_driver, workspace):
    """A row that cannot be decoded raises under strict and is skipped when relaxed.

    The relaxed scheduling page still consumes the bad row, so the cursor
    moves past it instead of terminating or rereading it.
    """
    store = _doc_store(workspace)
    await store.initialize()
    try:
        await store.upsert(
            {
                "bad": _doc_at(DocStatus.PENDING, "bad", "bad.txt", "2026-01-01T00:00:00"),
                "good": _doc_at(DocStatus.PENDING, "good", "good.txt", "2026-01-02T00:00:00"),
            }
        )
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await (
                await session.run(
                    as_literal_query(f"MATCH (n:`{store._label}` {{id: $id}}) SET n.data = $data"),
                    id="bad",
                    data="not-json",
                )
            ).consume()

        relaxed = await store.get_docs_by_statuses([DocStatus.PENDING], strict=False)
        assert set(relaxed) == {"good"}
        with pytest.raises(json.JSONDecodeError):
            await store.get_docs_by_statuses([DocStatus.PENDING], strict=True)

        assert await store.get_docs_by_ids(["bad"], strict=False) == {}
        with pytest.raises(json.JSONDecodeError):
            await store.get_docs_by_ids(["bad"], strict=True)
        assert set(await store.get_full_docs_by_ids(["good", "bad"], strict=False)) == {"good"}
        with pytest.raises(json.JSONDecodeError):
            await store.get_full_docs_by_ids(["bad"], strict=True)

        with pytest.raises(json.JSONDecodeError):
            await store.get_docs_by_statuses_page([DocStatus.PENDING], limit=2, strict=True)
        skipped = await store.get_docs_by_statuses_page([DocStatus.PENDING], limit=1, strict=False)
        assert skipped.docs == {}
        assert isinstance(skipped.next_position, CursorAfter)
        rest = await store.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=skipped.next_position, strict=False
        )
        assert list(rest.docs) == ["good"]
        # A full page is not exhaustion: the next call is the one that ends.
        assert isinstance(rest.next_position, CursorAfter)
        done = await store.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=rest.next_position, strict=False
        )
        assert done.docs == {}
        assert done.next_position is CURSOR_END
    finally:
        await store.drop()


# --- vector storage ----------------------------------------------------------


async def test_vector_roundtrip_and_nearest_neighbour(shared_driver, vector_search_supported, workspace):
    store = MemgraphVectorStorage(
        namespace="chunks",
        workspace=workspace,
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
        meta_fields={"content", "source_id"},
    )
    await store.initialize()
    try:
        await store.upsert(
            {
                "v-cat": {"content": "cat", "source_id": "s-cat"},
                "v-kitten": {"content": "kitten", "source_id": "s-kitten"},
                "v-dog": {"content": "dog", "source_id": "s-dog"},
                "v-car": {"content": "car", "source_id": "s-car"},
            }
        )

        # Direct fetch round-trips (embedding is stripped from the returned dict).
        got = await store.get_by_id("v-cat")
        assert got is not None
        assert got["content"] == "cat"
        assert got["source_id"] == "s-cat"
        assert "embedding" not in got

        many = await store.get_by_ids(["v-cat", "v-dog"])
        assert {m["id"] for m in many} == {"v-cat", "v-dog"}
        assert all("embedding" not in m for m in many)

        vectors = await store.get_vectors_by_ids(["v-cat"])
        assert "v-cat" in vectors
        assert len(vectors["v-cat"]) == _EMBED_DIM

        # Nearest-neighbour search for "cat": expect cat first (cos 1.0), then
        # kitten (cos 0.8). dog and car (cos 0.0) are below the 0.2 threshold
        # and must be excluded. This proves the vector_search.search path and
        # the similarity direction (higher = closer) end to end.
        results = await store.query("cat", top_k=4)
        ids = [r["id"] for r in results]
        assert ids == ["v-cat", "v-kitten"], f"unexpected NN order/result: {ids}"

        # Similarities must be monotonically non-increasing (nearest first) and
        # the closest match must score highest.
        sims = [r["distance"] for r in results]
        assert sims[0] >= sims[1]
        assert results[0]["id"] == "v-cat"
        # cat vs itself under cosine is ~1.0; allow numerical slack.
        assert sims[0] == pytest.approx(1.0, abs=1e-3)
        # dog / car are below threshold and must not appear.
        assert "v-dog" not in ids
        assert "v-car" not in ids

        # top_k limits the number of candidates considered.
        top1 = await store.query("cat", top_k=1)
        assert [r["id"] for r in top1] == ["v-cat"]

        await store.delete(["v-cat"])
        assert await store.get_by_id("v-cat") is None
        assert [r["id"] for r in await store.query("cat", top_k=4)] == ["v-kitten"]

        result = await store.drop()
        assert result["status"] == "success"
        assert await store.get_by_id("v-kitten") is None
    finally:
        await store.drop()


async def test_vector_respects_threshold(shared_driver, vector_search_supported, workspace):
    """A high cosine threshold filters out the merely-similar neighbour."""
    config = _global_config()
    config["vector_db_storage_cls_kwargs"]["cosine_better_than_threshold"] = 0.95
    store = MemgraphVectorStorage(
        namespace="chunks",
        workspace=workspace,
        global_config=config,
        embedding_func=_fake_embedding_func(),
        meta_fields={"content"},
    )
    await store.initialize()
    try:
        await store.upsert(
            {
                "v-cat": {"content": "cat"},
                "v-kitten": {"content": "kitten"},
            }
        )
        # cos(cat, kitten) = 0.8 < 0.95, so only the exact match survives.
        results = await store.query("cat", top_k=4)
        assert [r["id"] for r in results] == ["v-cat"]
    finally:
        await store.drop()


async def test_vector_query_logs_and_excludes_stale_candidates(
    shared_driver, vector_search_supported, workspace, caplog
):
    """A node deleted without nulling its embedding first (bypassing the
    store's own delete(), which nulls it precisely to avoid this) can still
    surface as a vector_search.search candidate even though it's gone from
    the graph. query() must exclude it instead of erroring, and warn with
    the raw-hit vs live-candidate counts.
    """
    store = MemgraphVectorStorage(
        namespace="chunks",
        workspace=workspace,
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
        meta_fields={"content"},
    )
    await store.initialize()
    try:
        await store.upsert(
            {
                "v-cat": {"content": "cat"},
                "v-kitten": {"content": "kitten"},
            }
        )

        # Bypass store.delete() (which nulls the embedding first) so the raw
        # DETACH DELETE leaves a stale entry in the native vector index.
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            query = as_literal_query(f"MATCH (n:`{store._label}` {{id: 'v-cat'}}) DETACH DELETE n")
            await (await session.run(query)).consume()

        # lightrag's logger has propagate=False, so caplog.at_level (which only
        # relies on root-logger propagation) never sees its records; attach the
        # capture handler directly instead.
        lightrag_logger = logging.getLogger("lightrag")
        lightrag_logger.addHandler(caplog.handler)
        try:
            results = await store.query("cat", top_k=4)
        finally:
            lightrag_logger.removeHandler(caplog.handler)

        # The stale candidate is excluded, not erroring and not silently
        # dropped without a trace.
        assert [r["id"] for r in results] == ["v-kitten"]
        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("vector_search.search returned" in m and "but only" in m for m in warnings), warnings
    finally:
        await store.drop()


# --- zero-config MemgraphLightRAGWrapper (#217 regression) -------------------

# All Memgraph-related names the wrapper/backends could possibly read, so the
# regression test below can genuinely start from zero Memgraph env vars.
_ALL_MEMGRAPH_ENV_NAMES = (
    "MEMGRAPH_URL",
    "MEMGRAPH_URI",
    "MEMGRAPH_USER",
    "MEMGRAPH_USERNAME",
    "MEMGRAPH_PASSWORD",
    "MEMGRAPH_DATABASE",
    "MEMGRAPH_HOST",
    "MEMGRAPH_PORT",
    "MEMGRAPH_WORKSPACE",
)


@pytest.fixture(scope="session")
def default_memgraph_reachable() -> None:
    """Skip unless a Memgraph is reachable at the true zero-config default:
    bolt://localhost:7687 with no auth. This is what full_memgraph_persistence=False
    must fall back to with no Memgraph env vars set at all, independent of whatever
    MEMGRAPH_URL the rest of this module resolves to.
    """
    from neo4j import GraphDatabase

    driver = None
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("", ""), connection_timeout=5)
        driver.verify_connectivity()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"No live Memgraph reachable at the zero-config default bolt://localhost:7687: {exc}")
    finally:
        if driver is not None:
            driver.close()


async def _dummy_llm_model_func(prompt, **kwargs):
    return "dummy response"


async def test_wrapper_full_persistence_false_works_with_zero_env_vars(
    default_memgraph_reachable, tmp_path, monkeypatch
):
    """Regression test for #217.

    MemgraphLightRAGWrapper(full_memgraph_persistence=False).initialize(...) must
    work with zero Memgraph-related env vars set, against a local default Memgraph,
    same as on main. Before the fix, _bridge_lightrag_env_names() left MEMGRAPH_URI
    unset in this case, and LightRAG's own MemgraphStorage graph backend raised
    ValueError("... requires the following environment variables: MEMGRAPH_URI")
    during initialize_storages().
    """
    for name in _ALL_MEMGRAPH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    wrapper = MemgraphLightRAGWrapper(full_memgraph_persistence=False)
    ws = "it_" + uuid.uuid4().hex[:10]
    try:
        await wrapper.initialize(
            working_dir=str(tmp_path),
            workspace=ws,
            embedding_func=_fake_embedding_func(),
            llm_model_func=_dummy_llm_model_func,
        )
        graph = wrapper.get_lightrag().chunk_entity_relation_graph
        # A real round trip against Memgraph, not just a successful construction.
        result = await graph.drop()
        assert result["status"] == "success"
    finally:
        if wrapper.rag is not None:
            await wrapper.afinalize()


# --- default embedding_func ---------------------------------------------------


async def test_wrapper_defaults_embedding_func_to_working_memgraph_sentence_embed(
    shared_driver, embeddings_module_supported, workspace, tmp_path, monkeypatch
):
    """Omitting embedding_func must not silently fall back to a billed
    openai_embed call. It must resolve to Memgraph's own sentence-transformer
    default, and that default must actually work end to end.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    wrapper = MemgraphLightRAGWrapper()
    try:
        await wrapper.initialize(
            working_dir=str(tmp_path),
            workspace=workspace,
            llm_model_func=_dummy_llm_model_func,
            # embedding_func intentionally omitted.
        )
        rag = wrapper.get_lightrag()
        # LightRAG re-wraps the passed EmbeddingFunc (concurrency-limiting the
        # underlying callable, unwrapping the EmbeddingFunc nesting), so identity
        # doesn't survive the round trip -- the declared model/dim and, most
        # importantly, actual behaviour do.
        embedding_func = rag.embedding_func
        assert embedding_func is not None
        assert embedding_func.model_name == memgraph_sentence_embed.model_name
        assert embedding_func.embedding_dim == DEFAULT_EMBEDDING_DIM
        vectors = await embedding_func(["hello world", "graph databases are fast"])
        assert vectors.shape == (2, DEFAULT_EMBEDDING_DIM)
    finally:
        if wrapper.rag is not None:
            await wrapper.afinalize()
