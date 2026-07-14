"""Tests for the Memgraph KV / vector / doc-status storage backends.

These tests exercise the import, registration and instantiation paths without a
live Memgraph server: only lifecycle/registration logic is checked, not any
Cypher execution.
"""

import numpy as np
import pytest
from lightrag.base import BaseKVStorage, BaseVectorStorage, DocStatusStorage
from lightrag.kg import verify_storage_implementation
from lightrag.utils import EmbeddingFunc


def _fake_embedding_func(dim: int = 8) -> EmbeddingFunc:
    async def _embed(texts):
        return np.ones((len(texts), dim), dtype=float)

    return EmbeddingFunc(embedding_dim=dim, func=_embed)


def _global_config() -> dict:
    return {
        "working_dir": "/tmp/lightrag_memgraph_test",
        "embedding_batch_num": 4,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


def test_register_is_idempotent():
    import lightrag.kg as kg

    from lightrag_memgraph import register_memgraph_storages

    register_memgraph_storages()
    register_memgraph_storages()  # second call must be a no-op

    for name, storage_type in [
        ("MemgraphKVStorage", "KV_STORAGE"),
        ("MemgraphVectorStorage", "VECTOR_STORAGE"),
        ("MemgraphDocStatusStorage", "DOC_STATUS_STORAGE"),
    ]:
        impls = kg.STORAGE_IMPLEMENTATIONS[storage_type]["implementations"]
        assert impls.count(name) == 1
        assert kg.STORAGES[name].startswith("lightrag_memgraph.")
        assert kg.STORAGE_ENV_REQUIREMENTS[name] == ["MEMGRAPH_URI"]


@pytest.mark.parametrize(
    "name,storage_type",
    [
        ("MemgraphKVStorage", "KV_STORAGE"),
        ("MemgraphVectorStorage", "VECTOR_STORAGE"),
        ("MemgraphDocStatusStorage", "DOC_STATUS_STORAGE"),
    ],
)
def test_verify_storage_implementation_accepts_names(name, storage_type):
    from lightrag_memgraph import register_memgraph_storages

    register_memgraph_storages()
    # Should not raise.
    verify_storage_implementation(storage_type, name)


def test_kv_storage_instantiation():
    from lightrag_memgraph import MemgraphKVStorage

    store = MemgraphKVStorage(
        namespace="full_docs",
        workspace="base",
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )
    assert isinstance(store, BaseKVStorage)
    assert store._label == "LightRAGKV_base_full_docs"


def test_vector_storage_instantiation():
    from lightrag_memgraph import MemgraphVectorStorage

    store = MemgraphVectorStorage(
        namespace="entities",
        workspace="base",
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
        meta_fields={"entity_name", "content"},
    )
    assert isinstance(store, BaseVectorStorage)
    assert store._label == "LightRAGVector_base_entities"
    assert store._index_name == "lightrag_vec_base_entities"
    assert store.cosine_better_than_threshold == 0.2
    assert store._dimension == 8


def test_doc_status_storage_instantiation():
    from lightrag_memgraph import MemgraphDocStatusStorage

    store = MemgraphDocStatusStorage(
        namespace="doc_status",
        workspace="base",
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )
    assert isinstance(store, DocStatusStorage)
    assert store._label == "LightRAGDocStatus_base"


def test_doc_status_reconstruction_roundtrip():
    """_to_status must rebuild a DocProcessingStatus from a stored dict."""
    from lightrag.base import DocStatus

    from lightrag_memgraph import MemgraphDocStatusStorage

    stored = {
        "content_summary": "hello",
        "content_length": 5,
        "file_path": "doc1",
        "status": "processed",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "chunks_list": ["c1", "c2"],
        "content": "should be dropped",
    }
    status = MemgraphDocStatusStorage._to_status(stored)
    assert status.status == DocStatus.PROCESSED
    assert status.file_path == "doc1"
    assert status.chunks_count is None
    assert status.metadata == {}


@pytest.mark.asyncio
async def test_get_doc_by_basename_and_hash_none_cases():
    """Empty input and the 'unknown_source' sentinel short-circuit to None.

    These paths return before touching Memgraph, so they are exercised here as
    unit tests; the positive round-trip lives in the live integration test.
    """
    from lightrag_memgraph import MemgraphDocStatusStorage

    store = MemgraphDocStatusStorage(
        namespace="doc_status",
        workspace="base",
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )
    assert await store.get_doc_by_file_basename("") is None
    assert await store.get_doc_by_file_basename("unknown_source") is None
    assert await store.get_doc_by_content_hash("") is None


def test_content_hash_is_indexed_field():
    """content_hash must be promoted to an indexed node property in 1.5.x."""
    from lightrag_memgraph.docstatus_impl import _INDEXED_FIELDS

    assert "content_hash" in _INDEXED_FIELDS


@pytest.mark.asyncio
async def test_get_all_status_counts_includes_all_key(monkeypatch):
    """get_all_status_counts must add an 'all' grand-total key (reference contract)."""
    from lightrag.base import DocStatus

    from lightrag_memgraph import MemgraphDocStatusStorage

    store = MemgraphDocStatusStorage(
        namespace="doc_status",
        workspace="base",
        global_config=_global_config(),
        embedding_func=_fake_embedding_func(),
    )

    async def _fake_status_counts():
        return {s.value: 0 for s in DocStatus} | {DocStatus.PROCESSED.value: 3, DocStatus.PENDING.value: 2}

    monkeypatch.setattr(store, "get_status_counts", _fake_status_counts)

    counts = await store.get_all_status_counts()
    assert counts["all"] == 5
    assert counts[DocStatus.PROCESSED.value] == 3
