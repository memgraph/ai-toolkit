"""Live-Memgraph tests for the default `embeddings.text()`-backed embedding function.

Mirrors test_integration_memgraph.py's pattern: skip the whole module (not fail)
if no live Memgraph is reachable, or if it has no `embeddings` module loaded
(e.g. a plain `memgraph` image instead of `memgraph-mage`).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import pytest_asyncio

from lightrag_memgraph._connection import close_driver, get_driver
from lightrag_memgraph.embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    build_memgraph_sentence_embed,
    memgraph_sentence_embed,
)
from memgraph_toolbox.api.memgraph import memgraph_env

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _connection_settings() -> tuple[str, str, str, str]:
    env = memgraph_env()
    return (env["MEMGRAPH_URL"], env["MEMGRAPH_USER"], env["MEMGRAPH_PASSWORD"], env["MEMGRAPH_DATABASE"])


def _resolve_uri() -> str:
    uri = os.environ.get("MEMGRAPH_URL")
    if not uri:
        host = os.environ.get("MEMGRAPH_HOST")
        if host:
            uri = f"bolt://{host}:{os.environ.get('MEMGRAPH_PORT') or 7687}"
    if uri:
        os.environ["MEMGRAPH_URL"] = uri
    else:
        uri = _connection_settings()[0]
    return uri


@pytest.fixture(scope="session")
def memgraph_uri() -> str:
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
def embeddings_module_supported(memgraph_uri: str) -> bool:
    """Skip if the server has no `embeddings` module (e.g. plain `memgraph`, not `memgraph-mage`)."""
    from neo4j import GraphDatabase

    _uri, username, password, database = _connection_settings()
    driver = GraphDatabase.driver(memgraph_uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            names = {record["name"] for record in session.run("CALL mg.procedures() YIELD name RETURN name")}
        if not any(n.startswith("embeddings.") for n in names):
            pytest.skip("Memgraph server has no `embeddings` module (needs the memgraph-mage image)")
    finally:
        driver.close()
    return True


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def shared_driver(memgraph_uri: str):
    driver = await get_driver()
    yield driver
    await close_driver()


async def test_memgraph_sentence_embed_returns_real_vectors(shared_driver, embeddings_module_supported):
    result = await memgraph_sentence_embed(["hello world", "graph databases are fast"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, DEFAULT_EMBEDDING_DIM)
    # Real model output, not a stub -- rows for different sentences must differ.
    assert not np.allclose(result[0], result[1])


async def test_memgraph_sentence_embed_model_name_is_default(shared_driver, embeddings_module_supported):
    assert memgraph_sentence_embed.model_name == DEFAULT_MODEL_NAME


async def test_memgraph_sentence_embed_handles_empty_input(shared_driver, embeddings_module_supported):
    result = await memgraph_sentence_embed([])
    assert result.shape == (0,) or result.size == 0


async def test_build_memgraph_sentence_embed_dimension_mismatch_raises(shared_driver, embeddings_module_supported):
    """A model_name/embedding_dim pair that doesn't match the module's real output
    must surface as an error (via EmbeddingFunc's own dimension check), not silently
    return misshapen vectors.
    """
    mismatched = build_memgraph_sentence_embed(model_name=DEFAULT_MODEL_NAME, embedding_dim=999)
    with pytest.raises(ValueError, match="dimension mismatch"):
        await mismatched(["hello world"])
