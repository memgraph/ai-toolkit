"""Tests for the async Memgraph client.

The construction/config tests run without a live server (the async driver is
created lazily and never connected here). The live query test skips cleanly
when no Memgraph server is reachable, so this module never requires a server.
"""

import pytest

from ..api.memgraph import AsyncMemgraph, Memgraph


def test_async_memgraph_constructs_without_connecting():
    """AsyncMemgraph builds a driver lazily and does not connect in __init__."""
    client = AsyncMemgraph(
        url="bolt://localhost:7687",
        username="",
        password="",
        database="memgraph",
    )
    assert client.driver is not None
    assert client.database == "memgraph"


def test_async_memgraph_resolves_config_from_env(monkeypatch):
    """Connection config is resolved via memgraph_env (env vars + defaults)."""
    monkeypatch.setenv("MEMGRAPH_DATABASE", "env-db")
    client = AsyncMemgraph()
    assert client.database == "env-db"
    # Explicit values still win over the environment.
    explicit = AsyncMemgraph(database="arg-db")
    assert explicit.database == "arg-db"


def test_async_memgraph_mirrors_sync_defaults():
    """AsyncMemgraph shares the sync client's user-agent default."""
    assert AsyncMemgraph.DEFAULT_USER_AGENT == Memgraph.DEFAULT_USER_AGENT


@pytest.mark.asyncio
async def test_async_memgraph_query_live():
    """Live round-trip; skips cleanly when no server is reachable."""
    import neo4j

    client = AsyncMemgraph(url="bolt://localhost:7687", username="", password="")
    try:
        await client.verify_connectivity()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.SessionExpired, OSError):
        await client.close()
        pytest.skip("No Memgraph server reachable at bolt://localhost:7687")

    try:
        result = await client.query("RETURN 1 AS value")
        assert result == [{"value": 1}]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_memgraph_query_raw_live():
    """query_raw returns raw neo4j records holding live graph objects, not flat dicts."""
    import neo4j
    from neo4j.graph import Node

    client = AsyncMemgraph(url="bolt://localhost:7687", username="", password="")
    try:
        await client.verify_connectivity()
    except (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.SessionExpired, OSError):
        await client.close()
        pytest.skip("No Memgraph server reachable at bolt://localhost:7687")

    try:
        await client.query("MATCH (n:AsyncRawNode) DETACH DELETE n")
        await client.query("CREATE (:AsyncRawNode {name: 'A'})")
        records = await client.query_raw("MATCH (n:AsyncRawNode) RETURN n")

        assert len(records) == 1
        node = records[0]["n"]
        assert isinstance(node, Node)
        assert "AsyncRawNode" in node.labels
        assert node.element_id is not None
    finally:
        await client.query("MATCH (n:AsyncRawNode) DETACH DELETE n")
        await client.close()
