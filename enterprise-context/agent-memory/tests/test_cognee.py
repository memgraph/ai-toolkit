"""Integration tests for the Cognee framework with Memgraph.

Requires:
    pip install cognee-community-graph-adapter-memgraph
    A running Memgraph instance.
    OPENAI_API_KEY set in the environment (cognee uses an LLM for extraction).
"""

import asyncio
import os
import pathlib

import pytest

cognee = pytest.importorskip("cognee")
register = pytest.importorskip(
    "cognee_community_graph_adapter_memgraph",
    reason="cognee-community-graph-adapter-memgraph not installed",
).register

pytestmark = [pytest.mark.cognee, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
async def _configure_cognee(
    memgraph_url, memgraph_username, memgraph_password, tmp_path
):
    """Register the Memgraph adapter and configure cognee for each test."""
    register()

    cognee.config.set_graph_database_provider("memgraph")
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": memgraph_url,
            "graph_database_username": memgraph_username or "memgraph",
            "graph_database_password": memgraph_password or "memgraph",
        }
    )

    # Cognee reads the LLM API key from the LLM_API_KEY env var.
    # Ensure it is set (fall back to OPENAI_API_KEY which CI provides).
    if not os.environ.get("LLM_API_KEY"):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            os.environ["LLM_API_KEY"] = api_key

    system_dir = tmp_path / ".cognee_system"
    data_dir = tmp_path / ".data_storage"
    system_dir.mkdir()
    data_dir.mkdir()
    cognee.config.system_root_directory(str(system_dir))
    cognee.config.data_root_directory(str(data_dir))
    yield


async def test_cognee_add_and_search(assert_graph_not_empty):
    """Add sample data, cognify, and verify nodes are persisted in Memgraph."""
    sample_data = [
        "Artificial intelligence is a branch of computer science.",
        "Machine learning is a subset of AI that focuses on algorithms that learn from data.",
    ]

    await cognee.add(sample_data, "test_knowledge")
    await cognee.cognify(["test_knowledge"])

    # Verify data landed in Memgraph
    assert_graph_not_empty(min_nodes=1, min_relationships=1)

    results = await cognee.search(
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        query_text="artificial intelligence",
    )

    assert results is not None
    assert len(results) > 0


async def test_cognee_graph_data_accessible(assert_graph_not_empty, run_cypher):
    """Verify that graph data can be retrieved after cognify and nodes exist in Memgraph."""
    from cognee.infrastructure.databases.graph import get_graph_engine

    await cognee.add(["Graph databases store data as nodes and edges."], "graph_data")
    await cognee.cognify(["graph_data"])

    # Verify via direct Cypher that Memgraph has data
    assert_graph_not_empty(min_nodes=1)

    records = run_cypher("MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt")
    assert len(records) > 0, "Expected at least one label group in Memgraph"

    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()

    assert graph_data is not None
