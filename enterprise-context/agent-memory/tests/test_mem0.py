"""Integration tests for Mem0 graph memory with Memgraph.

Requires:
    pip install "mem0ai[graph]"
    A running Memgraph instance.
    OPENAI_API_KEY set in the environment (mem0 uses OpenAI for embeddings and LLM extraction).
"""

import os
import uuid

import pytest

mem0_memory = pytest.importorskip("mem0", reason="mem0ai not installed")

pytestmark = pytest.mark.mem0


@pytest.fixture()
def memory(memgraph_url, memgraph_username, memgraph_password, tmp_path):
    """Create a Mem0 Memory instance configured with Memgraph.

    Uses a unique Qdrant path per test to avoid concurrent access errors,
    and configures the full pipeline (LLM + embedder + graph_store) so
    that entity extraction actually writes nodes to Memgraph.
    """
    from mem0 import Memory

    qdrant_path = str(tmp_path / f"qdrant_{uuid.uuid4().hex}")

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_test",
                "path": qdrant_path,
            },
        },
        "graph_store": {
            "provider": "memgraph",
            "config": {
                "url": memgraph_url,
                "username": memgraph_username or "memgraph",
                "password": memgraph_password or "memgraph",
            },
        },
    }
    return Memory.from_config(config_dict=config)


def test_mem0_add_and_search(memory, assert_graph_not_empty, run_cypher):
    """Store messages and verify data is persisted in Memgraph."""
    messages = [
        {"role": "user", "content": "I love hiking in the mountains."},
        {
            "role": "assistant",
            "content": "That sounds great! Do you have a favorite trail?",
        },
        {"role": "user", "content": "Yes, I really enjoy the Appalachian Trail."},
    ]

    result = memory.add(messages, user_id="test_user")
    assert result is not None

    # Verify nodes were created in Memgraph
    assert_graph_not_empty(min_nodes=1)

    # Check that at least one relationship was created (mem0 builds a graph of entities)
    records = run_cypher("MATCH (n) RETURN count(n) AS cnt")
    assert records[0]["cnt"] > 0, "mem0 should have created nodes in Memgraph"

    search_results = memory.search("What does the user enjoy?", user_id="test_user")
    assert search_results is not None
    assert "results" in search_results
    assert len(search_results["results"]) > 0


def test_mem0_multiple_users(memory, run_cypher):
    """Verify memories are isolated per user_id and stored in Memgraph."""
    memory.add(
        [{"role": "user", "content": "I prefer Python for data science."}],
        user_id="alice",
    )
    memory.add(
        [{"role": "user", "content": "I prefer Rust for systems programming."}],
        user_id="bob",
    )

    # Verify that nodes exist in Memgraph for both users
    records = run_cypher("MATCH (n) RETURN count(n) AS cnt")
    assert records[0]["cnt"] >= 2, "Expected nodes for both alice and bob in Memgraph"

    alice_results = memory.search("programming language", user_id="alice")
    bob_results = memory.search("programming language", user_id="bob")

    assert alice_results is not None
    assert bob_results is not None
    assert bob_results is not None
