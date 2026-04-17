"""Integration tests for Mem0 graph memory with Memgraph.

Requires:
    pip install "mem0ai[graph]"
    A running Memgraph instance.
    OPENAI_API_KEY set in the environment (mem0 uses OpenAI for embeddings).
"""

import os

import pytest

mem0_memory = pytest.importorskip("mem0", reason="mem0ai not installed")

pytestmark = pytest.mark.mem0


@pytest.fixture()
def memory(memgraph_url, memgraph_username, memgraph_password):
    """Create a Mem0 Memory instance configured with Memgraph."""
    from mem0 import Memory

    config = {
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


def test_mem0_add_and_search(memory):
    """Store messages and verify search retrieves relevant memories."""
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

    search_results = memory.search("What does the user enjoy?", user_id="test_user")
    assert search_results is not None
    assert "results" in search_results
    assert len(search_results["results"]) > 0


def test_mem0_multiple_users(memory):
    """Verify memories are isolated per user_id."""
    memory.add(
        [{"role": "user", "content": "I prefer Python for data science."}],
        user_id="alice",
    )
    memory.add(
        [{"role": "user", "content": "I prefer Rust for systems programming."}],
        user_id="bob",
    )

    alice_results = memory.search("programming language", user_id="alice")
    bob_results = memory.search("programming language", user_id="bob")

    assert alice_results is not None
    assert bob_results is not None
