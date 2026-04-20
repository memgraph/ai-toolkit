"""Integration tests for neo4j-agent-memory with Memgraph.

Tests whether the neo4j-agent-memory library (designed for Neo4j) works
against Memgraph via Bolt protocol compatibility.

Requires:
    pip install neo4j-agent-memory
    A running Memgraph instance.
"""

import pytest

neo4j_agent_memory = pytest.importorskip(
    "neo4j_agent_memory", reason="neo4j-agent-memory not installed"
)

pytestmark = [pytest.mark.neo4j_agent_memory, pytest.mark.asyncio]


@pytest.fixture()
def memory_settings(memgraph_url, memgraph_password):
    """Create MemorySettings pointed at Memgraph."""
    from neo4j_agent_memory import MemorySettings, ExtractionConfig, ExtractorType

    return MemorySettings(
        neo4j={
            "uri": memgraph_url,
            "password": memgraph_password or "",
            "database": "memgraph",
        },
        extraction=ExtractionConfig(extractor_type=ExtractorType.NONE),
    )


async def test_short_term_memory(memory_settings, assert_graph_not_empty, run_cypher):
    """Store and retrieve short-term (conversation) memory, verify in Memgraph."""
    from neo4j_agent_memory import MemoryClient

    async with MemoryClient(memory_settings) as memory:
        await memory.short_term.add_message(
            session_id="session-1",
            role="user",
            content="Hello, my name is Alice and I work at Memgraph.",
        )
        await memory.short_term.add_message(
            session_id="session-1",
            role="assistant",
            content="Nice to meet you, Alice!",
        )

        messages = await memory.short_term.get_messages(session_id="session-1")
        assert len(messages) >= 2

    # Verify messages were persisted as nodes in Memgraph
    assert_graph_not_empty(min_nodes=2)
    records = run_cypher("MATCH (n) RETURN count(n) AS cnt")
    assert records[0]["cnt"] >= 2, "Expected at least 2 message nodes in Memgraph"


async def test_long_term_memory_entities(
    memory_settings, assert_graph_not_empty, run_cypher
):
    """Add entities and preferences to long-term memory, verify in Memgraph."""
    from neo4j_agent_memory import MemoryClient

    async with MemoryClient(memory_settings) as memory:
        await memory.long_term.add_entity("Alice", "PERSON")
        await memory.long_term.add_preference(
            category="technology",
            preference="Prefers graph databases",
        )

        context = await memory.get_context(
            "What technology does Alice prefer?",
            session_id="session-1",
        )
        assert context is not None

    # Verify entity and preference nodes exist in Memgraph
    assert_graph_not_empty(min_nodes=1)
    records = run_cypher("MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt")
    assert len(records) > 0, "Expected labeled nodes for entities/preferences"


async def test_get_context(memory_settings, assert_graph_not_empty, run_cypher):
    """Verify get_context returns combined short+long term results from Memgraph."""
    from neo4j_agent_memory import MemoryClient

    async with MemoryClient(memory_settings) as memory:
        await memory.short_term.add_message(
            session_id="session-2",
            role="user",
            content="I enjoy reading about distributed systems.",
        )
        await memory.long_term.add_entity("DistributedSystems", "TOPIC")

        context = await memory.get_context(
            "distributed systems",
            session_id="session-2",
        )
        assert context is not None

    # Verify both message and entity nodes exist in Memgraph
    assert_graph_not_empty(min_nodes=2)
    records = run_cypher("MATCH (n) RETURN count(n) AS cnt")
    assert (
        records[0]["cnt"] >= 2
    ), "Expected nodes from both short-term and long-term memory"
