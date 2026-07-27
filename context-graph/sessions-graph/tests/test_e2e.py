"""End-to-end tests against a real Memgraph instance.

Requires Memgraph reachable at bolt://localhost:7687 (default) -- override
via MEMGRAPH_URL/MEMGRAPH_USER/MEMGRAPH_PASSWORD/MEMGRAPH_DATABASE. Skips
cleanly if unreachable (see conftest.py's `memgraph`/`graph` fixtures).

No LLM calls in this file -- see test_e2e_reconciliation.py for the tier that
exercises real entity extraction via reconcile_session().
"""

from __future__ import annotations

import pytest


def test_save_and_get_memories_round_trip(graph):
    mem = graph.save_memory("alice", "Prefers Python over TypeScript")

    memories = graph.get_memories("alice")

    assert len(memories) == 1
    assert memories[0].memory_id == mem.memory_id
    assert memories[0].content == "Prefers Python over TypeScript"


def test_search_memories_finds_matching_content(graph):
    graph.save_memory("alice", "Prefers Python over TypeScript")
    graph.save_memory("alice", "Likes hiking on weekends")

    results = graph.search_memories("alice", "Python")

    assert len(results) == 1
    assert "Python" in results[0].content


def test_update_and_delete_memory(graph):
    mem = graph.save_memory("alice", "Original content")

    updated = graph.update_memory(mem.memory_id, "Updated content")
    assert updated.content == "Updated content"

    graph.delete_memory(mem.memory_id)
    assert graph.get_memories("alice") == []


def test_get_memories_for_session(graph):
    graph.save_memory("alice", "Fact one", session_id="s-1")
    graph.save_memory("alice", "Fact two", session_id="s-1")
    graph.save_memory("alice", "Unrelated fact", session_id="s-2")

    session_memories = graph.get_memories_for_session("s-1")

    assert {m.content for m in session_memories} == {"Fact one", "Fact two"}


def test_connector_session_start_creates_user_and_session(graph, memgraph):
    pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")
    from sessions_graph.connector import SessionsGraphConnector

    from agent_context_graph.events import SessionStartEvent

    connector = SessionsGraphConnector(graph)
    connector.on_event(SessionStartEvent(session_id="s-1", user_id="alice"))

    rows = memgraph.query(
        "MATCH (u:User {user_id: 'alice'})-[:HAD_SESSION]->(s:Session {session_id: 's-1'}) RETURN count(*) AS count"
    )
    assert rows[0]["count"] == 1


def test_connector_session_end_marks_reconciliation_pending(graph, memgraph):
    pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")
    from sessions_graph.connector import SessionsGraphConnector

    from agent_context_graph.events import SessionEndEvent, SessionStartEvent

    connector = SessionsGraphConnector(graph)
    connector.on_event(SessionStartEvent(session_id="s-1", user_id="alice"))
    connector.on_event(SessionEndEvent(session_id="s-1"))

    rows = memgraph.query("MATCH (s:Session {session_id: 's-1'}) RETURN s.reconciliation_status AS status")
    assert rows[0]["status"] == "pending"

    pending = graph.get_pending_reconciliation_sessions()
    assert "s-1" in pending
