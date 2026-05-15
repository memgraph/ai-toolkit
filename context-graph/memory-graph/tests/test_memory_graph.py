"""Unit tests for Memory Graph models, core, and connector.

These tests use an in-memory stub for the Memgraph client so they run
without a live database.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from memory_graph.models import Memory, MemoryValidationError

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class TestMemory:
    def test_valid_memory_is_created(self):
        m = Memory(user_id="alice", content="Prefers Python")
        assert m.user_id == "alice"
        assert m.content == "Prefers Python"
        assert m.memory_id  # auto-generated
        assert m.created_at
        assert m.session_id is None

    def test_empty_content_raises(self):
        with pytest.raises(MemoryValidationError, match="content"):
            Memory(user_id="alice", content="")

    def test_whitespace_only_content_raises(self):
        with pytest.raises(MemoryValidationError, match="content"):
            Memory(user_id="alice", content="   ")

    def test_invalid_user_id_raises(self):
        with pytest.raises(MemoryValidationError, match="user_id"):
            Memory(user_id="alice bob", content="some fact")  # space not allowed

    def test_session_id_stored(self):
        m = Memory(user_id="alice", content="fact", session_id="s-1")
        assert m.session_id == "s-1"


# ---------------------------------------------------------------------------
# core (stubbed Memgraph)
# ---------------------------------------------------------------------------


def _stub_db(rows: list | None = None):
    db = MagicMock()
    db.query.return_value = rows or []
    return db


def _graph(rows=None):
    from memory_graph.core import MemoryGraph

    g = MemoryGraph.__new__(MemoryGraph)
    g._db = _stub_db(rows)
    return g


class TestMemoryGraphCore:
    def test_save_memory_runs_two_queries(self):
        g = _graph()
        mem = g.save_memory("alice", "Prefers dark mode")

        assert mem.user_id == "alice"
        assert mem.content == "Prefers dark mode"
        assert g._db.query.call_count == 1  # no session_id → one query only

    def test_save_memory_with_session_runs_two_queries(self):
        g = _graph()
        g.save_memory("alice", "Fact", session_id="s-1")
        assert g._db.query.call_count == 2  # main + provenance

    def test_save_memory_rejects_empty_content(self):
        g = _graph()
        with pytest.raises(MemoryValidationError):
            g.save_memory("alice", "")

    def test_get_memories_returns_empty_list(self):
        g = _graph(rows=[])
        result = g.get_memories("alice")
        assert result == []

    def test_get_memories_maps_rows(self):
        rows = [
            {
                "memory_id": "m-1",
                "user_id": "alice",
                "content": "Prefers Python",
                "created_at": "2026-01-01T00:00:00+00:00",
                "session_id": None,
            }
        ]
        g = _graph(rows=rows)
        result = g.get_memories("alice")
        assert len(result) == 1
        assert result[0].content == "Prefers Python"

    def test_search_memories_skips_empty_query(self):
        g = _graph()
        result = g.search_memories("alice", "")
        assert result == []
        g._db.query.assert_not_called()

    def test_delete_memory_calls_detach_delete(self):
        g = _graph()
        g.delete_memory("m-1")
        query_text = g._db.query.call_args.args[0]
        assert "DETACH DELETE" in query_text

    def test_update_memory_returns_none_when_not_found(self):
        g = _graph(rows=[])
        result = g.update_memory("m-1", "new content")
        assert result is None


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


class TestMemoryGraphConnector:
    def _make(self):
        pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")
        from memory_graph.connector import MemoryGraphConnector

        from agent_context_graph.events import SessionEndEvent, SessionStartEvent

        db = MagicMock()
        db.query.return_value = []

        from memory_graph.core import MemoryGraph

        graph = MemoryGraph.__new__(MemoryGraph)
        graph._db = db

        connector = MemoryGraphConnector(graph)
        return connector, graph, db, SessionStartEvent, SessionEndEvent

    def test_session_start_merges_user_and_session_nodes(self):
        connector, _graph, db, SessionStartEvent, _ = self._make()

        event = SessionStartEvent(session_id="s-1", user_id="alice")
        connector.on_event(event)

        assert connector.active_user_id == "alice"
        assert connector.active_session_id == "s-1"
        assert db.query.call_count == 2  # User MERGE + Session MERGE

    def test_session_start_without_user_id_only_merges_session(self):
        connector, _graph, db, SessionStartEvent, _ = self._make()

        event = SessionStartEvent(session_id="s-1")
        connector.on_event(event)

        assert connector.active_user_id is None
        assert connector.active_session_id == "s-1"
        assert db.query.call_count == 1  # Session MERGE only

    def test_session_end_clears_active_context(self):
        connector, _graph, _db, SessionStartEvent, SessionEndEvent = self._make()

        connector.on_event(SessionStartEvent(session_id="s-1", user_id="alice"))
        connector.on_event(SessionEndEvent(session_id="s-1"))

        assert connector.active_user_id is None
        assert connector.active_session_id is None

    def test_supports_session_events_only(self):
        connector, _, _, SessionStartEvent, SessionEndEvent = self._make()
        from agent_context_graph.events import ToolStartEvent

        assert connector.supports(SessionStartEvent(session_id="s-1"))
        assert connector.supports(SessionEndEvent(session_id="s-1"))
        assert not connector.supports(ToolStartEvent(session_id="s-1", tool_name="read_file"))
