"""Unit tests for Sessions Graph models, core, and connector.

These tests use an in-memory stub for the Memgraph client so they run
without a live database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sessions_graph.models import Memory, MemoryValidationError

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
    from sessions_graph.core import SessionsGraph

    g = SessionsGraph.__new__(SessionsGraph)
    g._db = _stub_db(rows)
    return g


class TestSessionsGraphCore:
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

    def test_update_memory_returns_none_when_not_found(self):
        g = _graph(rows=[])
        result = g.update_memory("m-1", "new content")
        assert result is None


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


class TestSessionsGraphConnector:
    @pytest.fixture()
    def context_graph_config(self, monkeypatch, tmp_path):
        """Point agent_context_graph's hook config at a temp dir so tests never
        touch a real ~/.config/context-graph/config.toml on the host running them.
        """
        pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")
        from agent_context_graph.adapters import _identity

        # The supported override (ADR 0003), not module privates: those were
        # removed when the config path became a lookup, and monkeypatching a
        # missing attribute raises -- while silencing that with raising=False
        # would point these tests at the real host config, which is exactly
        # what this fixture exists to prevent.
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "context-graph" / "config.toml"))
        _identity._reset_cache()
        yield _identity
        _identity._reset_cache()

    def _make(self):
        pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")
        from sessions_graph.connector import SessionsGraphConnector

        from agent_context_graph.events import SessionEndEvent, SessionStartEvent

        db = MagicMock()
        db.query.return_value = []

        from sessions_graph.core import SessionsGraph

        graph = SessionsGraph.__new__(SessionsGraph)
        graph._db = db

        connector = SessionsGraphConnector(graph)
        return connector, graph, db, SessionStartEvent, SessionEndEvent

    def test_session_start_merges_user_and_session_nodes(self):
        connector, _graph, db, SessionStartEvent, _ = self._make()

        event = SessionStartEvent(session_id="s-1", user_id="alice")
        connector.on_event(event)

        assert connector.active_user_id == "alice"
        assert connector.active_session_id == "s-1"
        assert db.query.call_count == 1  # single combined MERGE wiring User-[:HAD_SESSION]->Session

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

    def test_session_end_always_spawns_embedding_but_not_reconciliation_by_default(self):
        connector, _graph, _db, SessionStartEvent, SessionEndEvent = self._make()

        with patch("sessions_graph.connector.subprocess.Popen") as mock_popen:
            connector.on_event(SessionStartEvent(session_id="s-1", user_id="alice"))
            connector.on_event(SessionEndEvent(session_id="s-1"))

        # call_count and the commands only, never the whole call: its args
        # include env=dict(os.environ), real secrets and all -- which is
        # exactly how rendering it in a failure message got flagged in review.
        assert mock_popen.call_count == 1
        assert mock_popen.call_args.args[0][-3:] == ["embed", "--session", "s-1"]
        assert mock_popen.call_args.kwargs["start_new_session"] is True

    def test_auto_reconcile_true_spawns_detached_process(self, context_graph_config):
        from sessions_graph.connector import SessionsGraphConnector

        context_graph_config.write_full_config(
            memgraph_url="bolt://remote:7687",
            memgraph_user="admin",
            memgraph_password="secret",
            memgraph_database="mydb",
            openai_api_key="sk-test",
        )
        context_graph_config._reset_cache()

        _connector, graph, _db, SessionStartEvent, SessionEndEvent = self._make()
        connector = SessionsGraphConnector(graph, auto_reconcile=True)

        with patch("sessions_graph.connector.subprocess.Popen") as mock_popen:
            connector.on_event(SessionStartEvent(session_id="s-1", user_id="alice"))
            connector.on_event(SessionEndEvent(session_id="s-1"))

        assert mock_popen.call_count == 2
        embed, reconcile = mock_popen.call_args_list
        assert embed.args[0][-3:] == ["embed", "--session", "s-1"]
        assert reconcile.args[0][-3:] == ["reconcile", "--session", "s-1"]
        assert reconcile.kwargs["start_new_session"] is True
        env = reconcile.kwargs["env"]
        assert env["MEMGRAPH_URL"] == "bolt://remote:7687"
        assert env["MEMGRAPH_USER"] == "admin"
        assert env["MEMGRAPH_PASSWORD"] == "secret"
        assert env["MEMGRAPH_DATABASE"] == "mydb"
        assert env["OPENAI_API_KEY"] == "sk-test"

    def test_supports_session_events_only(self):
        connector, _, _, SessionStartEvent, SessionEndEvent = self._make()
        from agent_context_graph.events import ToolStartEvent

        assert connector.supports(SessionStartEvent(session_id="s-1"))
        assert connector.supports(SessionEndEvent(session_id="s-1"))
        assert not connector.supports(ToolStartEvent(session_id="s-1", tool_name="read_file"))


class TestReconciliationEnv:
    """Unit tests for connector._child_env(), the fix for the detached
    ``sessions-graph`` subprocesses never seeing resolved Memgraph/LLM config.
    """

    @pytest.fixture()
    def context_graph_config(self, monkeypatch, tmp_path):
        pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")
        from agent_context_graph.adapters import _identity

        # The supported override (ADR 0003), not module privates: those were
        # removed when the config path became a lookup, and monkeypatching a
        # missing attribute raises -- while silencing that with raising=False
        # would point these tests at the real host config, which is exactly
        # what this fixture exists to prevent.
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "context-graph" / "config.toml"))
        _identity._reset_cache()
        yield _identity
        _identity._reset_cache()

    def test_merges_config_onto_ambient_env(self, context_graph_config, monkeypatch):
        from sessions_graph.connector import _child_env

        monkeypatch.setenv("PATH", "/usr/bin")
        context_graph_config.write_full_config(memgraph_url="bolt://remote:7687", openai_api_key="sk-test")
        context_graph_config._reset_cache()

        env = _child_env(llm=True)

        assert env["PATH"] == "/usr/bin"  # ambient env preserved
        assert env["MEMGRAPH_URL"] == "bolt://remote:7687"
        assert env["OPENAI_API_KEY"] == "sk-test"

    def test_empty_config_clears_ambient_credentials(self, context_graph_config, monkeypatch):
        from sessions_graph.connector import _child_env

        monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
        monkeypatch.setenv("MEMGRAPH_USER", "ambient-user")
        monkeypatch.setenv("MEMGRAPH_PASSWORD", "ambient-password")
        monkeypatch.setenv("MEMGRAPH_URI", "bolt://ambient:7687")
        monkeypatch.setenv("MEMGRAPH_USERNAME", "ambient-user")

        env = _child_env(llm=True)

        assert env["OPENAI_API_KEY"] == ""
        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["MEMGRAPH_USER"] == ""
        assert env["MEMGRAPH_PASSWORD"] == ""
        assert "MEMGRAPH_URI" not in env
        assert "MEMGRAPH_USERNAME" not in env

    def test_embedding_env_gets_memgraph_but_not_configured_llm_keys(self, context_graph_config, monkeypatch):
        """Embedding runs inside Memgraph; the child needs the connection, not the keys."""
        from sessions_graph.connector import _child_env

        monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
        context_graph_config.write_full_config(memgraph_url="bolt://remote:7687", openai_api_key="sk-test")
        context_graph_config._reset_cache()

        env = _child_env(llm=False)

        assert env["MEMGRAPH_URL"] == "bolt://remote:7687"
        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env
