"""Unit tests for Sessions Graph models, core, and connector.

These tests use an in-memory stub for the Memgraph client so they run
without a live database.
"""

from __future__ import annotations

import os
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

    g = SessionsGraph(_stub_db(rows))
    return g


class TestSessionsGraphCore:
    def test_save_memory_rejects_empty_content(self):
        g = _graph()
        with pytest.raises(MemoryValidationError):
            g.save_memory("alice", "")

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


@pytest.fixture()
def context_graph_config():
    """The config module, already pointed at a temp file by conftest's autouse fixture."""
    pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")
    from memgraph_context_graph.adapters import _identity

    return _identity


class TestSessionsGraphConnector:
    def _make(self):
        pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")
        from sessions_graph.connector import SessionsGraphConnector

        from memgraph_context_graph.events import SessionEndEvent, SessionStartEvent

        db = MagicMock()
        db.query.return_value = []

        from sessions_graph.core import SessionsGraph

        graph = SessionsGraph(db)

        connector = SessionsGraphConnector(graph)
        return connector, graph, db, SessionStartEvent, SessionEndEvent

    def test_session_start_merges_user_and_session_nodes(self, graph, memgraph):
        pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")
        from sessions_graph.connector import SessionsGraphConnector

        from memgraph_context_graph.events import SessionStartEvent

        connector = SessionsGraphConnector(graph)
        connector.on_event(SessionStartEvent(session_id="s-1", user_id="alice"))

        assert connector.active_user_id == "alice"
        assert connector.active_session_id == "s-1"
        rows = memgraph.query("MATCH (u:User)-[:HAD_SESSION]->(s:Session) RETURN u.user_id AS user, s.session_id AS id")
        assert rows == [{"user": "alice", "id": "s-1"}]

    def test_session_start_without_user_id_only_merges_session(self, graph, memgraph):
        pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")
        from sessions_graph.connector import SessionsGraphConnector

        from memgraph_context_graph.events import SessionStartEvent

        connector = SessionsGraphConnector(graph)
        connector.on_event(SessionStartEvent(session_id="s-1"))

        assert connector.active_user_id is None
        assert connector.active_session_id == "s-1"
        assert memgraph.query("MATCH (s:Session) RETURN s.session_id AS id") == [{"id": "s-1"}]
        assert memgraph.query("MATCH (u:User) RETURN count(u) AS c") == [{"c": 0}]

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

        context_graph_config.write_config(
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
        # The child reads the file it is handed; no credential travels in its environment.
        assert env[context_graph_config.CONFIG_PATH_ENV] == str(context_graph_config.config_file())
        assert "secret" not in env.values() and "sk-test" not in env.values()

    def test_supports_session_events_only(self):
        connector, _, _, SessionStartEvent, SessionEndEvent = self._make()
        from memgraph_context_graph.events import ToolStartEvent

        assert connector.supports(SessionStartEvent(session_id="s-1"))
        assert connector.supports(SessionEndEvent(session_id="s-1"))
        assert not connector.supports(ToolStartEvent(session_id="s-1", tool_name="read_file"))


class TestReconciliationEnv:
    """connector._child_env(), then the child's own fill: what a detached ``sessions-graph`` run connects with."""

    def _child_fill(self, env, monkeypatch):
        """Run the child's side of the hand-off: start from *env*, fill from the file it names."""
        from sessions_graph.cli import _fill_env_from_context_graph_config

        for key in list(os.environ):
            monkeypatch.delenv(key)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _fill_env_from_context_graph_config()
        return dict(os.environ)

    def test_hands_down_the_config_file_not_its_values(self, context_graph_config, monkeypatch):
        from sessions_graph.connector import _child_env

        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("MEMGRAPH_URL", "bolt://ambient:7687")
        context_graph_config.write_config(memgraph_url="bolt://remote:7687", openai_api_key="sk-test")
        context_graph_config._reset_cache()

        env = _child_env()

        assert env["PATH"] == "/usr/bin"  # ambient env preserved
        assert env[context_graph_config.CONFIG_PATH_ENV] == str(context_graph_config.config_file())
        assert "MEMGRAPH_URL" not in env and "OPENAI_API_KEY" not in env
        context_graph_config._reset_cache()
        child = self._child_fill(env, monkeypatch)
        assert child["MEMGRAPH_URL"] == "bolt://remote:7687"
        assert child["OPENAI_API_KEY"] == "sk-test"

    def test_empty_config_clears_ambient_credentials(self, context_graph_config, monkeypatch):
        from sessions_graph.connector import _child_env

        monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
        monkeypatch.setenv("MEMGRAPH_USER", "ambient-user")
        monkeypatch.setenv("MEMGRAPH_PASSWORD", "ambient-password")
        monkeypatch.setenv("MEMGRAPH_URI", "bolt://ambient:7687")
        monkeypatch.setenv("MEMGRAPH_USERNAME", "ambient-user")

        child = self._child_fill(_child_env(), monkeypatch)

        assert child["OPENAI_API_KEY"] == ""
        assert child["ANTHROPIC_API_KEY"] == ""
        assert child["MEMGRAPH_USER"] == ""
        assert child["MEMGRAPH_PASSWORD"] == ""
        assert "MEMGRAPH_URI" not in child
        assert "MEMGRAPH_USERNAME" not in child

    def test_a_hand_run_keeps_what_its_environment_sets(self, context_graph_config, monkeypatch):
        """``MEMGRAPH_URL=... sessions-graph reconcile`` (dev-memgraph.sh reconcile) targets that instance."""
        from sessions_graph.cli import _fill_env_from_context_graph_config

        context_graph_config.write_config(memgraph_url="bolt://remote:7687", openai_api_key="sk-test")
        context_graph_config._reset_cache()
        monkeypatch.setenv("MEMGRAPH_URL", "bolt://localhost:7690")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        _fill_env_from_context_graph_config()

        assert os.environ["MEMGRAPH_URL"] == "bolt://localhost:7690"
        assert os.environ["OPENAI_API_KEY"] == "sk-test"


class TestConnectorRegistration:
    """The CONNECTOR spec the hub finds through the memgraph_context_graph.connectors entry point."""

    def _spec(self):
        pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")
        from sessions_graph.connector import CONNECTOR

        return CONNECTOR

    def test_is_registered_under_its_distribution_name(self):
        from memgraph_context_graph.connectors import load_connectors

        assert load_connectors()["sessions-graph"] is self._spec()

    @pytest.mark.parametrize("configured", [None, False, True])
    def test_builds_the_connector_with_the_configured_auto_reconcile(self, configured):
        from memgraph_context_graph.adapters._identity import HookConfig

        connector = self._spec().build(_graph(), HookConfig(auto_reconcile=configured))

        assert connector._auto_reconcile is bool(configured)

    def test_doctor_fails_without_a_user_and_says_how_to_set_one(self, monkeypatch):
        from memgraph_context_graph.adapters import _identity

        monkeypatch.setattr(_identity, "resolve_user_id", lambda payload: None)
        ok, detail = self._spec().check(_identity.HookConfig())

        assert not ok
        assert "config set identity.user_id" in detail
