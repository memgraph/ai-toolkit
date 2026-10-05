"""The ``recall`` tool against a real Memgraph: identity and widths from the config, never from the call."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("agent_context_graph")
pytest.importorskip("actions_graph")

from sessions_graph.embeddings import EmbeddingUnavailableError, check_available
from sessions_graph.tool import RecallTool

from actions_graph import MessageRole, Session
from agent_context_graph.adapters._identity import HookConfig
from agent_context_graph.tools import ToolError, load_tools


@pytest.fixture
def test_config(memgraph):
    """A HookConfig pointing at the test Memgraph, as the config file would."""

    def make(user_id, **overrides):
        return HookConfig(
            user_id=user_id,
            memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
            memgraph_user=os.environ.get("MEMGRAPH_USER", ""),
            memgraph_password=os.environ.get("MEMGRAPH_PASSWORD", ""),
            memgraph_database=os.environ.get("MEMGRAPH_DATABASE", "memgraph"),
            **overrides,
        )

    return make


@pytest.fixture
def remembered(graph, memgraph, actions_graph):
    """Two users who each said where staging lives, embedded the way session end does it."""
    try:
        check_available(memgraph)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"Memgraph can't embed (needs MAGE): {exc}")
    for user, cluster in (("u1", "kestrel"), ("u2", "osprey")):
        session = f"{user}-s"
        actions_graph.ensure_session(Session(session_id=session, started_at="2026-09-01T10:00:00+00:00"))
        memgraph.query(
            "MERGE (u:User {user_id: $u}) WITH u MATCH (s:Session {session_id: $s}) MERGE (u)-[:HAD_SESSION]->(s)",
            {"u": user, "s": session},
        )
        actions_graph.record_message(
            session_id=session,
            role=MessageRole.USER,
            content=f"Our staging cluster is called {cluster}.",
            timestamp="2026-09-01T10:00:00+00:00",
        )
        graph.embed_session(session)


def test_recall_reads_the_configured_users_memory(remembered, test_config):
    result = RecallTool().call({"question": "What is the staging cluster called?"}, test_config("u1"))

    assert "kestrel" in result.text
    assert "osprey" not in result.text
    assert result.text.startswith("Memory recall for: What is the staging cluster called?")
    assert [turn["text"] for turn in result.structured["turns"]] == ["Our staging cluster is called kestrel."]


def test_recall_takes_its_widths_from_the_config(remembered, test_config):
    result = RecallTool().call(
        {"question": "staging cluster"}, test_config("u1", recall_settings={"lanes": "text", "text_k": "1"})
    )

    assert len(result.structured["turns"]) == 1


@pytest.mark.parametrize(
    ("arguments", "user", "settings", "message"),
    [
        ({"question": "  "}, "u1", {}, "needs a question"),
        ({"question": "staging"}, None, {}, "No user is configured"),
        ({"question": "staging"}, "u1", {"lanes": "telepathy"}, "unknown recall lanes"),
    ],
)
def test_recall_refuses_what_it_cannot_answer(test_config, arguments, user, settings, message):
    with pytest.raises(ToolError, match=message):
        RecallTool().call(arguments, test_config(user, recall_settings=settings))


def test_sessions_graph_registers_recall():
    assert isinstance(load_tools()["recall"], RecallTool)
