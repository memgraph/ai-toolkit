"""Forgetting a session's Actions against a real Memgraph: its own Agents go, one another session shares stays."""

import contextlib

import pytest

from actions_graph import ActionsGraph
from actions_graph.connector import ActionsGraphConnector
from memgraph_context_graph import AgentLink
from memgraph_context_graph import session as session_node
from memgraph_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter


@pytest.fixture
def graph():
    g = ActionsGraph()
    with contextlib.suppress(Exception):
        g.setup()
    g.clear()
    g._db.query("MATCH (f:ForgottenSession) DELETE f")
    yield g
    g.clear()
    g._db.query("MATCH (f:ForgottenSession) DELETE f")


def _call(graph, session_id, tool_use_id, agent_id=None):
    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": session_id,
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
        "tool_input": {"command": "ls"},
        "cwd": "/repo",
    }
    if agent_id:
        payload |= {"agent_id": agent_id, "agent_type": "Explore"}
    ClaudeCodeHooksAdapter(link).handle_payload(payload)


def _count(graph, query):
    return graph._db.query(query)[0]["n"]


def test_forget_deletes_the_sessions_actions_and_its_own_agents(graph):
    _call(graph, "s-1", "direct")
    _call(graph, "s-1", "nested", agent_id="agent-own")
    _call(graph, "s-2", "other")

    assert sorted(graph.owned_action_ids("s-1")) == sorted(
        row["id"]
        for row in graph._db.query(
            "MATCH (a:Action) WHERE a.tool_use_id IN ['direct', 'nested'] RETURN a.action_id AS id"
        )
    )
    removed = graph.forget_session("s-1")

    assert removed == {"actions": 2, "agents": 1, "shared_agents": 0}
    assert _count(graph, "MATCH (a:Agent) RETURN count(a) AS n") == 0
    assert graph._db.query("MATCH (a:Action) RETURN a.tool_use_id AS id") == [{"id": "other"}]


def test_an_agent_another_session_shares_stays_with_its_actions(graph):
    _call(graph, "s-1", "one", agent_id="shared")
    _call(graph, "s-2", "two", agent_id="shared")

    assert graph.owned_action_ids("s-1") == []
    removed = graph.forget_session("s-1")

    assert removed == {"actions": 0, "agents": 0, "shared_agents": 1}
    assert _count(graph, "MATCH (:Agent {agent_id: 'shared'})-[:HAS_ACTION]->(a) RETURN count(a) AS n") == 2


def test_a_forgotten_session_gets_no_new_actions_or_agents(graph):
    _call(graph, "s-1", "before")
    session_node.mark_forgotten(graph._db, "s-1")
    graph.forget_session("s-1")
    session_node.delete(graph._db, "s-1")

    _call(graph, "s-1", "after")
    _call(graph, "s-1", "after-nested", agent_id="late-agent")

    assert _count(graph, "MATCH (n) WHERE n:Session OR n:Action OR n:Agent RETURN count(n) AS n") == 0
