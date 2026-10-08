"""Connections end to end: REFERENCES/CLOSES between stored Resources, and Touches to the Actions behind them.

Uses a real ActionsGraph (and its real connector) for the Touch -> Action join, never a fake.
"""

from __future__ import annotations

import pytest

from actions_graph import ActionsGraph
from actions_graph.connector import ActionsGraphConnector
from agent_context_graph.events import AgentStartEvent, SessionStartEvent, ToolStartEvent
from resources_graph.sweep import sweep


def view(emit, number, tool_use_id, *, session_id="s1", agent_name=None):
    emit(
        ToolStartEvent(
            session_id=session_id,
            tool_name="Bash",
            tool_input={"command": f"gh issue view {number} -R memgraph/memgraph"},
            tool_use_id=tool_use_id,
            agent_name=agent_name,
        )
    )


def edges(graph, relationship):
    rows = graph._db.query(
        f"MATCH (a:Resource)-[:{relationship}]->(b:Resource) RETURN a.number AS a, b.number AS b ORDER BY a, b"
    )
    return [(row["a"], row["b"]) for row in rows]


@pytest.mark.parametrize("order", [(4933, 4887), (4887, 4933)])
def test_references_and_closes_appear_whichever_end_is_stored_first(graph, harness, source, order):
    harness(SessionStartEvent(session_id="s1", user_id="ante"))
    first, second = order
    view(harness, first, "toolu_1")
    sweep(graph, source)
    assert edges(graph, "REFERENCES") == edges(graph, "CLOSES") == []  # the other end isn't stored: no edge, no stub
    assert graph._db.query("MATCH (r:Resource) WHERE r.number IS NOT NULL RETURN count(r) AS n")[0]["n"] == 1

    view(harness, second, "toolu_2")
    sweep(graph, source)

    assert (4933, 4887) in edges(graph, "REFERENCES")
    assert edges(graph, "CLOSES") == [(4933, 4887)]


def test_references_to_resources_not_in_memory_are_kept_as_urls(graph, harness, source):
    harness(SessionStartEvent(session_id="s1", user_id="ante"))
    view(harness, 4887, "toolu_1")
    sweep(graph, source)

    urls = graph._db.query("MATCH (i:Issue {number: 4887}) RETURN i.referenced_by_urls AS urls")[0]["urls"]
    assert "https://github.com/memgraph/memgraph/pull/4933" in urls


@pytest.fixture()
def with_actions(graph, harness):
    """The same hook run with actions-graph enabled too, as --connector actions-graph would."""
    actions = ActionsGraph()
    actions.setup()
    connector = ActionsGraphConnector(actions)

    def emit(event):
        if connector.supports(event):
            connector.on_event(event)
        harness(event)

    return emit


def test_touches_link_to_the_tool_call_and_the_subagent_that_made_them(graph, with_actions, source):
    with_actions(SessionStartEvent(session_id="s1", user_id="ante"))
    view(with_actions, 4887, "toolu_main")
    with_actions(AgentStartEvent(session_id="s1", agent_name="agent-7", agent_type="Explore"))
    view(with_actions, 4933, "toolu_sub", agent_name="agent-7")

    report = sweep(graph, source)

    caused = graph._db.query(
        "MATCH (t:Touch)-[:CAUSED_BY]->(a:ToolCall) RETURN t.address AS address, a.tool_use_id AS tool_use_id "
        "ORDER BY tool_use_id"
    )
    assert [(row["address"], row["tool_use_id"]) for row in caused] == [
        ("github:memgraph/memgraph#4887", "toolu_main"),
        ("github:memgraph/memgraph#4933", "toolu_sub"),
    ]
    by_agent = graph._db.query(
        "MATCH (t:Touch)-[:BY_AGENT]->(a:Agent) RETURN t.address AS address, a.agent_id AS agent"
    )
    assert [(row["address"], row["agent"]) for row in by_agent] == [("github:memgraph/memgraph#4933", "agent-7")]
    assert report.linked == 3

    assert sweep(graph, source).linked == 0  # links are drawn once


def test_touches_without_recorded_actions_stay_unlinked(graph, harness, source):
    harness(SessionStartEvent(session_id="s1", user_id="ante"))
    view(harness, 4887, "toolu_1")

    report = sweep(graph, source)

    assert report.linked == 0
    assert graph._db.query("MATCH (t:Touch {status: 'resolved'}) RETURN count(t) AS n")[0]["n"] == 1
