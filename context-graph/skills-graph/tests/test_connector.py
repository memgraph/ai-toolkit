from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")

from agent_context_graph.events import ToolEndEvent, ToolStartEvent
from skills_graph.connector import SkillGraphConnector


def _connector():
    graph = SimpleNamespace(_db=MagicMock())
    return SkillGraphConnector(graph), graph


def test_mcp_tool_name_is_treated_as_skill_tool():
    connector, graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="mcp__skills__get_skill",
        tool_input={"arguments": {"name": "cypher-basics"}},
        timestamp="2026-04-30T00:00:00+00:00",
    )

    assert connector.supports(event)

    connector.on_event(event)

    params = graph._db.query.call_args.kwargs["params"]
    assert params["session_id"] == "s1"
    assert params["skill_name"] == "cypher-basics"
    assert params["action"] == "get_skill"


def test_mcp_search_result_records_nested_json_skill_names():
    connector, graph = _connector()
    event = ToolEndEvent(
        session_id="s1",
        tool_name="mcp__skills__list_skills",
        result={"content": [{"text": '[{"name": "s1"}, {"name": "s2"}]'}]},
        timestamp="2026-04-30T00:00:00+00:00",
    )

    assert connector.supports(event)

    connector.on_event(event)

    params = [call.kwargs["params"] for call in graph._db.query.call_args_list]
    assert [param["skill_name"] for param in params] == ["s1", "s2"]
    assert {param["action"] for param in params} == {"list_skills_result"}


def test_non_skill_mcp_tool_is_ignored():
    connector, _graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="mcp__filesystem__read_file",
        tool_input={"path": "README.md"},
    )

    assert not connector.supports(event)
