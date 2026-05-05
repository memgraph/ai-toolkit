from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("agent_context_graph", reason="agent-context-graph not installed")

from agent_context_graph.events import ToolEndEvent, ToolStartEvent
from skills_graph.connector import SkillGraphConnector


def _connector():
    graph = SimpleNamespace(record_skill_usage=MagicMock())
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

    params = graph.record_skill_usage.call_args.kwargs
    assert params["session_id"] == "s1"
    assert params["skill_name"] == "cypher-basics"
    assert params["action"] == "get_skill"
    assert params["create_missing"] is False


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

    params = [call.kwargs for call in graph.record_skill_usage.call_args_list]
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


def test_add_and_delete_skill_tools_are_supported_by_default():
    connector, _graph = _connector()

    assert connector.supports(ToolStartEvent(session_id="s1", tool_name="add_skill", tool_input={"name": "s1"}))
    assert connector.supports(ToolStartEvent(session_id="s1", tool_name="delete_skill", tool_input={"name": "s1"}))


def test_deeply_nested_results_stop_at_depth_limit():
    result = {"content": []}
    current = result["content"]
    for _ in range(20):
        nested = {"content": []}
        current.append(nested)
        current = nested["content"]
    current.append({"name": "too-deep"})

    assert SkillGraphConnector._extract_result_skill_names(result) == []


def test_result_extraction_still_reads_reasonable_nested_json():
    result = {"content": [{"text": '{"results": [{"name": "s1"}]}'}]}

    assert SkillGraphConnector._extract_result_skill_names(result) == ["s1"]


def test_exec_command_reading_skill_file_is_recorded(tmp_path):
    skill_dir = tmp_path / "skills" / "memgraph-console"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: memgraph-console
description: Use mgconsole with Memgraph
tags:
  - memgraph
  - cypher
---

# Memgraph Console
""",
        encoding="utf-8",
    )
    connector, graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="exec_command",
        tool_input={"cmd": f"sed -n '1,220p' {skill_file}"},
        timestamp="2026-04-30T00:00:00+00:00",
    )

    assert connector.supports(event)

    connector.on_event(event)

    params = graph.record_skill_usage.call_args.kwargs
    assert params["session_id"] == "s1"
    assert params["skill_name"] == "memgraph-console"
    assert params["action"] == "read_skill_file"
    assert params["create_missing"] is True
    assert params["description"] == "Use mgconsole with Memgraph"
    assert params["source_path"] == str(skill_file)
    assert params["tags"] == ["memgraph", "cypher"]


def test_non_skill_file_read_is_ignored():
    connector, _graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="exec_command",
        tool_input={"cmd": "sed -n '1,80p' README.md"},
    )

    assert not connector.supports(event)
