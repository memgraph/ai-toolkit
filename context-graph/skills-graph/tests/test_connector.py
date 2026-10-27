from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")

from memgraph_context_graph.events import MessageEvent, ToolEndEvent, ToolStartEvent
from skills_graph.connector import SkillGraphConnector

if TYPE_CHECKING:
    from skills_graph import SkillGraph


def _connector():
    graph = SimpleNamespace(record_skill_usage=MagicMock())
    return SkillGraphConnector(cast("SkillGraph", graph)), graph


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
    assert params.get("create_missing", False) is False


def test_tool_call_inside_subagent_attaches_usage_to_agent_container():
    connector, graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="get_skill",
        tool_input={"name": "cypher-basics"},
        timestamp="2026-04-30T00:00:00+00:00",
        agent_name="agent-1",
    )

    connector.on_event(event)

    params = graph.record_skill_usage.call_args.kwargs
    assert params["container_agent_id"] == "agent-1"


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


def _write_skill(root, name, description="Does a thing"):
    skill_file = root / name / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n", encoding="utf-8")
    return skill_file


def test_claude_code_skill_tool_records_usage_from_the_installed_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skill_file = _write_skill(tmp_path / "home" / ".agents" / "skills", "grilling", "Grill the user")
    connector, graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="Skill",
        tool_input={"skill": "grilling"},
        metadata={"cwd": str(tmp_path / "project")},
    )

    assert connector.supports(event)
    connector.on_event(event)

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "grilling"
    assert params["action"] == "invoke_skill"
    assert params["create_missing"] is True
    assert params["description"] == "Grill the user"
    assert params["source_path"] == str(skill_file)


def test_claude_code_skill_tool_prefers_the_project_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / "home" / ".claude" / "skills", "release")
    project_skill = _write_skill(tmp_path / "project" / ".claude" / "skills", "release")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1",
            tool_name="Skill",
            tool_input={"skill": "release"},
            metadata={"cwd": str(tmp_path / "project")},
        )
    )

    assert graph.record_skill_usage.call_args.kwargs["source_path"] == str(project_skill)


def test_plugin_qualified_skill_reduces_to_its_name_and_resolves_from_the_plugin_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cache = tmp_path / "home" / ".claude" / "plugins" / "cache" / "market" / "context-graph" / "0.11.0" / "skills"
    skill_file = _write_skill(cache, "setup", "Set up capture")
    connector, graph = _connector()

    connector.on_event(ToolStartEvent(session_id="s1", tool_name="Skill", tool_input={"skill": "context-graph:setup"}))

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "setup"
    assert params["source_path"] == str(skill_file)


def test_claude_code_skill_tool_records_usage_when_no_skill_file_is_found(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    connector, graph = _connector()

    connector.on_event(ToolStartEvent(session_id="s1", tool_name="Skill", tool_input={"skill": "caveman"}))

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "caveman"
    assert params["create_missing"] is True
    assert params["source_path"] is None


def test_claude_code_skill_tool_result_is_not_a_second_usage():
    connector, _graph = _connector()
    event = ToolEndEvent(
        session_id="s1",
        tool_name="Skill",
        result={"success": True, "commandName": "grilling"},
        metadata={"tool_input": {"skill": "grilling"}},
    )

    assert not connector.supports(event)


def test_read_tool_records_skill_file_read(tmp_path):
    skill_file = _write_skill(tmp_path / "skills", "caveman", "Ultra-compressed communication mode.")
    connector, graph = _connector()

    connector.on_event(ToolStartEvent(session_id="s1", tool_name="Read", tool_input={"file_path": str(skill_file)}))

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "caveman"
    assert params["action"] == "read_skill_file"
    assert params["description"] == "Ultra-compressed communication mode."
    assert "# caveman" in params["content"]


def test_relative_skill_path_resolves_against_the_session_cwd(tmp_path):
    skill_file = _write_skill(tmp_path / "skills", "release")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1",
            tool_name="Bash",
            tool_input={"command": "cat skills/release/SKILL.md"},
            metadata={"cwd": str(tmp_path)},
        )
    )

    assert graph.record_skill_usage.call_args.kwargs["source_path"] == str(skill_file)


def test_every_skill_file_read_in_one_command_is_recorded(tmp_path):
    first = _write_skill(tmp_path, "first")
    second = _write_skill(tmp_path, "second")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(session_id="s1", tool_name="Bash", tool_input={"command": f"cat {first}; head -50 {second}"})
    )

    assert [call.kwargs["skill_name"] for call in graph.record_skill_usage.call_args_list] == ["first", "second"]


def test_codex_argv_command_reading_a_skill_file_is_recorded(tmp_path):
    skill_file = _write_skill(tmp_path, "diagnose")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1", tool_name="shell", tool_input={"command": ["bash", "-lc", f"sed -n '1,80p' {skill_file}"]}
        )
    )

    assert graph.record_skill_usage.call_args.kwargs["skill_name"] == "diagnose"


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Bash", {"command": "git add skills/release/SKILL.md"}),
        ("Bash", {"command": "git diff -- skills/release/SKILL.md"}),
        ("Bash", {"command": "rg -n TODO skills/release/SKILL.md"}),
        ("Bash", {"command": "ls ../plugins/*/skills/*/SKILL.md"}),
        ("Bash", {"command": 'for D in skills/*; do cat "$D/SKILL.md"; done'}),
        ("Bash", {"command": "sed -i 's/a/b/' skills/release/SKILL.md"}),
        ("Bash", {"command": "cat notes.md > skills/release/SKILL.md"}),
        ("Write", {"file_path": "/repo/skills/release/SKILL.md", "content": "---"}),
        ("Edit", {"file_path": "/repo/skills/release/SKILL.md", "old_string": "a", "new_string": "b"}),
        ("Agent", {"prompt": "read skills/release/SKILL.md and summarize it", "subagent_type": "Explore"}),
    ],
)
def test_a_skill_file_that_is_not_read_records_nothing(tool_name, tool_input):
    connector, graph = _connector()
    event = ToolStartEvent(session_id="s1", tool_name=tool_name, tool_input=tool_input)

    assert not connector.supports(event)
    connector.on_event(event)
    graph.record_skill_usage.assert_not_called()


def test_a_path_in_metadata_alone_is_not_a_read():
    connector, _graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="Agent",
        tool_input={"prompt": "summarize the release skill"},
        metadata={"tool_input": {"prompt": "cat skills/release/SKILL.md"}},
    )

    assert not connector.supports(event)


def test_reading_a_memory_file_with_name_and_description_records_nothing(tmp_path):
    memory_file = tmp_path / "memory" / "feedback.md"
    memory_file.parent.mkdir()
    memory_file.write_text("---\nname: feedback\ndescription: Prefer small PRs\n---\nBody\n", encoding="utf-8")
    connector, _graph = _connector()

    assert not connector.supports(
        ToolStartEvent(session_id="s1", tool_name="Read", tool_input={"file_path": str(memory_file)})
    )
    assert not connector.supports(
        ToolEndEvent(
            session_id="s1",
            tool_name="Read",
            result=memory_file.read_text(encoding="utf-8"),
            metadata={"tool_input": {"file_path": str(memory_file)}},
        )
    )


def test_injected_skill_content_with_the_expansion_marker_is_recorded():
    connector, graph = _connector()
    event = MessageEvent(
        session_id="s1",
        role="user",
        content="Base directory for this skill: /home/u/.agents/skills/caveman\n\n"
        "---\nname: caveman\ndescription: Ultra-compressed communication mode.\n---\n\n# Caveman\n",
    )

    assert connector.supports(event)
    connector.on_event(event)

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "caveman"
    assert params["action"] == "prompt_expansion"
    assert params["source_path"] == "/home/u/.agents/skills/caveman/SKILL.md"


def test_message_with_frontmatter_but_no_expansion_marker_is_ignored():
    connector, _graph = _connector()
    event = MessageEvent(
        session_id="s1", role="user", content="---\nname: feedback\ndescription: Prefer small PRs\n---\n"
    )

    assert not connector.supports(event)


def test_non_skill_file_read_is_ignored():
    connector, _graph = _connector()
    event = ToolStartEvent(
        session_id="s1",
        tool_name="exec_command",
        tool_input={"cmd": "sed -n '1,80p' README.md"},
    )

    assert not connector.supports(event)


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "Notes\ncat /repo/skills/release/SKILL.md\n"',
        "python3 - <<'PY'\ncat /repo/skills/release/SKILL.md\nPY",
        "cat <<EOF > notes.md\ncat /repo/skills/release/SKILL.md\nEOF",
    ],
)
def test_skill_file_read_inside_quoted_text_or_a_heredoc_records_nothing(command):
    connector, _graph = _connector()

    assert not connector.supports(ToolStartEvent(session_id="s1", tool_name="Bash", tool_input={"command": command}))


def test_command_after_a_heredoc_is_still_read(tmp_path):
    skill_file = _write_skill(tmp_path, "release")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1",
            tool_name="Bash",
            tool_input={"command": f"python3 - <<'PY'\nprint(1)\nPY\ncat {skill_file}"},
        )
    )

    assert graph.record_skill_usage.call_args.kwargs["source_path"] == str(skill_file)


def test_relative_read_resolves_against_the_codex_workdir(tmp_path):
    skill_file = _write_skill(tmp_path / "repo" / "skills", "release")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1",
            tool_name="exec_command",
            tool_input={"cmd": "cat skills/release/SKILL.md", "workdir": str(tmp_path / "repo")},
            metadata={"cwd": str(tmp_path)},
        )
    )

    params = graph.record_skill_usage.call_args.kwargs
    assert params["source_path"] == str(skill_file)
    assert params["description"] == "Does a thing"


def test_relative_read_follows_cd_in_the_command(tmp_path):
    skill_file = _write_skill(tmp_path / "repo" / "skills", "release")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1",
            tool_name="Bash",
            tool_input={"command": "cd repo && cat skills/release/SKILL.md"},
            metadata={"cwd": str(tmp_path)},
        )
    )

    assert graph.record_skill_usage.call_args.kwargs["source_path"] == str(skill_file)


def test_bare_skill_file_read_resolves_against_the_working_directory(tmp_path):
    skill_file = _write_skill(tmp_path, "release")
    connector, graph = _connector()

    connector.on_event(
        ToolStartEvent(
            session_id="s1",
            tool_name="Bash",
            tool_input={"command": "cat SKILL.md"},
            metadata={"cwd": str(skill_file.parent)},
        )
    )

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "release"
    assert params["source_path"] == str(skill_file)


def test_claude_code_skill_tool_keeps_the_invoked_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skill_file = tmp_path / "home" / ".claude" / "skills" / "release" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: other-name\ndescription: Ship it\n---\n", encoding="utf-8")
    connector, graph = _connector()

    connector.on_event(ToolStartEvent(session_id="s1", tool_name="Skill", tool_input={"skill": "release"}))

    params = graph.record_skill_usage.call_args.kwargs
    assert params["skill_name"] == "release"
    assert params["description"] == "Ship it"
