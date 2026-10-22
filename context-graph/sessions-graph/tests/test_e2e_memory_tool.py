"""The ``memory`` tool against a real Memgraph: on only for opted-in users, identity from the config."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

pytest.importorskip("memgraph_context_graph")

from sessions_graph.tool import MemoryTool

from memgraph_context_graph.adapters._identity import HookConfig
from memgraph_context_graph.tools import ToolError, available_tools, load_tools


@pytest.fixture
def config(memgraph):
    """A HookConfig for an opted-in user, pointing at the test Memgraph as the config file would."""
    return HookConfig(
        user_id="alice",
        memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        memgraph_user=os.environ.get("MEMGRAPH_USER", ""),
        memgraph_password=os.environ.get("MEMGRAPH_PASSWORD", ""),
        memgraph_database=os.environ.get("MEMGRAPH_DATABASE", "memgraph"),
        memory_backend="context-graph",
    )


def test_registered_but_available_only_after_opting_in(config):
    assert "memory" in load_tools()
    assert "memory" in available_tools(config)
    assert "memory" not in available_tools(replace(config, memory_backend=None))
    assert "memory" not in available_tools(replace(config, memory_backend="native"))
    assert "recall" in available_tools(replace(config, memory_backend=None))


def test_doctor_names_the_memory_tool_only_once_it_is_the_memory(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from memgraph_context_graph.adapters import _identity
    from memgraph_context_graph.cli import _check_mcp

    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity.write_config(user_id="alice")
    _identity._reset_cache()
    assert _check_mcp()["detail"] == "serves recall"

    _identity.write_config(memory_backend="context-graph")
    _identity._reset_cache()
    assert _check_mcp()["detail"] == "serves memory, recall"
    _identity._reset_cache()


def test_writes_land_in_the_configured_users_memory(config, graph):
    tool = MemoryTool()

    result = tool.call({"command": "create", "path": "/memories/prefs.md", "file_text": "Prefers uv"}, config)

    assert result.text == "File created successfully at: /memories/prefs.md"
    assert [f.content for f in graph.memory_store("alice").files()] == ["Prefers uv"]
    assert graph.memory_store("bob").files() == []


def test_command_errors_reach_the_model_as_tool_errors(config):
    with pytest.raises(ToolError, match="does not exist"):
        MemoryTool().call({"command": "view", "path": "/memories/nope.md"}, config)
    with pytest.raises(ToolError, match="must start with /memories"):
        MemoryTool().call({"command": "view", "path": "/etc/passwd"}, config)


def test_no_user_configured(config):
    with pytest.raises(ToolError, match="No user is configured"):
        MemoryTool().call({"command": "view", "path": "/memories"}, replace(config, user_id=None))


def test_community_has_no_shared_spaces(config):
    tool = MemoryTool()

    with pytest.raises(ToolError, match="needs the Enterprise edition"):
        tool.call({"command": "create", "path": "/memories/spaces/team/a.md", "file_text": "x"}, config)
    assert "No shared spaces" in tool.call({"command": "view", "path": "/memories/spaces"}, config).text
    index = tool.session_context(config, {})
    assert index is not None
    assert "Shared space" not in index
