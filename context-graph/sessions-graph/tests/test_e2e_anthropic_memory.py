"""MemgraphMemoryTool through the Anthropic SDK's own tool-call entry point, against a real Memgraph.

The model is not called: ``call()`` is what the SDK's tool runner invokes
with a ``tool_use`` block's input, so these exercise the same path a live
conversation takes, minus the API.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("anthropic.lib.tools")

from sessions_graph.anthropic_memory import AsyncMemgraphMemoryTool, MemgraphMemoryTool


def test_definition_is_the_api_memory_tool(graph):
    assert MemgraphMemoryTool(graph, "alice").to_dict() == {"type": "memory_20250818", "name": "memory"}


def test_tool_calls_read_and_write_the_users_graph_memory(graph):
    tool = MemgraphMemoryTool(graph, "alice", session_id="app-1")

    created = tool.call({"command": "create", "path": "/memories/prefs.md", "file_text": "Prefers uv"})
    viewed = tool.call({"command": "view", "path": "/memories/prefs.md"})

    assert created == "File created successfully at: /memories/prefs.md"
    assert "     1\tPrefers uv" in viewed
    assert [f.content for f in graph.memory_store("alice").files()] == ["Prefers uv"]
    assert graph.memory_store("bob").files() == []


def test_errors_come_back_as_tool_errors(graph):
    from sessions_graph.anthropic_memory import ToolError

    tool = MemgraphMemoryTool(graph, "alice")
    with pytest.raises(ToolError, match="does not exist"):
        tool.call({"command": "view", "path": "/memories/missing.md"})


def test_clear_all_memory(graph):
    tool = MemgraphMemoryTool(graph, "alice")
    tool.call({"command": "create", "path": "/memories/a.md", "file_text": "a"})
    tool.call({"command": "create", "path": "/memories/dir/b.md", "file_text": "b"})

    assert tool.clear_all_memory() == "All memory cleared"
    assert graph.memory_store("alice").files() == []


async def test_async_tool(graph):
    tool = AsyncMemgraphMemoryTool(graph, "alice")

    await tool.call({"command": "create", "path": "/memories/a.md", "file_text": "x"})
    renamed = await tool.call({"command": "rename", "old_path": "/memories/a.md", "new_path": "/memories/b.md"})

    assert renamed == "Successfully renamed /memories/a.md to /memories/b.md"
    assert [f.path for f in graph.memory_store("alice").files()] == ["/memories/b.md"]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_saves_a_memory_through_the_tool_runner(graph):
    """A live conversation: Claude decides on its own to write to the graph-backed memory."""
    from anthropic import Anthropic

    runner = Anthropic().beta.messages.tool_runner(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=[MemgraphMemoryTool(graph, "alice")],
        messages=[{"role": "user", "content": "Remember for future conversations: I prefer uv over pip."}],
    )
    runner.until_done()

    assert any("uv" in file.content for file in graph.memory_store("alice").files())
