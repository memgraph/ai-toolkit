"""The tool surface: the MCP server, the recall CLI, and the SessionStart hint.

The tool here is a stand-in: what recall itself returns is tested against a
real Memgraph in sessions-graph. These tests are about the plumbing every
tool goes through -- the MCP protocol, the CLI, the hook output.
"""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from agent_context_graph.adapters import _identity
from agent_context_graph.cli import main as top_level_main
from agent_context_graph.hooks.cli import main as hook_main
from agent_context_graph.link import AgentLink
from agent_context_graph.tools import ToolError, ToolResult


@dataclass
class _EchoTool:
    name: str = "recall"
    connector: str = "sessions-graph"
    description: str = "Echoes the question."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    }
    session_hint: str | None = "Call `recall` for past sessions."

    def call(self, arguments: dict[str, Any], config: _identity.HookConfig) -> ToolResult:
        if arguments["question"] == "fail":
            raise ToolError("no user configured")
        return ToolResult(
            text=f"rows for {arguments['question']} of {config.user_id}",
            structured={"question": arguments["question"], "user": config.user_id},
        )


@dataclass
class _OptInTool(_EchoTool):
    """A tool that, like ``memory``, exists only once the user opts into graph memory."""

    name: str = "memory"
    session_hint: str | None = "Use `memory`."

    def available(self, config: _identity.HookConfig) -> bool:
        return config.graph_memory


@pytest.fixture(autouse=True)
def _config(monkeypatch, tmp_path):
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(user_id="ante")
    yield
    _identity._reset_cache()


@pytest.fixture
def tools(monkeypatch):
    registered = {"recall": _EchoTool()}
    monkeypatch.setattr("agent_context_graph.tools.load_tools", lambda: registered)
    return registered


@asynccontextmanager
async def _connected(server):
    """A client session talking to ``server`` in memory, the same way under mcp 1.x and 2.x."""
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(lambda: server.run(*server_streams, server.create_initialization_options()))
        async with ClientSession(*client_streams) as client:
            await client.initialize()
            yield client
        tg.cancel_scope.cancel()


def _wire(model) -> dict:
    """A result as the harness receives it: mcp 2.x renamed the Python fields, not the JSON."""
    return model.model_dump(by_alias=True, exclude_none=True)


@pytest.mark.asyncio
async def test_mcp_lists_tools_and_returns_only_the_text(tools):
    """Text only: Claude Code and Codex both show structuredContent instead of the text when both are sent."""
    pytest.importorskip("mcp")
    from agent_context_graph.mcp_server import build_server

    async with _connected(build_server(tools)) as client:
        listed = _wire(await client.list_tools())
        result = _wire(await client.call_tool("recall", {"question": "deploy day"}))

    assert [(tool["name"], tool["inputSchema"]["required"]) for tool in listed["tools"]] == [("recall", ["question"])]
    assert not result.get("isError")
    assert result["content"] == [{"type": "text", "text": "rows for deploy day of ante"}]
    assert "structuredContent" not in result


@pytest.mark.asyncio
async def test_mcp_reports_a_tool_error_to_the_model(tools):
    pytest.importorskip("mcp")
    from agent_context_graph.mcp_server import build_server

    async with _connected(build_server(tools)) as client:
        result = _wire(await client.call_tool("recall", {"question": "fail"}))

    assert result["isError"]
    assert result["content"] == [{"type": "text", "text": "no user configured"}]


def test_recall_cli_prints_the_text_or_the_json(tools, capsys):
    assert top_level_main(["recall", "deploy", "day"]) == 0
    assert capsys.readouterr().out.strip() == "rows for deploy day of ante"

    assert top_level_main(["recall", "--json", "deploy day"]) == 0
    assert json.loads(capsys.readouterr().out) == {"question": "deploy day", "user": "ante"}


def test_recall_cli_without_a_recall_tool_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.tools.load_tools", dict)

    assert top_level_main(["recall", "anything"]) == 1
    assert "sessions-graph" in capsys.readouterr().err


def test_session_start_adds_the_hint_of_each_enabled_connectors_tool(tools, monkeypatch, capsys):
    payload = '{"hook_event_name":"SessionStart","session_id":"s1","source":"startup"}'

    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert hook_main(["run", "claude-code"]) == 0
    assert capsys.readouterr().out.strip() == ""

    monkeypatch.setattr("agent_context_graph.hooks.runner.create_link", lambda *args, **kwargs: AgentLink())
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert hook_main(["run", "codex", "--connector", "sessions-graph"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "Call `recall` for past sessions."}
    }


def test_the_hint_never_fails_the_hook(monkeypatch, capsys):
    def broken() -> dict:
        raise ImportError("a tool's package is half-installed")

    monkeypatch.setattr("agent_context_graph.tools.load_tools", broken)
    monkeypatch.setattr("agent_context_graph.hooks.runner.create_link", lambda *args, **kwargs: AgentLink())
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"SessionStart","session_id":"s1"}'))

    assert hook_main(["run", "claude-code", "--connector", "sessions-graph", "--strict"]) == 0
    assert capsys.readouterr().out.strip() == ""


def test_recall_settings_survive_config_set(tmp_path):
    """Widths are edited into [recall] by hand; `config set` and bootstrap rewrite the file and must keep them."""
    path = _identity.config_file()
    path.write_text(path.read_text() + '\n[recall]\nturns_k = "12"\nlanes = "turns,text"\n')
    _identity._reset_cache()

    _identity.write_config(embedding_model="BAAI/bge-small-en-v1.5")
    _identity.write_full_config(user_id="ante")
    _identity._reset_cache()

    config = _identity.load_config()
    assert config.recall_settings == {"turns_k": "12", "lanes": "turns,text"}
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"


@pytest.mark.asyncio
async def test_mcp_hides_and_refuses_a_tool_until_its_user_opts_in():
    pytest.importorskip("mcp")
    from agent_context_graph.mcp_server import build_server

    server = build_server({"recall": _EchoTool(), "memory": _OptInTool()})
    async with _connected(server) as client:
        before = [tool["name"] for tool in _wire(await client.list_tools())["tools"]]
        refused = _wire(await client.call_tool("memory", {"question": "x"}))
        _identity.write_config(memory_backend="context-graph")
        after = [tool["name"] for tool in _wire(await client.list_tools())["tools"]]
        allowed = _wire(await client.call_tool("memory", {"question": "x"}))

    assert before == ["recall"]
    assert refused["isError"] and refused["content"][0]["text"] == "Unknown tool: memory"
    assert after == ["recall", "memory"]
    assert not allowed.get("isError")


def test_session_hints_skip_tools_the_user_has_not_turned_on(monkeypatch):
    from agent_context_graph.tools import session_hints

    monkeypatch.setattr("agent_context_graph.tools.load_tools", lambda: {"memory": _OptInTool()})

    assert session_hints(["sessions-graph"]) == []
    _identity.write_config(memory_backend="context-graph")
    assert session_hints(["sessions-graph"], _identity.load_config()) == ["Use `memory`."]
