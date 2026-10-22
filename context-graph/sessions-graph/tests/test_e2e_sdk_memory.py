"""Context Graph as an SDK app's memory, with a real Memgraph: the SDKs' own hook objects, no model.

The Claude Agent SDK's callbacks get Claude Code's hook payloads, and the
OpenAI Agents SDK's ``RunHooks`` get its run objects; both are fed here as the
SDKs would, and the assertions are on the graph they leave.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, cast

import pytest

pytest.importorskip("memgraph_context_graph")
pytest.importorskip("actions_graph")
pytest.importorskip("claude_agent_sdk")
pytest.importorskip("agents")

import sessions_graph.connector
from agents.items import ModelResponse, TResponseInputItem
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool_context import ToolContext
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from agents import Agent, FunctionTool
from memgraph_context_graph import capture_log
from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.sdk import ContextGraphMemory
from memgraph_context_graph.sdk.claude import ClaudeAgentMemory
from memgraph_context_graph.sdk.openai import OpenAIAgentsMemory

PROJECT = "/tmp/sdk-app"


@pytest.fixture
def spawned(monkeypatch):
    """The background ``sessions-graph`` jobs a session end would start, recorded instead of run."""
    calls: list[list[str]] = []
    monkeypatch.setattr(sessions_graph.connector, "_spawn_detached", lambda args, env: calls.append(args))
    return calls


@pytest.fixture
def memory(graph, tmp_path, monkeypatch, spawned):
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(user_id="alice", memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    yield ContextGraphMemory(cwd=PROJECT, connectors=["actions-graph", "sessions-graph"])
    _identity._reset_cache()


def _turns(memgraph, session_id):
    rows = memgraph.query(
        "MATCH (:Session {session_id: $s})-[:HAS_ACTION]->(a:Message) RETURN a.text AS text ORDER BY a.timestamp",
        {"s": session_id},
    )
    return [row["text"] for row in rows]


def _payload(session_id, event, **fields):
    return {
        "session_id": session_id,
        "cwd": PROJECT,
        "transcript_path": "/dev/null",
        "hook_event_name": event,
        **fields,
    }


async def test_claude_agent_session_is_captured_and_its_memory_write_has_provenance(memory, memgraph, spawned):
    claude = ClaudeAgentMemory(memory)
    hook = claude.hooks()["UserPromptSubmit"][0].hooks[0]
    assert claude.allowed_tools() == ["mcp__context-graph__memory", "mcp__context-graph__recall"]

    first = await hook(_payload("c-1", "UserPromptSubmit", prompt="Always use pnpm here."), None, {"signal": None})
    write = {"command": "create", "path": "/memories/user/pnpm.md", "file_text": "Use pnpm, never npm.\n"}
    result = await claude._tool_handler("memory")(write)
    await hook(
        _payload("c-1", "PostToolUse", tool_name="mcp__context-graph__memory", tool_use_id="t-1", tool_input=write),
        "t-1",
        {"signal": None},
    )
    await hook(
        _payload("c-1", "Stop", stop_hook_active=False, last_assistant_message="Saved: pnpm."), None, {"signal": None}
    )
    again = await hook(_payload("c-1", "UserPromptSubmit", prompt="Thanks."), None, {"signal": None})
    claude.end_session()

    assert "memory" in cast("Any", first)["hookSpecificOutput"]["additionalContext"]
    assert again == {}, "only a session's first prompt carries the memory context"
    assert result["is_error"] is False
    assert _turns(memgraph, "c-1") == ["Always use pnpm here.", "Saved: pnpm.", "Thanks."]
    row = memgraph.query(
        "MATCH (u:User {user_id: 'alice'})-[:HAD_SESSION]->(s:Session {session_id: 'c-1'})"
        "-[:PRODUCED_MEMORY]->(m:Memory) RETURN m.path AS path, s.source_sdk AS source"
    )
    assert [r["path"] for r in row] == ["/memories/user/pnpm.md"]
    assert ["embed", "--session", "c-1"] in spawned


def test_claude_options_turn_the_claude_code_plugin_off_and_keep_the_apps_settings(memory, tmp_path):
    """The SDK loads user settings by default, and the user-level plugin's hooks would capture every event twice."""
    claude = ClaudeAgentMemory(memory)

    default = json.loads(cast("str", claude.options().settings))
    from_string = json.loads(cast("str", claude.options(settings='{"model": "haiku"}').settings))
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"enabledPlugins": {"other@market": true}}')
    from_file = json.loads(cast("str", claude.options(settings=str(settings_file)).settings))

    assert default == {"enabledPlugins": {"context-graph@context-graph-plugins": False}}
    assert from_string["model"] == "haiku"
    assert from_string["enabledPlugins"] == {"context-graph@context-graph-plugins": False}
    assert from_file["enabledPlugins"] == {"other@market": True, "context-graph@context-graph-plugins": False}


def _message(text):
    return ResponseOutputMessage(
        id="m-1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


async def test_openai_agents_run_is_captured_on_the_session(memory, memgraph, spawned):
    wired = OpenAIAgentsMemory(memory, session_id="o-1")
    hooks = wired.hooks()
    agent = Agent(name="Assistant", tools=wired.tools())
    history: list[TResponseInputItem] = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "Which package manager?"},
    ]
    tool = next(t for t in agent.tools if t.name == "recall")
    tool_context = ToolContext(
        None, tool_name="recall", tool_call_id="call-1", tool_arguments='{"question": "package manager"}'
    )

    await hooks.on_agent_start(AgentHookContext(None, turn_input=history), agent)
    await hooks.on_tool_start(tool_context, agent, tool)
    await hooks.on_tool_end(tool_context, agent, tool, "rows")
    await hooks.on_llm_end(
        RunContextWrapper(None), agent, ModelResponse(output=[_message("Use pnpm.")], usage=Usage(), response_id="r-1")
    )
    await hooks.on_agent_end(AgentHookContext(None), agent, "Use pnpm.")
    wired.end_session()

    assert _turns(memgraph, "o-1") == ["Which package manager?", "Use pnpm."], "only the new user message is captured"
    tools = memgraph.query(
        "MATCH (:Session {session_id: 'o-1'})-[:HAS_ACTION]->(a:ToolCall) RETURN a.tool_name AS name"
    )
    assert [r["name"] for r in tools] == ["recall"]
    assert memgraph.query("MATCH (:User {user_id: 'alice'})-[:HAD_SESSION]->(s:Session {session_id: 'o-1'}) RETURN s")
    assert ["embed", "--session", "o-1"] in spawned


async def test_a_memory_saved_through_one_sdk_is_in_the_others_instructions(memory):
    claude = ClaudeAgentMemory(memory)
    await claude._tool_handler("memory")(
        {"command": "create", "path": "/memories/user/editor.md", "file_text": "---\ndescription: Uses Helix\n---\n"}
    )

    instructions = OpenAIAgentsMemory(memory).instructions("Be brief.")
    text = await instructions(RunContextWrapper(None), Agent(name="Assistant"))

    assert text.startswith("Be brief.\n\n")
    assert "/memories/user/editor.md — Uses Helix" in text


async def test_a_run_after_one_that_failed_still_captures_its_user_message(memory, memgraph):
    # A run that raises (MaxTurnsExceeded, a guardrail) never reaches on_agent_end.
    hooks = OpenAIAgentsMemory(memory, session_id="o-2").hooks()
    agent = Agent(name="Assistant")
    failed = AgentHookContext(None, usage=Usage(), turn_input=[{"role": "user", "content": "first, which fails"}])
    handoff = AgentHookContext(None, usage=failed.usage, turn_input=failed.turn_input)
    retry = AgentHookContext(None, usage=Usage(), turn_input=[{"role": "user", "content": "second"}])

    await hooks.on_agent_start(failed, agent)
    await hooks.on_agent_start(handoff, Agent(name="Specialist"))
    await hooks.on_agent_start(retry, agent)

    assert _turns(memgraph, "o-2") == ["first, which fails", "second"], "a handoff isn't a new run; a new Usage is"


async def test_openai_tools_are_strict_and_an_unused_field_arrives_as_left_out(memory, graph):
    tools = {tool.name: tool for tool in OpenAIAgentsMemory(memory).tools()}
    memory_tool = tools["memory"]
    assert isinstance(memory_tool, FunctionTool)
    assert memory_tool.strict_json_schema
    assert memory_tool.params_json_schema["required"] == list(memory_tool.params_json_schema["properties"])

    # Strict mode makes the model send every field, the unused ones as null.
    arguments = {name: None for name in memory_tool.params_json_schema["properties"]}
    arguments |= {"command": "create", "path": "/memories/user/tea.md", "file_text": "Likes tea.\n"}
    context = ToolContext(None, tool_name="memory", tool_call_id="c-1", tool_arguments=json.dumps(arguments))
    result = await memory_tool.on_invoke_tool(context, json.dumps(arguments))

    assert "created" in str(result).lower()
    assert [f.content for f in graph.memory_store("alice").files()] == ["Likes tea.\n"]


async def test_an_unreachable_memgraph_never_fails_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(user_id="alice", memgraph_url="bolt://127.0.0.1:1")
    memory = ContextGraphMemory(cwd=PROJECT, connectors=["actions-graph", "sessions-graph"])
    claude = ClaudeAgentMemory(memory)
    hook = claude.hooks()["UserPromptSubmit"][0].hooks[0]

    first = await hook(_payload("c-down", "UserPromptSubmit", prompt="hello"), None, {"signal": None})
    started = time.monotonic()
    await hook(_payload("c-down", "Stop", stop_hook_active=False, last_assistant_message="hi"), None, {"signal": None})
    backed_off = time.monotonic() - started
    text, is_error = await memory.call_tool_async("recall", {"question": "anything"})
    _identity._reset_cache()

    context = cast("Any", first)["hookSpecificOutput"]["additionalContext"]
    assert "`recall`" in context, "the model is still told which tools it has"
    assert "Memory index" not in context, "but not an index it couldn't read"
    assert backed_off < 1, "capture waits out no second connect timeout while Memgraph is down"
    assert is_error
    assert text.startswith("recall failed:")
    assert [f.stage for f in capture_log.read_recent()].count("connect") == 1
