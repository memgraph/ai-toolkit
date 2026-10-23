"""Context Graph memory for apps built on the Claude Agent SDK.

The Agent SDK runs Claude Code in-process, and its hook callbacks receive the
same payloads Claude Code's command hooks do, so capture translates them with
the ``claude-code`` runtime's own rules, only recorded under a source of its
own (``claude-agent-sdk``). Where the SDK differs:

- Python callbacks get no ``SessionStart``/``SessionEnd``: a session starts at
  its first hook and the memory context goes in with the first prompt
  (``UserPromptSubmit``'s ``additionalContext``); the app ends it with
  :meth:`ClaudeAgentMemory.end_session`.
- ``recall`` and ``memory`` are served by an in-process SDK MCP server, not
  the stdio one.

Needs the ``memgraph-context-graph[claude]`` extra::

    from claude_agent_sdk import ClaudeSDKClient
    from memgraph_context_graph.link import AgentLink
    from memgraph_context_graph.sdk import ContextGraphMemory
    from memgraph_context_graph.sdk.claude import ClaudeAgentMemory

    memory = ClaudeAgentMemory(ContextGraphMemory(user_id="alice"))
    async with ClaudeSDKClient(memory.options(model="claude-sonnet-5-5")) as client:
        await client.query("What did we decide about the retry policy?")
        async for message in client.receive_response():
            print(message)
    memory.end_session()

With ``setting_sources`` unset, Claude Code loads the user's settings, and
with them the Context Graph Claude Code plugin, whose hooks would capture
every event a second time; :meth:`ClaudeAgentMemory.options` turns that
plugin off for the session.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, SdkMcpTool, create_sdk_mcp_server

from memgraph_context_graph import capture_log
from memgraph_context_graph.adapters._spec import SpecAdapter
from memgraph_context_graph.adapters.claude_code import SPEC as CLAUDE_CODE_SPEC
from memgraph_context_graph.setup.agents import PLUGIN_ID

if TYPE_CHECKING:
    from claude_agent_sdk.types import HookEvent, McpSdkServerConfig

    from memgraph_context_graph.link import AgentLink
    from memgraph_context_graph.sdk import ContextGraphMemory

SOURCE_SDK = "claude-agent-sdk"
SERVER_NAME = "context-graph"

#: The claude-code rules re-labelled; payloads are the same, the source tells the two apart in the graph.
SPEC = replace(CLAUDE_CODE_SPEC, name=SOURCE_SDK, source_sdk=SOURCE_SDK)

#: The hook events Python callbacks receive that the claude-code rules translate.
CAPTURED_HOOKS: tuple[HookEvent, ...] = (
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "SubagentStart",
    "SubagentStop",
    "Stop",
)


class _Adapter(SpecAdapter):
    SPEC = SPEC


class ClaudeAgentMemory:
    """Wires a :class:`~memgraph_context_graph.sdk.ContextGraphMemory` into ``ClaudeAgentOptions``.

    Args:
        memory: Whose memory, and where capture goes.
    """

    def __init__(self, memory: ContextGraphMemory) -> None:
        self.memory = memory
        # The adapter only calls emit(); the memory's adds ordering and the unreachable-Memgraph backoff.
        self._adapter = _Adapter(cast("AgentLink", memory))

    def options(self, **kwargs: Any) -> ClaudeAgentOptions:
        """``ClaudeAgentOptions(**kwargs)`` with the memory's hooks, MCP server and tool permissions added.

        Hooks, MCP servers and allowed tools given in *kwargs* are kept; the
        memory's are added after them. Claude Code's own auto memory is turned
        off unless *kwargs*' ``env`` says otherwise: Context Graph is the memory.
        The Context Graph Claude Code plugin is turned off in *kwargs*'
        ``settings`` (a JSON string or a settings file), so its hooks don't
        capture the session a second time.
        """
        hooks: dict[Any, list[HookMatcher]] = dict(kwargs.pop("hooks", None) or {})
        for event, matchers in self.hooks().items():
            hooks[event] = [*hooks.get(event, []), *matchers]
        mcp_servers = {**(kwargs.pop("mcp_servers", None) or {}), SERVER_NAME: self.mcp_server()}
        allowed_tools = [*(kwargs.pop("allowed_tools", None) or []), *self.allowed_tools()]
        env = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", **(kwargs.pop("env", None) or {})}
        settings = _without_the_plugin(kwargs.pop("settings", None))
        return ClaudeAgentOptions(
            hooks=hooks, mcp_servers=mcp_servers, allowed_tools=allowed_tools, env=env, settings=settings, **kwargs
        )

    def hooks(self) -> dict[HookEvent, list[HookMatcher]]:
        """Hook callbacks that capture the session and give its first prompt the memory context."""
        callback = cast("Any", self._on_hook)
        return {event: [HookMatcher(hooks=[callback])] for event in CAPTURED_HOOKS}

    def mcp_server(self) -> McpSdkServerConfig:
        """An in-process MCP server with the memory's tools."""
        tools = [
            SdkMcpTool(name=tool.name, description=tool.description, input_schema=tool.input_schema, handler=handler)
            for tool in self.memory.tools().values()
            for handler in [self._tool_handler(tool.name)]
        ]
        return create_sdk_mcp_server(name=SERVER_NAME, tools=tools)

    def allowed_tools(self) -> list[str]:
        """The memory tools' names as the SDK permits them (``mcp__context-graph__recall``)."""
        return [f"mcp__{SERVER_NAME}__{name}" for name in self.memory.tools()]

    def end_session(self, session_id: str | None = None) -> None:
        """End *session_id*, or every session captured so far: the SDK has no session-end callback.

        Each finished turn is already embedded for recall; ending the session
        is what queues it for reconciliation into extracted entities.
        """
        self.memory.end_session(session_id)

    async def aend_session(self, session_id: str | None = None) -> None:
        """:meth:`end_session` off the event loop."""
        await asyncio.to_thread(self.end_session, session_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _on_hook(self, input_data: dict[str, Any], _tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        session_id = str(input_data.get("session_id") or "")
        first = False
        if session_id:
            first = await asyncio.to_thread(self.memory.start_session, session_id, source_sdk=SOURCE_SDK)
        try:
            await asyncio.to_thread(self._adapter.handle_payload, input_data)
        except Exception as exc:  # Capture never fails the agent; the failure log has it.
            capture_log.record_failure(
                "translate",
                exc,
                runtime=SOURCE_SDK,
                hook_event=str(input_data.get("hook_event_name")),
                session_id=session_id,
            )
        if first and input_data.get("hook_event_name") == "UserPromptSubmit":
            text = await asyncio.to_thread(self.memory.session_context, session_id)
            if text:
                return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}}
        return {}

    def _tool_handler(self, name: str) -> Any:
        async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            text, is_error = await self.memory.call_tool_async(name, arguments)
            return {"content": [{"type": "text", "text": text}], "is_error": is_error}

        return handler


def _without_the_plugin(settings: str | None) -> str:
    """*settings* as a JSON string, with the Context Graph Claude Code plugin turned off.

    The SDK passes them as ``--settings``, which outranks the user's settings,
    so this wins over the plugin being enabled there; the rest is kept.

    Args:
        settings: ``ClaudeAgentOptions.settings``: a JSON object, a settings file's path, or None.

    Raises:
        ValueError: *settings* isn't a JSON object, or names a file that doesn't hold one.
    """
    if settings is None:
        merged: Any = {}
    else:
        text = settings if settings.lstrip().startswith("{") else Path(settings).read_text(encoding="utf-8")
        merged = json.loads(text)
    if not isinstance(merged, dict):
        raise ValueError(f"settings must be a JSON object, got {settings!r}")
    merged["enabledPlugins"] = {**(merged.get("enabledPlugins") or {}), PLUGIN_ID: False}
    return json.dumps(merged)
