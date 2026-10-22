"""Context Graph memory for apps built on the OpenAI Agents SDK.

The Agents SDK has no hook that sees the conversation as such, so capture
reads it off the run lifecycle (``RunHooks``):

- the user's message is the trailing user items of the run's input
  (``AgentHookContext.turn_input``), taken when the run's first agent starts;
- every model response's text is an assistant message (``on_llm_end``);
- tools and handoffs are their own hooks; the final output ends the turn,
  which embeds it for recall.

The agent running the conversation is its main thread, not a subagent, so
events name it in ``metadata["agent"]`` rather than ``agent_name``, which
actions-graph files under a subagent of the session.

The memory context goes into the agent's instructions (a dynamic
``instructions`` callable), and ``recall``/``memory`` are function tools.

Needs the ``memgraph-context-graph[openai]`` extra::

    from agents import Agent, Runner
    from memgraph_context_graph.sdk import ContextGraphMemory
    from memgraph_context_graph.sdk.openai import OpenAIAgentsMemory

    memory = OpenAIAgentsMemory(ContextGraphMemory(user_id="alice"), session_id="chat-42")
    agent = Agent(name="Assistant", instructions=memory.instructions("You are helpful."), tools=memory.tools())
    result = await Runner.run(agent, "What did we decide about retries?", hooks=memory.hooks())
    memory.end_session()

``session_id`` is the app's conversation id: pass the same one (an Agents SDK
``Session``'s ``session_id`` fits) for every run of one conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from agents import FunctionTool, RunHooks, Tool
from memgraph_context_graph.events import (
    HandoffEvent,
    MessageEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from memgraph_context_graph.sdk import ContextGraphMemory

SOURCE_SDK = "openai-agents"


class OpenAIAgentsMemory:
    """Wires a :class:`~memgraph_context_graph.sdk.ContextGraphMemory` into an Agents SDK app.

    Args:
        memory: Whose memory, and where capture goes.
        session_id: The conversation every run belongs to; a fresh one when omitted.
    """

    def __init__(self, memory: ContextGraphMemory, *, session_id: str | None = None) -> None:
        self.memory = memory
        self.session_id = session_id or f"openai-agents-{uuid.uuid4()}"
        self._context: str | None = None
        self._context_read = False

    def instructions(self, base: Any = None) -> Callable[[Any, Any], Awaitable[str]]:
        """Dynamic instructions: *base* (a string or the SDK's instructions callable), then the memory context."""

        async def instructions(run_context: Any, agent: Any) -> str:
            text = base(run_context, agent) if callable(base) else base
            if hasattr(text, "__await__"):
                text = await text
            context = await self._session_context()
            return "\n\n".join(part for part in (text, context) if part)

        return instructions

    def tools(self) -> list[Tool]:
        """The memory's tools as function tools."""
        return [
            FunctionTool(
                name=tool.name,
                description=tool.description,
                params_json_schema=strict_schema(tool.input_schema),
                on_invoke_tool=self._invoker(tool.name),
            )
            for tool in self.memory.tools().values()
        ]

    def hooks(self) -> RunHooks:
        """``RunHooks`` that capture each run into the session."""
        return _CaptureHooks(self)

    def end_session(self) -> None:
        """End the conversation: the Agents SDK has no session end of its own.

        Each finished run is already embedded for recall; ending the session
        is what queues it for reconciliation into extracted entities.
        """
        self.memory.end_session(self.session_id)

    async def aend_session(self) -> None:
        """:meth:`end_session` off the event loop."""
        await asyncio.to_thread(self.end_session)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _session_context(self) -> str | None:
        if not self._context_read:
            self._context = await asyncio.to_thread(self.memory.session_context, self.session_id)
            self._context_read = True
        return self._context

    def _invoker(self, name: str) -> Callable[[Any, str], Awaitable[str]]:
        async def invoke(_tool_context: Any, arguments: str) -> str:
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return f"{name} needs its arguments as a JSON object."
            arguments_given = parsed if isinstance(parsed, dict) else {}
            # Strict mode sends every optional field, as null when unused; the tools expect it left out.
            given = {key: value for key, value in arguments_given.items() if value is not None}
            text, _is_error = await self.memory.call_tool_async(name, given)
            return text

        return invoke

    async def _emit(self, event: Any) -> None:
        await self.memory.emit_async(event)


class _CaptureHooks(RunHooks):
    """Run lifecycle → Event Protocol, for one conversation."""

    def __init__(self, owner: OpenAIAgentsMemory) -> None:
        self._owner = owner
        # A run's agents (one per handoff) all share its Usage object, and every run gets a new one, so it names
        # the run even when one fails before reaching on_agent_end. Held, so its id can't be reused.
        self._run_usage: object | None = None

    @property
    def _sid(self) -> str:
        return self._owner.session_id

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        owner = self._owner
        usage = getattr(context, "usage", None)
        if usage is None or usage is not self._run_usage:
            self._run_usage = usage
            model = getattr(agent, "model", None)
            await asyncio.to_thread(
                owner.memory.start_session,
                self._sid,
                source_sdk=SOURCE_SDK,
                model=model if isinstance(model, str) else None,
            )
            for text in _trailing_user_messages(getattr(context, "turn_input", None) or []):
                await owner._emit(MessageEvent(session_id=self._sid, source_sdk=SOURCE_SDK, role="user", content=text))

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        text = _response_text(response)
        if text:
            await self._owner._emit(
                MessageEvent(
                    session_id=self._sid,
                    source_sdk=SOURCE_SDK,
                    role="assistant",
                    content=text,
                    metadata={"agent": _name(agent)},
                )
            )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        await self._owner._emit(TurnEndEvent(session_id=self._sid, source_sdk=SOURCE_SDK))

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        await self._owner._emit(
            ToolStartEvent(
                session_id=self._sid,
                source_sdk=SOURCE_SDK,
                tool_name=_name(tool),
                tool_input=_tool_input(context),
                tool_use_id=getattr(context, "tool_call_id", None),
                metadata={"agent": _name(agent)},
            )
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        await self._owner._emit(
            ToolEndEvent(
                session_id=self._sid,
                source_sdk=SOURCE_SDK,
                tool_name=_name(tool),
                tool_use_id=getattr(context, "tool_call_id", None),
                result=result if isinstance(result, str) else str(result),
                # sessions-graph reads the written paths here for the memory tool's provenance.
                metadata={"agent": _name(agent), "tool_input": _tool_input(context)},
            )
        )

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        await self._owner._emit(
            HandoffEvent(
                session_id=self._sid,
                source_sdk=SOURCE_SDK,
                from_agent=_name(from_agent),
                to_agent=_name(to_agent),
            )
        )


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """*schema* as OpenAI's strict mode takes it: every property required, the optional ones nullable.

    Without strict mode the model may leave out a required field (gpt-4.1-mini
    called ``memory`` with no ``command``); strict mode makes it follow the schema.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", ()))
    return {
        **schema,
        "properties": {
            name: prop if name in required else {**prop, "type": [prop["type"], "null"]}
            for name, prop in properties.items()
        },
        "required": list(properties),
        "additionalProperties": False,
    }


def _name(obj: Any) -> str:
    return str(getattr(obj, "name", None) or type(obj).__name__)


def _tool_input(context: Any) -> Any:
    """A function tool's arguments, parsed; ``ToolContext`` carries them as a JSON string."""
    arguments = getattr(context, "tool_arguments", None)
    if isinstance(arguments, str):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(arguments)
    return arguments


def _trailing_user_messages(items: Any) -> list[str]:
    """The user messages at the end of a run's input: what the user just said.

    An app that carries history passes it back as input (``to_input_list()``),
    so only the trailing user items are new; earlier ones were captured by the
    run that first saw them.
    """
    if isinstance(items, str):
        return [items] if items.strip() else []
    texts: list[str] = []
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("role") != "user":
            break
        text = _content_text(item.get("content"))
        if text:
            texts.append(text)
    return list(reversed(texts))


def _response_text(response: Any) -> str:
    """The text of a model response's message items."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            parts.append(_content_text(getattr(item, "content", None)))
    return "\n".join(part for part in parts if part)


def _content_text(content: Any) -> str:
    """Text from a message's content: a string, or a list of text parts (input or output, dicts or objects)."""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for part in content or []:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)
