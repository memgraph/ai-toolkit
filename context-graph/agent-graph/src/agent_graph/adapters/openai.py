"""OpenAI Agents SDK adapter for agent-graph.

Translates OpenAI Agents SDK lifecycle hooks (``RunHooksBase``) into the
common Event protocol and forwards them to the AgentLink hub.

Usage::

    from agent_graph import AgentLink
    from agent_graph.adapters.openai import OpenAIAdapter

    link = AgentLink()
    adapter = OpenAIAdapter(link, session_id="s-1")
    hooks = adapter.get_sdk_hooks()

    # Pass hooks to Runner
    result = await Runner.run(agent, "Hello", run_config=RunConfig(hooks=hooks))
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from agent_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
    HandoffEvent,
    LLMEndEvent,
    LLMStartEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_graph.protocols import SDKAdapter

if TYPE_CHECKING:
    from agent_graph.link import AgentLink

_SOURCE = "openai"


class OpenAIAdapter(SDKAdapter):
    """Adapter that converts OpenAI Agents SDK lifecycle hooks into agent-graph events.

    Args:
        link: The AgentLink hub to emit events to.
        session_id: Session identifier for all events.
        auto_session: If ``True``, emit SessionStartEvent on creation.
        session_kwargs: Extra fields for SessionStartEvent (model, tags, …).
    """

    def __init__(
        self,
        link: AgentLink,
        session_id: str,
        *,
        auto_session: bool = True,
        session_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._link = link
        self._session_id = session_id

        if auto_session:
            kw = session_kwargs or {}
            self._link.emit(
                SessionStartEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    model=kw.get("model"),
                    working_directory=kw.get("working_directory"),
                    tags=kw.get("tags", []),
                    metadata=kw.get("metadata", {}),
                )
            )

    def get_sdk_hooks(self) -> Any:
        """Return a ``RunHooksBase`` subclass for ``RunConfig(hooks=...)``."""
        try:
            from agents.lifecycle import RunHooksBase
        except ImportError:
            msg = "openai-agents is required for OpenAIAdapter. Install it with: pip install agent-graph[openai]"
            raise ImportError(msg)  # noqa: B904

        adapter = self

        class _Hooks(RunHooksBase):  # type: ignore[type-arg]
            async def on_agent_start(
                self,
                context: Any,
                agent: Any,
            ) -> None:
                adapter._link.emit(
                    AgentStartEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        agent_name=getattr(agent, "name", str(agent)),
                        agent_type=type(agent).__name__,
                    )
                )

            async def on_agent_end(
                self,
                context: Any,
                agent: Any,
                output: Any,
            ) -> None:
                adapter._link.emit(
                    AgentEndEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        agent_name=getattr(agent, "name", str(agent)),
                        agent_type=type(agent).__name__,
                        output=str(output) if output else None,
                    )
                )

            async def on_tool_start(
                self,
                context: Any,
                agent: Any,
                tool: Any,
            ) -> None:
                tool_name = getattr(tool, "name", str(tool))
                # ToolContext exposes tool_call_id and tool_arguments (JSON string)
                tool_call_id = getattr(context, "tool_call_id", None)
                tool_arguments = getattr(context, "tool_arguments", None)

                # Parse JSON string to dict for downstream connectors
                tool_input: Any = tool_arguments
                if isinstance(tool_arguments, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        tool_input = json.loads(tool_arguments)

                adapter._link.emit(
                    ToolStartEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_use_id=tool_call_id,
                        agent_name=getattr(agent, "name", None),
                    )
                )

            async def on_tool_end(
                self,
                context: Any,
                agent: Any,
                tool: Any,
                result: str,
            ) -> None:
                tool_name = getattr(tool, "name", str(tool))
                tool_call_id = getattr(context, "tool_call_id", None)

                adapter._link.emit(
                    ToolEndEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        tool_name=tool_name,
                        tool_use_id=tool_call_id,
                        result=result,
                        agent_name=getattr(agent, "name", None),
                    )
                )

            async def on_handoff(
                self,
                context: Any,
                from_agent: Any,
                to_agent: Any,
            ) -> None:
                adapter._link.emit(
                    HandoffEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        from_agent=getattr(from_agent, "name", str(from_agent)),
                        to_agent=getattr(to_agent, "name", str(to_agent)),
                    )
                )

            async def on_llm_start(
                self,
                context: Any,
                agent: Any,
                system_prompt: str | None,
                input_items: list[Any],
            ) -> None:
                adapter._link.emit(
                    LLMStartEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        agent_name=getattr(agent, "name", None),
                        system_prompt=system_prompt,
                        input_items=input_items,
                    )
                )

            async def on_llm_end(
                self,
                context: Any,
                agent: Any,
                response: Any,
            ) -> None:
                # Extract usage from response if available
                usage = getattr(response, "usage", None)
                adapter._link.emit(
                    LLMEndEvent(
                        session_id=adapter._session_id,
                        source_sdk=_SOURCE,
                        agent_name=getattr(agent, "name", None),
                        model=getattr(response, "model", None),
                        input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                        output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                        response=response,
                    )
                )

        return _Hooks()

    def end_session(
        self,
        *,
        status: str = "completed",
        total_cost_usd: float | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
    ) -> None:
        """Manually signal session end (call after Runner.run completes)."""
        self._link.emit(
            SessionEndEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                status=status,
                total_cost_usd=total_cost_usd,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )
        )
