"""Claude Agent SDK adapter for agent-graph.

Translates Claude Agent SDK hook callbacks into the common Event protocol
and forwards them to the AgentLink hub.

Usage::

    from agent_graph import AgentLink
    from agent_graph.adapters.claude import ClaudeAdapter

    link = AgentLink()
    adapter = ClaudeAdapter(link, session_id="s-1")
    hooks = adapter.get_sdk_hooks()

    # Pass hooks to ClaudeAgentOptions
    options = ClaudeAgentOptions(hooks=hooks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_graph.protocols import SDKAdapter

if TYPE_CHECKING:
    from agent_graph.link import AgentLink

_SOURCE = "claude"


class ClaudeAdapter(SDKAdapter):
    """Adapter that converts Claude Agent SDK hooks into agent-graph events.

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
        self._tool_use_to_action: dict[str, str] = {}

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

    # ------------------------------------------------------------------
    # SDK interface
    # ------------------------------------------------------------------

    def get_sdk_hooks(self) -> dict[str, list[Any]]:
        """Return a hooks dict suitable for ``ClaudeAgentOptions(hooks=...)``."""
        try:
            from claude_agent_sdk import HookMatcher
        except ImportError:

            class HookMatcher:  # type: ignore[no-redef]
                def __init__(
                    self,
                    matcher: str | None = None,
                    hooks: list[Any] | None = None,
                    timeout: float | None = None,
                ):
                    self.matcher = matcher
                    self.hooks = hooks or []
                    self.timeout = timeout

        return {
            "PreToolUse": [HookMatcher(hooks=[self._pre_tool_use])],
            "PostToolUse": [HookMatcher(hooks=[self._post_tool_use])],
            "PostToolUseFailure": [HookMatcher(hooks=[self._post_tool_use_failure])],
            "UserPromptSubmit": [HookMatcher(hooks=[self._user_prompt_submit])],
            "SubagentStart": [HookMatcher(hooks=[self._subagent_start])],
            "SubagentStop": [HookMatcher(hooks=[self._subagent_stop])],
            "Notification": [HookMatcher(hooks=[self._notification])],
            "Stop": [HookMatcher(hooks=[self._stop])],
        }

    # ------------------------------------------------------------------
    # Hook callbacks (async, matching Claude SDK signature)
    # ------------------------------------------------------------------

    async def _pre_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        actual_id = input_data.get("tool_use_id") or tool_use_id

        self._link.emit(
            ToolStartEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=actual_id,
                agent_name=input_data.get("agent_id"),
                metadata={k: input_data.get(k) for k in ("agent_type", "cwd") if input_data.get(k) is not None},
            )
        )
        return {}

    async def _post_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        actual_id = input_data.get("tool_use_id") or tool_use_id
        tool_response = input_data.get("tool_response")

        is_error = False
        error_message = None
        content = tool_response
        if isinstance(tool_response, dict):
            is_error = tool_response.get("is_error", False)
            error_message = tool_response.get("error")
            content = tool_response.get("content", tool_response)

        self._link.emit(
            ToolEndEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                tool_name=tool_name,
                tool_use_id=actual_id,
                result=content,
                is_error=is_error,
                error_message=error_message,
                agent_name=input_data.get("agent_id"),
            )
        )
        return {}

    async def _post_tool_use_failure(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        actual_id = input_data.get("tool_use_id") or tool_use_id
        error = input_data.get("error", "Unknown error")

        self._link.emit(
            ErrorOccurredEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                error_type="tool_failure",
                error_message=str(error),
                error_details={
                    "tool_name": tool_name,
                    "tool_use_id": actual_id,
                    "tool_input": input_data.get("tool_input", {}),
                    "is_interrupt": input_data.get("is_interrupt", False),
                },
                recoverable=not input_data.get("is_interrupt", False),
            )
        )
        return {}

    async def _user_prompt_submit(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self._link.emit(
            MessageEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                role="user",
                content=input_data.get("prompt", ""),
                metadata={k: input_data.get(k) for k in ("cwd", "permission_mode") if input_data.get(k) is not None},
            )
        )
        return {}

    async def _subagent_start(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self._link.emit(
            AgentStartEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                agent_name=input_data.get("agent_id", ""),
                agent_type=input_data.get("agent_type", ""),
            )
        )
        return {}

    async def _subagent_stop(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self._link.emit(
            AgentEndEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                agent_name=input_data.get("agent_id", ""),
                agent_type=input_data.get("agent_type", ""),
                metadata={
                    "stop_hook_active": input_data.get("stop_hook_active"),
                    "agent_transcript_path": input_data.get("agent_transcript_path"),
                },
            )
        )
        return {}

    async def _notification(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self._link.emit(
            MessageEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                role="system",
                content=input_data.get("message", ""),
                metadata={
                    "title": input_data.get("title"),
                    "notification_type": input_data.get("notification_type"),
                },
            )
        )
        return {}

    async def _stop(
        self,
        _input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        self._link.emit(
            SessionEndEvent(
                session_id=self._session_id,
                source_sdk=_SOURCE,
                status="completed",
            )
        )
        return {}
