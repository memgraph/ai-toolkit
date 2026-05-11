"""Agent Context Graph connector for ActionsGraph.

The connector lives inside actions-graph because ActionsGraph owns the
interpretation of Event Protocol events as persisted action/session graph data.
Agent Context Graph only routes normalized runtime events.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    Event,
    EventType,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_context_graph.protocols import GraphConnector

from .models import (
    ActionStatus,
    ActionType,
    ErrorEvent,
    Message,
    MessageRole,
    Session,
    SubagentEvent,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from .core import ActionsGraph


_SUPPORTED_EVENTS = {
    EventType.SESSION_START,
    EventType.SESSION_END,
    EventType.TOOL_START,
    EventType.TOOL_END,
    EventType.AGENT_START,
    EventType.AGENT_END,
    EventType.MESSAGE,
    EventType.ERROR,
}


class ActionsGraphConnector(GraphConnector):
    """Receives Event Protocol events and records session observability data."""

    def __init__(self, graph: ActionsGraph) -> None:
        self._graph = graph

    def supports(self, event: Event) -> bool:
        return event.event_type in _SUPPORTED_EVENTS

    def on_event(self, event: Event) -> None:
        if isinstance(event, SessionStartEvent):
            self._on_session_start(event)
        elif isinstance(event, SessionEndEvent):
            self._on_session_end(event)
        elif isinstance(event, ToolStartEvent):
            self._on_tool_start(event)
        elif isinstance(event, ToolEndEvent):
            self._on_tool_end(event)
        elif isinstance(event, AgentStartEvent):
            self._on_agent_start(event)
        elif isinstance(event, AgentEndEvent):
            self._on_agent_end(event)
        elif isinstance(event, MessageEvent):
            self._on_message(event)
        elif isinstance(event, ErrorOccurredEvent):
            self._on_error(event)

    def _on_session_start(self, event: SessionStartEvent) -> None:
        if self._graph.get_session(event.session_id) is not None:
            return
        self._graph.create_session(
            Session(
                session_id=event.session_id,
                started_at=event.timestamp,
                model=event.model,
                working_directory=event.working_directory,
                tags=event.tags,
                metadata=self._metadata(event),
            )
        )

    def _on_session_end(self, event: SessionEndEvent) -> None:
        self._ensure_session(event)
        self._graph.end_session(
            event.session_id,
            status=self._status(event.status),
            total_cost_usd=event.total_cost_usd,
            total_input_tokens=event.total_input_tokens,
            total_output_tokens=event.total_output_tokens,
        )

    def _on_tool_start(self, event: ToolStartEvent) -> None:
        self._ensure_session(event)
        self._graph.record_action(
            ToolCall(
                action_id=self._action_id(event),
                session_id=event.session_id,
                timestamp=event.timestamp,
                status=ActionStatus.IN_PROGRESS,
                tool_name=event.tool_name,
                tool_input=self._dict_or_wrapped(event.tool_input),
                tool_use_id=event.tool_use_id,
                metadata=self._metadata(event, agent_name=event.agent_name),
            )
        )

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        self._ensure_session(event)
        self._graph.record_action(
            ToolResult(
                action_id=self._action_id(event),
                session_id=event.session_id,
                timestamp=event.timestamp,
                status=ActionStatus.FAILED if event.is_error else ActionStatus.COMPLETED,
                parent_action_id=self._tool_start_action_id(event) if event.tool_use_id else None,
                tool_use_id=event.tool_use_id or "",
                tool_name=event.tool_name,
                content=self._content(event.result),
                is_error=event.is_error,
                error_message=event.error_message,
                metadata=self._metadata(event, agent_name=event.agent_name),
            )
        )

    def _on_agent_start(self, event: AgentStartEvent) -> None:
        self._ensure_session(event)
        action = SubagentEvent(
            action_id=self._action_id(event),
            session_id=event.session_id,
            timestamp=event.timestamp,
            agent_id=event.agent_name,
            agent_type=event.agent_type,
            description=event.parent_agent_name or "",
            metadata=self._metadata(event),
        )
        action.action_type = ActionType.SUBAGENT_START
        self._graph.record_action(action)

    def _on_agent_end(self, event: AgentEndEvent) -> None:
        self._ensure_session(event)
        action = SubagentEvent(
            action_id=self._action_id(event),
            session_id=event.session_id,
            timestamp=event.timestamp,
            agent_id=event.agent_name,
            agent_type=event.agent_type,
            result=self._content(event.output),
            metadata=self._metadata(event),
        )
        action.action_type = ActionType.SUBAGENT_STOP
        self._graph.record_action(action)

    def _on_message(self, event: MessageEvent) -> None:
        self._ensure_session(event)
        self._graph.record_action(
            Message(
                action_id=self._action_id(event),
                session_id=event.session_id,
                timestamp=event.timestamp,
                role=self._role(event.role),
                content=event.content,
                model=event.model,
                metadata=self._metadata(event),
            )
        )

    def _on_error(self, event: ErrorOccurredEvent) -> None:
        self._ensure_session(event)
        self._graph.record_action(
            ErrorEvent(
                action_id=self._action_id(event),
                session_id=event.session_id,
                timestamp=event.timestamp,
                error_type=event.error_type,
                error_message=event.error_message,
                error_details=event.error_details,
                recoverable=event.recoverable,
                metadata=self._metadata(event),
            )
        )

    def _ensure_session(self, event: Event) -> None:
        if self._graph.get_session(event.session_id) is not None:
            return
        self._graph.create_session(
            Session(
                session_id=event.session_id,
                started_at=event.timestamp,
                metadata=self._metadata(event),
            )
        )

    @staticmethod
    def _metadata(event: Event, **extra: Any) -> dict[str, Any]:
        metadata = dict(event.metadata)
        if event.source_sdk:
            metadata["source_sdk"] = event.source_sdk
        for key, value in extra.items():
            if value is not None:
                metadata[key] = value
        return metadata

    @staticmethod
    def _status(value: str) -> ActionStatus:
        try:
            return ActionStatus(value)
        except ValueError:
            return ActionStatus.COMPLETED

    @staticmethod
    def _role(value: str) -> MessageRole:
        try:
            return MessageRole(value)
        except ValueError:
            return MessageRole.USER

    @staticmethod
    def _dict_or_wrapped(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {"value": value}

    @staticmethod
    def _content(value: Any) -> str | list[dict[str, Any]] | None:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
        return str(value)

    @classmethod
    def _action_id(cls, event: Event) -> str:
        if isinstance(event, (ToolStartEvent, ToolEndEvent)) and event.tool_use_id:
            return cls._stable_id(event.session_id, event.event_type.value, event.tool_use_id)
        return cls._stable_id(event.session_id, event.event_type.value, event.timestamp)

    @classmethod
    def _tool_start_action_id(cls, event: ToolEndEvent) -> str:
        return cls._stable_id(event.session_id, EventType.TOOL_START.value, event.tool_use_id or "")

    @staticmethod
    def _stable_id(*parts: str) -> str:
        payload = "\0".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
