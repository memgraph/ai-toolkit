"""Memgraph Context Graph connector for ActionsGraph.

The connector lives inside actions-graph because ActionsGraph owns the
interpretation of Event Protocol events as persisted action/session graph data.
Memgraph Context Graph only routes normalized runtime events.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

from memgraph_context_graph.events import (
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
from memgraph_context_graph.protocols import GraphConnector

from .models import (
    ActionStatus,
    Agent,
    ErrorEvent,
    Message,
    MessageRole,
    Session,
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


#: What a Session keeps from the event that created it.
_SESSION_METADATA_KEYS = ("cwd", "permission_mode", "source", "model")


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
        self._graph.record_session_start(
            Session(
                session_id=event.session_id,
                started_at=event.timestamp,
                model=event.model,
                working_directory=event.working_directory,
                metadata=self._session_metadata(event),
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
            ),
            container_agent_id=self._container(event, event.agent_name),
        )

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        self._ensure_session(event)
        self._graph.record_action(
            ToolResult(
                action_id=self._action_id(event),
                session_id=event.session_id,
                timestamp=event.timestamp,
                status=ActionStatus.FAILED if event.is_error else ActionStatus.COMPLETED,
                parent_action_id=self._tool_call_for(event),
                tool_use_id=event.tool_use_id or "",
                tool_name=event.tool_name,
                content=self._content(event.result),
                is_error=event.is_error,
                error_message=event.error_message,
                metadata=self._metadata(event, agent_name=event.agent_name),
            ),
            container_agent_id=self._container(event, event.agent_name),
        )

    def _on_agent_start(self, event: AgentStartEvent) -> None:
        self._ensure_session(event)
        self._graph.start_agent(
            Agent(
                agent_id=event.agent_name,
                agent_type=event.agent_type,
                session_id=event.session_id,
                started_at=event.timestamp,
                description=self._metadata_value(event, "description", ""),
                metadata=self._metadata(event, parent_agent_name=event.parent_agent_name),
            )
        )

    def _on_agent_end(self, event: AgentEndEvent) -> None:
        self._ensure_session(event)
        self._graph.end_agent(
            event.agent_name,
            # AgentEndEvent.output is Any, but every adapter sets a str or None.
            last_assistant_message=cast("str | None", self._content(event.output)),
            metadata=self._metadata(event),
        )

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
                metadata=self._metadata(event, agent_name=event.agent_name),
            ),
            container_agent_id=self._container(event, event.agent_name),
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
        self._graph.ensure_session(
            Session(
                session_id=event.session_id,
                started_at=event.timestamp,
                metadata=self._session_metadata(event),
            )
        )

    def _container(self, event: Event, agent_name: str | None) -> str | None:
        """The subagent an action belongs to, creating its Agent node if the runtime named one we haven't seen.

        Only when the runtime itself said the event came from a subagent (its
        payload carried ``agent_id``): some adapters name the running agent on
        every event, top-level included, and those must stay on the Session.
        """
        ensure_agent = getattr(self._graph, "ensure_agent", None)
        if agent_name and ensure_agent is not None and event.metadata.get("agent_id") == agent_name:
            ensure_agent(
                agent_name,
                event.session_id,
                agent_type=self._metadata_value(event, "agent_type", ""),
                started_at=event.timestamp,
            )
        return agent_name

    @staticmethod
    def _session_metadata(event: Event) -> dict[str, Any]:
        """Session-level facts only. The first event's full metadata can be a tool call's input and output
        or the last assistant message, which don't belong on the Session and may hold secrets."""
        metadata = {key: event.metadata[key] for key in _SESSION_METADATA_KEYS if event.metadata.get(key) is not None}
        if event.source_sdk:
            metadata["source_sdk"] = event.source_sdk
        return metadata

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
    def _metadata_value(event: Event, key: str, default: str) -> str:
        value = event.metadata.get(key)
        return value if isinstance(value, str) else default

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

    def _tool_call_for(self, event: ToolEndEvent) -> str | None:
        if event.tool_use_id:
            return self._tool_start_action_id(event)
        # No tool-call id (e.g. Copilot CLI): pair with the oldest unanswered
        # call of the same tool in this session.
        return self._graph.find_unanswered_tool_call(event.session_id, event.tool_name)

    @classmethod
    def _tool_start_action_id(cls, event: ToolEndEvent) -> str:
        return cls._stable_id(event.session_id, EventType.TOOL_START.value, event.tool_use_id or "")

    @staticmethod
    def _stable_id(*parts: str) -> str:
        payload = "\0".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
