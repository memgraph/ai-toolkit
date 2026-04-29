"""Connector that routes agent-link events to an ActionsGraph.

Maps the common Event protocol onto the ActionsGraph API:
- SessionStartEvent  → create_session()
- SessionEndEvent    → end_session()
- ToolStartEvent     → record_tool_call()
- ToolEndEvent       → record_tool_result()
- MessageEvent       → record_message()
- AgentStart/End     → record_action() (SubagentEvent)
- ErrorOccurredEvent → record_action() (ErrorEvent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_link.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    Event,
    EventType,
    LLMEndEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_link.protocols import GraphConnector

if TYPE_CHECKING:
    from actions_graph import ActionsGraph

_SUPPORTED_EVENTS = {
    EventType.SESSION_START,
    EventType.SESSION_END,
    EventType.TOOL_START,
    EventType.TOOL_END,
    EventType.AGENT_START,
    EventType.AGENT_END,
    EventType.MESSAGE,
    EventType.ERROR,
    EventType.LLM_END,
}


class ActionsConnector(GraphConnector):
    """Writes agent-link events to an :class:`ActionsGraph`.

    Args:
        graph: An initialised ActionsGraph instance.
    """

    def __init__(self, graph: ActionsGraph) -> None:
        self._graph = graph
        # Track tool_use_id → action_id for correlating results
        self._tool_use_to_action: dict[str, str] = {}

    def supports(self, event: Event) -> bool:
        return event.event_type in _SUPPORTED_EVENTS

    def on_event(self, event: Event) -> None:
        _handlers: dict[EventType, Any] = {
            EventType.SESSION_START: self._handle_session_start,
            EventType.SESSION_END: self._handle_session_end,
            EventType.TOOL_START: self._handle_tool_start,
            EventType.TOOL_END: self._handle_tool_end,
            EventType.AGENT_START: self._handle_agent_start,
            EventType.AGENT_END: self._handle_agent_end,
            EventType.MESSAGE: self._handle_message,
            EventType.ERROR: self._handle_error,
            EventType.LLM_END: self._handle_llm_end,
        }
        handler = _handlers.get(event.event_type)
        if handler:
            handler(event)

    # ------------------------------------------------------------------

    def _handle_session_start(self, event: SessionStartEvent) -> None:
        from actions_graph import Session

        session = Session(
            session_id=event.session_id,
            model=event.model,
            working_directory=event.working_directory,
            tags=event.tags,
            metadata=event.metadata,
        )
        self._graph.create_session(session)

    def _handle_session_end(self, event: SessionEndEvent) -> None:
        from actions_graph import ActionStatus

        status_map = {
            "completed": ActionStatus.COMPLETED,
            "failed": ActionStatus.FAILED,
        }
        self._graph.end_session(
            event.session_id,
            status=status_map.get(event.status, ActionStatus.COMPLETED),
            total_cost_usd=event.total_cost_usd,
            total_input_tokens=event.total_input_tokens,
            total_output_tokens=event.total_output_tokens,
        )

    def _handle_tool_start(self, event: ToolStartEvent) -> None:
        action = self._graph.record_tool_call(
            session_id=event.session_id,
            tool_name=event.tool_name,
            tool_input=event.tool_input or {},
            tool_use_id=event.tool_use_id,
        )
        if event.tool_use_id:
            self._tool_use_to_action[event.tool_use_id] = action.action_id

    def _handle_tool_end(self, event: ToolEndEvent) -> None:
        self._graph.record_tool_result(
            session_id=event.session_id,
            tool_use_id=event.tool_use_id or "",
            tool_name=event.tool_name,
            content=event.result if isinstance(event.result, (str, list)) else str(event.result),
            is_error=event.is_error,
            error_message=event.error_message,
        )

    def _handle_agent_start(self, event: AgentStartEvent) -> None:
        from actions_graph.models import ActionStatus, ActionType, SubagentEvent

        action = SubagentEvent(
            session_id=event.session_id,
            agent_id=event.agent_name,
            agent_type=event.agent_type,
            status=ActionStatus.IN_PROGRESS,
            metadata=event.metadata,
        )
        action.action_type = ActionType.SUBAGENT_START
        self._graph.record_action(action)

    def _handle_agent_end(self, event: AgentEndEvent) -> None:
        from actions_graph.models import ActionStatus, ActionType, SubagentEvent

        action = SubagentEvent(
            session_id=event.session_id,
            agent_id=event.agent_name,
            agent_type=event.agent_type,
            status=ActionStatus.COMPLETED,
            metadata=event.metadata,
        )
        action.action_type = ActionType.SUBAGENT_STOP
        self._graph.record_action(action)

    def _handle_message(self, event: MessageEvent) -> None:
        from actions_graph.models import MessageRole

        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        self._graph.record_message(
            session_id=event.session_id,
            role=role_map.get(event.role, MessageRole.USER),
            content=event.content,
            model=event.model,
        )

    def _handle_error(self, event: ErrorOccurredEvent) -> None:
        from actions_graph.models import ErrorEvent

        action = ErrorEvent(
            session_id=event.session_id,
            error_type=event.error_type,
            error_message=event.error_message,
            error_details=event.error_details,
            recoverable=event.recoverable,
            metadata=event.metadata,
        )
        self._graph.record_action(action)

    def _handle_llm_end(self, event: LLMEndEvent) -> None:
        from actions_graph.models import Message, MessageRole

        if event.response is not None:
            action = Message(
                session_id=event.session_id,
                role=MessageRole.ASSISTANT,
                content=str(event.response),
                model=event.model,
                metadata={
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    **event.metadata,
                },
            )
            self._graph.record_action(action)
