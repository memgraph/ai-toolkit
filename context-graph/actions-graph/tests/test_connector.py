"""Tests for ActionsGraphConnector."""

from actions_graph.connector import ActionsGraphConnector
from actions_graph.models import ActionStatus, ErrorEvent, Message, Session, ToolCall, ToolResult
from agent_context_graph.events import (
    ErrorOccurredEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)


class FakeActionsGraph:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.actions = []
        self.ended_sessions = []

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def create_session(self, session: Session):
        self.sessions[session.session_id] = session
        return session

    def end_session(self, session_id: str, **kwargs):
        self.ended_sessions.append((session_id, kwargs))
        session = self.sessions[session_id]
        session.status = kwargs["status"]
        session.ended_at = "ended"
        return session

    def record_action(self, action):
        self.actions.append(action)
        return action


def test_records_session_start():
    graph = FakeActionsGraph()
    connector = ActionsGraphConnector(graph)

    connector.on_event(
        SessionStartEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:00+00:00",
            model="gpt-5",
            working_directory="/repo",
            tags=["local"],
            source_sdk="codex",
        )
    )

    session = graph.sessions["session-1"]
    assert session.model == "gpt-5"
    assert session.working_directory == "/repo"
    assert session.tags == ["local"]
    assert session.metadata["source_sdk"] == "codex"


def test_records_tool_start_as_tool_call_action():
    graph = FakeActionsGraph()
    connector = ActionsGraphConnector(graph)

    connector.on_event(
        ToolStartEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:00+00:00",
            tool_name="Read",
            tool_input={"file_path": "CONTEXT.md"},
            tool_use_id="tool-1",
            agent_name="codex",
        )
    )

    assert graph.sessions["session-1"].session_id == "session-1"
    action = graph.actions[0]
    assert isinstance(action, ToolCall)
    assert action.tool_name == "Read"
    assert action.tool_input == {"file_path": "CONTEXT.md"}
    assert action.tool_use_id == "tool-1"
    assert action.status == ActionStatus.IN_PROGRESS
    assert action.metadata["agent_name"] == "codex"


def test_records_tool_end_as_tool_result_with_stable_parent_id():
    graph = FakeActionsGraph()
    connector = ActionsGraphConnector(graph)

    connector.on_event(
        ToolStartEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:00+00:00",
            tool_name="Read",
            tool_use_id="tool-1",
        )
    )
    connector.on_event(
        ToolEndEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:01+00:00",
            tool_name="Read",
            tool_use_id="tool-1",
            result="ok",
        )
    )

    tool_call = graph.actions[0]
    tool_result = graph.actions[1]
    assert isinstance(tool_result, ToolResult)
    assert tool_result.tool_name == "Read"
    assert tool_result.content == "ok"
    assert tool_result.parent_action_id == tool_call.action_id


def test_records_tool_error_as_failed_tool_result():
    graph = FakeActionsGraph()
    connector = ActionsGraphConnector(graph)

    connector.on_event(
        ToolEndEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:01+00:00",
            tool_name="Write",
            tool_use_id="tool-2",
            is_error=True,
            error_message="denied",
        )
    )

    action = graph.actions[0]
    assert isinstance(action, ToolResult)
    assert action.status == ActionStatus.FAILED
    assert action.is_error is True
    assert action.error_message == "denied"


def test_records_message_and_error_events():
    graph = FakeActionsGraph()
    connector = ActionsGraphConnector(graph)

    connector.on_event(
        MessageEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:00+00:00",
            role="assistant",
            content="done",
            model="gpt-5",
        )
    )
    connector.on_event(
        ErrorOccurredEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:01+00:00",
            error_type="runtime",
            error_message="boom",
            error_details={"code": "E"},
            recoverable=False,
        )
    )

    assert isinstance(graph.actions[0], Message)
    assert graph.actions[0].model == "gpt-5"
    assert isinstance(graph.actions[1], ErrorEvent)
    assert graph.actions[1].error_details == {"code": "E"}
    assert graph.actions[1].recoverable is False


def test_records_session_end():
    graph = FakeActionsGraph()
    connector = ActionsGraphConnector(graph)

    connector.on_event(
        SessionEndEvent(
            session_id="session-1",
            timestamp="2026-01-01T00:00:00+00:00",
            status="completed",
            total_input_tokens=10,
            total_output_tokens=5,
        )
    )

    assert graph.ended_sessions == [
        (
            "session-1",
            {
                "status": ActionStatus.COMPLETED,
                "total_cost_usd": None,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
            },
        )
    ]
