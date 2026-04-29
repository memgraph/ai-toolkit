"""Unit tests for the event models."""

from agent_link.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    EventType,
    HandoffEvent,
    LLMEndEvent,
    LLMStartEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)


class TestEventTypes:
    """Verify each event subclass has the correct event_type."""

    def test_session_start(self):
        e = SessionStartEvent(session_id="s1", model="gpt-4")
        assert e.event_type == EventType.SESSION_START
        assert e.model == "gpt-4"
        assert e.session_id == "s1"

    def test_session_end(self):
        e = SessionEndEvent(session_id="s1", status="failed", total_cost_usd=0.05)
        assert e.event_type == EventType.SESSION_END
        assert e.status == "failed"
        assert e.total_cost_usd == 0.05

    def test_tool_start(self):
        e = ToolStartEvent(
            session_id="s1",
            tool_name="Read",
            tool_input={"path": "/file.py"},
            tool_use_id="tu-1",
        )
        assert e.event_type == EventType.TOOL_START
        assert e.tool_name == "Read"
        assert e.tool_input == {"path": "/file.py"}

    def test_tool_end(self):
        e = ToolEndEvent(
            session_id="s1",
            tool_name="Read",
            tool_use_id="tu-1",
            result="contents",
            is_error=False,
        )
        assert e.event_type == EventType.TOOL_END
        assert e.result == "contents"
        assert not e.is_error

    def test_tool_end_error(self):
        e = ToolEndEvent(
            session_id="s1",
            tool_name="Write",
            is_error=True,
            error_message="Permission denied",
        )
        assert e.is_error
        assert e.error_message == "Permission denied"

    def test_agent_start(self):
        e = AgentStartEvent(
            session_id="s1",
            agent_name="Coder",
            agent_type="sub",
        )
        assert e.event_type == EventType.AGENT_START
        assert e.agent_name == "Coder"

    def test_agent_end(self):
        e = AgentEndEvent(
            session_id="s1",
            agent_name="Coder",
            output="done",
        )
        assert e.event_type == EventType.AGENT_END
        assert e.output == "done"

    def test_llm_start(self):
        e = LLMStartEvent(
            session_id="s1",
            agent_name="Assistant",
            system_prompt="You are helpful.",
        )
        assert e.event_type == EventType.LLM_START
        assert e.system_prompt == "You are helpful."

    def test_llm_end(self):
        e = LLMEndEvent(
            session_id="s1",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=200,
        )
        assert e.event_type == EventType.LLM_END
        assert e.input_tokens == 100
        assert e.output_tokens == 200

    def test_handoff(self):
        e = HandoffEvent(
            session_id="s1",
            from_agent="Planner",
            to_agent="Coder",
        )
        assert e.event_type == EventType.HANDOFF
        assert e.from_agent == "Planner"
        assert e.to_agent == "Coder"

    def test_message(self):
        e = MessageEvent(
            session_id="s1",
            role="user",
            content="Hello",
        )
        assert e.event_type == EventType.MESSAGE
        assert e.role == "user"

    def test_error(self):
        e = ErrorOccurredEvent(
            session_id="s1",
            error_type="tool_failure",
            error_message="timeout",
            recoverable=True,
        )
        assert e.event_type == EventType.ERROR
        assert e.recoverable

    def test_source_sdk_default(self):
        e = ToolStartEvent(session_id="s1", tool_name="X")
        assert e.source_sdk == ""

    def test_source_sdk_set(self):
        e = ToolStartEvent(session_id="s1", tool_name="X", source_sdk="claude")
        assert e.source_sdk == "claude"

    def test_timestamp_auto_generated(self):
        e = ToolStartEvent(session_id="s1", tool_name="X")
        assert e.timestamp  # non-empty ISO string
        assert "T" in e.timestamp

    def test_metadata_default(self):
        e = SessionStartEvent(session_id="s1")
        assert e.metadata == {}
