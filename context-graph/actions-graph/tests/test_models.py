"""Tests for actions_graph.models module."""

import pytest
from actions_graph.models import (
    ActionStatus,
    ActionType,
    ActionValidationError,
    ErrorEvent,
    Message,
    MessageRole,
    PermissionRequest,
    RateLimitEvent,
    Session,
    StructuredOutput,
    SubagentEvent,
    ToolCall,
    ToolResult,
)


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Test creating a valid session."""
        session = Session(session_id="test-session-123")
        assert session.session_id == "test-session-123"
        assert session.status == ActionStatus.IN_PROGRESS
        assert session.started_at is not None
        assert session.ended_at is None

    def test_create_session_with_all_fields(self):
        """Test creating a session with all fields."""
        session = Session(
            session_id="test-session-456",
            model="claude-sonnet-4-20250514",
            working_directory="/path/to/project",
            git_branch="main",
            tags=["test", "demo"],
            metadata={"key": "value"},
        )
        assert session.model == "claude-sonnet-4-20250514"
        assert session.working_directory == "/path/to/project"
        assert session.git_branch == "main"
        assert session.tags == ["test", "demo"]
        assert session.metadata == {"key": "value"}

    def test_invalid_session_id(self):
        """Test that invalid session IDs raise an error."""
        with pytest.raises(ActionValidationError):
            Session(session_id="")

        with pytest.raises(ActionValidationError):
            Session(session_id="invalid session id with spaces")

    def test_forked_session(self):
        """Test creating a forked session."""
        session = Session(
            session_id="forked-session",
            parent_session_id="original-session",
        )
        assert session.parent_session_id == "original-session"


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            session_id="session-123",
            tool_name="Read",
            tool_input={"file_path": "/path/to/file.py"},
            tool_use_id="tool-use-001",
        )
        assert tool_call.tool_name == "Read"
        assert tool_call.tool_input == {"file_path": "/path/to/file.py"}
        assert tool_call.tool_use_id == "tool-use-001"
        assert tool_call.action_type == ActionType.TOOL_CALL

    def test_mcp_tool_detection(self):
        """Test automatic MCP tool detection."""
        tool_call = ToolCall(
            session_id="session-123",
            tool_name="mcp__playwright__browser_click",
            tool_input={"selector": "button"},
        )
        assert tool_call.is_mcp is True
        assert tool_call.mcp_server == "playwright"

    def test_non_mcp_tool(self):
        """Test non-MCP tool detection."""
        tool_call = ToolCall(
            session_id="session-123",
            tool_name="Write",
            tool_input={"content": "hello"},
        )
        assert tool_call.is_mcp is False
        assert tool_call.mcp_server is None


class TestToolResult:
    """Tests for ToolResult model."""

    def test_create_tool_result(self):
        """Test creating a tool result."""
        result = ToolResult(
            session_id="session-123",
            tool_use_id="tool-use-001",
            tool_name="Read",
            content="file contents here",
        )
        assert result.tool_use_id == "tool-use-001"
        assert result.content == "file contents here"
        assert result.is_error is False
        assert result.action_type == ActionType.TOOL_RESULT

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            session_id="session-123",
            tool_use_id="tool-use-002",
            tool_name="Write",
            is_error=True,
            error_message="Permission denied",
        )
        assert result.is_error is True
        assert result.error_message == "Permission denied"
        assert result.status == ActionStatus.FAILED


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        message = Message(
            session_id="session-123",
            role=MessageRole.USER,
            content="Hello, Claude!",
        )
        assert message.role == MessageRole.USER
        assert message.content == "Hello, Claude!"
        assert message.action_type == ActionType.USER_MESSAGE

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        message = Message(
            session_id="session-123",
            role=MessageRole.ASSISTANT,
            content=[{"type": "text", "text": "Hello!"}],
            model="claude-sonnet-4-20250514",
        )
        assert message.role == MessageRole.ASSISTANT
        assert message.action_type == ActionType.ASSISTANT_MESSAGE
        assert message.model == "claude-sonnet-4-20250514"

    def test_create_system_message(self):
        """Test creating a system message."""
        message = Message(
            session_id="session-123",
            role=MessageRole.SYSTEM,
            content="Session started",
        )
        assert message.action_type == ActionType.SYSTEM_MESSAGE


class TestStructuredOutput:
    """Tests for StructuredOutput model."""

    def test_create_structured_output(self):
        """Test creating a structured output."""
        output = StructuredOutput(
            session_id="session-123",
            output_type="code_review",
            output_data={"issues": [], "suggestions": ["Add tests"]},
            validation_passed=True,
        )
        assert output.output_type == "code_review"
        assert output.output_data["suggestions"] == ["Add tests"]
        assert output.action_type == ActionType.STRUCTURED_OUTPUT


class TestSubagentEvent:
    """Tests for SubagentEvent model."""

    def test_create_subagent_start(self):
        """Test creating a subagent start event."""
        event = SubagentEvent(
            session_id="session-123",
            agent_id="subagent-001",
            agent_type="code-reviewer",
            description="Review the code changes",
        )
        event.action_type = ActionType.SUBAGENT_START
        assert event.agent_id == "subagent-001"
        assert event.action_type == ActionType.SUBAGENT_START

    def test_create_subagent_stop(self):
        """Test creating a subagent stop event."""
        event = SubagentEvent(
            session_id="session-123",
            agent_id="subagent-001",
            agent_type="code-reviewer",
            result="Review completed successfully",
        )
        event.action_type = ActionType.SUBAGENT_STOP
        assert event.result == "Review completed successfully"


class TestErrorEvent:
    """Tests for ErrorEvent model."""

    def test_create_error_event(self):
        """Test creating an error event."""
        error = ErrorEvent(
            session_id="session-123",
            error_type="api_error",
            error_message="Rate limit exceeded",
            recoverable=True,
        )
        assert error.error_type == "api_error"
        assert error.error_message == "Rate limit exceeded"
        assert error.recoverable is True
        assert error.status == ActionStatus.FAILED
        assert error.action_type == ActionType.ERROR


class TestPermissionRequest:
    """Tests for PermissionRequest model."""

    def test_create_permission_request(self):
        """Test creating a permission request."""
        request = PermissionRequest(
            session_id="session-123",
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp/test"},
        )
        assert request.tool_name == "Bash"
        assert request.action_type == ActionType.PERMISSION_REQUEST


class TestRateLimitEvent:
    """Tests for RateLimitEvent model."""

    def test_create_rate_limit_event(self):
        """Test creating a rate limit event."""
        event = RateLimitEvent(
            session_id="session-123",
            rate_limit_status="allowed_warning",
            rate_limit_type="five_hour",
            utilization=0.8,
        )
        assert event.rate_limit_status == "allowed_warning"
        assert event.utilization == 0.8
        assert event.action_type == ActionType.RATE_LIMIT
