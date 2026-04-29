"""End-to-end tests for actions_graph with Memgraph.

These tests require a running Memgraph instance.
"""

import pytest
from actions_graph import (
    ActionsGraph,
    ActionStatus,
    ActionType,
    MessageRole,
    Session,
    ToolCall,
)


@pytest.fixture
def graph():
    """Create a fresh ActionsGraph instance for testing."""
    import contextlib

    g = ActionsGraph()
    with contextlib.suppress(Exception):
        g.setup()  # Constraints may already exist
    g.clear()
    yield g
    g.clear()


class TestActionsGraphSetup:
    """Tests for ActionsGraph setup and teardown."""

    def test_setup_and_drop(self, graph: ActionsGraph):
        """Test setting up and dropping the schema."""
        # Schema should already be set up by fixture
        # Just verify we can create a session
        session = Session(session_id="test-setup-session")
        graph.create_session(session)
        retrieved = graph.get_session("test-setup-session")
        assert retrieved is not None


class TestSessionOperations:
    """Tests for session CRUD operations."""

    def test_create_and_get_session(self, graph: ActionsGraph):
        """Test creating and retrieving a session."""
        session = Session(
            session_id="test-session-001",
            model="claude-sonnet-4-20250514",
            working_directory="/test/project",
            tags=["test", "e2e"],
        )
        graph.create_session(session)

        retrieved = graph.get_session("test-session-001")
        assert retrieved is not None
        assert retrieved.session_id == "test-session-001"
        assert retrieved.model == "claude-sonnet-4-20250514"
        assert "test" in retrieved.tags

    def test_end_session(self, graph: ActionsGraph):
        """Test ending a session."""
        session = Session(session_id="test-end-session")
        graph.create_session(session)

        ended = graph.end_session(
            "test-end-session",
            status=ActionStatus.COMPLETED,
            total_cost_usd=0.05,
            total_input_tokens=1000,
            total_output_tokens=500,
        )

        assert ended is not None
        assert ended.status == ActionStatus.COMPLETED
        assert ended.ended_at is not None
        assert ended.total_cost_usd == 0.05

    def test_list_sessions(self, graph: ActionsGraph):
        """Test listing sessions."""
        for i in range(3):
            session = Session(
                session_id=f"list-test-{i}",
                tags=["list-test"],
            )
            graph.create_session(session)

        sessions = graph.list_sessions(tag="list-test")
        assert len(sessions) == 3

    def test_forked_session(self, graph: ActionsGraph):
        """Test creating a forked session."""
        # Create parent session
        parent = Session(session_id="parent-session")
        graph.create_session(parent)

        # Create forked session
        forked = Session(
            session_id="forked-session",
            parent_session_id="parent-session",
        )
        graph.create_session(forked)

        retrieved = graph.get_session("forked-session")
        assert retrieved is not None
        assert retrieved.parent_session_id == "parent-session"


class TestActionOperations:
    """Tests for action CRUD operations."""

    def test_record_tool_call(self, graph: ActionsGraph):
        """Test recording a tool call."""
        session = Session(session_id="tool-call-session")
        graph.create_session(session)

        tool_call = graph.record_tool_call(
            session_id="tool-call-session",
            tool_name="Read",
            tool_input={"file_path": "/test/file.py"},
            tool_use_id="tool-001",
        )

        assert tool_call.tool_name == "Read"
        assert tool_call.action_type == ActionType.TOOL_CALL

        # Retrieve and verify
        retrieved = graph.get_action(tool_call.action_id)
        assert retrieved is not None
        assert isinstance(retrieved, ToolCall)
        assert retrieved.tool_name == "Read"

    def test_record_tool_result(self, graph: ActionsGraph):
        """Test recording a tool result."""
        session = Session(session_id="tool-result-session")
        graph.create_session(session)

        result = graph.record_tool_result(
            session_id="tool-result-session",
            tool_use_id="tool-001",
            tool_name="Read",
            content="file contents",
        )

        assert result.content == "file contents"
        assert result.is_error is False

    def test_record_message(self, graph: ActionsGraph):
        """Test recording a message."""
        session = Session(session_id="message-session")
        graph.create_session(session)

        message = graph.record_message(
            session_id="message-session",
            role=MessageRole.USER,
            content="Hello!",
        )

        assert message.role == MessageRole.USER
        assert message.action_type == ActionType.USER_MESSAGE

    def test_action_sequence(self, graph: ActionsGraph):
        """Test that actions form a sequence with FOLLOWED_BY."""
        session = Session(session_id="sequence-session")
        graph.create_session(session)

        # Record multiple actions
        graph.record_message(
            session_id="sequence-session",
            role=MessageRole.USER,
            content="First message",
        )
        graph.record_tool_call(
            session_id="sequence-session",
            tool_name="Read",
            tool_input={"file_path": "/test.py"},
        )
        graph.record_message(
            session_id="sequence-session",
            role=MessageRole.ASSISTANT,
            content="Response",
        )

        # Get sequence
        sequence = graph.get_action_sequence("sequence-session")
        assert len(sequence) == 3

        # Verify FOLLOWED_BY relationships
        assert sequence[0]["next_action_id"] == sequence[1]["action_id"]
        assert sequence[1]["next_action_id"] == sequence[2]["action_id"]

    def test_get_session_actions(self, graph: ActionsGraph):
        """Test getting all actions for a session."""
        session = Session(session_id="get-actions-session")
        graph.create_session(session)

        # Record several actions
        for i in range(5):
            graph.record_tool_call(
                session_id="get-actions-session",
                tool_name=f"Tool{i}",
                tool_input={},
            )

        actions = graph.get_session_actions("get-actions-session")
        assert len(actions) == 5

    def test_filter_actions_by_type(self, graph: ActionsGraph):
        """Test filtering actions by type."""
        session = Session(session_id="filter-actions-session")
        graph.create_session(session)

        # Record mixed actions
        graph.record_message(
            session_id="filter-actions-session",
            role=MessageRole.USER,
            content="Message",
        )
        graph.record_tool_call(
            session_id="filter-actions-session",
            tool_name="Read",
            tool_input={},
        )
        graph.record_tool_call(
            session_id="filter-actions-session",
            tool_name="Write",
            tool_input={},
        )

        # Filter by tool call
        tool_calls = graph.get_session_actions(
            "filter-actions-session",
            action_type=ActionType.TOOL_CALL,
        )
        assert len(tool_calls) == 2


class TestAnalytics:
    """Tests for analytics queries."""

    def test_tool_usage_stats(self, graph: ActionsGraph):
        """Test getting tool usage statistics."""
        session = Session(session_id="stats-session")
        graph.create_session(session)

        # Record tool calls
        for _ in range(3):
            graph.record_tool_call(
                session_id="stats-session",
                tool_name="Read",
                tool_input={},
            )
        for _ in range(2):
            graph.record_tool_call(
                session_id="stats-session",
                tool_name="Write",
                tool_input={},
            )

        stats = graph.get_tool_usage_stats("stats-session")
        assert len(stats) == 2

        # Read should have more calls
        read_stats = next(s for s in stats if s["tool_name"] == "Read")
        assert read_stats["call_count"] == 3

    def test_session_summary(self, graph: ActionsGraph):
        """Test getting a session summary."""
        session = Session(session_id="summary-session")
        graph.create_session(session)

        # Add various actions
        graph.record_message(
            session_id="summary-session",
            role=MessageRole.USER,
            content="Hello",
        )
        graph.record_tool_call(
            session_id="summary-session",
            tool_name="Read",
            tool_input={},
        )
        graph.record_message(
            session_id="summary-session",
            role=MessageRole.ASSISTANT,
            content="Hi!",
        )

        summary = graph.get_session_summary("summary-session")
        assert summary["action_count"] == 3
        assert summary["user_message_count"] == 1
        assert summary["assistant_message_count"] == 1
        assert summary["tool_call_count"] == 1


class TestMCPTools:
    """Tests for MCP tool handling."""

    def test_mcp_tool_tracking(self, graph: ActionsGraph):
        """Test that MCP tools are correctly identified and tracked."""
        session = Session(session_id="mcp-session")
        graph.create_session(session)

        tool_call = graph.record_tool_call(
            session_id="mcp-session",
            tool_name="mcp__playwright__browser_click",
            tool_input={"selector": "button"},
        )

        assert tool_call.is_mcp is True
        assert tool_call.mcp_server == "playwright"

        # Verify in tool stats
        stats = graph.get_tool_usage_stats("mcp-session")
        assert len(stats) == 1
        assert stats[0]["is_mcp"] is True
        assert stats[0]["mcp_server"] == "playwright"
