"""End-to-end tests for actions_graph with Memgraph.

These tests require a running Memgraph instance.
"""

import json

import pytest

from actions_graph import (
    ActionsGraph,
    ActionStatus,
    ActionType,
    Agent,
    MessageRole,
    Session,
    ToolCall,
    ToolResult,
)
from actions_graph.connector import ActionsGraphConnector
from agent_context_graph import AgentLink
from agent_context_graph.adapters.antigravity_cli import AntigravityCLIHooksAdapter
from agent_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter
from agent_context_graph.adapters.codex import CodexHooksAdapter
from agent_context_graph.adapters.copilot_cli import CopilotCLIHooksAdapter
from agent_context_graph.adapters.cursor import CursorHooksAdapter
from agent_context_graph.adapters.grok import GrokHooksAdapter
from agent_context_graph.adapters.opencode import OpenCodeHooksAdapter


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
        )
        graph.create_session(session)

        retrieved = graph.get_session("test-session-001")
        assert retrieved is not None
        assert retrieved.session_id == "test-session-001"
        assert retrieved.model == "claude-sonnet-4-20250514"

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
            )
            graph.create_session(session)

        sessions = graph.list_sessions()
        assert len(sessions) == 3


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


_CWD = "/test/project"
_LS = {"command": "ls"}


def _runtime_case(adapter_class, source_sdk, payloads, *, expected_result="README.md", **marks):
    return pytest.param(adapter_class, source_sdk, payloads, expected_result, id=source_sdk, **marks)


# One session start, tool start, and tool end per runtime, each in that
# runtime's own documented payload shape -- so field-name and casing drift in
# an adapter fails against a real graph, not just a recording connector.
_RUNTIME_CASES = [
    _runtime_case(
        CodexHooksAdapter,
        "codex",
        [
            {"hook_event_name": "SessionStart", "session_id": "S", "cwd": _CWD, "model": "gpt-5", "source": "startup"},
            {
                "hook_event_name": "PreToolUse",
                "session_id": "S",
                "tool_name": "Bash",
                "tool_input": _LS,
                "tool_use_id": "t1",
            },
            {
                "hook_event_name": "PostToolUse",
                "session_id": "S",
                "tool_name": "Bash",
                "tool_input": _LS,
                "tool_response": "README.md",
                "tool_use_id": "t1",
            },
        ],
    ),
    _runtime_case(
        ClaudeCodeHooksAdapter,
        "claude-code",
        [
            {"hook_event_name": "SessionStart", "session_id": "S", "cwd": _CWD, "source": "startup"},
            {
                "hook_event_name": "PreToolUse",
                "session_id": "S",
                "tool_name": "Bash",
                "tool_input": _LS,
                "tool_use_id": "t1",
            },
            {
                "hook_event_name": "PostToolUse",
                "session_id": "S",
                "tool_name": "Bash",
                "tool_input": _LS,
                "tool_response": {"stdout": "README.md", "stderr": "", "interrupted": False},
                "tool_use_id": "t1",
            },
        ],
    ),
    _runtime_case(
        CopilotCLIHooksAdapter,
        "copilot-cli",
        # Native camelCase payloads carry no event name; the runner injects
        # it from --event-name, reproduced here as hook_event_name. No tool id.
        [
            {"hook_event_name": "sessionStart", "sessionId": "S", "timestamp": 1, "cwd": _CWD, "source": "new"},
            {
                "hook_event_name": "preToolUse",
                "sessionId": "S",
                "cwd": _CWD,
                "toolName": "bash",
                "toolArgs": '{"command": "ls"}',
            },
            {
                "hook_event_name": "postToolUse",
                "sessionId": "S",
                "cwd": _CWD,
                "toolName": "bash",
                "toolArgs": '{"command": "ls"}',
                "toolResult": {"resultType": "success", "textResultForLlm": "README.md"},
            },
        ],
    ),
    _runtime_case(
        CursorHooksAdapter,
        "cursor",
        [
            {"hook_event_name": "sessionStart", "conversation_id": "S", "session_id": "S", "workspace_roots": [_CWD]},
            {
                "hook_event_name": "preToolUse",
                "conversation_id": "S",
                "tool_name": "Shell",
                "tool_input": _LS,
                "tool_use_id": "t1",
            },
            {
                "hook_event_name": "postToolUse",
                "conversation_id": "S",
                "tool_name": "Shell",
                "tool_input": _LS,
                "tool_output": "README.md",
                "tool_use_id": "t1",
            },
        ],
    ),
    _runtime_case(
        OpenCodeHooksAdapter,
        "opencode",
        # As normalized by _opencode_plugin.js from OpenCode V2's hook and bus events.
        [
            {"hook_event_name": "session.created", "session_id": "S", "cwd": _CWD, "model": "claude-sonnet-5"},
            {
                "hook_event_name": "tool.execute.before",
                "session_id": "S",
                "tool_name": "bash",
                "tool_input": _LS,
                "tool_use_id": "t1",
            },
            {
                "hook_event_name": "tool.execute.after",
                "session_id": "S",
                "tool_name": "bash",
                "tool_input": _LS,
                "tool_use_id": "t1",
                # V2's result shape, as captured from a live OpenCode 2.0.18 session.
                "tool_result": {
                    "output": {"exit": 0, "truncated": False, "output": "README.md", "status": "completed"},
                    "content": [{"type": "text", "text": "README.md"}],
                    "metadata": {"status": "completed", "truncated": False, "exit": 0},
                },
                "is_error": False,
            },
        ],
    ),
    _runtime_case(
        AntigravityCLIHooksAdapter,
        "antigravity-cli",
        # As captured from a live agy 1.2.13 session: no event name (injected
        # via --event-name), no tool-call id (stepIdx pairs the two), and no
        # tool result, so the recorded result is empty.
        [
            {
                "hook_event_name": "PreInvocation",
                "conversationId": "S",
                "modelName": "gemini-3.8-flash-high",
                "workspacePaths": [_CWD],
                "invocationNum": 0,
                "initialNumSteps": 1,
            },
            {
                "hook_event_name": "PreToolUse",
                "conversationId": "S",
                "workspacePaths": [_CWD],
                "stepIdx": 2,
                "toolCall": {"name": "run_command", "args": _LS},
            },
            {
                "hook_event_name": "PostToolUse",
                "conversationId": "S",
                "workspacePaths": [_CWD],
                "stepIdx": 2,
                "toolCall": {"name": "run_command", "args": _LS},
                "error": "",
            },
        ],
        expected_result=None,
    ),
    _runtime_case(
        GrokHooksAdapter,
        "grok",
        # As captured from a live grok 1.0.40 session: every field arrives in
        # both camelCase and snake_case, and tool results are typed objects.
        [
            {"hook_event_name": "SessionStart", "session_id": "S", "sessionId": "S", "cwd": _CWD, "source": "new"},
            {
                "hook_event_name": "PreToolUse",
                "session_id": "S",
                "tool_name": "run_terminal_command",
                "tool_input": _LS,
                "tool_use_id": "call-1",
            },
            {
                "hook_event_name": "PostToolUse",
                "session_id": "S",
                "tool_name": "run_terminal_command",
                "tool_input": _LS,
                "tool_response": {"type": "Bash", "output_for_prompt": "README.md", "exit_code": 0},
                "tool_use_id": "call-1",
            },
        ],
    ),
]


def _with_session_id(payload: dict, session_id: str) -> dict:
    return {key: session_id if value == "S" else value for key, value in payload.items()}


@pytest.mark.parametrize(("adapter_class", "source_sdk", "payloads", "expected_result"), _RUNTIME_CASES)
def test_runtime_hook_tool_events_persist_as_actions(
    graph: ActionsGraph,
    adapter_class,
    source_sdk: str,
    payloads: list[dict],
    expected_result,
):
    """Every command-hook runtime persists its native tool activity in Memgraph."""
    session_id = f"{source_sdk}-hook-e2e"
    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
    adapter = adapter_class(link)

    for payload in payloads:
        adapter.handle_payload(_with_session_id(payload, session_id))

    session = graph.get_session(session_id)
    assert session is not None
    assert session.working_directory == _CWD

    actions = graph.get_session_actions(session_id)
    assert [action.action_type for action in actions] == [ActionType.TOOL_CALL, ActionType.TOOL_RESULT]
    tool_call, tool_result = actions
    assert isinstance(tool_call, ToolCall)
    assert isinstance(tool_result, ToolResult)
    assert tool_call.tool_name == tool_result.tool_name
    assert tool_call.tool_input == _LS
    assert tool_result.content == expected_result
    assert tool_result.is_error is False
    assert all(action.metadata["source_sdk"] == source_sdk for action in actions)
    # The result must be linked to its call even when the runtime sends no
    # tool-call id (Copilot).
    linked = graph._db.query(
        "MATCH (c:ToolCall {action_id: $call})-[:PARENT_OF]->(r:ToolResult {action_id: $result}) RETURN count(*) AS n",
        params={"call": tool_call.action_id, "result": tool_result.action_id},
    )
    assert linked[0]["n"] == 1


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


class TestAgentOperations:
    """Tests for the Agent node lifecycle, SPAWNED inference, and containment."""

    def test_start_and_end_agent(self, graph: ActionsGraph):
        session = Session(session_id="agent-lifecycle-session")
        graph.create_session(session)

        graph.start_agent(Agent(agent_id="agent-1", agent_type="Explore", session_id="agent-lifecycle-session"))
        fetched = graph.get_agent("agent-1")
        assert fetched is not None
        assert fetched.status == ActionStatus.IN_PROGRESS
        assert fetched.session_id == "agent-lifecycle-session"

        graph.end_agent("agent-1", last_assistant_message="Found it")
        ended = graph.get_agent("agent-1")
        assert ended is not None
        assert ended.status == ActionStatus.COMPLETED
        assert ended.last_assistant_message == "Found it"

    def test_agent_unconditionally_linked_to_session(self, graph: ActionsGraph):
        session = Session(session_id="unconditional-agent-session")
        graph.create_session(session)

        # No open Task call exists -- SPAWNED can't be inferred, but HAS_AGENT
        # must still exist so an ambiguous Agent stays reachable.
        graph.start_agent(
            Agent(agent_id="agent-orphan", agent_type="Explore", session_id="unconditional-agent-session")
        )

        agents = graph.get_session_agents("unconditional-agent-session")
        assert [a.agent_id for a in agents] == ["agent-orphan"]

    def test_spawned_inference_links_unambiguous_open_task_call(self, graph: ActionsGraph):
        session = Session(session_id="spawn-session")
        graph.create_session(session)

        spawning_call = graph.record_tool_call(
            session_id="spawn-session",
            tool_name="Task",
            tool_input={"subagent_type": "Explore"},
            tool_use_id="task-1",
        )

        graph.start_agent(Agent(agent_id="agent-2", agent_type="Explore", session_id="spawn-session"))

        rows = graph._db.query(
            "MATCH (parent:Action {action_id: $action_id})-[:SPAWNED]->(a:Agent {agent_id: $agent_id}) "
            "RETURN count(*) AS c",
            params={"action_id": spawning_call.action_id, "agent_id": "agent-2"},
        )
        assert rows[0]["c"] == 1

    def test_spawned_inference_matches_the_real_tool_name_agent_not_task(self, graph: ActionsGraph):
        """Verified against a real live Claude Code session (2026-08-17):
        the tool that actually launches a subagent reports tool_name "Agent" in
        hook payloads -- "Task" is how Claude Code's CLI/UI refers to it, not
        what shows up in PreToolUse/PostToolUse. Locks in the fix to
        agent_spawning_tool_names so it can't silently regress back to "Task"-only."""
        session = Session(session_id="spawn-real-tool-name-session")
        graph.create_session(session)

        spawning_call = graph.record_tool_call(
            session_id="spawn-real-tool-name-session",
            tool_name="Agent",
            tool_input={"subagent_type": "Explore"},
            tool_use_id="agent-tool-1",
        )

        graph.start_agent(Agent(agent_id="agent-3", agent_type="Explore", session_id="spawn-real-tool-name-session"))

        rows = graph._db.query(
            "MATCH (parent:Action {action_id: $action_id})-[:SPAWNED]->(a:Agent {agent_id: $agent_id}) "
            "RETURN count(*) AS c",
            params={"action_id": spawning_call.action_id, "agent_id": "agent-3"},
        )
        assert rows[0]["c"] == 1

    def test_spawned_inference_disambiguates_by_agent_type(self, graph: ActionsGraph):
        session = Session(session_id="spawn-disambiguate-session")
        graph.create_session(session)

        explore_call = graph.record_tool_call(
            session_id="spawn-disambiguate-session",
            tool_name="Task",
            tool_input={"subagent_type": "Explore"},
            tool_use_id="task-explore",
        )
        graph.record_tool_call(
            session_id="spawn-disambiguate-session",
            tool_name="Task",
            tool_input={"subagent_type": "general-purpose"},
            tool_use_id="task-general",
        )

        graph.start_agent(
            Agent(agent_id="agent-explore", agent_type="Explore", session_id="spawn-disambiguate-session")
        )

        rows = graph._db.query(
            "MATCH (parent:Action)-[:SPAWNED]->(a:Agent {agent_id: $agent_id}) RETURN parent.action_id AS action_id",
            params={"agent_id": "agent-explore"},
        )
        assert rows[0]["action_id"] == explore_call.action_id

    def test_spawned_inference_returns_none_on_genuine_ambiguity(self, graph: ActionsGraph):
        session = Session(session_id="spawn-ambiguous-session")
        graph.create_session(session)

        graph.record_tool_call(
            session_id="spawn-ambiguous-session",
            tool_name="Task",
            tool_input={"subagent_type": "Explore"},
            tool_use_id="task-a",
        )
        graph.record_tool_call(
            session_id="spawn-ambiguous-session",
            tool_name="Task",
            tool_input={"subagent_type": "Explore"},
            tool_use_id="task-b",
        )

        graph.start_agent(Agent(agent_id="agent-ambiguous", agent_type="Explore", session_id="spawn-ambiguous-session"))

        rows = graph._db.query(
            "MATCH ()-[:SPAWNED]->(a:Agent {agent_id: $agent_id}) RETURN count(*) AS c",
            params={"agent_id": "agent-ambiguous"},
        )
        assert rows[0]["c"] == 0

    def test_get_session_actions_includes_actions_nested_under_an_agent(self, graph: ActionsGraph):
        session = Session(session_id="nested-actions-session")
        graph.create_session(session)

        graph.record_message(
            session_id="nested-actions-session",
            role=MessageRole.USER,
            content="top-level message",
        )
        graph.start_agent(Agent(agent_id="agent-nested", agent_type="Explore", session_id="nested-actions-session"))
        graph.record_action(
            ToolCall(
                session_id="nested-actions-session",
                tool_name="Read",
                tool_input={"file_path": "a.py"},
            ),
            container_agent_id="agent-nested",
        )

        actions = graph.get_session_actions("nested-actions-session")
        assert len(actions) == 2
        tool_names = {getattr(a, "tool_name", None) for a in actions}
        assert "Read" in tool_names

    def test_followed_by_scoped_per_agent_container(self, graph: ActionsGraph):
        session = Session(session_id="followed-by-session")
        graph.create_session(session)

        graph.start_agent(Agent(agent_id="agent-chain", agent_type="Explore", session_id="followed-by-session"))
        first = graph.record_action(
            ToolCall(session_id="followed-by-session", tool_name="Read", tool_input={}),
            container_agent_id="agent-chain",
        )
        second = graph.record_action(
            ToolCall(session_id="followed-by-session", tool_name="Write", tool_input={}),
            container_agent_id="agent-chain",
        )
        # A top-level action recorded after the agent's actions must not
        # join the agent's own FOLLOWED_BY chain.
        graph.record_message(
            session_id="followed-by-session",
            role=MessageRole.USER,
            content="top-level, separate chain",
        )

        rows = graph._db.query(
            "MATCH (a:Action {action_id: $first})-[:FOLLOWED_BY]->(b:Action) RETURN b.action_id AS next_id",
            params={"first": first.action_id},
        )
        assert rows[0]["next_id"] == second.action_id

    def test_record_action_falls_back_to_session_when_agent_node_does_not_exist(self, graph: ActionsGraph):
        """Some adapters (e.g. OpenAI Agents SDK) set an agent name for every
        running agent, not just genuine nested subagents, and no Agent node may
        exist for it at all. A hard MATCH on the Agent would silently orphan
        the Action (no HAS_ACTION link to anything); it must fall back to the
        Session instead."""
        session = Session(session_id="no-agent-node-session")
        graph.create_session(session)

        action = graph.record_action(
            ToolCall(session_id="no-agent-node-session", tool_name="Read", tool_input={}),
            container_agent_id="top-level-agent-with-no-node",
        )

        rows = graph._db.query(
            "MATCH (:Session {session_id: $sid})-[:HAS_ACTION]->(a:Action {action_id: $action_id}) RETURN count(a) AS c",
            params={"sid": "no-agent-node-session", "action_id": action.action_id},
        )
        assert rows[0]["c"] == 1


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


def _connector(graph: ActionsGraph) -> AgentLink:
    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
    return link


def test_session_start_after_the_first_event_still_records_start_fields(graph: ActionsGraph):
    """Copilot delivers the first prompt before sessionStart (#368)."""
    from agent_context_graph.events import MessageEvent, SessionStartEvent

    link = _connector(graph)
    link.emit(MessageEvent(session_id="late-start", role="user", content="hi", source_sdk="copilot-cli"))
    link.emit(
        SessionStartEvent(session_id="late-start", working_directory=_CWD, model="gpt-5", source_sdk="copilot-cli")
    )

    session = graph.get_session("late-start")
    assert session is not None
    assert session.working_directory == _CWD
    assert session.model == "gpt-5"


def test_concurrent_first_events_create_one_session_and_keep_every_event():
    """OpenCode runs hooks in parallel; each hook is its own process and connection (#370)."""
    import threading

    from agent_context_graph.events import MessageEvent, SessionStartEvent, ToolStartEvent

    setup_graph = ActionsGraph()
    setup_graph.clear()
    sessions = [f"race-{index}" for index in range(10)]

    def emit(event) -> None:
        # A fresh graph per event, like one hook subprocess per event.
        _connector(ActionsGraph()).emit(event)

    threads = []
    for session_id in sessions:
        for event in (
            SessionStartEvent(session_id=session_id, working_directory=_CWD, source_sdk="opencode"),
            MessageEvent(session_id=session_id, role="user", content="go", source_sdk="opencode"),
            ToolStartEvent(session_id=session_id, tool_name="bash", tool_use_id="t1", source_sdk="opencode"),
        ):
            threads.append(threading.Thread(target=emit, args=(event,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for session_id in sessions:
        rows = setup_graph._db.query(
            "MATCH (s:Session {session_id: $sid}) RETURN count(s) AS n", params={"sid": session_id}
        )
        assert rows[0]["n"] == 1
        assert len(setup_graph.get_session_actions(session_id)) == 2
    setup_graph.clear()


def test_results_without_ids_pair_with_the_oldest_open_call_of_the_same_tool(graph: ActionsGraph):
    """Copilot sends no tool-call id; parallel calls of one tool pair in start order."""
    from agent_context_graph.events import ToolEndEvent, ToolStartEvent

    link = _connector(graph)
    for tool_name, tool_input in (
        ("bash", {"command": "ls"}),
        ("view", {"path": "README.md"}),
        ("bash", {"command": "pwd"}),
    ):
        link.emit(
            ToolStartEvent(session_id="no-ids", source_sdk="copilot-cli", tool_name=tool_name, tool_input=tool_input)
        )
    for tool_name, result in (("bash", "README.md"), ("view", "# Demo"), ("bash", "/work")):
        link.emit(ToolEndEvent(session_id="no-ids", source_sdk="copilot-cli", tool_name=tool_name, result=result))

    pairs = graph._db.query(
        """
        MATCH (s:Session {session_id: 'no-ids'})-[:HAS_ACTION]->(c:ToolCall)-[:PARENT_OF]->(r:ToolResult)
        RETURN c.properties AS call, r.properties AS result
        """
    )
    linked = sorted(
        ((json.loads(row["call"])["tool_input"], json.loads(row["result"])["content"]) for row in pairs), key=str
    )
    assert linked == sorted(
        [({"command": "ls"}, "README.md"), ({"path": "README.md"}, "# Demo"), ({"command": "pwd"}, "/work")],
        key=str,
    )
