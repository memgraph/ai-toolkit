"""Tests for standalone Claude SDK action hooks."""

from types import SimpleNamespace

import pytest

from actions_graph.hooks import ActionTracker, create_message_handler, create_tracking_hooks
from actions_graph.models import (
    ActionStatus,
    ErrorEvent,
    Message,
    PermissionRequest,
    RateLimitEvent,
    Session,
    SubagentEvent,
    ToolCall,
    ToolResult,
)


class FakeActionsGraph:
    def __init__(self):
        self.sessions: list[Session] = []
        self.actions = []
        self.ended_sessions = []

    def create_session(self, session: Session):
        self.sessions.append(session)
        return session

    def record_action(self, action):
        self.actions.append(action)
        return action

    def end_session(self, session_id: str, **kwargs):
        self.ended_sessions.append((session_id, kwargs))
        return None


@pytest.mark.asyncio
async def test_tool_hooks_record_call_result_and_failure():
    graph = FakeActionsGraph()
    tracker = ActionTracker(graph, "session-1")

    await tracker.pre_tool_use(
        {"tool_name": "Read", "tool_input": {"file_path": "a.py"}, "tool_use_id": "tool-1"},
        None,
        {},
    )
    await tracker.post_tool_use(
        {"tool_name": "Read", "tool_response": {"content": "ok"}, "tool_use_id": "tool-1"},
        None,
        {},
    )
    await tracker.post_tool_use_failure(
        {"tool_name": "Write", "tool_use_id": "tool-1", "error": "denied", "is_interrupt": True},
        None,
        {},
    )

    assert isinstance(graph.actions[0], ToolCall)
    assert graph.actions[0].tool_name == "Read"
    assert isinstance(graph.actions[1], ToolResult)
    assert graph.actions[1].parent_action_id == graph.actions[0].action_id
    assert isinstance(graph.actions[2], ErrorEvent)
    assert graph.actions[2].parent_action_id == graph.actions[0].action_id
    assert graph.actions[2].recoverable is False


@pytest.mark.asyncio
async def test_message_subagent_permission_notification_and_stop_hooks():
    graph = FakeActionsGraph()
    tracker = ActionTracker(graph, "session-1")

    await tracker.user_prompt_submit({"prompt": "hello", "cwd": "/repo"}, None, {})
    await tracker.subagent_start({"agent_id": "agent-1", "agent_type": "reviewer"}, None, {})
    await tracker.subagent_stop(
        {"agent_id": "agent-1", "agent_type": "reviewer", "agent_transcript_path": "/tmp/t"},
        None,
        {},
    )
    await tracker.permission_request({"tool_name": "Bash", "tool_input": {"cmd": "ls"}}, None, {})
    await tracker.notification({"message": "notice", "notification_type": "info"}, None, {})
    await tracker.stop({}, None, {})

    assert isinstance(graph.actions[0], Message)
    assert graph.actions[0].content == "hello"
    assert isinstance(graph.actions[1], SubagentEvent)
    assert graph.actions[1].action_type.value == "subagent_start"
    assert isinstance(graph.actions[2], SubagentEvent)
    assert graph.actions[2].action_type.value == "subagent_stop"
    assert graph.actions[2].parent_action_id == graph.actions[1].action_id
    assert isinstance(graph.actions[3], PermissionRequest)
    assert isinstance(graph.actions[4], Message)
    assert graph.ended_sessions == [("session-1", {"status": ActionStatus.COMPLETED})]


def test_create_tracking_hooks_creates_session_and_hook_map():
    graph = FakeActionsGraph()

    hooks = create_tracking_hooks(graph, "session-1", session_kwargs={})

    assert graph.sessions[0].session_id == "session-1"
    assert set(hooks) == {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "UserPromptSubmit",
        "SubagentStart",
        "SubagentStop",
        "PermissionRequest",
        "Notification",
        "Stop",
    }


def test_message_handler_records_assistant_result_and_rate_limit():
    graph = FakeActionsGraph()
    handler = create_message_handler(graph, "session-1")

    handler(
        SimpleNamespace(
            type="AssistantMessage",
            content=[SimpleNamespace(text="hi"), SimpleNamespace(thinking="think")],
            model="claude",
            usage={"input_tokens": 1},
            message_id="msg-1",
        )
    )
    handler(
        SimpleNamespace(
            type="ResultMessage",
            subtype="success",
            total_cost_usd=0.01,
            usage={"input_tokens": 2, "output_tokens": 3},
        )
    )
    handler(
        SimpleNamespace(
            type="RateLimitEvent",
            rate_limit_info=SimpleNamespace(
                status="allowed_warning",
                rate_limit_type="five_hour",
                resets_at=123,
                utilization=0.8,
            ),
        )
    )

    assert isinstance(graph.actions[0], Message)
    assert graph.actions[0].content == [{"type": "text", "text": "hi"}, {"type": "thinking", "thinking": "think"}]
    assert graph.ended_sessions[0][1]["total_input_tokens"] == 2
    assert graph.ended_sessions[0][1]["total_output_tokens"] == 3
    assert isinstance(graph.actions[1], RateLimitEvent)
