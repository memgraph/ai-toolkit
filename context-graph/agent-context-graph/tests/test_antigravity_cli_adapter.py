"""Tests for the Antigravity CLI command-hook runtime adapter.

Payload shapes are trimmed from a live agy 1.2.13 session.
"""

import json

from agent_context_graph import AgentLink
from agent_context_graph.adapters.antigravity_cli import PLUGIN, AntigravityCLIHooksAdapter, init
from agent_context_graph.events import ErrorOccurredEvent, SessionStartEvent, ToolEndEvent, ToolStartEvent
from agent_context_graph.protocols import GraphConnector

_COMMON = {
    "conversationId": "conv-1",
    "modelName": "gemini-3.8-flash-high",
    "workspacePaths": ["/work/project"],
    "transcriptPath": "/home/.gemini/antigravity-cli/brain/conv-1/transcript.jsonl",
}
_LS = {"name": "run_command", "args": {"CommandLine": "ls", "Cwd": "/work/project", "toolSummary": "Directory listing"}}


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def _adapter():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    return AntigravityCLIHooksAdapter(link), connector


def test_first_invocation_starts_the_session_and_later_ones_do_not():
    adapter, connector = _adapter()

    adapter.handle_payload({"hook_event_name": "PreInvocation", **_COMMON, "invocationNum": 0, "initialNumSteps": 1})
    adapter.handle_payload({"hook_event_name": "PreInvocation", **_COMMON, "invocationNum": 1, "initialNumSteps": 3})

    (session_start,) = connector.events
    assert isinstance(session_start, SessionStartEvent)
    assert session_start.session_id == "conv-1"
    assert session_start.model == "gemini-3.8-flash-high"
    assert session_start.working_directory == "/work/project"
    assert session_start.source_sdk == "antigravity-cli"


def test_tool_start_and_end_pair_on_step_index():
    adapter, connector = _adapter()

    adapter.handle_payload({"hook_event_name": "PreToolUse", **_COMMON, "stepIdx": 2, "toolCall": _LS})
    adapter.handle_payload({"hook_event_name": "PostToolUse", **_COMMON, "stepIdx": 2, "toolCall": _LS, "error": ""})
    adapter.handle_payload(
        {"hook_event_name": "PostToolUse", **_COMMON, "stepIdx": 4, "toolCall": _LS, "error": "exit status 1"}
    )

    tool_start, tool_end, failed_end = connector.events
    assert isinstance(tool_start, ToolStartEvent)
    assert tool_start.tool_name == "run_command"
    assert tool_start.tool_input == _LS["args"]
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.tool_use_id == tool_start.tool_use_id == "step-2"
    assert tool_end.is_error is False
    assert tool_end.error_message is None
    assert isinstance(failed_end, ToolEndEvent)
    assert failed_end.is_error is True
    assert failed_end.error_message == "exit status 1"


def test_stop_records_only_failed_executions():
    adapter, connector = _adapter()
    stop = {"hook_event_name": "Stop", **_COMMON, "executionNum": 0, "fullyIdle": True}

    assert adapter.handle_payload({**stop, "terminationReason": "NO_TOOL_CALL", "error": ""}) == []
    adapter.handle_payload({**stop, "terminationReason": "ERROR", "error": "model unavailable"})

    (error,) = connector.events
    assert isinstance(error, ErrorOccurredEvent)
    assert error.error_type == "ERROR"
    assert error.error_message == "model unavailable"


def test_responses_never_answer_a_permission_decision():
    assert PLUGIN.response_for_payload({"hook_event_name": "PostToolUse"}) == {}
    # Empty PreToolUse output leaves agy's own permission prompt in charge.
    assert PLUGIN.response_for_payload({"hook_event_name": "PreToolUse"}) is None
    assert PLUGIN.response_for_payload({"hook_event_name": "Stop"}) is None


def test_init_writes_a_named_group_with_flat_lifecycle_hooks(tmp_path):
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    (agents_dir / "hooks.json").write_text(json.dumps({"my-linter": {"PostToolUse": []}}), encoding="utf-8")

    init(tmp_path, ["actions-graph"], hook_command="capture", timeout=12)

    config = json.loads((agents_dir / "hooks.json").read_text())
    assert config["my-linter"] == {"PostToolUse": []}
    group = config["agent-context-graph"]
    assert set(group) == {"PreInvocation", "PreToolUse", "PostToolUse", "Stop"}
    assert group["PreToolUse"] == [
        {"hooks": [{"command": "capture --event-name PreToolUse", "type": "command", "timeout": 12}]}
    ]
    # agy rejects a nested entry on a lifecycle event, and with it the whole file.
    assert group["Stop"] == [{"command": "capture --event-name Stop", "type": "command", "timeout": 12}]
    assert group["PreInvocation"] == [
        {"command": "capture --event-name PreInvocation", "type": "command", "timeout": 12}
    ]
