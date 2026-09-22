"""Tests for the GitHub Copilot CLI command-hook runtime adapter."""

import io
import json

from agent_context_graph import AgentLink
from agent_context_graph.adapters.copilot_cli import PLUGIN, CopilotCLIHooksAdapter, init
from agent_context_graph.events import AgentEndEvent, ErrorOccurredEvent, SessionEndEvent, ToolEndEvent
from agent_context_graph.hooks.runner import run_hook
from agent_context_graph.protocols import GraphConnector


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_copilot_vscode_compatible_payloads_emit_events():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = CopilotCLIHooksAdapter(link)

    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "copilot-1",
            "tool_name": "Read",
            "tool_input": {"path": "README.md"},
            "tool_result": {"result_type": "success", "text_result_for_llm": "contents"},
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "copilot-1",
            "agent_id": "research-1",
            "agent_type": "research",
            "last_assistant_message": "Done",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "ErrorOccurred",
            "session_id": "copilot-1",
            "error": {"name": "ToolError", "message": "failed", "stack": "trace"},
            "error_context": "tool_execution",
            "recoverable": True,
        }
    )

    tool_end, agent_end, error = connector.events
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.result == "contents"
    assert isinstance(agent_end, AgentEndEvent)
    assert agent_end.output == "Done"
    assert isinstance(error, ErrorOccurredEvent)
    assert error.error_type == "ToolError"
    assert error.error_message == "failed"


def test_copilot_init_writes_versioned_project_hook_file(tmp_path):
    init(tmp_path, ["actions-graph"], hook_command="capture", timeout=9)

    config = json.loads((tmp_path / ".github" / "hooks" / "agent-context-graph.json").read_text())
    assert config["version"] == 1
    assert config["hooks"]["preToolUse"][0] == {
        "command": "capture --event-name preToolUse",
        "type": "command",
        "timeoutSec": 9,
        "matcher": ".*",
    }
    assert "agentStop" not in config["hooks"]


def test_copilot_only_ends_session_for_session_end():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = CopilotCLIHooksAdapter(link)

    assert adapter.handle_payload({"hook_event_name": "agentStop", "sessionId": "copilot-1"}) == []

    events = adapter.handle_payload({"hook_event_name": "sessionEnd", "sessionId": "copilot-1", "reason": "complete"})
    assert len(events) == 1
    assert isinstance(events[0], SessionEndEvent)


def test_runner_injects_event_name_for_native_copilot_payload(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"sessionId":"copilot-1","agentName":"research"}'))

    assert run_hook(PLUGIN, ["--event-name", "subagentStop"]) == 0

    assert json.loads(capsys.readouterr().out) == {"decision": "allow"}
