"""Tests for the Cursor command-hook runtime adapter."""

import json

from agent_context_graph import AgentLink
from agent_context_graph.adapters.cursor import CursorHooksAdapter, init
from agent_context_graph.events import AgentEndEvent, MessageEvent, SessionEndEvent, ToolEndEvent
from agent_context_graph.protocols import GraphConnector


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_cursor_uses_conversation_id_and_maps_agent_activity():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = CursorHooksAdapter(link)

    adapter.handle_payload(
        {
            "hook_event_name": "postToolUse",
            "conversation_id": "conversation-1",
            "generation_id": "generation-2",
            "tool_name": "Shell",
            "tool_output": '{"exitCode":0,"stdout":"ok"}',
            "tool_use_id": "tool-1",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "subagentStop",
            "conversation_id": "conversation-1",
            "subagent_type": "explore",
            "status": "completed",
            "summary": "Found the implementation",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "afterAgentResponse",
            "conversation_id": "conversation-1",
            "text": "Finished",
        }
    )

    tool_end, agent_end, message = connector.events
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.session_id == "conversation-1"
    assert tool_end.metadata["generation_id"] == "generation-2"
    assert isinstance(agent_end, AgentEndEvent)
    assert agent_end.output == "Found the implementation"
    assert isinstance(message, MessageEvent)
    assert message.role == "assistant"


def test_cursor_init_writes_versioned_hooks_file(tmp_path):
    init(tmp_path, ["actions-graph"], hook_command="capture", timeout=7)

    config = json.loads((tmp_path / ".cursor" / "hooks.json").read_text())
    assert config["version"] == 1
    assert config["hooks"]["preToolUse"][0] == {
        "command": "capture",
        "type": "command",
        "timeout": 7,
        "matcher": "*",
    }
    assert "stop" not in config["hooks"]


def test_cursor_init_preserves_unrelated_settings_and_hooks(tmp_path):
    settings_path = tmp_path / ".cursor" / "hooks.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"custom": True, "hooks": {"workspaceOpen": [{"command": "existing"}]}}))

    init(tmp_path, ["actions-graph"], hook_command="capture")

    config = json.loads(settings_path.read_text())
    assert config["custom"] is True
    assert config["hooks"]["workspaceOpen"] == [{"command": "existing"}]
    assert "preToolUse" in config["hooks"]


def test_cursor_only_ends_session_for_session_end():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = CursorHooksAdapter(link)

    assert adapter.handle_payload({"hook_event_name": "stop", "conversation_id": "cursor-1"}) == []

    events = adapter.handle_payload({"hook_event_name": "sessionEnd", "session_id": "cursor-1", "reason": "completed"})
    assert len(events) == 1
    assert isinstance(events[0], SessionEndEvent)
