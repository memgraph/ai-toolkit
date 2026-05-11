"""Tests for the Claude Code hooks adapter."""

import io

from agent_context_graph import AgentLink
from agent_context_graph.adapters.claude_code import (
    ClaudeCodeHooksAdapter,
    build_hooks_config,
    load_payload,
    response_for_payload,
)
from agent_context_graph.events import Event, EventType
from agent_context_graph.protocols import GraphConnector


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events: list[Event] = []

    def on_event(self, event: Event) -> None:
        self.events.append(event)


def test_session_start_payload_emits_session_start_event():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "SessionStart",
            "session_id": "s1",
            "cwd": "/repo",
            "source": "startup",
        }
    )

    event = rec.events[0]
    assert event.event_type == EventType.SESSION_START
    assert event.source_sdk == "claude-code"
    assert event.session_id == "s1"
    assert event.working_directory == "/repo"


def test_user_prompt_payload_emits_message_event():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s1",
            "prompt": "Use the cypher skill",
            "cwd": "/repo",
        }
    )

    event = rec.events[0]
    assert event.event_type == EventType.MESSAGE
    assert event.role == "user"
    assert event.content == "Use the cypher skill"


def test_tool_payloads_emit_tool_events():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "session_id": "s1",
            "tool_name": "Read",
            "tool_input": {"file_path": "/skills/cypher/SKILL.md"},
            "tool_use_id": "tu-1",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "s1",
            "tool_name": "Read",
            "tool_input": {"file_path": "/skills/cypher/SKILL.md"},
            "tool_response": "skill body",
            "tool_use_id": "tu-1",
        }
    )

    assert [event.event_type for event in rec.events] == [EventType.TOOL_START, EventType.TOOL_END]
    assert rec.events[0].tool_name == "Read"
    assert rec.events[0].tool_input == {"file_path": "/skills/cypher/SKILL.md"}
    assert rec.events[1].result == "skill body"


def test_failed_tool_payload_emits_error_tool_end():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUseFailure",
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "npm test"},
            "tool_use_id": "tu-1",
            "error": "Command exited with non-zero status code 1",
        }
    )

    event = rec.events[0]
    assert event.event_type == EventType.TOOL_END
    assert event.is_error
    assert event.error_message == "Command exited with non-zero status code 1"


def test_subagent_payloads_emit_agent_events():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "SubagentStart",
            "session_id": "s1",
            "agent_id": "agent-1",
            "agent_type": "Explore",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "s1",
            "agent_id": "agent-1",
            "agent_type": "Explore",
            "last_assistant_message": "Done",
        }
    )

    assert [event.event_type for event in rec.events] == [EventType.AGENT_START, EventType.AGENT_END]
    assert rec.events[0].agent_name == "agent-1"
    assert rec.events[1].output == "Done"


def test_stop_payload_emits_session_end_and_json_response():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    payload = {"hook_event_name": "Stop", "session_id": "s1"}
    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload(payload)

    assert rec.events[0].event_type == EventType.SESSION_END
    assert response_for_payload(payload) == {"continue": True}


def test_build_hooks_config_uses_command_for_supported_hooks():
    config = build_hooks_config("python hook.py")

    assert "SessionStart" in config
    assert "PreToolUse" in config
    assert "PostToolUseFailure" in config
    assert config["SessionStart"][0]["matcher"] == "startup|resume|clear"
    assert config["PreToolUse"][0]["matcher"] == "*"
    assert "matcher" not in config["Stop"][0]
    assert config["PreToolUse"][0]["hooks"][0]["command"] == "python hook.py"


def test_load_payload_requires_json_object():
    payload = load_payload(io.StringIO('{"hook_event_name": "Stop"}'))

    assert payload == {"hook_event_name": "Stop"}
