"""Tests for the OpenAI Codex hooks adapter."""

import io

from agent_context_graph import AgentLink
from agent_context_graph.adapters.codex import PLUGIN, CodexHooksAdapter
from agent_context_graph.events import (
    Event,
    EventType,
    MessageEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_context_graph.hooks.runner import load_payload
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

    adapter = CodexHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "SessionStart",
            "session_id": "s1",
            "model": "gpt-5.4",
            "cwd": "/repo",
            "source": "startup",
        }
    )

    assert len(rec.events) == 1
    event = rec.events[0]
    assert isinstance(event, SessionStartEvent)
    assert event.event_type == EventType.SESSION_START
    assert event.source_sdk == "codex"
    assert event.session_id == "s1"
    assert event.model == "gpt-5.4"
    assert event.working_directory == "/repo"
    assert event.metadata["source"] == "startup"


def test_user_prompt_payload_emits_message_event():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = CodexHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s1",
            "prompt": "Use the cypher skill",
            "cwd": "/repo",
        }
    )

    event = rec.events[0]
    assert isinstance(event, MessageEvent)
    assert event.event_type == EventType.MESSAGE
    assert event.role == "user"
    assert event.content == "Use the cypher skill"
    assert event.source_sdk == "codex"


def test_tool_payloads_emit_tool_events():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = CodexHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "session_id": "s1",
            "tool_name": "mcp__skills__get_skill",
            "tool_input": {"name": "cypher-basics"},
            "tool_use_id": "tu-1",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "s1",
            "tool_name": "mcp__skills__get_skill",
            "tool_response": {"content": "skill body"},
            "tool_use_id": "tu-1",
        }
    )

    assert [event.event_type for event in rec.events] == [EventType.TOOL_START, EventType.TOOL_END]
    tool_start, tool_end = rec.events
    assert isinstance(tool_start, ToolStartEvent)
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_start.tool_name == "mcp__skills__get_skill"
    assert tool_start.tool_input == {"name": "cypher-basics"}
    assert tool_end.result == "skill body"


def test_post_tool_use_error_result_marks_error():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    adapter = CodexHooksAdapter(link)
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_response": {"exit_code": 2, "stderr": "nope"},
        }
    )

    event = rec.events[0]
    assert isinstance(event, ToolEndEvent)
    assert event.event_type == EventType.TOOL_END
    assert event.is_error
    assert event.error_message == "nope"


def test_stop_ends_the_turn_with_the_reply_and_keeps_the_session_open():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    # Stop fires after every turn, so it must never end the session.
    payload = {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": "Done."}
    adapter = CodexHooksAdapter(link)
    adapter.handle_payload(payload)

    reply, turn_end = rec.events
    assert isinstance(reply, MessageEvent)
    assert reply.role == "assistant"
    assert reply.content == "Done."
    assert turn_end.event_type == EventType.TURN_END
    assert all(event.event_type != EventType.SESSION_END for event in rec.events)
    assert PLUGIN.response_for_payload(payload) == {"continue": True}


def test_stop_records_the_turns_reply_before_the_turn_end():
    """The reply is the answer recall needs; the turn end that follows spawns its embedding."""
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    CodexHooksAdapter(link).handle_payload(
        {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": "We deploy on Thursdays."}
    )

    assert [event.event_type for event in rec.events] == [EventType.MESSAGE, EventType.TURN_END]
    reply = rec.events[0]
    assert isinstance(reply, MessageEvent)
    assert (reply.session_id, reply.role, reply.content) == ("s1", "assistant", "We deploy on Thursdays.")
    assert "last_assistant_message" not in reply.metadata


def test_stop_without_a_reply_records_only_the_turn_end():
    link = AgentLink()
    rec = _RecordingConnector()
    link.add_connector(rec)

    CodexHooksAdapter(link).handle_payload(
        {"hook_event_name": "Stop", "session_id": "s1", "last_assistant_message": "  "}
    )

    assert [event.event_type for event in rec.events] == [EventType.TURN_END]


def test_build_hooks_config_uses_command_for_supported_hooks():
    config = PLUGIN.build_hooks_config("python hook.py")

    assert "SessionStart" in config
    assert "PreToolUse" in config
    assert config["SessionStart"][0]["matcher"] == "startup|resume|clear"
    assert config["PreToolUse"][0]["matcher"] == "*"
    assert "matcher" not in config["Stop"][0]
    assert config["PreToolUse"][0]["hooks"][0]["command"] == "python hook.py"


def test_load_payload_requires_json_object():
    payload = load_payload(io.StringIO('{"hook_event_name": "Stop"}'))

    assert payload == {"hook_event_name": "Stop"}
