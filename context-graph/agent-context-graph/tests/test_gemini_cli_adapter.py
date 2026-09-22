"""Tests for the Gemini CLI command-hook runtime adapter."""

import json

from agent_context_graph import AgentLink
from agent_context_graph.adapters.gemini_cli import GeminiCLIHooksAdapter, build_hooks_config, init
from agent_context_graph.events import EventType, MessageEvent, ToolEndEvent
from agent_context_graph.protocols import GraphConnector


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_gemini_payloads_emit_tool_and_message_events():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = GeminiCLIHooksAdapter(link)

    adapter.handle_payload({"hook_event_name": "BeforeAgent", "session_id": "gemini-1", "prompt": "Read the docs"})
    adapter.handle_payload(
        {
            "hook_event_name": "AfterTool",
            "session_id": "gemini-1",
            "tool_name": "read_file",
            "tool_input": {"path": "README.md"},
            "tool_response": {"llmContent": "contents"},
        }
    )

    message, tool_end = connector.events
    assert isinstance(message, MessageEvent)
    assert message.role == "user"
    assert message.source_sdk == "gemini-cli"
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.result == "contents"
    assert tool_end.metadata["tool_input"] == {"path": "README.md"}


def test_gemini_config_uses_millisecond_timeouts_and_merges_settings(tmp_path):
    settings_dir = tmp_path / ".gemini"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(json.dumps({"theme": "Default"}), encoding="utf-8")

    init(tmp_path, ["actions-graph"], hook_command="capture", timeout=12)

    settings = json.loads((settings_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["theme"] == "Default"
    assert settings["hooks"]["BeforeTool"][0]["hooks"][0]["timeout"] == 12_000
    assert settings["hooks"]["BeforeTool"][0]["matcher"] == ".*"


def test_gemini_probe_is_its_native_session_end_event():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)

    GeminiCLIHooksAdapter(link).handle_payload(
        {"hook_event_name": "SessionEnd", "session_id": "doctor", "reason": "exit"}
    )

    assert connector.events[0].event_type == EventType.SESSION_END
    assert build_hooks_config("capture")["SessionEnd"]
