"""Tests for the OpenCode V2 plugin runtime adapter."""

from agent_context_graph import AgentLink
from agent_context_graph.adapters.opencode import OpenCodeHooksAdapter, init
from agent_context_graph.events import ErrorOccurredEvent, MessageEvent, SessionEndEvent, ToolEndEvent
from agent_context_graph.protocols import GraphConnector


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_opencode_shim_payloads_emit_events():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = OpenCodeHooksAdapter(link)

    adapter.handle_payload(
        {
            "hook_event_name": "session.prompt",
            "session_id": "open-1",
            "prompt": "Inspect the graph",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "tool.execute.after",
            "session_id": "open-1",
            "tool_name": "read",
            "tool_input": {"filePath": "README.md"},
            "tool_result": "contents",
            "tool_use_id": "call-1",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "session.execution.failed",
            "session_id": "open-1",
            "error": {"type": "ProviderError", "message": "unavailable", "status": 503},
        }
    )

    message, tool_end, error = connector.events
    assert isinstance(message, MessageEvent)
    assert message.role == "user"
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.result == "contents"
    assert isinstance(error, ErrorOccurredEvent)
    assert error.error_type == "ProviderError"
    assert error.error_details == {"status": 503}


def test_opencode_init_installs_v2_plugin_with_capture_command(tmp_path):
    init(tmp_path, ["actions-graph"], hook_command="capture --strict")

    plugin = (tmp_path / ".opencode" / "plugins" / "agent-context-graph" / "index.js").read_text()
    assert 'id: "memgraph.agent-context-graph"' in plugin
    assert "export default {" in plugin
    # Only Node built-ins: OpenCode resolves imports next to the plugin file.
    assert 'from "@opencode/plugin"' not in plugin
    assert 'const command = ["capture", "--strict"]' in plugin
    assert '"-lc"' not in plugin
    assert "ctx.tool.hook" in plugin
    assert "ctx.event.subscribe" in plugin
    assert "event.data" in plugin
    assert "tool_use_id: event.id" in plugin
    assert '"session.idle"' not in plugin


def test_opencode_records_completed_assistant_text_and_ignores_unmapped_events():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = OpenCodeHooksAdapter(link)

    assert adapter.handle_payload({"hook_event_name": "session.text.delta", "session_id": "open-1"}) == []
    assert adapter.handle_payload({"hook_event_name": "session.idle", "session_id": "open-1"}) == []

    message = adapter.handle_payload({"hook_event_name": "session.text.ended", "session_id": "open-1", "text": "done"})[
        0
    ]
    session_end = adapter.handle_payload({"hook_event_name": "session.deleted", "session_id": "open-1"})[0]

    assert isinstance(message, MessageEvent)
    assert message.role == "assistant"
    assert message.content == "done"
    assert isinstance(session_end, SessionEndEvent)


def test_opencode_tool_result_records_model_text_and_flags_nonzero_exit():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = OpenCodeHooksAdapter(link)

    (tool_end,) = adapter.handle_payload(
        {
            "hook_event_name": "tool.execute.after",
            "session_id": "open-1",
            "tool_name": "shell",
            "tool_input": {"command": "false"},
            "tool_use_id": "call-2",
            "tool_result": {
                "output": {"exit": 1, "truncated": False, "output": "", "status": "completed"},
                "content": [{"type": "text", "text": "<exited with code 1>"}],
                "metadata": {"exit": 1},
            },
            "is_error": False,
        }
    )

    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.result == "<exited with code 1>"
    assert tool_end.is_error is True
