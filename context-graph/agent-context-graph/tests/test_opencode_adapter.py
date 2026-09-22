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
            "hook_event_name": "session.error",
            "session_id": "open-1",
            "error": {"name": "ProviderError", "message": "unavailable"},
        }
    )

    message, tool_end, error = connector.events
    assert isinstance(message, MessageEvent)
    assert message.role == "user"
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.result == "contents"
    assert isinstance(error, ErrorOccurredEvent)
    assert error.error_type == "ProviderError"


def test_opencode_init_installs_v2_plugin_with_capture_command(tmp_path):
    init(tmp_path, ["actions-graph"], hook_command="capture --strict")

    plugin = (tmp_path / ".opencode" / "plugins" / "agent-context-graph" / "index.js").read_text()
    assert 'id: "memgraph.agent-context-graph"' in plugin
    assert "Plugin.define" in plugin
    assert '"capture --strict"' in plugin
    assert "ctx.tool.hook" in plugin
    assert "ctx.event.subscribe" in plugin
    assert '"session.idle"' not in plugin


def test_opencode_records_assistant_updates_without_ending_idle_session():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    adapter = OpenCodeHooksAdapter(link)

    assert (
        adapter.handle_payload(
            {"hook_event_name": "message.updated", "session_id": "open-1", "role": "user", "content": "duplicate"}
        )
        == []
    )
    assert adapter.handle_payload({"hook_event_name": "session.idle", "session_id": "open-1"}) == []

    message = adapter.handle_payload(
        {"hook_event_name": "message.updated", "session_id": "open-1", "role": "assistant", "content": "done"}
    )[0]
    session_end = adapter.handle_payload({"hook_event_name": "session.deleted", "session_id": "open-1"})[0]

    assert isinstance(message, MessageEvent)
    assert message.role == "assistant"
    assert isinstance(session_end, SessionEndEvent)
