"""Tests for the OpenCode V2 plugin runtime adapter."""

import json
import shlex
import shutil
import subprocess
import sys

import pytest

from memgraph_context_graph import AgentLink
from memgraph_context_graph.adapters.opencode import OpenCodeHooksAdapter, init
from memgraph_context_graph.events import ErrorOccurredEvent, MessageEvent, SessionEndEvent, ToolEndEvent, TurnEndEvent
from memgraph_context_graph.protocols import GraphConnector


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

    message, tool_end, error, failed_turn_end = connector.events
    assert isinstance(message, MessageEvent)
    assert message.role == "user"
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.result == "contents"
    assert isinstance(error, ErrorOccurredEvent)
    assert error.error_type == "ProviderError"
    assert error.error_details == {"status": 503}
    assert isinstance(failed_turn_end, TurnEndEvent)


def test_opencode_init_installs_v2_plugin_with_capture_command(tmp_path):
    init(tmp_path, ["actions-graph"], hook_command="capture --strict")

    plugin = (tmp_path / ".opencode" / "plugins" / "memgraph-context-graph" / "index.js").read_text()
    assert 'id: "memgraph.memgraph-context-graph"' in plugin
    assert "export default {" in plugin
    # Only Node built-ins: OpenCode resolves imports next to the plugin file.
    assert 'from "@opencode/plugin"' not in plugin
    assert 'const command = ["capture", "--strict"]' in plugin
    assert '"-lc"' not in plugin
    assert "ctx.tool.hook" in plugin
    assert "ctx.event.subscribe" in plugin
    assert "event.data" in plugin
    assert "tool_use_id: event.id" in plugin
    # `opencode run` can exit before the bus delivers the turn end; shutdown sends it.
    assert 'capture({ hook_event_name: "session.execution.succeeded", session_id: sessionID })' in plugin
    assert '"session.idle"' not in plugin


# Drives the installed plugin with a stand-in for OpenCode's plugin context:
# the bus yields the events in argv[2], then the prompt hook fires once.
_PLUGIN_DRIVER = """
const [plugin, events] = [process.argv[1], JSON.parse(process.argv[2])]
const { default: { setup } } = await import(plugin)
const hooks = {}
const registrar = (domain) => ({
  hook: async (name, fn) => { hooks[`${domain}.${name}`] = fn; return { dispose() {} } },
})
const cleanup = await setup({
  location: { directory: "/plugin/dir" },
  session: registrar("session"),
  tool: registrar("tool"),
  event: { subscribe: async function* () { for (const event of events) yield event } },
})
await new Promise((resolve) => setTimeout(resolve, 100))
await hooks["session.prompt"]({ sessionID: "ses_1", prompt: { text: "hi" } })
await cleanup()
"""


def _run_plugin(tmp_path, bus_events):
    """The payloads the plugin captures for one prompt, after the bus delivers *bus_events*."""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    captured = tmp_path / "captured.jsonl"
    recorder = f"import sys; open({str(captured)!r}, 'a').write(sys.stdin.read() + '\\n')"
    init(tmp_path, ["actions-graph"], hook_command=shlex.join([sys.executable, "-c", recorder]))
    plugin = tmp_path / ".opencode" / "plugins" / "memgraph-context-graph" / "index.js"
    module = plugin.rename(plugin.with_suffix(".mjs"))
    subprocess.run(
        ["node", "--input-type=module", "-e", _PLUGIN_DRIVER, module.as_uri(), json.dumps(bus_events)],
        check=True,
        timeout=30,
    )
    return [json.loads(line) for line in captured.read_text().splitlines() if line]


def test_opencode_plugin_starts_the_session_from_the_first_prompt_when_the_bus_missed_it(tmp_path):
    payloads = _run_plugin(tmp_path, [])

    starts = [payload for payload in payloads if payload["hook_event_name"] == "session.created"]
    assert starts == [{"hook_event_name": "session.created", "session_id": "ses_1", "cwd": "/plugin/dir"}]
    assert [payload["hook_event_name"] for payload in payloads][:2] == ["session.created", "session.prompt"]


def test_opencode_plugin_records_the_bus_session_start_once(tmp_path):
    created = {"type": "session.created", "data": {"sessionID": "ses_1", "location": {"directory": "/session/dir"}}}

    payloads = _run_plugin(tmp_path, [created])

    starts = [payload for payload in payloads if payload["hook_event_name"] == "session.created"]
    assert starts == [{"hook_event_name": "session.created", "session_id": "ses_1", "cwd": "/session/dir"}]


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


def test_execution_succeeded_ends_the_turn_without_ending_the_session():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)

    (turn_end,) = OpenCodeHooksAdapter(link).handle_payload(
        {"hook_event_name": "session.execution.succeeded", "session_id": "open-1"}
    )

    assert isinstance(turn_end, TurnEndEvent)
