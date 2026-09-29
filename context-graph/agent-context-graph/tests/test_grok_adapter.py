"""Tests for the Grok Build command-hook runtime adapter.

Payload shapes are trimmed from a live grok 1.0.40 session, which sends every
field in both camelCase and snake_case.
"""

import json

from agent_context_graph import AgentLink
from agent_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter
from agent_context_graph.adapters.cursor import CursorHooksAdapter
from agent_context_graph.adapters.grok import PLUGIN, GrokHooksAdapter, init
from agent_context_graph.events import MessageEvent, SessionEndEvent, SessionStartEvent, ToolEndEvent, ToolStartEvent
from agent_context_graph.protocols import GraphConnector

_LS = {"command": "ls", "description": "List files in the workspace"}
_BASH_RESULT = {
    "type": "Bash",
    "output": [82, 69, 65, 68, 77, 69, 46, 109, 100, 10],
    "output_for_prompt": "exit: 0\nREADME.md\napp.py\n",
    "exit_code": 0,
}


def _payload(event, **fields):
    return {
        "hook_event_name": event,
        "hookEventName": event,
        "session_id": "grok-1",
        "sessionId": "grok-1",
        "cwd": "/work/project",
        "workspaceRoot": "/work/project/",
        "permissionMode": "auto",
        **fields,
    }


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def _adapter():
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)
    return GrokHooksAdapter(link), connector


def test_session_prompt_and_tool_round_trip():
    adapter, connector = _adapter()

    adapter.handle_payload(_payload("SessionStart", source="new"))
    adapter.handle_payload(_payload("UserPromptSubmit", prompt="Run 'ls'"))
    adapter.handle_payload(
        _payload("PreToolUse", tool_name="run_terminal_command", tool_input=_LS, tool_use_id="call-1")
    )
    adapter.handle_payload(
        _payload(
            "PostToolUse",
            tool_name="run_terminal_command",
            tool_input=_LS,
            tool_response=_BASH_RESULT,
            toolResult=_BASH_RESULT,
            tool_use_id="call-1",
        )
    )

    session_start, prompt, tool_start, tool_end = connector.events
    assert isinstance(session_start, SessionStartEvent)
    assert session_start.working_directory == "/work/project"
    assert session_start.source_sdk == "grok"
    assert isinstance(prompt, MessageEvent)
    assert prompt.content == "Run 'ls'"
    assert isinstance(tool_start, ToolStartEvent)
    assert tool_start.tool_input == _LS
    assert isinstance(tool_end, ToolEndEvent)
    assert tool_end.tool_use_id == "call-1"
    assert tool_end.result == "exit: 0\nREADME.md\napp.py\n"
    assert tool_end.is_error is False


def test_typed_results_use_model_text_and_flag_nonzero_exit():
    adapter, connector = _adapter()
    read = {"type": "ReadFile", "FileContent": {"content": "1→# Demo project\n", "content_concise": "1→# Demo"}}

    adapter.handle_payload(_payload("PostToolUse", tool_name="read_file", tool_response=read, tool_use_id="c-1"))
    adapter.handle_payload(
        _payload(
            "PostToolUse",
            tool_name="run_terminal_command",
            tool_response={**_BASH_RESULT, "exit_code": 2, "output_for_prompt": "exit: 2\n"},
            tool_use_id="c-2",
        )
    )

    read_end, failed_end = connector.events
    assert read_end.result == "1→# Demo project\n"
    assert read_end.is_error is False
    assert failed_end.result == "exit: 2\n"
    assert failed_end.is_error is True


def test_stop_is_a_turn_boundary_and_session_end_closes_the_session():
    adapter, connector = _adapter()

    adapter.handle_payload(_payload("Stop", reason="end_turn", lastAssistantMessage="Two files.", stopHookActive=False))
    # Grok fires Stop again at shutdown with no reply.
    assert adapter.handle_payload(_payload("Stop", reason="shutdown", stopHookActive=False)) == []
    adapter.handle_payload(_payload("SessionEnd", reason="shutdown"))

    reply, session_end = connector.events
    assert isinstance(reply, MessageEvent)
    assert reply.role == "assistant"
    assert reply.content == "Two files."
    assert isinstance(session_end, SessionEndEvent)


def test_capture_hooks_write_no_output():
    for event in ("PreToolUse", "PostToolUse", "Stop", "SessionEnd"):
        assert PLUGIN.response_for_payload({"hook_event_name": event}) is None


def test_init_writes_a_dedicated_project_hook_file(tmp_path):
    init(tmp_path, ["actions-graph"], hook_command="capture", timeout=15)

    config = json.loads((tmp_path / ".grok" / "hooks" / "agent-context-graph.json").read_text())
    assert config["hooks"]["PreToolUse"] == [{"hooks": [{"command": "capture", "type": "command", "timeout": 15}]}]
    assert "SessionEnd" in config["hooks"]


def test_claude_and_cursor_adapters_ignore_grok_running_their_hook_files():
    # Grok runs a project's .claude/settings.json and .cursor/hooks.json hooks
    # with its own payloads; recording them there would duplicate the session.
    link = AgentLink()
    connector = _RecordingConnector()
    link.add_connector(connector)

    grok_session_start = _payload("SessionStart", source="new")
    assert ClaudeCodeHooksAdapter(link).handle_payload(grok_session_start) == []
    assert CursorHooksAdapter(link).handle_payload({**grok_session_start, "hook_event_name": "sessionStart"}) == []
    assert connector.events == []
