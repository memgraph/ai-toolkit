"""Context before a tool runs: connectors add a line, runtimes that can show it do, and nothing ever blocks."""

from __future__ import annotations

import io
import json

import pytest

from agent_context_graph.hooks.cli import main as hook_main
from agent_context_graph.hooks.runtime_plugin import load_runtime_plugins
from agent_context_graph.link import AgentLink
from agent_context_graph.protocols import GraphConnector

PRE_TOOL = {
    "hook_event_name": "PreToolUse",
    "session_id": "s1",
    "tool_name": "Bash",
    "tool_input": {"command": "gh issue view 1 -R a/b"},
    "tool_use_id": "toolu_1",
}


class _Says(GraphConnector):
    """A connector that answers every tool start with ``line`` (or raises it)."""

    def __init__(self, line):
        self.line = line

    def on_event(self, event):
        pass

    def context_before_tool(self, event):
        if isinstance(self.line, Exception):
            raise self.line
        return self.line


def run(monkeypatch, runtime, connectors, payload=PRE_TOOL):
    link = AgentLink()
    for connector in connectors:
        link.add_connector(connector)
    monkeypatch.setattr("agent_context_graph.hooks.runner.create_link", lambda *args, **kwargs: link)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return hook_main(["run", runtime, "--strict"])


@pytest.mark.parametrize("runtime", ["claude-code", "codex"])
def test_lines_reach_runtimes_that_can_show_them(monkeypatch, capsys, runtime):
    assert run(monkeypatch, runtime, [_Says("first"), _Says(None), _Says("second")]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "first\nsecond"}
    }


def test_a_failing_connector_adds_nothing_and_never_fails_the_hook(monkeypatch, capsys):
    assert run(monkeypatch, "claude-code", [_Says(RuntimeError("memgraph down")), _Says("still here")]) == 0

    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"] == "still here"


def test_no_line_no_output(monkeypatch, capsys):
    assert run(monkeypatch, "claude-code", [_Says(None)]) == 0
    assert capsys.readouterr().out.strip() == ""


def test_only_tool_starts_are_asked(monkeypatch, capsys):
    stop = {"hook_event_name": "Stop", "session_id": "s1"}
    assert run(monkeypatch, "claude-code", [_Says("never")], stop) == 0
    assert json.loads(capsys.readouterr().out) == {"continue": True}


def test_which_runtimes_show_context_before_a_tool():
    """Only where it is documented to leave the call alone (verified against each harness's hook docs)."""
    showing = {
        name
        for name, plugin in load_runtime_plugins().items()
        if getattr(plugin, "context_before_tool_response", lambda _text: None)("x")
    }
    assert showing == {"claude-code", "codex"}
