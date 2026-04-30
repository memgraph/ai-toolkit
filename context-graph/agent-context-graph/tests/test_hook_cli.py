"""Tests for the generic command hook CLI."""

import io

from agent_context_graph.hooks.cli import main


def test_generic_cli_dispatches_codex_hook(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert main(["codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_generic_cli_requires_runtime(capsys):
    assert main([]) == 2

    assert "runtime" in capsys.readouterr().out
