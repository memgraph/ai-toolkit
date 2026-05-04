"""Tests for the generic command hook CLI."""

import io
import json

from agent_context_graph.cli import main as top_level_main
from agent_context_graph.hooks.cli import main


def test_generic_cli_dispatches_codex_hook(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert main(["run", "codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_top_level_cli_dispatches_hook_run(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert top_level_main(["hook", "run", "codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_generic_cli_requires_runtime(capsys):
    assert main([]) == 2

    assert "command" in capsys.readouterr().out


def test_init_codex_writes_private_config(tmp_path, capsys):
    assert (
        main(
            [
                "init",
                "codex",
                "--project-dir",
                str(tmp_path),
                "--connector",
                "skills-graph",
                "--hook-command",
                "/venv/bin/agent-context-graph hook run codex --connector skills-graph",
            ]
        )
        == 0
    )

    assert (tmp_path / ".codex" / "config.toml").read_text() == "[features]\ncodex_hooks = true\n"
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == "/venv/bin/agent-context-graph hook run codex --connector skills-graph"
    assert hooks["hooks"]["SessionStart"][0]["matcher"] == "startup|resume|clear"
    assert "Wrote" in capsys.readouterr().out


def test_init_codex_refuses_to_overwrite_without_force(tmp_path, capsys):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text("existing", encoding="utf-8")

    assert main(["init", "codex", "--project-dir", str(tmp_path)]) == 1

    assert (codex_dir / "config.toml").read_text() == "existing"
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_init_codex_force_overwrites(tmp_path):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text("existing", encoding="utf-8")

    assert main(["init", "codex", "--project-dir", str(tmp_path), "--force"]) == 0

    assert (codex_dir / "config.toml").read_text() == "[features]\ncodex_hooks = true\n"
