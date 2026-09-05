"""Tests for driving a real Claude Code session.

The session itself is not exercised here -- it is billed and model-dependent,
and belongs to the gold-slice run. What is tested is everything around it that
could otherwise affect the machine it runs on, or produce a fixture that looks
valid without being one.
"""

import json
from pathlib import Path

import pytest
from context_graph_eval import live


def test_the_users_real_config_is_never_touched(tmp_path, monkeypatch):
    """The bug this replaced: repointing the global config file redirected every
    Claude Code session on the machine, not just the driven one. An unrelated
    session's activity was recorded into the graph under test."""
    real_config = tmp_path / "real-config.toml"
    real_config.write_text('url = "bolt://localhost:7687"', encoding="utf-8")
    monkeypatch.setenv(live.CONFIG_PATH_ENV, str(real_config))

    with live.hooks_pointed_at("bolt://localhost:7689") as env:
        assert env[live.CONFIG_PATH_ENV] != str(real_config)

    assert real_config.read_text(encoding="utf-8") == 'url = "bolt://localhost:7687"'


def test_the_judges_credential_does_not_reach_the_session(monkeypatch):
    """#304 runs the judge on Anthropic, so the eval process legitimately holds
    an ANTHROPIC_API_KEY. Inherited by the driven session it takes precedence
    over the CLI's own login and makes it refuse to start -- and the judge's
    credential has no business steering the session under test."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-judge-key")

    with live.hooks_pointed_at("bolt://localhost:7689") as env:
        assert "ANTHROPIC_API_KEY" not in env


def test_the_session_config_names_the_eval_instance(tmp_path):
    with live.hooks_pointed_at("bolt://localhost:7689") as env:
        written = Path(env[live.CONFIG_PATH_ENV]).read_text(encoding="utf-8")

    assert "bolt://localhost:7689" in written


def test_the_session_config_sets_a_user_id(tmp_path):
    """sessions-graph only writes (:User)-[:HAD_SESSION]->(:Session) when a
    user_id resolves, so without one the session is only half recorded."""
    with live.hooks_pointed_at("bolt://localhost:7689", user_id="someone") as env:
        written = Path(env[live.CONFIG_PATH_ENV]).read_text(encoding="utf-8")

    assert 'user_id = "someone"' in written


def test_the_throwaway_config_does_not_outlive_the_run():
    with live.hooks_pointed_at("bolt://localhost:7689") as env:
        path = Path(env[live.CONFIG_PATH_ENV])
        assert path.exists()

    assert not path.exists()


def test_hook_settings_name_every_connector(tmp_path: Path):
    """`hook run` silently does nothing without explicit --connector flags -- a
    gap that made an earlier live verification record no data at all."""
    settings = json.loads(live.hooks_settings(tmp_path / "s.json").read_text(encoding="utf-8"))

    command = json.dumps(settings)
    for connector in live.CONNECTORS:
        assert f"--connector {connector}" in command


def test_hook_settings_resolve_this_checkout(tmp_path: Path):
    """A globally installed agent-context-graph can shadow the checkout under
    test, so the session would exercise the wrong version of the code."""
    settings = live.hooks_settings(tmp_path / "s.json").read_text(encoding="utf-8")

    assert "uv run agent-context-graph" in settings


def test_hook_settings_do_not_narrow_the_environment(tmp_path: Path):
    """`uv run --package agent-context-graph` re-resolves to that package's own
    closure, which excludes the connector packages. Every hook then raised
    ImportError and -- because the runner swallows hook errors so a broken hook
    cannot break the harness -- exited 0 having written nothing. Two billed
    sessions reported success and recorded no data."""
    settings = live.hooks_settings(tmp_path / "s.json").read_text(encoding="utf-8")

    assert "--package" not in settings


def test_the_transcript_is_returned_not_discarded(monkeypatch, tmp_path):
    """A zero exit says the CLI did not crash, not that the model delegated or
    that hooks recorded anything. When a run plants nothing the transcript is the
    only way to tell those apart, and re-running to find out costs another billed
    session."""

    class _Done:
        returncode = 0
        stdout = '{"result": "delegated to one subagent"}'
        stderr = ""

    monkeypatch.setattr(live.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(live.subprocess, "run", lambda *a, **kw: _Done())

    session_id, transcript = live.drive_session("anything", repo_root=tmp_path)

    assert session_id
    assert "delegated" in transcript


def test_the_session_environment_is_passed_through(monkeypatch, tmp_path):
    """The config override only isolates the session if it actually reaches the
    subprocess -- otherwise hooks fall back to the global file and the
    contamination returns silently."""
    captured = {}

    class _Done:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def _run(*args, **kwargs):
        captured.update(kwargs.get("env") or {})
        return _Done()

    monkeypatch.setattr(live.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(live.subprocess, "run", _run)

    with live.hooks_pointed_at("bolt://localhost:7689") as env:
        live.drive_session("anything", repo_root=tmp_path, env=env)

    assert captured[live.CONFIG_PATH_ENV].endswith("config.toml")


def test_a_missing_claude_cli_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(live.shutil, "which", lambda name: None)

    with pytest.raises(live.LiveSessionError, match="claude"):
        live.drive_session("anything", repo_root=tmp_path)
