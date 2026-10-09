"""Exercise checkout shell workflows without touching a user's config or Docker."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DEV = REPO / "scripts" / "dev-memgraph.sh"
INSTALL = REPO / "context-graph" / "scripts" / "install.sh"


@pytest.fixture
def shell_env(tmp_path):
    """Use the installed checkout Python through tiny command-boundary shims."""
    bins = tmp_path / "bin"
    bins.mkdir()
    commands = {
        "uv": f'while [ "$1" != python ] && [ "$1" != python3 ]; do shift; done\nshift\nexec {shlex.quote(sys.executable)} "$@"',
        "agent-context-graph": f'exec {shlex.quote(sys.executable)} -m agent_context_graph.cli "$@"',
    }
    for name, body in commands.items():
        command = bins / name
        command.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        command.chmod(0o700)
    env = dict(os.environ)
    env.update(PATH=f"{bins}:/usr/bin:/bin", CONTEXT_GRAPH_CONFIG=str(tmp_path / "selected config.toml"))
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    return env, bins


def test_dogfood_exports_only_private_config_path(shell_env, monkeypatch):
    """The emitted export must actually enable hooks, preserving the source config."""
    from agent_context_graph.adapters import _identity

    env, _ = shell_env
    monkeypatch.setenv("CONTEXT_GRAPH_CONFIG", env["CONTEXT_GRAPH_CONFIG"])
    _identity.write_config(
        user_id="alice", anthropic_api_key="private-anthropic-key", github_token="private-github-token"
    )
    source = Path(env["CONTEXT_GRAPH_CONFIG"])
    original = source.read_bytes()
    env["AI_TOOLKIT_DEV_PORT"] = "7791"
    result = subprocess.run(["bash", str(DEV), "dogfood-env"], env=env, capture_output=True, text=True, check=True)
    words = shlex.split(result.stdout)
    assert words[0] == "export" and len(words) == 2
    assert words[1].startswith("CONTEXT_GRAPH_CONFIG=")
    generated = Path(words[1].split("=", 1)[1])
    try:
        assert generated != source
        assert generated.stat().st_mode & 0o777 == 0o600
        monkeypatch.setenv("CONTEXT_GRAPH_CONFIG", str(generated))
        _identity._reset_cache()
        config = _identity.load_config()
        assert config.auto_reconcile is True
        assert config.memgraph_url == "bolt://localhost:7791"
        assert config.user_id == "alice"
        assert config.anthropic_api_key == "private-anthropic-key"
        assert config.github_token == "private-github-token"
        assert "private-anthropic-key" not in result.stdout
        assert "private-github-token" not in result.stdout
        assert source.read_bytes() == original
    finally:
        generated.unlink()
        _identity._reset_cache()


@pytest.mark.parametrize("existing", [True, False])
def test_hooks_restore_selected_config(shell_env, existing):
    """Restoration must address the selected path, including its original absence."""
    env, _ = shell_env
    source = Path(env["CONTEXT_GRAPH_CONFIG"])
    original = b'[identity]\nuser_id = "original"\n'
    if existing:
        source.write_bytes(original)
    for command in ("hooks-local", "hooks-restore"):
        subprocess.run(["bash", str(DEV), command], env=env, capture_output=True, text=True, check=True)
    assert source.exists() == existing
    if existing:
        assert source.read_bytes() == original
    assert not list(source.parent.glob("*.pre-local-test-backup*"))


def test_installer_preserves_container_with_different_port(shell_env, tmp_path):
    """An unreachable endpoint must never authorize removing a named container."""
    env, bins = shell_env
    calls = tmp_path / "docker-calls"
    docker = bins / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {shlex.quote(str(calls))}\n'
        'case "$1" in\ninfo) exit 0;;\ninspect) [ "$2" != --format ] || echo 7687; exit 0;;\n'
        "*) exit 99;;\nesac\n"
    )
    docker.chmod(0o700)
    env.update(MEMGRAPH_HOST="127.0.0.1", MEMGRAPH_PORT="65530", CONTEXT_GRAPH_MEMGRAPH_CONTAINER="preserve-me")
    result = subprocess.run(["bash", str(INSTALL)], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "container preserved" in result.stderr
    assert "rm " not in calls.read_text()
    assert "run " not in calls.read_text()
    assert "start " not in calls.read_text()
