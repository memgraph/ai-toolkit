import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from memgraph_context_graph import capture_log, spaces
from memgraph_context_graph.adapters import _identity

_FAKE_AGENT = Path(__file__).with_name("fake_agent_cli.py")


@pytest.fixture(autouse=True)
def _capture_log_in_tmp(monkeypatch, tmp_path):
    """No test appends to the user's real capture-failure log; each gets its own."""
    log = tmp_path / "capture-failures.jsonl"
    monkeypatch.setattr(capture_log, "_path_override", log)
    return log


@pytest.fixture(autouse=True)
def _no_real_config(monkeypatch, tmp_path):
    """No test reads the user's real config file, whose Memgraph a hook run would write to.

    A hook run that names no ``--connector`` enables ``[capture] connectors``, else the
    defaults, so this config enables none: a test that wants one names it, or writes its own config.
    """
    config = tmp_path / "isolated-config.toml"
    config.write_text('[capture]\nconnectors = ""\n')
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config))
    _identity._reset_cache()
    spaces._reset_cache()
    yield
    _identity._reset_cache()
    spaces._reset_cache()


@pytest.fixture(autouse=True)
def _graph_version_offline(monkeypatch):
    """The check that reads the config's graph allows everything: that graph defaults to the user's real Memgraph.

    Checks through a connection a test made itself (``graph_version.check``) still run.
    """
    from memgraph_context_graph import graph_version

    monkeypatch.setattr(graph_version, "check_configured", lambda **_: graph_version.Rules())


@pytest.fixture
def default_connectors(monkeypatch, tmp_path):
    """A config that sets no ``[capture] connectors``, for tests about what setup does with the defaults."""
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "unset.toml"))
    _identity._reset_cache()


@pytest.fixture
def setup_home(tmp_path, monkeypatch):
    """A throwaway home and config file for setup and uninstall; Docker still finds its real context."""
    real_docker_config = Path.home() / ".docker"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    if real_docker_config.is_dir() and "DOCKER_CONFIG" not in os.environ:
        monkeypatch.setenv("DOCKER_CONFIG", str(real_docker_config))
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    spaces._reset_cache()
    yield home
    _identity._reset_cache()
    spaces._reset_cache()


class FakeAgents:
    """The fake `claude` and `codex` on PATH, and what each was asked to do."""

    def __init__(self, state: Path) -> None:
        self.state = state

    def of(self, agent: str) -> dict:
        everyone = json.loads(self.state.read_text()) if self.state.exists() else {}
        return everyone.get(agent, {"marketplaces": [], "plugins": [], "calls": []})


@pytest.fixture
def fake_agents(tmp_path, monkeypatch):
    """Only fake `claude` and `codex` on PATH (plus what setup shells out to), so no real agent is ever touched.

    PATH is rebuilt rather than prepended to: setup detects agents by what's on PATH, and the developer's
    own Copilot, Cursor or OpenCode would otherwise turn up in every plan.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for agent in ("claude", "codex"):
        script = bin_dir / agent
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{_FAKE_AGENT}" {agent} "$@"\n')
        script.chmod(0o755)
    # Linked one by one, not by directory: ~/.local/bin and Homebrew's bin hold agent CLIs too.
    for tool in ("docker", "git"):
        if found := shutil.which(tool):
            (bin_dir / tool).symlink_to(found)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]))
    state = tmp_path / "fake-agents.json"
    monkeypatch.setenv("FAKE_AGENT_STATE", str(state))
    return FakeAgents(state)
