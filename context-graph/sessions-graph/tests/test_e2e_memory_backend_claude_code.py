"""Switching Claude Code's memory to Context Graph and back, with a real Memgraph and a throwaway home."""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

pytest.importorskip("agent_context_graph")

from agent_context_graph.adapters import _identity
from agent_context_graph.adapters.claude_code import PLUGIN
from agent_context_graph.cli import main as top_level_main
from agent_context_graph.memory_backend import MemoryBackendError, use_graph_memory, use_native_memory


@pytest.fixture
def home(tmp_path, monkeypatch, memgraph):
    """A home with Claude Code settings and one project's auto memory, and a config for an existing user."""
    home = tmp_path / "home"
    checkout = (tmp_path / "repos" / "ai-toolkit").resolve()
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "git@github.com:memgraph/ai-toolkit.git"], check=True
    )

    memory = home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(checkout)) / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("- [x](feedback_trailers.md) — index\n")
    (memory / "feedback_trailers.md").write_text(
        "---\nname: trailers\ndescription: No attribution trailers\nmetadata:\n  type: feedback\n---\nNever add them.\n"
    )
    (memory / "project_map.md").write_text("---\ndescription: The map\nmetadata:\n  type: project\n---\nMap #484.\n")
    gone = home / ".claude" / "projects" / "-nowhere-old-repo" / "memory"
    gone.mkdir(parents=True)
    (gone / "reference_dash.md").write_text("---\ndescription: dashboard\ntype: reference\n---\nhttps://example\n")

    (home / ".claude" / "settings.json").write_text(json.dumps({"theme": "dark"}))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(user_id="alice", memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    yield home
    _identity._reset_cache()


def _settings(home):
    return json.loads((home / ".claude" / "settings.json").read_text())


def test_opting_in_imports_turns_auto_memory_off_and_turns_ours_on(home, graph):
    lines = use_graph_memory(PLUGIN, home=home)

    files = {f.path: f.content for f in graph.memory_store("alice").files()}
    assert set(files) == {
        "/memories/feedback/trailers.md",
        "/memories/projects/github.com_memgraph_ai-toolkit/map.md",
        "/memories/projects/nowhere-old-repo/dash.md",
    }
    assert files["/memories/feedback/trailers.md"].endswith("Never add them.\n")
    assert _settings(home) == {"theme": "dark", "autoMemoryEnabled": False}
    assert json.loads((home / ".claude" / "settings.json.context-graph.bak").read_text()) == {"theme": "dark"}
    assert _identity.load_config().graph_memory
    assert "Imported /memories/feedback/trailers.md" in lines
    # The originals stay, so handing memory back loses nothing.
    assert (next((home / ".claude" / "projects").glob("*ai-toolkit/memory")) / "MEMORY.md").exists()


def test_rerunning_keeps_what_the_graph_already_has(home, graph):
    store = graph.memory_store("alice")
    store.create("/memories/feedback/trailers.md", "edited since import")

    lines = use_graph_memory(PLUGIN, home=home)
    use_graph_memory(PLUGIN, home=home)

    assert "Kept existing /memories/feedback/trailers.md" in lines
    assert {f.path: f.content for f in store.files()}["/memories/feedback/trailers.md"] == "edited since import"
    assert len(store.files()) == 3


def test_handing_memory_back(home):
    use_graph_memory(PLUGIN, home=home)

    use_native_memory(PLUGIN, home=home)

    assert _settings(home) == {"theme": "dark"}
    assert not _identity.load_config().graph_memory
    assert _identity.load_config().memory_backend == "native"


def test_no_user_means_no_switch_and_nothing_changed(home):
    _identity.write_config(user_id="")

    with pytest.raises(MemoryBackendError, match="No user is configured"):
        use_graph_memory(PLUGIN, home=home)

    assert _settings(home) == {"theme": "dark"}
    assert not _identity.load_config().graph_memory


def test_setup_cli_and_doctor_agree(home, capsys):
    assert top_level_main(["setup", "claude-code", "--memory-backend", "context-graph"]) == 0
    assert "Set memory.backend = context-graph" in capsys.readouterr().out
    _identity._reset_cache()

    from agent_context_graph.cli import _check_memory_backend

    assert _check_memory_backend("claude-code")["ok"]
    (home / ".claude" / "settings.json").write_text("{}")
    check = _check_memory_backend("claude-code")
    assert not check["ok"] and "two memories" in check["detail"]
