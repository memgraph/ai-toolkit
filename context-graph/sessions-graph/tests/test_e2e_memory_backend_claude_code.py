"""Switching Claude Code's memory to Context Graph and back, with a real Memgraph and a throwaway home."""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

pytest.importorskip("memgraph_context_graph")

from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.adapters.claude_code import PLUGIN
from memgraph_context_graph.cli import main as top_level_main
from memgraph_context_graph.memory_backend import MemoryBackendError, use_graph_memory, use_native_memory

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


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
    # Claude Code keeps one memory folder per repository; a worktree's folder holds only transcripts.
    worktree = (tmp_path / "repos" / "ai-toolkit-wt").resolve()
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "--allow-empty", "-m", "init"], check=True, env=_GIT_ENV
    )
    subprocess.run(["git", "-C", str(checkout), "worktree", "add", "-q", str(worktree)], check=True)
    (home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(worktree))).mkdir(parents=True)

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
    use_graph_memory(PLUGIN, home=home)
    store = graph.memory_store("alice")
    store.str_replace("/memories/feedback/trailers.md", "Never add them.", "Edited since import.")

    lines = use_graph_memory(PLUGIN, home=home)

    assert "Kept existing /memories/feedback/trailers.md" in lines
    assert store.read("/memories/feedback/trailers.md").endswith("Edited since import.\n")
    assert len(store.files()) == 3


def _add_project(home, tmp_path, name, files):
    """A second checkout with its own remote and auto memory."""
    checkout = (tmp_path / "repos" / name).resolve()
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", f"git@github.com:acme/{name}.git"], check=True
    )
    memory = home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(checkout)) / "memory"
    memory.mkdir(parents=True)
    for filename, text in files.items():
        (memory / filename).write_text(text)
    return memory


def test_same_named_memories_from_two_projects_are_both_kept(home, tmp_path, graph):
    other = "---\nname: trailers\ndescription: Sign off\nmetadata:\n  type: feedback\n---\nAlways sign off.\n"
    _add_project(home, tmp_path, "zeta", {"feedback_trailers.md": other})

    use_graph_memory(PLUGIN, home=home)

    files = {f.path: f.content for f in graph.memory_store("alice").files()}
    assert files["/memories/feedback/trailers.md"].endswith("Never add them.\n")
    assert files["/memories/projects/github.com_acme_zeta/feedback_trailers.md"] == other


def _memory_dir(home, project):
    return next((home / ".claude" / "projects").glob(f"*{project}")) / "memory"


def test_handing_memory_back(home):
    use_graph_memory(PLUGIN, home=home)

    use_native_memory(PLUGIN, home=home)

    assert _settings(home) == {"theme": "dark"}
    assert not _identity.load_config().graph_memory
    assert _identity.load_config().memory_backend == "native"


def test_handing_memory_back_copies_what_the_graph_learned_since(home, graph):
    use_graph_memory(PLUGIN, home=home)
    graph.record_session_start("s-toolkit", user_id="alice", project_key="github.com_memgraph_ai-toolkit")
    store = graph.memory_store("alice", session_id="s-toolkit")
    store.create("/memories/user/role.md", "---\nname: role\ndescription: Works on Memgraph\ntype: user\n---\nHi.\n")
    store.str_replace("/memories/projects/github.com_memgraph_ai-toolkit/map.md", "Map #484.", "Map #484 closed.")
    store.create(
        "/memories/projects/github.com_memgraph_ai-toolkit/plan.md", "---\ndescription: The plan\n---\nShip.\n"
    )
    store.create("/memories/projects/never-opened/x.md", "A project Claude Code has no folder for.\n")
    store.create("/memories/codex/memory.md", "Another agent's file.\n")
    graph.memory_store("alice").create("/memories/feedback/nowhere.md", "Written by no session.\n")

    lines = use_native_memory(PLUGIN, home=home)

    toolkit, old_repo = _memory_dir(home, "ai-toolkit"), _memory_dir(home, "old-repo")
    assert (toolkit / "project_map.md").read_text().endswith("Map #484 closed.\n")
    assert (toolkit / "project_plan.md").read_text().endswith("Ship.\n")
    # A user-wide memory goes back where it came from, or to the repository of the session that wrote it.
    assert (toolkit / "feedback_trailers.md").read_text().endswith("Never add them.\n")
    assert (toolkit / "user_role.md").read_text().endswith("Hi.\n")
    assert sorted(path.name for path in old_repo.iterdir()) == ["reference_dash.md"]
    index = (toolkit / "MEMORY.md").read_text().splitlines()
    assert "- [role](user_role.md) — Works on Memgraph" in index
    assert "- [project_plan](project_plan.md) — The plan" in index
    assert len(index) == 4
    assert "Copied back /memories/projects/github.com_memgraph_ai-toolkit/plan.md" in lines
    assert any(
        line.startswith("Kept in Memgraph only")
        and "/memories/codex/memory.md" in line
        and "never-opened" in line
        and "/memories/feedback/nowhere.md" in line
        for line in lines
    )


def test_handing_memory_back_writes_only_the_repository_folder(home):
    use_graph_memory(PLUGIN, home=home)

    use_native_memory(PLUGIN, home=home)

    worktree = next((home / ".claude" / "projects").glob("*ai-toolkit-wt"))
    assert not (worktree / "memory").exists()


def test_handing_memory_back_round_trips_same_named_memories(home, tmp_path):
    other = "---\nname: trailers\ndescription: Sign off\nmetadata:\n  type: feedback\n---\nAlways sign off.\n"
    zeta = _add_project(home, tmp_path, "zeta", {"feedback_trailers.md": other})
    toolkit = _memory_dir(home, "ai-toolkit")
    before = (toolkit / "feedback_trailers.md").read_text()

    use_graph_memory(PLUGIN, home=home)
    use_native_memory(PLUGIN, home=home)

    assert (zeta / "feedback_trailers.md").read_text() == other
    assert (toolkit / "feedback_trailers.md").read_text() == before


def test_handing_memory_back_keeps_a_native_file_the_graph_doesnt_descend_from(home, graph):
    use_graph_memory(PLUGIN, home=home)
    graph.memory_store("alice").str_replace("/memories/feedback/trailers.md", "Never add them.", "Never, ever.")
    native = _memory_dir(home, "ai-toolkit") / "feedback_trailers.md"
    native.write_text("Changed outside the graph.\n")

    lines = use_native_memory(PLUGIN, home=home)

    assert native.read_text() == "Changed outside the graph.\n"
    assert any(line.startswith(f"Kept {native}") for line in lines)
    assert "Copied back /memories/feedback/trailers.md" not in lines


def test_handing_memory_back_again_never_overwrites_newer_native_files(home, graph):
    use_graph_memory(PLUGIN, home=home)
    use_native_memory(PLUGIN, home=home)
    edited = _memory_dir(home, "ai-toolkit") / "project_map.md"
    edited.write_text("Edited by Claude Code since.\n")

    use_native_memory(PLUGIN, home=home)

    assert edited.read_text() == "Edited by Claude Code since.\n"


def test_handing_memory_back_without_memgraph_stays_on_the_graph(home):
    use_graph_memory(PLUGIN, home=home)
    _identity.write_config(memgraph_url="bolt://localhost:1")

    with pytest.raises(MemoryBackendError, match="Can't reach Memgraph"):
        use_native_memory(PLUGIN, home=home)

    assert _settings(home)["autoMemoryEnabled"] is False
    assert _identity.load_config().graph_memory


def test_no_user_means_no_switch_and_nothing_changed(home):
    _identity.write_config(user_id="")

    with pytest.raises(MemoryBackendError, match="No user is configured"):
        use_graph_memory(PLUGIN, home=home)

    assert _settings(home) == {"theme": "dark"}
    assert not _identity.load_config().graph_memory


@pytest.fixture
def claude_with_plugin(tmp_path, monkeypatch):
    """A `claude` on PATH that reports the Context Graph plugin installed, so setup never runs the real one."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text(
        "#!/bin/sh\n"
        'if [ "$*" = "plugin list --json" ]; then echo \'[{"id": "context-graph@context-graph-plugins"}]\'; exit 0; fi\n'
        'if [ "$*" = "plugin marketplace list --json" ]; then echo "[]"; exit 0; fi\n'
        'if [ "$1 $2" = "plugin uninstall" ]; then exit 0; fi\n'
        "exit 1\n"
    )
    claude.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")


def test_setup_cli_and_doctor_agree(home, claude_with_plugin, capsys, monkeypatch):
    top_level_main(["setup", "claude-code", "--memory-backend", "context-graph", "--no-tui", "--no-viewer"])
    out = capsys.readouterr().out
    assert "OK    Claude Code memory: memory.backend = context-graph" in out
    assert "Set memory.backend = context-graph" in out
    _identity._reset_cache()

    from memgraph_context_graph.cli import _check_memory_backend

    check = _check_memory_backend("claude-code")
    assert check is not None and check["ok"]

    # A project's settings outrank the user's, so one can turn auto memory back on there.
    project = home.parent / "project"
    (project / ".claude").mkdir(parents=True)
    monkeypatch.chdir(project)
    (project / ".claude" / "settings.json").write_text(json.dumps({"autoMemoryEnabled": True}))
    check = _check_memory_backend("claude-code")
    assert check is not None and not check["ok"] and str(project / ".claude" / "settings.json") in check["detail"]
    (project / ".claude" / "settings.local.json").write_text(json.dumps({"autoMemoryEnabled": False}))
    check = _check_memory_backend("claude-code")
    assert check is not None and check["ok"]

    (home / ".claude" / "settings.json").write_text("{}")
    check = _check_memory_backend("claude-code")
    assert check is not None and not check["ok"] and "two memories" in check["detail"]


def test_uninstall_gives_claude_code_its_own_memory_back(home, claude_with_plugin, capsys):
    """Without the plugin there would be no memory at all, so uninstall turns Claude Code's own back on."""
    use_graph_memory(PLUGIN, home=home)
    assert _settings(home)["autoMemoryEnabled"] is False

    assert top_level_main(["uninstall", "claude-code"]) == 0

    out = capsys.readouterr().out
    assert "OK    Claude Code memory: its own memory is back on" in out
    assert _settings(home) == {"theme": "dark"}
    assert (_memory_dir(home, "ai-toolkit") / "feedback_trailers.md").exists()
    assert _identity.load_config().memory_backend == "native"


def test_setup_hands_every_named_agent_back_before_the_shared_backend_changes(home, claude_with_plugin, monkeypatch):
    """Codex first, then Claude Code: Claude Code's copy-back must still see the graph as the backend."""
    from memgraph_context_graph.adapters.codex import PLUGIN as CODEX_PLUGIN

    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    use_graph_memory(PLUGIN, home=home)
    use_graph_memory(CODEX_PLUGIN, home=home)
    # Setup reads the graph the config names, so the file to copy back is one the import put there.
    native = _memory_dir(home, "ai-toolkit") / "project_map.md"
    native.unlink()

    top_level_main(["setup", "codex", "claude-code", "--memory-backend", "native", "--no-tui", "--no-viewer"])

    assert native.read_text().endswith("Map #484.\n")
    assert _settings(home) == {"theme": "dark"}
    assert _identity.load_config().memory_backend == "native"


@pytest.mark.parametrize("runtimes", [("codex", "claude-code"), ("claude-code", "codex")])
def test_a_failed_hand_back_keeps_the_graph_backend_until_retry(home, claude_with_plugin, monkeypatch, runtimes):
    from memgraph_context_graph.adapters.codex import PLUGIN as CODEX_PLUGIN

    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    use_graph_memory(PLUGIN, home=home)
    use_graph_memory(CODEX_PLUGIN, home=home)
    backup = home / ".codex" / "config.toml.context-graph-memory.bak"
    original = backup.read_bytes()
    backup.write_text("[broken")
    native = _memory_dir(home, "ai-toolkit") / "project_map.md"
    native.unlink()
    command = ["setup", *runtimes, "--memory-backend", "native", "--no-tui", "--no-viewer"]

    top_level_main(command)

    assert _identity.load_config().graph_memory
    backup.write_bytes(original)

    top_level_main(command)

    assert native.read_text().endswith("Map #484.\n")
    assert _settings(home) == {"theme": "dark"}
    assert _identity.load_config().memory_backend == "native"
