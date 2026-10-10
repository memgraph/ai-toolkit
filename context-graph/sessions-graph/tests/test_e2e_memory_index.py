"""The session-start memory index, generated from a real Memgraph's memory files."""

from __future__ import annotations

import os
import subprocess

import pytest
from sessions_graph.memory_index import MAX_INDEX_LINES, _key_from_remote, project_key_for, render_index

GUIDANCE = "Memory lives in Context Graph."


@pytest.fixture
def store(graph):
    return graph.memory_store("alice")


@pytest.mark.parametrize(
    "remote",
    [
        "git@github.com:memgraph/ai-toolkit.git",
        "https://github.com/memgraph/ai-toolkit.git",
        "https://user:token@github.com/memgraph/ai-toolkit",
        "ssh://git@github.com:22/memgraph/ai-toolkit.git",
        "https://GitHub.com/memgraph/ai-toolkit/",
    ],
)
def test_every_clone_url_of_a_repo_maps_to_one_key(remote):
    assert _key_from_remote(remote) == "github.com_memgraph_ai-toolkit"


def test_project_key_from_a_checkout(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "sub").mkdir()

    assert project_key_for(repo / "sub") == "checkout"
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:memgraph/ai-toolkit.git"], check=True
    )
    assert project_key_for(repo / "sub") == "github.com_memgraph_ai-toolkit"


def test_project_key_outside_git_and_for_missing_directories(tmp_path):
    plain = tmp_path / "notes dir"
    plain.mkdir()
    assert project_key_for(plain) == "notes_dir"
    assert project_key_for(tmp_path / "missing") is None


def test_index_lists_root_and_current_project_only(store):
    store.create("/memories/feedback/tests.md", "---\ndescription: Prefer real Memgraph\ntype: feedback\n---\nbody")
    store.create("/memories/projects/repo/decisions.md", "# Decisions\nwe chose X")
    store.create("/memories/projects/other/secret.md", "not this project")

    text = render_index(store, "repo", guidance=GUIDANCE)

    assert text.split("\n") == [
        GUIDANCE,
        "Current project folder: /memories/projects/repo/",
        "Memory index (view a file for its full text):",
        "- /memories/feedback/tests.md — Prefer real Memgraph",
        "- /memories/projects/repo/decisions.md — Decisions",
    ]


def test_index_reads_descriptions_nested_like_claude_codes_own_files(store):
    store.create("/memories/u.md", "---\nname: x\nmetadata:\n  type: user\n  description: nested one\n---\nbody")
    assert "- /memories/u.md — nested one" in render_index(store, None, guidance=GUIDANCE)


def test_pinned_files_are_shown_whole(store):
    store.create(
        "/memories/feedback/trailers.md",
        "---\ndescription: No trailers\npin: true\n---\nNever add attribution trailers.",
    )
    store.create("/memories/other.md", "---\ndescription: not pinned\n---\nhidden body")

    text = render_index(store, None, guidance=GUIDANCE)

    assert '<memory path="/memories/feedback/trailers.md">\nNever add attribution trailers.\n</memory>' in text
    assert "hidden body" not in text


def test_index_is_capped(store):
    for n in range(MAX_INDEX_LINES + 5):
        store.create(f"/memories/n/{n:04d}.md", f"note {n}")

    lines = render_index(store, None, guidance=GUIDANCE).split("\n")

    assert sum(line.startswith("- /memories/") for line in lines) == MAX_INDEX_LINES
    assert lines[-1] == "- ...and 5 more; view /memories to list them."


def test_empty_memory_says_so(store):
    assert render_index(store, "repo", guidance=GUIDANCE).endswith(
        "No memory files yet in /memories or /memories/projects/repo/."
    )


def test_session_start_hook_output_carries_the_index(graph, tmp_path, monkeypatch, capsys):
    """The SessionStart hook as a harness runs it, for an opted-in user in a git checkout."""
    pytest.importorskip("agent_context_graph")
    from agent_context_graph.adapters import _identity
    from agent_context_graph.hooks.runner import session_start_context

    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(
        user_id="alice",
        memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        memory_backend="context-graph",
    )
    repo = tmp_path / "ai-toolkit"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    graph.memory_store("alice").create("/memories/projects/ai-toolkit/plan.md", "---\ndescription: The plan\n---\n")

    context = session_start_context(["sessions-graph"], {"cwd": str(repo)})["hookSpecificOutput"]["additionalContext"]

    assert "Current project folder: /memories/projects/ai-toolkit/" in context
    assert "- /memories/projects/ai-toolkit/plan.md — The plan" in context
    assert "call the `recall` tool" in context
