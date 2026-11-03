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
        "Memory index (view a file for its full text; this project's files first, then the newest):",
        "- /memories/projects/repo/decisions.md — Decisions",
        "- /memories/feedback/tests.md — Prefer real Memgraph",
    ]


def test_index_ranks_this_project_then_what_it_wrote_then_the_newest(graph):
    """After a user-wide import, other projects' root notes must not crowd out this one's."""
    importer = graph.memory_store("alice")
    for n in range(100):
        importer.create(f"/memories/feedback/{n:04d}-elsewhere.md", _long_description(n))
    importer.create("/memories/projects/repo/decisions.md", "---\ndescription: We chose X\n---\n")
    graph.record_session_start("s-here", user_id="alice", project_key="repo")
    graph.memory_store("alice", session_id="s-here").create(
        "/memories/feedback/here.md", "---\ndescription: Mine\n---\n"
    )
    graph.record_session_start("s-there", user_id="alice", project_key="other")
    graph.memory_store("alice", session_id="s-there").create(
        "/memories/feedback/there.md", "---\ndescription: Not\n---\n"
    )

    lines = render_index(importer, "repo", guidance=GUIDANCE, max_chars=2_000).split("\n")

    entries = [line.split(" — ")[0] for line in lines if line.startswith("- /memories/")]
    assert entries[:4] == [
        "- /memories/projects/repo/decisions.md",
        "- /memories/feedback/here.md",
        "- /memories/feedback/there.md",
        "- /memories/feedback/0099-elsewhere.md",
    ]
    assert lines[-1] == f"- ...and {103 - len(entries)} more; view /memories to list them."


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


def _long_description(n: int) -> str:
    return f"---\ndescription: note {n} {'about something worth remembering ' * 5}\n---\nbody"


def test_index_lists_every_file_and_describes_the_most_relevant_first(store):
    for n in range(60):
        store.create(f"/memories/feedback/{n:04d}.md", _long_description(n))

    text = render_index(store, None, guidance=GUIDANCE, max_chars=10_000)

    entries = [line for line in text.split("\n") if line.startswith("- /memories/")]
    described = [" — note " in entry for entry in entries]
    assert len(text) <= 10_000
    assert len(entries) == 60
    assert entries[0].startswith("- /memories/feedback/0059.md — note 59")
    assert 0 < sum(described) < 60
    assert described == sorted(described, reverse=True)
    assert len(render_index(store, None, guidance=GUIDANCE)) > 10_000


def test_index_cut_to_the_budget_says_how_many_it_left_out(store):
    for n in range(400):
        store.create(f"/memories/n/{n:04d}.md", f"note {n}")

    lines = render_index(store, None, guidance=GUIDANCE, max_chars=2_000).split("\n")

    listed = sum(line.startswith("- /memories/") for line in lines)
    assert len("\n".join(lines)) <= 2_000
    assert lines[-1] == f"- ...and {400 - listed} more; view /memories to list them."
    assert lines[-2].split(" — ")[0] == f"- /memories/n/{400 - listed:04d}.md"


def test_pinned_text_takes_at_most_half_the_budget(store):
    store.create("/memories/a-rules.md", "---\npin: true\n---\nShort rule.")
    store.create("/memories/b-essay.md", "---\npin: true\n---\n" + "long " * 1_500)
    store.create("/memories/c-note.md", "---\ndescription: still listed\n---\n")

    text = render_index(store, None, guidance=GUIDANCE, max_chars=10_000)

    assert "Short rule." in text
    assert '<memory path="/memories/b-essay.md">' not in text
    assert "[/memories/b-essay.md is over the pinned-memory budget; view it.]" in text
    assert "- /memories/c-note.md — still listed" in text


def test_empty_memory_says_so(store):
    assert render_index(store, "repo", guidance=GUIDANCE).endswith(
        "No memory files yet in /memories or /memories/projects/repo/."
    )


def test_session_start_hook_output_carries_the_index(graph, tmp_path, monkeypatch, capsys):
    """The SessionStart hook as a harness runs it, for an opted-in user in a git checkout."""
    pytest.importorskip("memgraph_context_graph")
    from memgraph_context_graph.adapters import _identity
    from memgraph_context_graph.hooks.runner import session_start_text

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

    context = session_start_text(["sessions-graph"], {"cwd": str(repo)})

    assert context is not None

    assert "Current project folder: /memories/projects/ai-toolkit/" in context
    assert "- /memories/projects/ai-toolkit/plan.md — The plan" in context
    assert "call the `recall` tool" in context
