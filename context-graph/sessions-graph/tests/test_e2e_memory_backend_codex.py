"""Switching Codex's memory to Context Graph and back, with a real Memgraph and a throwaway home."""

from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("memgraph_context_graph")

import tomlkit

from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.adapters.codex import PLUGIN, SPEC
from memgraph_context_graph.memory_backend import MemoryBackendError, use_graph_memory, use_native_memory

ORIGINAL = """# my settings
model = "gpt-x"

[features]
memories = true # I like them

[mcp_servers.context-graph]
command = "memgraph-context-graph"
args = ["mcp"]
enabled_tools = ["recall"]
"""


@pytest.fixture
def home(tmp_path, monkeypatch, memgraph):
    home = tmp_path / "home"
    codex = home / ".codex"
    (codex / "memories").mkdir(parents=True)
    (codex / "config.toml").write_text(ORIGINAL)
    (codex / "memories" / "memory_summary.md").write_text("v1\n## User Profile\nPrefers uv.\n")
    (codex / "memories" / "MEMORY.md").write_text("# Task Group: tooling\nscope: global\n")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(user_id="alice", memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    yield home
    _identity._reset_cache()


def _config(home):
    return tomlkit.parse((home / ".codex" / "config.toml").read_text())


def test_opting_in_turns_codex_memories_off_lets_the_tool_through_and_imports(home, graph):
    assert PLUGIN.native_memory.enabled(home)

    use_graph_memory(PLUGIN, home=home)

    config = _config(home)
    assert config["features"]["memories"] is False
    assert config["memories"] == {"use_memories": False, "generate_memories": False}
    assert config["mcp_servers"]["context-graph"]["enabled_tools"] == ["recall", "memory"]
    assert config["model"] == "gpt-x"
    assert "# my settings" in (home / ".codex" / "config.toml").read_text()
    assert not PLUGIN.native_memory.enabled(home)
    assert _identity.load_config().graph_memory

    files = {f.path: f.content for f in graph.memory_store("alice").files()}
    # Codex's summary spans every project, so it stays a Codex reference rather than a user-wide memory.
    assert set(files) == {"/memories/codex/memory-summary.md", "/memories/codex/memory.md"}
    assert files["/memories/codex/memory-summary.md"].startswith("---\ndescription: What Codex had learned")
    assert files["/memories/codex/memory-summary.md"].endswith("Prefers uv.\n")


def test_handing_memory_back_restores_what_was_there(home):
    use_graph_memory(PLUGIN, home=home)
    use_native_memory(PLUGIN, home=home)

    config = _config(home)
    assert config["features"]["memories"] is True
    assert "use_memories" not in config.get("memories", {})
    assert PLUGIN.native_memory.enabled(home)
    assert not _identity.load_config().graph_memory


def test_a_round_trip_gives_back_the_config_byte_for_byte_with_no_empty_table(home):
    config = home / ".codex" / "config.toml"
    original = b'# mine\r\nmodel = "gpt-x"\r\n\r\n[features]\r\nhooks = true\r\n'
    config.write_bytes(original)

    use_graph_memory(PLUGIN, home=home)
    assert b"\r\n[memories]\r\n" in config.read_bytes()
    use_native_memory(PLUGIN, home=home)

    assert config.read_bytes() == original
    assert not (home / ".codex" / "config.toml.context-graph-memory.bak").exists()


def test_handing_back_restores_the_state_before_the_switch_not_an_older_setup_backup(home):
    """Setup's backup of config.toml can be weeks older than the switch; it must not decide what comes back."""
    (home / ".codex" / "config.toml.context-graph.bak").write_text("[features]\nmemories = false\n")

    use_graph_memory(PLUGIN, home=home)
    use_graph_memory(PLUGIN, home=home)  # setup re-run while on the graph: the keys are off, the backup stays
    use_native_memory(PLUGIN, home=home)

    assert _config(home)["features"]["memories"] is True
    assert PLUGIN.native_memory.enabled(home)


def _notes(home, store="memories"):
    folder = home / ".codex" / store / "extensions" / "ad_hoc" / "notes"
    return {path.name: path.read_text() for path in sorted(folder.glob("*.md"))}


def test_handing_memory_back_leaves_the_graphs_memories_as_notes_codex_consolidates(home, graph):
    use_graph_memory(PLUGIN, home=home)
    store = graph.memory_store("alice")
    store.create("/memories/feedback/trailers.md", "---\ndescription: No attribution trailers\n---\nNever add them.\n")
    store.create("/memories/projects/github.com_memgraph_ai-toolkit/map.md", "Map #484.\n")
    store.str_replace("/memories/codex/memory.md", "scope: global", "scope: everywhere")

    lines = use_native_memory(PLUGIN, home=home)

    notes = _notes(home)
    # Codex's note-file contract, which its own add_ad_hoc_note tool enforces.
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]{0,79}\.md", name) for name in notes)
    by_slug = {name[20:]: text for name, text in notes.items()}
    # The summary came from Codex and is unchanged there, so it isn't noted again; the edited handbook is.
    assert set(by_slug) == {
        "context-graph-feedback-trailers.md",
        "context-graph-projects-github-com-memgraph-ai-toolkit-map.md",
        "context-graph-codex-memory.md",
    }
    assert by_slug["context-graph-feedback-trailers.md"].endswith("Never add them.\n")
    assert "applies to the user everywhere" in by_slug["context-graph-feedback-trailers.md"]
    assert (
        "only in the project `github.com_memgraph_ai-toolkit`"
        in by_slug["context-graph-projects-github-com-memgraph-ai-toolkit-map.md"]
    )
    assert "scope: everywhere" in by_slug["context-graph-codex-memory.md"]
    assert "Copied back /memories/feedback/trailers.md" in lines
    assert any("folds them into its memory at its next consolidation" in line for line in lines)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_handing_memory_back_twice_never_leaves_the_same_note_twice(home, graph, newline):
    use_graph_memory(PLUGIN, home=home)
    graph.memory_store("alice").create(
        "/memories/user/role.md", newline.join(["---", "description: My role", "---", "Works on Memgraph.", ""])
    )
    use_native_memory(PLUGIN, home=home)

    use_graph_memory(PLUGIN, home=home)
    use_native_memory(PLUGIN, home=home)

    assert len(_notes(home)) == 1


def test_notes_go_to_the_v2_store_and_say_they_wait_while_codex_memories_were_off(home, graph):
    (home / ".codex" / "config.toml").write_text('[memories]\nversion = "v2"\n')
    use_graph_memory(PLUGIN, home=home)
    graph.memory_store("alice").create("/memories/user/role.md", "Works on Memgraph.\n")

    lines = use_native_memory(PLUGIN, home=home)

    assert not _notes(home) and len(_notes(home, "memories_v2")) == 1
    assert any("folds them in only once its memories are on" in line for line in lines)


def test_codex_home_override_is_honoured(home, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(other))

    PLUGIN.native_memory.disable(home)

    assert tomlkit.parse((other / "config.toml").read_text())["features"]["memories"] is False
    assert _config(home)["features"]["memories"] is True


def test_broken_config_stops_the_switch_before_anything_changes(home):
    (home / ".codex" / "config.toml").write_text("[broken")

    with pytest.raises(MemoryBackendError, match="not valid TOML"):
        PLUGIN.native_memory.disable(home)


def test_session_start_fires_after_compaction_too():
    assert SPEC.config.matchers["SessionStart"] == "startup|resume|clear|compact"
