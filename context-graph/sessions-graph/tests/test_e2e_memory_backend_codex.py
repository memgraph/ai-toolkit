"""Switching Codex's memory to Context Graph and back, with a real Memgraph and a throwaway home."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("agent_context_graph")

import tomlkit

from agent_context_graph.adapters import _identity
from agent_context_graph.adapters.codex import PLUGIN, SPEC
from agent_context_graph.memory_backend import MemoryBackendError, use_graph_memory, use_native_memory

ORIGINAL = """# my settings
model = "gpt-x"

[features]
memories = true # I like them

[mcp_servers.context-graph]
command = "agent-context-graph"
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
    assert set(files) == {"/memories/user/codex-memory-summary.md", "/memories/codex/memory.md"}
    assert files["/memories/user/codex-memory-summary.md"].startswith("---\ndescription: What Codex had learned")
    assert files["/memories/user/codex-memory-summary.md"].endswith("Prefers uv.\n")


def test_handing_memory_back_restores_what_was_there(home):
    use_graph_memory(PLUGIN, home=home)
    use_native_memory(PLUGIN, home=home)

    config = _config(home)
    assert config["features"]["memories"] is True
    assert "use_memories" not in config.get("memories", {})
    assert PLUGIN.native_memory.enabled(home)
    assert not _identity.load_config().graph_memory


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
