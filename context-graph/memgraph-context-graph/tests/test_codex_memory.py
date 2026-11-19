"""Codex memory switches preserve user settings in a throwaway home."""

from __future__ import annotations

import pytest

from memgraph_context_graph.adapters.codex_memory import CodexMemory


@pytest.fixture
def codex_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    return tmp_path, path


@pytest.mark.parametrize(
    "original",
    [
        "[features]\nmemories = true\n",
        "[features]\nmemories = false\n[memories]\nuse_memories = false\ngenerate_memories = false\n",
    ],
)
def test_repeating_hand_back_preserves_restored_settings(codex_config, original):
    home, config = codex_config
    config.write_text(original)
    # Setup's older backup must not become the restoration source on a later retry.
    config.with_name(config.name + ".context-graph.bak").write_text("[features]\nmemories = true\n")
    native = CodexMemory()
    native.disable(home)
    native.enable(home)
    restored = config.read_bytes()

    native.enable(home)

    assert config.read_bytes() == restored


def test_hand_back_without_a_switch_preserves_native_settings(codex_config):
    home, config = codex_config
    original = b"[features]\nmemories = true\n[memories]\ngenerate_memories = false\n"
    config.write_bytes(original)

    CodexMemory().enable(home)

    assert config.read_bytes() == original


def test_hand_back_without_a_config_does_not_create_one(codex_config):
    home, config = codex_config

    CodexMemory().enable(home)

    assert not config.exists()


def test_a_legacy_backup_is_not_restored_as_the_missing_switch_backup(codex_config):
    home, config = codex_config
    original = b"[features]\nmemories = false\n[memories]\nuse_memories = false\ngenerate_memories = false\n"
    config.write_bytes(original)
    backup = config.with_name(config.name + ".context-graph.bak")
    backup.write_text("[features]\nmemories = true\n")

    lines = CodexMemory().enable(home)

    assert config.read_bytes() == original
    assert backup.read_text() == "[features]\nmemories = true\n"
    assert any("To turn Codex memories on" in line and "wasn't restored automatically" in line for line in lines)
