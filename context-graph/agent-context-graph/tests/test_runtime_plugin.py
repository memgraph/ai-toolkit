"""Tests for the runtime plugin registry (entry_points-based discovery)."""

from __future__ import annotations

import pytest

from agent_context_graph.hooks.runtime_plugin import (
    UnknownRuntimeError,
    get_runtime_plugin,
    load_runtime_plugins,
)


def test_load_runtime_plugins_discovers_builtin_codex_and_claude_code():
    plugins = load_runtime_plugins()

    assert set(plugins) == {"codex", "claude-code"}
    assert plugins["codex"].name == "codex"
    assert plugins["claude-code"].name == "claude-code"


def test_get_runtime_plugin_normalizes_underscores_and_hyphens():
    assert get_runtime_plugin("claude_code").name == "claude-code"
    assert get_runtime_plugin("claude-code").name == "claude-code"
    assert get_runtime_plugin("codex").name == "codex"


def test_get_runtime_plugin_raises_on_unknown_runtime():
    with pytest.raises(UnknownRuntimeError, match="Unknown runtime: 'made-up'"):
        get_runtime_plugin("made-up")


def test_codex_plugin_exposes_full_protocol():
    plugin = get_runtime_plugin("codex")

    from agent_context_graph.adapters.codex import CodexHooksAdapter

    assert plugin.adapter_class is CodexHooksAdapter
    assert plugin.response_for_payload({"hook_event_name": "Stop"}) == {"continue": True}
    assert "SessionStart" in plugin.build_hooks_config("some-command")
    assert callable(plugin.init)


def test_claude_code_plugin_has_no_init():
    plugin = get_runtime_plugin("claude-code")

    from agent_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter

    assert plugin.adapter_class is ClaudeCodeHooksAdapter
    assert getattr(plugin, "init", None) is None
