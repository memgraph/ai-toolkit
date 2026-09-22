"""Tests for the runtime plugin registry (entry_points-based discovery)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_context_graph import AgentLink
from agent_context_graph.hooks.runtime_plugin import (
    RuntimeCLIPlugin,
    UnknownRuntimeError,
    get_runtime_plugin,
    load_runtime_plugins,
)
from agent_context_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from agent_context_graph.events import Event

EXPECTED_RUNTIMES = {"codex", "claude-code", "gemini-cli", "copilot-cli", "cursor", "opencode"}


class _RecordingConnector(GraphConnector):
    def __init__(self):
        self.events: list[Event] = []

    def on_event(self, event: Event) -> None:
        self.events.append(event)


def test_load_runtime_plugins_discovers_all_builtin_runtimes():
    plugins = load_runtime_plugins()

    assert set(plugins) == EXPECTED_RUNTIMES
    assert plugins["codex"].name == "codex"
    assert plugins["claude-code"].name == "claude-code"


@pytest.mark.parametrize("runtime", sorted(EXPECTED_RUNTIMES))
def test_runtime_plugin_contract(runtime):
    plugin = get_runtime_plugin(runtime)
    connector = _RecordingConnector()
    link = AgentLink()
    link.add_connector(connector)

    assert isinstance(plugin, RuntimeCLIPlugin)
    adapter = plugin.adapter_class(link)
    adapter.handle_payload(dict(plugin.probe_payload))

    assert connector.events
    assert all(event.source_sdk for event in connector.events)
    assert isinstance(plugin.build_hooks_config("capture"), dict)


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
    assert callable(getattr(plugin, "init", None))


def test_claude_code_plugin_has_no_init():
    plugin = get_runtime_plugin("claude-code")

    from agent_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter

    assert plugin.adapter_class is ClaudeCodeHooksAdapter
    assert getattr(plugin, "init", None) is None
