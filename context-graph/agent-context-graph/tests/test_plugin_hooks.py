"""The shipped Runtime Plugins' static hooks.json files must match their adapters.

Plugin users get hooks from these files, not from `build_hooks_config`, so a
hook the adapter handles but the plugin never registers silently never fires.
"""

import json
from pathlib import Path

import pytest

from agent_context_graph.adapters import claude_code, codex

_PLUGINS = Path(__file__).resolve().parents[2] / "plugins"


@pytest.mark.parametrize(
    ("plugin_dir", "spec"),
    [
        pytest.param("agent-context-graph-claude", claude_code.SPEC, id="claude-code"),
        pytest.param("agent-context-graph-codex", codex.SPEC, id="codex"),
    ],
)
def test_plugin_registers_every_adapter_hook(plugin_dir, spec):
    hooks = json.loads((_PLUGINS / plugin_dir / "hooks" / "hooks.json").read_text())["hooks"]

    assert set(hooks) == set(spec.hooks)
    for event, entries in hooks.items():
        commands = [handler["command"] for entry in entries for handler in entry["hooks"]]
        assert all(f"hook run {spec.name} " in f"{command} " for command in commands), event
        assert [entry.get("matcher") for entry in entries] == [spec.config.matchers.get(event)], event
