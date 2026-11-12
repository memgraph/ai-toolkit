"""The shipped Runtime Plugins' static hooks.json files must match their adapters.

Plugin users get hooks from these files, not from `build_hooks_config`, so a
hook the adapter handles but the plugin never registers silently never fires.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memgraph_context_graph.adapters import claude_code, codex

_PLUGINS = Path(__file__).resolve().parents[2] / "plugins"


@pytest.mark.parametrize(
    ("plugin_dir", "spec"),
    [
        pytest.param("memgraph-context-graph-claude", claude_code.SPEC, id="claude-code"),
        pytest.param("memgraph-context-graph-codex", codex.SPEC, id="codex"),
    ],
)
def test_plugin_registers_every_adapter_hook(plugin_dir, spec):
    hooks = json.loads((_PLUGINS / plugin_dir / "hooks" / "hooks.json").read_text())["hooks"]

    assert set(hooks) == set(spec.hooks)
    for event, entries in hooks.items():
        commands = [handler["command"] for entry in entries for handler in entry["hooks"]]
        assert all(re.search(rf"hook run {spec.name}([ ;]|$)", command) for command in commands), event
        assert [entry.get("matcher") for entry in entries] == [spec.config.matchers.get(event)], event


_PLUGIN_DIRS = ["memgraph-context-graph-claude", "memgraph-context-graph-codex"]
_SH = shutil.which("sh") or "/bin/sh"


def _hook_commands(plugin_dir):
    hooks = json.loads((_PLUGINS / plugin_dir / "hooks" / "hooks.json").read_text())["hooks"]
    return [(event, handler["command"]) for event, entries in hooks.items() for e in entries for handler in e["hooks"]]


@pytest.mark.parametrize("plugin_dir", _PLUGIN_DIRS)
def test_hooks_without_the_tool_stay_quiet_and_session_start_says_why(plugin_dir, tmp_path):
    """A plugin installed from the marketplace without the tool must not fail on every event."""
    for event, command in _hook_commands(plugin_dir):
        result = subprocess.run(
            [_SH, "-c", command], input="{}", capture_output=True, text=True, env={"PATH": str(tmp_path)}
        )
        assert result.returncode == 0, event
        if event == "SessionStart":
            assert "memgraph-context-graph tool its hooks call is not" in result.stdout
        else:
            assert result.stdout == "", event


@pytest.mark.parametrize("plugin_dir", _PLUGIN_DIRS)
def test_hooks_with_the_tool_pass_the_event_to_it(plugin_dir, tmp_path):
    tool = tmp_path / "memgraph-context-graph"
    tool.write_text('#!/bin/sh\necho "args: $*"\ncat\n')
    tool.chmod(0o755)
    for event, command in _hook_commands(plugin_dir):
        result = subprocess.run(
            [_SH, "-c", command],
            input='{"event": 1}',
            capture_output=True,
            text=True,
            env={"PATH": f"{tmp_path}{os.pathsep}/usr/bin{os.pathsep}/bin"},
        )
        assert result.returncode == 0, event
        assert result.stdout.startswith("args: hook run "), event
        assert result.stdout.endswith('{"event": 1}'), event


def test_codex_hooks_keep_the_plain_command_on_windows():
    """Codex may not run a POSIX shell on Windows, so it gets the command without the check."""
    hooks = json.loads((_PLUGINS / "memgraph-context-graph-codex" / "hooks" / "hooks.json").read_text())["hooks"]
    for event, entries in hooks.items():
        for handler in (h for e in entries for h in e["hooks"]):
            assert handler["commandWindows"].startswith("memgraph-context-graph hook run codex"), event
            assert handler["commandWindows"] in handler["command"], event
