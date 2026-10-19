"""The project-wired agents' user-level MCP registration and memory switches, against a throwaway home.

Pure file edits on each agent's own config: what is added, that the user's own entries survive, and that
undoing it leaves the file as it was. Whether the agent then really calls the tools is the live harness's
job (``scripts/live-hooks-e2e``).
"""

from __future__ import annotations

import json

import pytest
import tomlkit

from memgraph_context_graph.adapters.agent_mcp import AGENT_MCP, SERVER, server_argv
from memgraph_context_graph.adapters.agent_memory import COPILOT_MEMORY, GROK_MEMORY, NO_BUILT_IN_MEMORY
from memgraph_context_graph.memory_backend import MemoryBackendError
from memgraph_context_graph.setup import agents
from memgraph_context_graph.setup.agents import BY_RUNTIME


class Files:
    """An in-memory ``/memories`` tree with the two calls an import makes."""

    def __init__(self, existing=()):
        self.files: dict[str, str] = dict.fromkeys(existing, "")

    def view(self, path, view_range=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def create(self, path, file_text):
        self.files[path] = file_text
        return "created"


def _json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("runtime", "config", "servers", "allow_file", "rule"),
    [
        ("copilot-cli", ".copilot/mcp-config.json", ("mcpServers",), None, None),
        ("cursor", ".cursor/mcp.json", ("mcpServers",), ".cursor/cli-config.json", "Mcp(context-graph:*)"),
        (
            "antigravity-cli",
            ".gemini/config/mcp_config.json",
            ("mcpServers",),
            ".gemini/antigravity-cli/settings.json",
            "mcp(context-graph/*)",
        ),
    ],
)
def test_json_agents_register_the_server_beside_the_users_own(tmp_path, runtime, config, servers, allow_file, rule):
    wiring = AGENT_MCP[runtime]
    path = tmp_path / config
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"mcpServers": {"mine": {"command": "x"}}, "other": 1}))
    if allow_file:
        (tmp_path / allow_file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / allow_file).write_text(json.dumps({"permissions": {"allow": ["shell(ls)"]}, "theme": "dark"}))

    wiring.register(tmp_path)
    wiring.register(tmp_path)

    document = _json(path)
    assert document["other"] == 1
    assert document["mcpServers"]["mine"] == {"command": "x"}
    entry = document["mcpServers"][SERVER]
    assert [entry["command"], *entry["args"]] == server_argv()
    assert wiring.registered(tmp_path)
    if allow_file:
        assert _json(tmp_path / allow_file) == {"permissions": {"allow": ["shell(ls)", rule]}, "theme": "dark"}

    wiring.unregister(tmp_path)

    assert _json(path) == {"mcpServers": {"mine": {"command": "x"}}, "other": 1}
    assert not wiring.registered(tmp_path)
    if allow_file:
        assert _json(tmp_path / allow_file) == {"permissions": {"allow": ["shell(ls)"]}, "theme": "dark"}


def test_copilot_server_lists_every_tool(tmp_path):
    AGENT_MCP["copilot-cli"].register(tmp_path)
    entry = _json(tmp_path / ".copilot" / "mcp-config.json")["mcpServers"][SERVER]
    assert entry["type"] == "local"
    assert entry["tools"] == ["*"]


def test_grok_gets_a_server_that_sends_the_session_text_and_an_allow_rule(tmp_path):
    path = tmp_path / ".grok" / "config.toml"
    path.parent.mkdir()
    original = '[ui]\npermission_mode = "default"\n\n[mcp_servers.mine]\ncommand = "x"\n'
    path.write_text(original)
    wiring = AGENT_MCP["grok"]

    wiring.register(tmp_path)
    wiring.register(tmp_path)

    config = tomlkit.parse(path.read_text())
    server = config["mcp_servers"][SERVER]
    assert server["args"][-2:] == ["mcp", "--session-context"]
    assert config["mcp_servers"]["mine"]["command"] == "x"
    assert list(config["permission"]["allow"]) == ["MCPTool(context-graph__*)"]

    wiring.unregister(tmp_path)

    config = tomlkit.parse(path.read_text())
    assert SERVER not in config["mcp_servers"]
    assert list(config["permission"]["allow"]) == []


def test_a_broken_config_is_reported_not_overwritten(tmp_path):
    path = tmp_path / ".cursor" / "mcp.json"
    path.parent.mkdir()
    path.write_text("{not json")
    with pytest.raises(MemoryBackendError, match="not valid JSON"):
        AGENT_MCP["cursor"].register(tmp_path)
    assert path.read_text() == "{not json"


def test_copilot_memory_turns_off_and_back_to_what_it_was(tmp_path):
    settings = tmp_path / ".copilot" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"model": "gpt-5"}))
    assert COPILOT_MEMORY.enabled(tmp_path)

    COPILOT_MEMORY.disable(tmp_path)
    assert _json(settings) == {"model": "gpt-5", "memory": False}
    assert not COPILOT_MEMORY.enabled(tmp_path)

    COPILOT_MEMORY.enable(tmp_path)
    assert _json(settings) == {"model": "gpt-5"}
    assert COPILOT_MEMORY.import_into(Files(), tmp_path).imported == []


def test_grok_memory_turns_off_and_back_to_what_it_was(tmp_path):
    path = tmp_path / ".grok" / "config.toml"
    path.parent.mkdir()
    path.write_text('[ui]\ntheme = "dark"\n')
    assert GROK_MEMORY.enabled(tmp_path)

    GROK_MEMORY.disable(tmp_path)
    assert tomlkit.parse(path.read_text())["memory_v2"]["enabled"] is False
    assert not GROK_MEMORY.enabled(tmp_path)

    GROK_MEMORY.enable(tmp_path)
    assert tomlkit.parse(path.read_text()) == {"ui": {"theme": "dark"}}


def test_grok_topics_are_imported_with_a_description_and_never_overwrite(tmp_path):
    root = tmp_path / ".grok" / "memory-v2"
    (root / "global" / "topics").mkdir(parents=True)
    (root / "global" / "topics" / "style.md").write_text("# Coding style\nPrefer small functions.\n")
    (root / "global" / "MEMORY.md").write_text("# generated index\n")
    (root / "workspaces" / "billing-ae57e6f0" / "topics").mkdir(parents=True)
    (root / "workspaces" / "billing-ae57e6f0" / "topics" / "deploy.md").write_text("Deploys go through Argo.\n")
    files = Files(existing=["/memories/grok/billing/deploy.md"])

    report = GROK_MEMORY.import_into(files, tmp_path)

    assert report.imported == ["/memories/grok/style.md"]
    assert report.skipped == ["/memories/grok/billing/deploy.md"]
    assert files.files["/memories/grok/style.md"].startswith("---\ndescription: Coding style\ntype: reference\n---\n")


def test_an_agent_without_memory_of_its_own_has_nothing_to_switch(tmp_path):
    assert not NO_BUILT_IN_MEMORY.enabled(tmp_path)
    assert NO_BUILT_IN_MEMORY.disable(tmp_path) == []
    assert NO_BUILT_IN_MEMORY.import_into(Files(), tmp_path).imported == []


def test_setup_registers_the_memory_tools_and_uninstall_removes_them(setup_home, tmp_path):
    project = tmp_path / "billing"
    project.mkdir()
    copilot = BY_RUNTIME["copilot-cli"]

    assert agents.register_mcp(copilot, lambda _: None) == "memory tools registered"
    assert agents.register_mcp(copilot, lambda _: None) == "memory tools already registered"
    assert agents.mcp_registered(copilot)

    assert agents.unregister_mcp(copilot, lambda _: None)
    assert not agents.mcp_registered(copilot)
    assert _json(setup_home / ".copilot" / "mcp-config.json") == {"mcpServers": {}}


def test_opencode_gets_its_v2_server_shape_with_tools_of_their_own(tmp_path):
    path = tmp_path / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"model": "anthropic/claude-sonnet-5-5", "permissions": [{"action": "*", "effect": "ask"}]})
    )
    wiring = AGENT_MCP["opencode"]

    wiring.register(tmp_path)
    wiring.register(tmp_path)

    document = _json(path)
    entry = document["mcp"]["servers"][SERVER]
    assert entry["command"] == server_argv()
    assert entry["codemode"] is False
    assert document["permissions"] == [
        {"action": "*", "effect": "ask"},
        {"action": "context-graph_*", "resource": "*", "effect": "allow"},
    ]

    wiring.unregister(tmp_path)

    assert _json(path) == {
        "model": "anthropic/claude-sonnet-5-5",
        "permissions": [{"action": "*", "effect": "ask"}],
        "mcp": {"servers": {}},
    }
