"""Registering the ``context-graph`` MCP server with agents that have no Context Graph plugin.

Claude Code and Codex get the server from their plugin. Copilot CLI, Cursor,
OpenCode, Antigravity CLI and Grok Build are wired per project by hook files,
which reach no MCP config, so without this their model never sees ``recall`` or
``memory``. Grok Build starts it with ``--session-context``: no hook output of
Grok's reaches the model at session start, but it puts an MCP server's
instructions in the system prompt. (OpenCode does too, but reads them before a
server that takes a second to start is connected, so its plugin asks the hook.) Each registers MCP servers in its own user-level file. The server
goes there, not into one project, because memory belongs to the user and
follows them into every project, the same scope ``[memory] backend`` has.

Where an agent keeps a persistent allow-list, the server's tools are added to
it as well: in a headless run an agent denies a tool that needs approval, so
recall would fail for exactly the scripted sessions that rely on it. The tools
only read the user's own memory, or write their ``/memories`` files.

Every edit keeps the rest of the file as the user has it, writes a one-time
``.context-graph.bak`` copy next to it, and is undone by :meth:`unregister`.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.memory_backend import MemoryBackendError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

SERVER = "context-graph"
_BACKUP_SUFFIX = ".context-graph.bak"


def server_argv(*, session_context: bool = False) -> list[str]:
    """The command an agent runs to start the server: the installed CLI, else this interpreter."""
    executable = shutil.which("memgraph-context-graph")
    base = [executable] if executable else [sys.executable, "-m", "memgraph_context_graph.cli"]
    return [*base, "mcp", *(["--session-context"] if session_context else [])]


@dataclass(frozen=True)
class JsonAllowRule:
    """One rule in a JSON settings file's allow-list, naming every tool of the server."""

    path: Callable[[Path], Path]
    keys: tuple[str, ...]
    rule: Any


@dataclass(frozen=True)
class JsonMcpServers:
    """An agent that lists MCP servers under one key of a user-level JSON file.

    ``entry`` renders the server's entry from its argv; ``allow`` is the
    agent's persistent allow-list, if it has one. ``session_context`` starts
    the server with ``--session-context``.
    """

    path: Callable[[Path], Path]
    entry: Callable[[list[str]], dict[str, Any]]
    servers_key: tuple[str, ...] = ("mcpServers",)
    allow: JsonAllowRule | None = None
    session_context: bool = False

    def registered(self, home: Path) -> bool:
        """Whether the file lists a ``context-graph`` server."""
        return SERVER in _json_get(_read_json(self.path(home)), self.servers_key, {})

    def register(self, home: Path) -> list[str]:
        """Add or refresh the server entry, and the allow rule.

        Raises:
            MemoryBackendError: a file isn't a JSON object.
        """
        path = self.path(home)
        document = _read_json(path)
        _json_table(document, self.servers_key)[SERVER] = self.entry(server_argv(session_context=self.session_context))
        lines = _write_json(path, document)
        if self.allow is not None:
            lines += _add_rule(self.allow, home)
        return lines

    def unregister(self, home: Path) -> list[str]:
        """Remove the server entry and the allow rule, leaving everything else.

        Raises:
            MemoryBackendError: a file isn't a JSON object.
        """
        lines: list[str] = []
        path = self.path(home)
        if path.exists():
            document = _read_json(path)
            servers = _json_get(document, self.servers_key, {})
            if SERVER in servers:
                del servers[SERVER]
                lines += _write_json(path, document)
        if self.allow is not None:
            lines += _remove_rule(self.allow, home)
        return lines


@dataclass(frozen=True)
class GrokMcp:
    """Grok Build: ``[mcp_servers.context-graph]`` and a ``[permission] allow`` rule in ``~/.grok/config.toml``.

    Grok's session-start hooks can't put text in front of the model, so this
    server is started with ``--session-context`` and sends that text as its
    MCP instructions, which Grok does read.
    """

    rule: str = f"MCPTool({SERVER}__*)"

    def registered(self, home: Path) -> bool:
        """Whether the config lists a ``context-graph`` server."""
        return SERVER in _read_toml(_grok_config(home)).get("mcp_servers", {})

    def register(self, home: Path) -> list[str]:
        """Add or refresh the server and its allow rule.

        Raises:
            MemoryBackendError: the config isn't valid TOML.
        """
        import tomlkit

        path = _grok_config(home)
        config = _read_toml(path)
        argv = server_argv(session_context=True)
        server = tomlkit.table()
        server["command"] = argv[0]
        server["args"] = argv[1:]
        _toml_table(config, "mcp_servers", super_table=True)[SERVER] = server
        allow = _toml_table(config, "permission").setdefault("allow", tomlkit.array())
        if self.rule not in allow:
            allow.append(self.rule)
        return _write_toml(path, config)

    def unregister(self, home: Path) -> list[str]:
        """Remove the server and its allow rule.

        Raises:
            MemoryBackendError: the config isn't valid TOML.
        """
        path = _grok_config(home)
        if not path.exists():
            return []
        config = _read_toml(path)
        servers = config.get("mcp_servers", {})
        changed = False
        if SERVER in servers:
            del servers[SERVER]
            changed = True
            if not servers:
                del config["mcp_servers"]
        allow = config.get("permission", {}).get("allow")
        if allow is not None and self.rule in allow:
            allow.remove(self.rule)
            changed = True
        return _write_toml(path, config) if changed else []


def _copilot_entry(argv: list[str]) -> dict[str, Any]:
    return {"type": "local", "command": argv[0], "args": argv[1:], "tools": ["*"]}


def _stdio_entry(argv: list[str]) -> dict[str, Any]:
    return {"type": "stdio", "command": argv[0], "args": argv[1:]}


def _command_entry(argv: list[str]) -> dict[str, Any]:
    return {"command": argv[0], "args": argv[1:]}


def _opencode_entry(argv: list[str]) -> dict[str, Any]:
    # Without codemode off, OpenCode hides MCP tools behind its `execute` tool; with it, the model sees
    # `context-graph_memory` and `context-graph_recall` as tools of their own.
    return {"type": "local", "command": argv, "codemode": False}


#: The agents setup wires per project, by runtime name.
AGENT_MCP: dict[str, JsonMcpServers | GrokMcp] = {
    # Copilot keeps no allow-list between runs: an interactive user approves once, a script passes
    # --allow-tool=context-graph.
    "copilot-cli": JsonMcpServers(path=lambda home: home / ".copilot" / "mcp-config.json", entry=_copilot_entry),
    "cursor": JsonMcpServers(
        path=lambda home: home / ".cursor" / "mcp.json",
        entry=_stdio_entry,
        allow=JsonAllowRule(
            path=lambda home: home / ".cursor" / "cli-config.json",
            keys=("permissions", "allow"),
            rule=f"Mcp({SERVER}:*)",
        ),
    ),
    "antigravity-cli": JsonMcpServers(
        path=lambda home: home / ".gemini" / "config" / "mcp_config.json",
        entry=_command_entry,
        allow=JsonAllowRule(
            path=lambda home: home / ".gemini" / "antigravity-cli" / "settings.json",
            keys=("permissions", "allow"),
            rule=f"mcp({SERVER}/*)",
        ),
    ),
    "opencode": JsonMcpServers(
        path=lambda home: home / ".config" / "opencode" / "opencode.json",
        entry=_opencode_entry,
        servers_key=("mcp", "servers"),
        # OpenCode's default agent allows every tool, but a user rule of `*: ask` would stop a scripted run.
        allow=JsonAllowRule(
            path=lambda home: home / ".config" / "opencode" / "opencode.json",
            keys=("permissions",),
            rule={"action": f"{SERVER}_*", "resource": "*", "effect": "allow"},
        ),
    ),
    "grok": GrokMcp(),
}


# --- JSON ---------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise MemoryBackendError(f"{path} is not valid JSON ({exc}); fix it, then re-run.") from exc
    if not isinstance(document, dict):
        raise MemoryBackendError(f"{path} is not a JSON object; fix it, then re-run.")
    return document


def _json_get(document: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    node: Any = document
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _json_table(document: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    node = document
    for key in keys:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise MemoryBackendError(f"`{'.'.join(keys)}` is not an object; fix it, then re-run.")
        node = child
    return node


def _add_rule(allow: JsonAllowRule, home: Path) -> list[str]:
    path = allow.path(home)
    document = _read_json(path)
    *parents, last = allow.keys
    rules = _json_table(document, tuple(parents)).setdefault(last, [])
    if allow.rule in rules:
        return [f"Unchanged {path}"]
    rules.append(allow.rule)
    return _write_json(path, document)


def _remove_rule(allow: JsonAllowRule, home: Path) -> list[str]:
    path = allow.path(home)
    if not path.exists():
        return []
    document = _read_json(path)
    rules = _json_get(document, allow.keys, [])
    if allow.rule not in rules:
        return []
    rules.remove(allow.rule)
    return _write_json(path, document)


def _write_json(path: Path, document: dict[str, Any]) -> list[str]:
    rendered = json.dumps(document, indent=2) + "\n"
    return _write(path, rendered)


# --- TOML ---------------------------------------------------------------------


def _grok_config(home: Path) -> Path:
    return home / ".grok" / "config.toml"


def _read_toml(path: Path) -> Any:
    import tomlkit
    from tomlkit.exceptions import ParseError

    if not path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except ParseError as exc:
        raise MemoryBackendError(f"{path} is not valid TOML ({exc}); fix it, then re-run.") from exc


def _toml_table(config: Any, key: str, *, super_table: bool = False) -> Any:
    import tomlkit

    if key not in config:
        config[key] = tomlkit.table(is_super_table=super_table)
    return config[key]


def _write_toml(path: Path, config: Any) -> list[str]:
    import tomlkit

    return _write(path, tomlkit.dumps(config))


def _write(path: Path, rendered: str) -> list[str]:
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return [f"Unchanged {path}"]
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return [f"Wrote {path}"]
