"""The agents setup knows how to wire, how to find them, and how to wire and unwire each.

Claude Code and Codex are wired once per user, by installing the Context Graph plugin from the Memgraph
marketplace (memgraph/marketplace) with the agent's own CLI. The other five are wired per project, by the hook file ``hook init``
writes (#533), and get the ``context-graph`` MCP server once per user in their own MCP config
(:mod:`memgraph_context_graph.adapters.agent_mcp`), since the plugin that brings it to the first two
doesn't exist for them.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from memgraph_context_graph.adapters._spec import RuntimeSpec, SpecPlugin

MARKETPLACE = "memgraph"
PLUGIN_ID = f"context-graph@{MARKETPLACE}"
MARKETPLACE_SOURCE = "memgraph/marketplace"
#: Where the plugin came from before the Memgraph marketplace: this repo's own marketplace. Setup and update
#: move an install from there, so an agent never runs two copies of the plugin's hooks.
LEGACY_MARKETPLACE = "context-graph-plugins"
LEGACY_PLUGIN_ID = f"context-graph@{LEGACY_MARKETPLACE}"


class WiringError(Exception):
    """An agent's CLI refused a step; the message carries its last line of output."""


@dataclass(frozen=True)
class Agent:
    """One agent runtime as setup sees it."""

    runtime: str
    label: str
    binaries: tuple[str, ...]
    wiring: Literal["plugin", "project"]

    def binary(self) -> str | None:
        """The agent's CLI on PATH, if it is installed."""
        for name in self.binaries:
            if found := shutil.which(name):
                return found
        return None


AGENTS: tuple[Agent, ...] = (
    Agent("claude-code", "Claude Code", ("claude",), "plugin"),
    Agent("codex", "Codex", ("codex",), "plugin"),
    Agent("copilot-cli", "GitHub Copilot CLI", ("copilot",), "project"),
    Agent("cursor", "Cursor", ("cursor", "cursor-agent"), "project"),
    Agent("opencode", "OpenCode", ("opencode",), "project"),
    Agent("antigravity-cli", "Antigravity CLI", ("agy",), "project"),
    Agent("grok", "Grok Build", ("grok",), "project"),
)
BY_RUNTIME = {agent.runtime: agent for agent in AGENTS}


def detected() -> list[Agent]:
    """The agents whose CLI is on PATH, in table order."""
    return [agent for agent in AGENTS if agent.binary()]


# --- plugin agents ------------------------------------------------------------


def plugin_wired(agent: Agent) -> bool:
    """Whether the agent's CLI lists the Context Graph plugin as installed, from either marketplace."""
    return installed_plugin_id(agent) is not None


def plugin_enabled(agent: Agent) -> bool:
    """Whether the plugin is installed and not switched off; a listing that doesn't say counts as on."""
    entries = _plugin_entries(agent)
    entry = next((entries[plugin_id] for plugin_id in (PLUGIN_ID, LEGACY_PLUGIN_ID) if plugin_id in entries), None)
    return entry is not None and entry.get("enabled", True) is not False


def installed_plugin_id(agent: Agent) -> str | None:
    """The installed plugin's ID: :data:`PLUGIN_ID`, :data:`LEGACY_PLUGIN_ID` until setup or update moves it, or None."""
    entries = _plugin_entries(agent)
    return next((plugin_id for plugin_id in (PLUGIN_ID, LEGACY_PLUGIN_ID) if plugin_id in entries), None)


def _plugin_entries(agent: Agent) -> dict[str, dict]:
    """The agent's installed plugins by ID, from one listing."""
    binary = agent.binary()
    if binary is None:
        return {}
    listing = _json([binary, "plugin", "list", "--json"])
    if agent.runtime == "claude-code":
        return {entry.get("id"): entry for entry in listing or []}
    installed = (listing or {}).get("installed", []) if isinstance(listing, dict) else []
    return {entry.get("pluginId"): entry for entry in installed}


def wire_plugin(agent: Agent, log: Callable[[str], None]) -> str:
    """Register the marketplace and install the plugin; a no-op for what is already there.

    An install from :data:`LEGACY_MARKETPLACE` is removed first, with that marketplace.

    Raises:
        WiringError: the agent's CLI is missing or refused a step.
    """
    binary = _require(agent)
    moved = _remove_legacy(agent, binary, log)
    if PLUGIN_ID in _plugin_entries(agent):
        _allow_recall(agent)
        return "plugin already installed"
    if not _marketplace_registered(binary, MARKETPLACE):
        _stream([binary, "plugin", "marketplace", "add", MARKETPLACE_SOURCE], log)
    if agent.runtime == "claude-code":
        _stream([binary, "plugin", "install", PLUGIN_ID, "-y"], log)
    else:
        _stream([binary, "plugin", "add", PLUGIN_ID], log)
    _allow_recall(agent)
    return "plugin moved to the Memgraph marketplace" if moved else "plugin installed"


def unwire_plugin(agent: Agent, log: Callable[[str], None]) -> str:
    """Uninstall the plugin, and an install from :data:`LEGACY_MARKETPLACE` with that marketplace.

    The Memgraph marketplace stays registered: it also serves the ``memgraph`` plugin.

    Raises:
        WiringError: the agent's CLI is missing or refused a step.
    """
    binary = _require(agent)
    removed = False
    if PLUGIN_ID in _plugin_entries(agent):
        _stream([binary, "plugin", _uninstall_verb(agent), PLUGIN_ID], log)
        removed = True
    if agent.runtime == "codex":
        _codex_memory().disallow_plugin_tools(Path.home(), PLUGIN_ID)
    removed = _remove_legacy(agent, binary, log) or removed
    return "removed the plugin" if removed else "wasn't installed"


def update_plugin(agent: Agent, log: Callable[[str], None]) -> str:
    """Refresh the marketplace and bring the installed plugin (its hooks and MCP entry) to the latest release.

    Claude Code updates the plugin in place; a running session needs a restart to load it. Codex has no update
    verb, so the plugin is added again from the refreshed marketplace. An install from
    :data:`LEGACY_MARKETPLACE` is moved to the Memgraph marketplace instead.

    Raises:
        WiringError: the agent's CLI is missing, or refused a step.
    """
    binary = _require(agent)
    plugin_id = installed_plugin_id(agent)
    if plugin_id is None:
        return "plugin not installed; nothing to update"
    if plugin_id == LEGACY_PLUGIN_ID:
        return wire_plugin(agent, log)
    if agent.runtime == "claude-code":
        _stream([binary, "plugin", "marketplace", "update", MARKETPLACE], log)
        _stream([binary, "plugin", "update", PLUGIN_ID], log)
    else:
        _stream([binary, "plugin", "marketplace", "upgrade", MARKETPLACE], log)
        _stream([binary, "plugin", "add", PLUGIN_ID], log)
    _allow_recall(agent)
    return "plugin updated"


def _remove_legacy(agent: Agent, binary: str, log: Callable[[str], None]) -> bool:
    """Uninstall the plugin from :data:`LEGACY_MARKETPLACE` and remove that marketplace; whether there was any."""
    removed = False
    if LEGACY_PLUGIN_ID in _plugin_entries(agent):
        _stream([binary, "plugin", _uninstall_verb(agent), LEGACY_PLUGIN_ID], log)
        removed = True
    if agent.runtime == "codex":
        _codex_memory().disallow_plugin_tools(Path.home(), LEGACY_PLUGIN_ID)
    if _marketplace_registered(binary, LEGACY_MARKETPLACE):
        _stream([binary, "plugin", "marketplace", "remove", LEGACY_MARKETPLACE], log)
        removed = True
    return removed


def _uninstall_verb(agent: Agent) -> str:
    return "uninstall" if agent.runtime == "claude-code" else "remove"


def _allow_recall(agent: Agent) -> None:
    """Codex refuses an MCP tool that needs approval when its policy is ``never``; recall is read-only, so allow it."""
    if agent.runtime == "codex":
        _codex_memory().allow_plugin_tools(Path.home(), PLUGIN_ID)


def _codex_memory():
    from memgraph_context_graph.adapters import codex_memory

    return codex_memory


def _marketplace_registered(binary: str, name: str) -> bool:
    listing = _json([binary, "plugin", "marketplace", "list", "--json"])
    entries = listing.get("marketplaces", []) if isinstance(listing, dict) else listing or []
    return any(entry.get("name") == name for entry in entries)


# --- project agents -----------------------------------------------------------


def project_wired(agent: Agent, project_dir: Path) -> bool:
    """Whether *project_dir* holds this agent's capture wiring."""
    from memgraph_context_graph.adapters import opencode
    from memgraph_context_graph.adapters._spec import has_hook_config

    if agent.runtime == "opencode":
        return (project_dir / opencode.PLUGIN_PATH).is_file()
    return has_hook_config(_spec(agent), project_dir)


def wire_project(agent: Agent, project_dir: Path, log: Callable[[str], None]) -> str:
    """Write the agent's capture wiring into *project_dir*, replacing only what an earlier run wrote.

    Raises:
        WiringError: *project_dir* is the home directory or isn't a directory, or its hook file isn't a
            JSON object.
    """
    check_project_dir(project_dir)
    already = project_wired(agent, project_dir)
    init = _plugin(agent).init
    if init is None:
        raise WiringError(f"{agent.label} has no project wiring")
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            # OpenCode's plugin is a generated file of its own; the JSON hook files are merged instead.
            # No connectors named: the hooks enable the config file's [capture] connectors.
            init(project_dir, [], force=agent.runtime == "opencode")
    except (ValueError, FileExistsError) as exc:
        raise WiringError(str(exc)) from exc
    for line in output.getvalue().splitlines():
        log(line)
    return f"{'rewired' if already else 'wired'} {project_dir}"


def unwire_project(agent: Agent, project_dir: Path) -> bool:
    """Remove this agent's capture wiring from *project_dir*, keeping the user's own hooks.

    Raises:
        WiringError: the hook file isn't a JSON object.
    """
    from memgraph_context_graph.adapters import opencode
    from memgraph_context_graph.adapters._spec import remove_hook_config

    if agent.runtime == "opencode":
        plugin = project_dir / opencode.PLUGIN_PATH
        if not plugin.is_file():
            return False
        plugin.unlink()
        return True
    try:
        return remove_hook_config(_spec(agent), project_dir)
    except ValueError as exc:
        raise WiringError(str(exc)) from exc


def mcp_registered(agent: Agent) -> bool:
    """Whether the agent's user-level MCP config lists the ``context-graph`` server."""
    from memgraph_context_graph.adapters.agent_mcp import AGENT_MCP
    from memgraph_context_graph.memory_backend import MemoryBackendError

    wiring = AGENT_MCP.get(agent.runtime)
    try:
        return wiring is not None and wiring.registered(Path.home())
    except MemoryBackendError:
        return False


def register_mcp(agent: Agent, log: Callable[[str], None]) -> str:
    """Add the ``context-graph`` MCP server to the agent's user-level config, and allow its tools.

    Raises:
        WiringError: the agent's config file can't be read as written.
    """
    from memgraph_context_graph.adapters.agent_mcp import AGENT_MCP
    from memgraph_context_graph.memory_backend import MemoryBackendError

    wiring = AGENT_MCP.get(agent.runtime)
    if wiring is None:
        return ""
    try:
        already = wiring.registered(Path.home())
        lines = wiring.register(Path.home())
    except MemoryBackendError as exc:
        raise WiringError(str(exc)) from exc
    for line in lines:
        log(line)
    return "memory tools already registered" if already else "memory tools registered"


def unregister_mcp(agent: Agent, log: Callable[[str], None]) -> bool:
    """Remove what :func:`register_mcp` added; returns whether anything was there.

    Raises:
        WiringError: the agent's config file can't be read as written.
    """
    from memgraph_context_graph.adapters.agent_mcp import AGENT_MCP
    from memgraph_context_graph.memory_backend import MemoryBackendError

    wiring = AGENT_MCP.get(agent.runtime)
    if wiring is None:
        return False
    try:
        lines = wiring.unregister(Path.home())
    except MemoryBackendError as exc:
        raise WiringError(str(exc)) from exc
    for line in lines:
        log(line)
    return bool(lines)


def check_project_dir(project_dir: Path) -> None:
    """Refuse a project directory that would capture far more than one project.

    Raises:
        WiringError: *project_dir* isn't a directory, or is the home directory, where wiring would capture
            every session started anywhere below it.
    """
    if not project_dir.is_dir():
        raise WiringError(f"{project_dir} is not a directory")
    if project_dir.resolve() == Path.home().resolve():
        raise WiringError(f"refusing to wire {project_dir}: it's your home directory, so pass a project instead")


def _plugin(agent: Agent) -> SpecPlugin:
    from memgraph_context_graph.adapters._spec import SpecPlugin
    from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin

    plugin = get_runtime_plugin(agent.runtime)
    if not isinstance(plugin, SpecPlugin):
        raise WiringError(f"{agent.runtime} is registered by a plugin setup doesn't know how to wire")
    return plugin


def _spec(agent: Agent) -> RuntimeSpec:
    return _plugin(agent).spec


# --- subprocesses -------------------------------------------------------------


def _require(agent: Agent) -> str:
    binary = agent.binary()
    if binary is None:
        raise WiringError(f"{' or '.join(agent.binaries)} isn't on PATH; install {agent.label} first")
    return binary


def _json(argv: list[str]):
    """What an agent CLI's ``--json`` listing printed, or None when it didn't print one.

    A listing that fails once (the CLI busy updating itself, a notice printed ahead of the JSON) would read as
    "not installed" and mislabel a wired agent, so it is asked twice and the JSON is taken from where it starts.
    """
    for attempt in range(2):
        if attempt:
            time.sleep(0.5)
        # Not the terminal: Claude Code on macOS crashes (kqueue EINVAL) on a stdin that setup's questions
        # left in cbreak mode, and every listing read as "not installed".
        result = subprocess.run(argv, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
        if result.returncode == 0 and (found := _json_in(result.stdout)) is not None:
            return found
    return None


def _json_in(text: str):
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def _stream(argv: list[str], log: Callable[[str], None]) -> None:
    """Run *argv*, sending each output line to *log*.

    Raises:
        WiringError: the command exited non-zero.
    """
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL
    )
    last = ""
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            last = line
            log(line)
    if process.wait() != 0:
        raise WiringError(f"`{' '.join(argv[1:])}` failed: {last or f'exit {process.returncode}'}")
