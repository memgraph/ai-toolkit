"""The agents setup knows how to wire, how to find them, and how to wire and unwire each.

Claude Code and Codex are wired once per user, by installing the Context Graph plugin from this repo's
marketplace with the agent's own CLI. The other five are wired per project, by the hook file ``hook init``
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

MARKETPLACE = "context-graph-plugins"
PLUGIN_ID = f"context-graph@{MARKETPLACE}"
MARKETPLACE_SOURCE = "memgraph/ai-toolkit"
CONNECTORS = ("skills-graph", "actions-graph", "sessions-graph")


class WiringError(Exception):
    """An agent's CLI refused a step; the message carries its last line of output."""


@dataclass(frozen=True)
class Agent:
    """One agent runtime as setup sees it."""

    runtime: str
    label: str
    binaries: tuple[str, ...]
    wiring: Literal["plugin", "project"]
    #: For plugin agents: the marketplace's sparse path in this repo.
    sparse: str = ""

    def binary(self) -> str | None:
        """The agent's CLI on PATH, if it is installed."""
        for name in self.binaries:
            if found := shutil.which(name):
                return found
        return None


AGENTS: tuple[Agent, ...] = (
    Agent("claude-code", "Claude Code", ("claude",), "plugin", sparse=".claude-plugin"),
    Agent("codex", "Codex", ("codex",), "plugin", sparse=".agents/plugins"),
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
    """Whether the agent's CLI lists the Context Graph plugin as installed."""
    binary = agent.binary()
    if binary is None:
        return False
    listing = _json([binary, "plugin", "list", "--json"])
    if agent.runtime == "claude-code":
        return any(entry.get("id") == PLUGIN_ID for entry in listing or [])
    installed = (listing or {}).get("installed", []) if isinstance(listing, dict) else []
    return any(entry.get("pluginId") == PLUGIN_ID for entry in installed)


def wire_plugin(agent: Agent, log: Callable[[str], None]) -> str:
    """Register the marketplace and install the plugin; a no-op for what is already there.

    Raises:
        WiringError: the agent's CLI is missing or refused a step.
    """
    binary = _require(agent)
    if plugin_wired(agent):
        _allow_recall(agent)
        return "plugin already installed"
    if not _marketplace_registered(agent, binary):
        _stream([binary, "plugin", "marketplace", "add", MARKETPLACE_SOURCE, "--sparse", agent.sparse], log)
    if agent.runtime == "claude-code":
        _stream([binary, "plugin", "install", PLUGIN_ID, "-y"], log)
    else:
        _stream([binary, "plugin", "add", PLUGIN_ID], log)
    _allow_recall(agent)
    return "plugin installed"


def unwire_plugin(agent: Agent, log: Callable[[str], None]) -> str:
    """Uninstall the plugin and remove the marketplace.

    Raises:
        WiringError: the agent's CLI is missing or refused a step.
    """
    binary = _require(agent)
    removed = []
    if plugin_wired(agent):
        verb = "uninstall" if agent.runtime == "claude-code" else "remove"
        _stream([binary, "plugin", verb, PLUGIN_ID], log)
        removed.append("plugin")
    if agent.runtime == "codex":
        _codex_memory().disallow_plugin_tools(Path.home(), PLUGIN_ID)
    if _marketplace_registered(agent, binary):
        _stream([binary, "plugin", "marketplace", "remove", MARKETPLACE], log)
        removed.append("marketplace")
    return f"removed the {' and '.join(removed)}" if removed else "wasn't installed"


def _allow_recall(agent: Agent) -> None:
    """Codex refuses an MCP tool that needs approval when its policy is ``never``; recall is read-only, so allow it."""
    if agent.runtime == "codex":
        _codex_memory().allow_plugin_tools(Path.home(), PLUGIN_ID)


def _codex_memory():
    from memgraph_context_graph.adapters import codex_memory

    return codex_memory


def _marketplace_registered(agent: Agent, binary: str) -> bool:
    listing = _json([binary, "plugin", "marketplace", "list", "--json"])
    entries = listing.get("marketplaces", []) if isinstance(listing, dict) else listing or []
    return any(entry.get("name") == MARKETPLACE for entry in entries)


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
            init(project_dir, list(CONNECTORS), force=agent.runtime == "opencode")
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
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
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
