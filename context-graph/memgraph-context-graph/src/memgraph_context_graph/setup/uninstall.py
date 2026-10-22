"""Undo setup: the wiring always, and the graph, the config and the tool only when the user says so.

Unwiring (plugins, project hook files, MCP registrations, the memory switch) is what ``uninstall`` means.
Deleting the graph loses every session for good, so it is never a side effect (#533): each of the other
three is its own yes, asked or passed as a flag, and :func:`leftovers` says how to remove what stays.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from memgraph_context_graph.setup import agents as agents_mod
from memgraph_context_graph.setup import tool as tool_mod
from memgraph_context_graph.setup.engine import Outcome
from memgraph_context_graph.setup.memgraph import CONTAINER

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def wired(project_dirs: Sequence[Path]) -> list[str]:
    """The runtimes that are wired now: plugins installed, or hook files in one of *project_dirs*."""
    found = []
    for agent in agents_mod.AGENTS:
        if agent.wiring == "plugin":
            if agent.binary() and agents_mod.plugin_wired(agent):
                found.append(agent.runtime)
        elif any(agents_mod.project_wired(agent, path) for path in project_dirs):
            found.append(agent.runtime)
    return found


@dataclass(frozen=True)
class Removals:
    """What goes beyond the wiring; each only on the user's yes."""

    #: The Memgraph container setup starts, and its data volume: every session and memory in it.
    graph: bool = False
    #: The config file and the capture-failure log.
    config: bool = False
    #: The ``memgraph-context-graph`` uv tool, with the connectors and viewer installed into it.
    tool: bool = False


#: Only the wiring: what uninstall removes when nothing else was asked for.
WIRING_ONLY = Removals()


def removable(container: str = CONTAINER) -> Removals:
    """Which of the removals have something to remove here."""
    return Removals(
        graph=_graph_exists(container),
        config=any(path.exists() for path in _config_paths()),
        tool=tool_mod.uv_tool_environment(),
    )


def run(
    runtimes: Sequence[str],
    project_dirs: Sequence[Path],
    log: Callable[[str], None],
    report: Callable[[Outcome], None],
    *,
    removals: Removals = WIRING_ONLY,
    container: str = CONTAINER,
) -> list[Outcome]:
    """Unwire each of *runtimes*, then carry out *removals*; one step failing doesn't stop the rest."""
    outcomes = []
    for runtime in runtimes:
        agent = agents_mod.BY_RUNTIME[runtime]
        outcome = _unwire(agent, project_dirs, log)
        outcomes.append(outcome)
        report(outcome)
        if (memory := _memory_back(agent)) is not None:
            outcomes.append(memory)
            report(memory)
    if (backend := _config_back(runtimes)) is not None and not removals.config:
        outcomes.append(backend)
        report(backend)
    steps: list[Callable[[], Outcome]] = []
    if removals.graph:
        steps.append(lambda: _remove_graph(container))
    if removals.config:
        steps.append(_remove_config)
    if removals.tool:
        # Last: this process runs from the tool, so nothing may need to import from it afterwards.
        steps.append(lambda: _remove_tool(log))
    for step in steps:
        outcome = step()
        outcomes.append(outcome)
        report(outcome)
    return outcomes


def leftovers(removals: Removals = WIRING_ONLY, container: str = CONTAINER) -> list[str]:
    """What uninstall left, with the commands that remove it; empty when nothing is left."""
    from memgraph_context_graph.adapters._identity import config_file

    kept = []
    if not removals.graph:
        kept.append(f"  the graph:   docker rm -f {container} && docker volume rm {container}-data")
    if not removals.config:
        kept.append(f"  the config:  rm {config_file()}")
    if not removals.tool:
        kept.append("  the tool:    uv tool uninstall memgraph-context-graph")
    if not kept:
        return []
    return ["Kept; remove them yourself, or all at once with: memgraph-context-graph uninstall --all", *kept]


def _graph_exists(container: str) -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    return any(
        _docker(docker, *argv).returncode == 0
        for argv in (("container", "inspect", container), ("volume", "inspect", f"{container}-data"))
    )


def _remove_graph(container: str) -> Outcome:
    """Delete setup's container and its volume, by name only: a Memgraph the user runs elsewhere is never touched."""
    fix = f"docker rm -f {container} && docker volume rm {container}-data"
    docker = shutil.which("docker")
    if docker is None:
        # Setup starts its Memgraph only in Docker; without Docker there is no container of ours.
        return Outcome("Graph", "skip", "Docker isn't installed, so setup never started a Memgraph here")
    deleted = []
    for argv, what in (
        (("rm", "-f", container), f"container {container}"),
        (("volume", "rm", f"{container}-data"), f"volume {container}-data"),
    ):
        result = _docker(docker, *argv)
        if result.returncode == 0:
            deleted.append(what)
        elif "no such" not in result.stderr.lower():
            last = (result.stderr.strip().splitlines() or [f"exit {result.returncode}"])[-1]
            return Outcome("Graph", "fail", f"couldn't delete the {what}: {last}", fix=fix)
    return Outcome("Graph", "ok", f"deleted the {' and the '.join(deleted)}" if deleted else "nothing was there")


def _config_paths() -> list[Path]:
    from memgraph_context_graph import capture_log
    from memgraph_context_graph.adapters._identity import config_file

    return [config_file(), capture_log.path()]


def _remove_config() -> Outcome:
    removed = []
    for path in _config_paths():
        if path.exists():
            path.unlink()
            removed.append(str(path))
        # Only our own directory, and only once it is empty: a config path the user chose may sit among their files.
        if path.parent.name == "context-graph":
            with contextlib.suppress(OSError):
                path.parent.rmdir()
    return Outcome("Config", "ok", f"deleted {', '.join(removed)}" if removed else "nothing was there")


def _remove_tool(log: Callable[[str], None]) -> Outcome:
    import sys

    command = "uv tool uninstall memgraph-context-graph"
    if not tool_mod.uv_tool_environment():
        return Outcome("Tool", "skip", "not running as the uv tool; remove it the way you installed it")
    if sys.platform == "win32":
        # Windows won't delete the memgraph-context-graph.exe that is running this.
        return Outcome("Tool", "warn", f"Windows can't remove a running program; now run: {command}", fix=command)
    uv = shutil.which("uv")
    if uv is None:
        return Outcome("Tool", "fail", "uv isn't on PATH", fix=command)
    result = subprocess.run([uv, "tool", "uninstall", "memgraph-context-graph"], capture_output=True, text=True)
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip():
            log(line.rstrip())
    if result.returncode != 0:
        return Outcome("Tool", "fail", "uv couldn't uninstall it", fix=command)
    return Outcome("Tool", "ok", "uninstalled memgraph-context-graph and everything installed with it")


def _docker(docker: str, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([docker, *argv], capture_output=True, text=True, check=False)


def _unwire(agent: agents_mod.Agent, project_dirs: Sequence[Path], log: Callable[[str], None]) -> Outcome:
    try:
        if agent.wiring == "plugin":
            return Outcome(agent.label, "ok", agents_mod.unwire_plugin(agent, log))
        removed = [str(path) for path in project_dirs if agents_mod.unwire_project(agent, path)]
        if agents_mod.unregister_mcp(agent, log):
            removed.append("the memory tools")
    except agents_mod.WiringError as exc:
        return Outcome(agent.label, "fail", str(exc), fix=f"memgraph-context-graph uninstall {agent.runtime}")
    if not removed:
        return Outcome(agent.label, "skip", "no wiring found in " + ", ".join(str(path) for path in project_dirs))
    return Outcome(agent.label, "ok", f"unwired {', '.join(removed)}")


def _memory_back(agent: agents_mod.Agent) -> Outcome | None:
    """Turn the agent's own memory back on, if setup turned it off; without the plugin it would have none.

    The graph's memories are copied into it first, where it has a place for them. When Memgraph can't be read
    the memory still comes back on, with what it held before the switch, and the outcome says so.
    """
    from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin
    from memgraph_context_graph.memory_backend import MemoryBackendError, export_graph_memory, native_memory_of

    native = native_memory_of(get_runtime_plugin(agent.runtime))
    if native is None:
        return None
    home = Path.home()
    try:
        if native.enabled(home):
            return None
    except MemoryBackendError as exc:
        return Outcome(f"{agent.label} memory", "fail", str(exc))
    try:
        copied = export_graph_memory(native, agent.label, home=home)
        not_copied = None
    except MemoryBackendError as exc:
        copied, not_copied = [], str(exc)
    try:
        lines = copied + native.enable(home)
    except MemoryBackendError as exc:
        return Outcome(f"{agent.label} memory", "fail", str(exc))
    if not lines:  # An agent with no memory of its own: nothing came back on.
        return None
    if not_copied:
        summary = f"its own memory is back on, without the graph's memories: {not_copied}"
        return Outcome(f"{agent.label} memory", "warn", summary, lines=tuple(lines))
    return Outcome(f"{agent.label} memory", "ok", "its own memory is back on", lines=tuple(lines))


def _config_back(runtimes: Sequence[str]) -> Outcome | None:
    """Hand ``memory.backend`` back to native once no agent that could use the graph's memory is left wired."""
    from memgraph_context_graph.adapters._identity import MEMORY_BACKEND_NATIVE, load_config, write_config

    if not load_config().graph_memory:
        return None
    remaining = [
        agent
        for agent in agents_mod.AGENTS
        if agent.runtime not in runtimes
        and (
            agents_mod.mcp_registered(agent)
            if agent.wiring == "project"
            else agent.binary() is not None and agents_mod.plugin_wired(agent)
        )
    ]
    if remaining:
        return None
    write_config(memory_backend=MEMORY_BACKEND_NATIVE)
    return Outcome("Config", "ok", f"memory.backend = {MEMORY_BACKEND_NATIVE}")
