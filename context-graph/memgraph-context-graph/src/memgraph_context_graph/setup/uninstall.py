"""Undo setup's wiring: plugins, project hook files, MCP registrations, and the memory switch. Never data (#533).

The graph, the config file and the uv tool stay. :func:`leftovers` prints how to remove them, because
removing a graph is not something to do as a side effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from memgraph_context_graph.setup import agents as agents_mod
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


def run(
    runtimes: Sequence[str], project_dirs: Sequence[Path], log: Callable[[str], None], report: Callable[[Outcome], None]
) -> list[Outcome]:
    """Unwire each of *runtimes*; one failing doesn't stop the rest."""
    outcomes = []
    for runtime in runtimes:
        agent = agents_mod.BY_RUNTIME[runtime]
        outcome = _unwire(agent, project_dirs, log)
        outcomes.append(outcome)
        report(outcome)
        if (memory := _memory_back(agent)) is not None:
            outcomes.append(memory)
            report(memory)
    if (backend := _config_back(runtimes)) is not None:
        outcomes.append(backend)
        report(backend)
    return outcomes


def leftovers(container: str = CONTAINER) -> list[str]:
    """What uninstall leaves on purpose, with the commands that remove it."""
    from memgraph_context_graph.adapters._identity import config_file

    return [
        "Kept on purpose; remove them yourself if you want them gone:",
        f"  the graph:   docker rm -f {container} && docker volume rm {container}-data",
        f"  the config:  rm {config_file()}",
        "  the tool:    uv tool uninstall memgraph-context-graph",
    ]


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
    """Turn the agent's own memory back on, if setup turned it off; without the plugin it would have none."""
    from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin
    from memgraph_context_graph.memory_backend import MemoryBackendError, native_memory_of

    native = native_memory_of(get_runtime_plugin(agent.runtime))
    if native is None:
        return None
    home = Path.home()
    try:
        if native.enabled(home):
            return None
        lines = native.enable(home)
    except MemoryBackendError as exc:
        return Outcome(f"{agent.label} memory", "fail", str(exc))
    if not lines:  # An agent with no memory of its own: nothing came back on.
        return None
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
