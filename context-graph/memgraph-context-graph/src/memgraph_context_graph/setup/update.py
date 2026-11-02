"""Bring an existing install up to the installed release: plugins, hook files, MCP registrations and schema.

Upgrading the package moves the code hooks and the MCP server run, but what setup wrote once stays as it was:
a plugin's hook list, a project's generated hook file, an agent's MCP registration, the graph's indexes. This
re-applies each of them for what is wired now. Every step is idempotent, and none touches data.
"""

from __future__ import annotations

import shutil
import subprocess
from importlib import metadata
from typing import TYPE_CHECKING

from memgraph_context_graph.setup import agents as agents_mod
from memgraph_context_graph.setup import engine, uninstall
from memgraph_context_graph.setup import tool as tool_mod
from memgraph_context_graph.setup.engine import Outcome

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

#: What the user does once the steps are done: a running agent still holds the hooks it loaded at start.
RESTART_NOTE = "Restart running agent sessions so they load the updated hooks."


def wired(project_dirs: Sequence[Path]) -> list[str]:
    """The runtimes that are wired now: what ``uninstall`` would remove, and so what update refreshes."""
    return uninstall.wired(project_dirs)


def upgrade_packages(log: Callable[[str], None]) -> Outcome:
    """Move the uv tool and its connectors to their newest releases; only the tool's own environment is touched."""
    before = metadata.version("memgraph-context-graph")
    if not tool_mod.uv_tool_environment():
        return Outcome(
            "Packages",
            "warn",
            f"memgraph-context-graph {before} isn't the uv tool's environment, so upgrade it yourself",
        )
    try:
        tool_mod.install(viewer=False, log=log, upgrade=True)
    except tool_mod.ToolError as exc:
        return Outcome("Packages", "fail", str(exc), fix=f"{tool_mod.install_command()} --upgrade")
    after = _installed_version() or before
    summary = f"memgraph-context-graph {before} -> {after}" if after != before else f"already at {after}"
    return Outcome("Packages", "ok", summary)


def rerun(runtimes: Sequence[str], project_dirs: Sequence[Path]) -> int:
    """Run ``update --no-upgrade`` with the freshly installed executable, since this process still holds the old code.

    Returns:
        The exit code of that run; 1 when the executable can't be found.
    """
    executable = shutil.which("memgraph-context-graph")
    if executable is None:
        return 1
    argv = [executable, "update", "--no-upgrade", *runtimes]
    for path in project_dirs:
        argv += ["--project-dir", str(path)]
    return subprocess.run(argv, check=False).returncode


def _installed_version() -> str | None:
    """The version the ``memgraph-context-graph`` executable on PATH reports, which a new install may have changed."""
    executable = shutil.which("memgraph-context-graph")
    if executable is None:
        return None
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


def run(
    runtimes: Sequence[str], project_dirs: Sequence[Path], log: Callable[[str], None], report: Callable[[Outcome], None]
) -> list[Outcome]:
    """Refresh each of *runtimes*, the graph's schema and kept events, then check; one failing doesn't stop the rest."""
    outcomes: list[Outcome] = []

    def done(outcome: Outcome) -> None:
        outcomes.append(outcome)
        report(outcome)

    for runtime in runtimes:
        done(_refresh(agents_mod.BY_RUNTIME[runtime], project_dirs, log))
    done(_schema())
    if (replayed := replay_kept()) is not None:
        done(replayed)
    for outcome in engine.verify(runtimes):
        done(outcome)
    return outcomes


def _refresh(agent: agents_mod.Agent, project_dirs: Sequence[Path], log: Callable[[str], None]) -> Outcome:
    fix = f"memgraph-context-graph update {agent.runtime}"
    try:
        if agent.wiring == "plugin":
            return Outcome(agent.label, "ok", agents_mod.update_plugin(agent, log))
        refreshed = []
        # The registration holds the path of the executable, so it is what a moved install leaves stale.
        if agents_mod.mcp_registered(agent):
            refreshed.append(agents_mod.register_mcp(agent, log))
        refreshed += [
            agents_mod.wire_project(agent, path, log) for path in project_dirs if agents_mod.project_wired(agent, path)
        ]
    except agents_mod.WiringError as exc:
        return Outcome(agent.label, "fail", str(exc), fix=fix)
    if not refreshed:
        return Outcome(agent.label, "skip", "nothing wired in " + ", ".join(str(path) for path in project_dirs))
    return Outcome(agent.label, "ok", "; ".join(refreshed))


def _schema() -> Outcome:
    """Create the indexes and constraints a newer release added; each ``setup()`` skips what already exists."""
    from memgraph_context_graph import graph_version
    from memgraph_context_graph.adapters._identity import resolve_memgraph_env
    from memgraph_context_graph.connectors import configured_connectors
    from memgraph_context_graph.hooks.runner import setup_connector_schemas

    try:
        memgraph_env = resolve_memgraph_env()
        graph_version.setup_at(memgraph_env)
        setup_connector_schemas(configured_connectors(), memgraph_env=memgraph_env)
    except Exception as exc:  # the driver raises its own error types for an unreachable or refusing server
        return Outcome("Schema", "fail", f"{type(exc).__name__}: {exc}", fix=_schema_fix(exc))
    return Outcome("Schema", "ok", f"up to date on {memgraph_env['MEMGRAPH_URL']}")


def replay_kept() -> Outcome | None:
    """Add what hooks kept while the graph refused them (``refused_events``); None when nothing was kept."""
    from memgraph_context_graph import refused_events
    from memgraph_context_graph.adapters._identity import resolve_memgraph_env

    if not refused_events.pending():
        return None
    try:
        result = refused_events.replay(resolve_memgraph_env())
    except Exception as exc:  # the driver raises its own error types for an unreachable or refusing server
        return Outcome("Kept events", "fail", f"{type(exc).__name__}: {exc}", fix="memgraph-context-graph update")
    summary = f"added {result.replayed}"
    if result.waiting:
        summary += f"; {result.waiting} still wait for a newer Context Graph"
    if result.dropped:
        summary += f"; dropped {result.dropped} (see the capture-failure log)"
    return Outcome("Kept events", "warn" if result.waiting or result.dropped else "ok", summary)


def _schema_fix(exc: Exception) -> str:
    """What to do about a failed schema step; running it again only helps when the server was the problem."""
    if "unique constraint" in str(exc):
        # Running again meets the same nodes, so the hint has to say what to remove first.
        return (
            "nodes already in the graph share the value the constraint above makes unique; "
            "delete the extra copies, then run memgraph-context-graph update"
        )
    return "make Memgraph reachable, then run memgraph-context-graph update"
