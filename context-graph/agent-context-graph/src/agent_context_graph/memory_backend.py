"""Switching a user's harness memory between the harness's own and Context Graph.

Neither Claude Code nor Codex lets another store stand in for its built-in
memory, so "Context Graph as the memory backend" means: turn the harness's
memory off in its own user-level config, record ``[memory] backend =
"context-graph"`` in ours (which turns on the ``memory`` tool and the
session-start index), and bring the harness's existing memories across once,
so opting in never starts from empty.

A runtime plugin takes part by exposing ``native_memory``, an object with the
:class:`NativeMemory` shape. Runtimes without one can't switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agent_context_graph.adapters._identity import (
    MEMORY_BACKEND_GRAPH,
    MEMORY_BACKEND_NATIVE,
    load_config,
    write_config,
)

if TYPE_CHECKING:
    from agent_context_graph.hooks.runtime_plugin import RuntimeCLIPlugin


@dataclass
class ImportReport:
    """What an import brought across, and what it left alone."""

    imported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class MemoryFiles(Protocol):
    """The part of sessions-graph's ``MemoryStore`` an import writes through."""

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str: ...

    def create(self, path: str, file_text: str) -> str: ...


class NativeMemory(Protocol):
    """A harness's built-in memory, as far as switching away from it needs."""

    def enabled(self, home: Path) -> bool:
        """Whether the harness's own memory is on for this user."""
        ...

    def disable(self, home: Path) -> list[str]:
        """Turn the harness's memory off in its user-level config; returns what changed."""
        ...

    def enable(self, home: Path) -> list[str]:
        """Undo :meth:`disable`; returns what changed."""
        ...

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Copy the harness's existing memories into *files*, never overwriting a path."""
        ...


class MemoryBackendError(Exception):
    """The switch can't be made; the message says why and what to do."""


def native_memory_of(plugin: RuntimeCLIPlugin) -> NativeMemory | None:
    """The runtime's built-in memory, or None for a runtime that can't switch."""
    return getattr(plugin, "native_memory", None)


def use_graph_memory(plugin: RuntimeCLIPlugin, *, home: Path | None = None) -> list[str]:
    """Make Context Graph *plugin*'s memory for this user: harness memory off, ours on, old memories imported.

    The import runs first, so a failure (Memgraph down, sessions-graph
    missing) leaves the user on their harness memory rather than on an
    empty one. Re-running is safe: imported paths are skipped.

    Raises:
        MemoryBackendError: the runtime can't switch, no user is configured,
            sessions-graph isn't installed, or Memgraph can't be reached.
    """
    native = _require_native(plugin)
    home = home or Path.home()
    config = load_config()
    if not config.user_id:
        raise MemoryBackendError(
            "No user is configured, so there is no memory to move into. "
            "Set one first: agent-context-graph config set identity.user_id <name>"
        )
    report = native.import_into(_memory_files(config), home)
    lines = [f"Imported {path}" for path in report.imported]
    lines += [f"Kept existing {path}" for path in report.skipped]
    lines += native.disable(home)
    write_config(memory_backend=MEMORY_BACKEND_GRAPH)
    lines.append(
        f"Set memory.backend = {MEMORY_BACKEND_GRAPH}; the memory tool is on from the next {plugin.name} session."
    )
    return lines


def use_native_memory(plugin: RuntimeCLIPlugin, *, home: Path | None = None) -> list[str]:
    """Hand memory back to *plugin*'s own: harness memory on, ours off. Graph memories stay in Memgraph.

    Raises:
        MemoryBackendError: the runtime can't switch.
    """
    native = _require_native(plugin)
    lines = native.enable(home or Path.home())
    write_config(memory_backend=MEMORY_BACKEND_NATIVE)
    lines.append(
        f"Set memory.backend = {MEMORY_BACKEND_NATIVE}; {plugin.name}'s own memory is back from its next session."
    )
    return lines


def _require_native(plugin: RuntimeCLIPlugin) -> NativeMemory:
    native = native_memory_of(plugin)
    if native is None:
        raise MemoryBackendError(f"{plugin.name} has no memory Context Graph can replace.")
    return native


def _memory_files(config: Any) -> MemoryFiles:
    try:
        from sessions_graph import SessionsGraph
    except ImportError as exc:
        raise MemoryBackendError(
            "Context Graph memory needs sessions-graph: "
            "uv tool install 'agent-context-graph[mcp]' --with 'sessions-graph[agent-context-graph]'"
        ) from exc
    try:
        graph = SessionsGraph(
            url=config.memgraph_url,
            username=config.memgraph_user,
            password=config.memgraph_password,
            database=config.memgraph_database,
        )
        graph.setup()
    except Exception as exc:
        raise MemoryBackendError(f"Can't reach Memgraph at {config.memgraph_url}: {exc}") from exc
    return graph.memory_store(config.user_id)
