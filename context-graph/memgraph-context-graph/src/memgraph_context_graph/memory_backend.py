"""Switching a user's harness memory between the harness's own and Context Graph.

Neither Claude Code nor Codex lets another store stand in for its built-in
memory, so "Context Graph as the memory backend" means: turn the harness's
memory off in its own user-level config, record ``[memory] backend =
"context-graph"`` in ours (which turns on the ``memory`` tool and the
session-start index), and bring the harness's existing memories across once,
so opting in never starts from empty. Handing memory back carries the graph's
memories into the harness's own store, where it has one, so switching back
doesn't start from the day the user switched away. The graph is the one
source of truth while it is the backend; nothing writes to both.

A runtime plugin takes part by exposing ``native_memory``, an object with the
:class:`NativeMemory` shape. Runtimes without one can't switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from memgraph_context_graph.adapters._identity import (
    MEMORY_BACKEND_GRAPH,
    MEMORY_BACKEND_NATIVE,
    load_config,
    write_config,
)
from memgraph_context_graph.spaces import spaces_for

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sessions_graph.memory_store import MemoryStore

    from memgraph_context_graph.hooks.runtime_plugin import RuntimeCLIPlugin


@dataclass
class ImportReport:
    """What an import brought across, and what it left alone."""

    imported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class ExportReport:
    """What an export wrote into the harness's store, which graph paths it had no place for, what it kept, and what
    to tell."""

    exported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Native files left as they were because they differ from the graph's copy and it doesn't descend from them.
    kept: list[str] = field(default_factory=list)
    #: Lines for the user about how the harness takes the exported files in, when that isn't immediate.
    notes: list[str] = field(default_factory=list)


class StoredMemory(Protocol):
    """One file of the graph's memory, as an export reads it."""

    @property
    def path(self) -> str: ...

    @property
    def content(self) -> str: ...


class EarlierText(Protocol):
    """Text a write to a graph memory file replaced."""

    @property
    def content(self) -> str: ...


class MemorySource(Protocol):
    """The part of sessions-graph's ``MemoryStore`` an export reads through."""

    def files(self) -> Sequence[StoredMemory]: ...

    def versions(self, path: str) -> Sequence[EarlierText]: ...

    def written_in(self, path: str) -> list[str]: ...


class MemoryFiles(Protocol):
    """The part of sessions-graph's ``MemoryStore`` an import writes through."""

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str: ...

    def read(self, path: str) -> str | None: ...

    def versions(self, path: str) -> Sequence[EarlierText]: ...

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

    def export_from(self, source: MemorySource, home: Path) -> ExportReport:
        """Write *source*'s memories into the harness's own store, where it has a place for them."""
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
    user_id = spaces_for(config).user_id()
    if not user_id:
        raise MemoryBackendError(
            "No user is configured, so there is no memory to move into. "
            "Set one first: memgraph-context-graph config set identity.user_id <name>"
        )
    report = native.import_into(_memory_files(config, user_id), home)
    lines = [f"Imported {path}" for path in report.imported]
    lines += [f"Kept existing {path}" for path in report.skipped]
    lines += native.disable(home)
    write_config(memory_backend=MEMORY_BACKEND_GRAPH)
    lines.append(
        f"Set memory.backend = {MEMORY_BACKEND_GRAPH}; the memory tool is on from the next {plugin.name} session."
    )
    return lines


def use_native_memory(plugin: RuntimeCLIPlugin, *, home: Path | None = None) -> list[str]:
    """Hand memory back to *plugin*'s own: graph memories copied into it, harness memory on, ours off.

    The copy runs first and only while the graph is the backend, so a failure (Memgraph down) leaves the
    user on the graph's memory. ``memory.backend`` is one value for every runtime, so hand the others
    back with :func:`hand_back_memory` first: once it is native, nothing is copied.

    Raises:
        MemoryBackendError: the runtime can't switch, or the graph's memories can't be read.
    """
    lines = hand_back_memory(plugin, home=home)
    write_config(memory_backend=MEMORY_BACKEND_NATIVE)
    lines.append(
        f"Set memory.backend = {MEMORY_BACKEND_NATIVE}; {plugin.name}'s own memory is back from its next session."
    )
    return lines


def hand_back_memory(plugin: RuntimeCLIPlugin, *, home: Path | None = None) -> list[str]:
    """Copy the graph's memories into *plugin*'s own store and turn it back on, leaving ``memory.backend`` as is.

    Raises:
        MemoryBackendError: the runtime can't switch, or the graph's memories can't be read.
    """
    native = _require_native(plugin)
    home = home or Path.home()
    return export_graph_memory(native, plugin.name, home=home) + native.enable(home)


def export_graph_memory(native: NativeMemory, name: str, *, home: Path) -> list[str]:
    """Copy the graph's memories into *native*'s store while the graph is the backend; returns what happened.

    Raises:
        MemoryBackendError: sessions-graph isn't installed, or Memgraph can't be reached.
    """
    config = load_config()
    user_id = spaces_for(config).user_id()
    if not config.graph_memory or not user_id:
        return []
    report = native.export_from(_memory_files(config, user_id), home)
    lines = [f"Copied back {path}" for path in report.exported]
    if report.skipped:
        lines.append(f"Kept in Memgraph only, with no place in {name}'s memory: {', '.join(report.skipped)}")
    lines += [f"Kept {path}: it differs from the graph's copy, which doesn't descend from it" for path in report.kept]
    return lines + report.notes


def keep_in_graph(source: MemorySource) -> ExportReport:
    """The export of a harness with no store to write into: every memory stays in the graph."""
    return ExportReport(skipped=[memory.path for memory in source.files()])


def _require_native(plugin: RuntimeCLIPlugin) -> NativeMemory:
    native = native_memory_of(plugin)
    if native is None:
        raise MemoryBackendError(f"{plugin.name} has no memory Context Graph can replace.")
    return native


def _memory_files(config: Any, user_id: str) -> MemoryStore:
    try:
        from sessions_graph import SessionsGraph
    except ImportError as exc:
        raise MemoryBackendError(
            "Context Graph memory needs sessions-graph: "
            "uv tool install 'memgraph-context-graph[mcp]' --with 'sessions-graph[memgraph-context-graph]'"
        ) from exc
    target = spaces_for(config).personal().target
    try:
        graph = SessionsGraph(**target.client_kwargs())
        graph.setup()
    except Exception as exc:
        raise MemoryBackendError(f"Can't reach Memgraph at {target.url}: {exc}") from exc
    return graph.memory_store(user_id)
