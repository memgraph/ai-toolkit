"""Registry for command-hook runtime plugins.

A command-hook runtime (Codex, Claude Code, ...) registers itself by exposing
an object with this shape under the ``agent_context_graph.runtimes`` entry
point group, in its own package's ``pyproject.toml``:

    [project.entry-points."agent_context_graph.runtimes"]
    codex = "agent_context_graph.adapters.codex:PLUGIN"

No changes to agent-context-graph itself are needed to add a new runtime.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

_ENTRY_POINT_GROUP = "agent_context_graph.runtimes"


@runtime_checkable
class RuntimeCLIPlugin(Protocol):
    """What a runtime must supply to be usable by agent-context-graph's CLI.

    ``init`` is optional -- a runtime that has no project-local hook-config
    file to generate (or hasn't implemented one yet) may omit it; callers
    should use ``getattr(plugin, "init", None)`` rather than assume it exists.
    """

    name: str
    adapter_class: type[Any]

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None: ...

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]: ...

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None: ...


class UnknownRuntimeError(KeyError):
    """Raised when no plugin is registered for a requested runtime name."""


def _normalize(name: str) -> str:
    return name.strip().replace("_", "-")


def load_runtime_plugins() -> dict[str, RuntimeCLIPlugin]:
    """Discover every registered runtime plugin, keyed by its normalized ``name``."""
    plugins: dict[str, RuntimeCLIPlugin] = {}
    for entry_point in metadata.entry_points(group=_ENTRY_POINT_GROUP):
        plugin = entry_point.load()
        plugins[_normalize(plugin.name)] = plugin
    return plugins


def get_runtime_plugin(name: str) -> RuntimeCLIPlugin:
    """Look up a registered runtime plugin by name.

    Raises ``UnknownRuntimeError`` (listing what *is* registered) if none
    matches -- the CLI catches this to print a clear error instead of a
    traceback.
    """
    plugins = load_runtime_plugins()
    normalized = _normalize(name)
    if normalized not in plugins:
        available = ", ".join(sorted(plugins)) or "(none registered)"
        msg = f"Unknown runtime: {name!r}. Available: {available}"
        raise UnknownRuntimeError(msg)
    return plugins[normalized]
