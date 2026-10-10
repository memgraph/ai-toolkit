"""Tools that context-graph components offer to a harness's model.

``agent-context-graph mcp`` serves every registered tool over MCP, and the CLI
runs them by name (``agent-context-graph recall``). A component registers a
tool under the ``agent_context_graph.tools`` entry-point group, in its own
``pyproject.toml``:

    [project.entry-points."agent_context_graph.tools"]
    recall = "sessions_graph.tool:RECALL"

A tool reads its identity and Memgraph connection from the config file
(:class:`~agent_context_graph.adapters._identity.HookConfig`), never from its
arguments: the model can't ask on another user's behalf (#394).

A tool may also define ``available(config) -> bool``; while it returns False
the tool is neither listed nor callable, nor is its session hint shown. The
``memory`` tool uses this to exist only for users who opted into graph memory.

A tool may define ``session_context(config, payload) -> str | None`` to write
its session-start text from the SessionStart payload (e.g. its ``cwd``) instead
of the fixed ``session_hint``; if it raises, the fixed hint is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_context_graph.adapters._identity import HookConfig

_ENTRY_POINT_GROUP = "agent_context_graph.tools"


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns: text for the model, and the same data as JSON for programs (``--json``)."""

    text: str
    structured: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """A tool a component contributes; instances are long-lived and may hold a connection."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str:
        """What the model reads to decide when to call the tool."""
        ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    @property
    def connector(self) -> str:
        """The connector whose graph the tool reads, e.g. ``sessions-graph``."""
        ...

    @property
    def session_hint(self) -> str | None:
        """One line added to the model's context at session start, or None."""
        ...

    def call(self, arguments: dict[str, Any], config: HookConfig) -> ToolResult:
        """Run the tool.

        Raises:
            ToolError: the call can't be answered (bad arguments, missing config);
                the message is shown to the model.
        """
        ...


class ToolError(Exception):
    """A tool call that can't be answered, with a message meant for the model."""


def load_tools() -> dict[str, Tool]:
    """Every registered tool, keyed by name."""
    tools: dict[str, Tool] = {}
    for entry_point in metadata.entry_points(group=_ENTRY_POINT_GROUP):
        tool = entry_point.load()
        tools[tool.name] = tool
    return tools


def available_tools(config: HookConfig) -> dict[str, Tool]:
    """The registered tools *config* turns on, keyed by name."""
    return {name: tool for name, tool in load_tools().items() if is_available(tool, config)}


def is_available(tool: Tool, config: HookConfig) -> bool:
    """Whether *tool* is on for *config*; tools without an ``available`` check always are."""
    available = getattr(tool, "available", None)
    return available is None or bool(available(config))


def session_hints(
    connectors: list[str], config: HookConfig | None = None, *, payload: dict[str, Any] | None = None
) -> list[str]:
    """The session-start text of the available tools that read one of ``connectors``' graphs."""
    if config is None:
        from agent_context_graph.adapters._identity import load_config

        config = load_config()
    enabled = {_normalize(connector) for connector in connectors}
    hints = []
    for tool in available_tools(config).values():
        if _normalize(tool.connector) in enabled and (hint := _session_text(tool, config, payload or {})):
            hints.append(hint)
    return hints


def _session_text(tool: Tool, config: HookConfig, payload: dict[str, Any]) -> str | None:
    session_context = getattr(tool, "session_context", None)
    if session_context is not None:
        try:
            return session_context(config, payload)
        except Exception:  # Never fail the hook over memory context; fall back to the fixed hint.
            return tool.session_hint
    return tool.session_hint


def _normalize(name: str) -> str:
    return name.strip().replace("_", "-")
