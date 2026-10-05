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


def session_hints(connectors: list[str]) -> list[str]:
    """The session-start lines of the tools that read one of ``connectors``' graphs."""
    enabled = {_normalize(connector) for connector in connectors}
    return [
        tool.session_hint
        for tool in load_tools().values()
        if tool.session_hint and _normalize(tool.connector) in enabled
    ]


def _normalize(name: str) -> str:
    return name.strip().replace("_", "-")
