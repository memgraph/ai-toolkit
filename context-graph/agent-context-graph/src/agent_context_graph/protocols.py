"""Protocols (interfaces) for runtime adapters and graph connectors.

These are the two extension points of agent-context-graph:
1. **RuntimeAdapter** — translates runtime hooks/callbacks into ``Event`` objects.
2. **GraphConnector** — receives ``Event`` objects and writes to a graph component.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import Event


class GraphConnector(ABC):
    """Receives events and persists them to a context-graph component.

    Subclass this for each graph component (actions-graph, skills-graph, …).
    A connector may choose to ignore events it does not care about.
    """

    @abstractmethod
    def on_event(self, event: Event) -> None:
        """Process an event and persist relevant data to the graph.

        This is called synchronously.  If the underlying graph driver
        is async, the connector should manage its own event loop or
        buffering strategy.
        """

    def supports(self, event: Event) -> bool:
        """Return ``True`` if the connector handles events of this type.

        The default implementation returns ``True`` for all events.
        Override to filter on ``event.event_type``.
        """
        return True


class RuntimeAdapter(ABC):
    """Translates runtime-specific hooks or callbacks into :class:`Event` objects.

    Subclass this for each agent runtime integration, including agent
    development SDKs (Claude, OpenAI Agents SDK, ...) and command-hook
    runtimes (Codex, Claude Code, ...).
    """

    @abstractmethod
    def get_runtime_hooks(self) -> Any:
        """Return the runtime-specific hook object(s).

        For Claude Agent SDK this is a ``dict[str, list[HookMatcher]]``.
        For OpenAI Agents SDK this is a ``RunHooksBase`` subclass.
        For command-hook runtimes this may be a generated hook config shape.
        The caller passes the returned value to the SDK's run API.
        """
