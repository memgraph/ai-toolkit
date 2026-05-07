"""AgentLink — the central hub that routes events to graph connectors.

An ``AgentLink`` instance holds a list of :class:`GraphConnector` instances.
Runtime adapters call :meth:`emit` to broadcast events to every registered
connector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import Event
    from .protocols import GraphConnector


class AgentLink:
    """Routes :class:`Event` objects from runtime adapters to graph connectors.

    Usage::

        link = AgentLink()
        link.add_connector(SkillGraphConnector(skill_graph))
        link.add_connector(MyGraphConnector(graph))

        adapter = ClaudeAdapter(link, session_id="s-1")
        hooks = adapter.get_runtime_hooks()
    """

    def __init__(self) -> None:
        self._connectors: list[GraphConnector] = []

    def add_connector(self, connector: GraphConnector) -> None:
        """Register a graph connector."""
        self._connectors.append(connector)

    def remove_connector(self, connector: GraphConnector) -> None:
        """Unregister a graph connector."""
        self._connectors.remove(connector)

    @property
    def connectors(self) -> list[GraphConnector]:
        """Return all registered connectors (read-only snapshot)."""
        return list(self._connectors)

    def emit(self, event: Event) -> None:
        """Broadcast an event to all connectors that support it."""
        for connector in self._connectors:
            if connector.supports(event):
                connector.on_event(event)
