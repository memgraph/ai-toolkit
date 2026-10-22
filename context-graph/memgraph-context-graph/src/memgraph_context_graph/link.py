"""AgentLink — the central hub that routes events to graph connectors.

An ``AgentLink`` instance holds a list of :class:`GraphConnector` instances.
Runtime adapters call :meth:`emit` to broadcast events to every registered
connector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .capture_policy import CapturePolicy
    from .events import Event
    from .protocols import GraphConnector

    #: Told about one connector failing one event; the link carries on with the rest.
    ConnectorErrorHandler = Callable[[GraphConnector, Event, Exception], None]


class ConnectorFailuresError(Exception):
    """Connectors that failed while every other connector still ran, raised only when asked for (strict mode)."""

    def __init__(self, failures: list[tuple[str, BaseException]]) -> None:
        self.failures = failures
        names = ", ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in failures)
        super().__init__(f"{len(failures)} connector failure(s): {names}")


class AgentLink:
    """Routes :class:`Event` objects from runtime adapters to graph connectors.

    Usage::

        link = AgentLink()
        link.add_connector(SkillGraphConnector(skill_graph))
        link.add_connector(MyGraphConnector(graph))

        memory = ContextGraphMemory(link=link)  # memgraph_context_graph.sdk
    """

    def __init__(self, on_error: ConnectorErrorHandler | None = None, *, policy: CapturePolicy | None = None) -> None:
        """Create a link.

        Args:
            on_error: When given, a connector that raises is reported here and
                the event still reaches every later connector (and later events
                still run), so one broken connector never costs the others their
                capture. Without it the exception propagates, as a library caller
                driving the link directly expects.
            policy: Rewrites (redacts, clips) or drops each event before any
                connector sees it. The hook runner passes the config file's;
                without one, events reach connectors as emitted.
        """
        self._connectors: list[GraphConnector] = []
        self._on_error = on_error
        self._policy = policy

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
        """Broadcast an event, as the capture policy leaves it, to all connectors that support it."""
        if self._policy is not None:
            policed = self._policy.apply(event)
            if policed is None:
                return
            event = policed
        for connector in self._connectors:
            try:
                if connector.supports(event):
                    connector.on_event(event)
            except Exception as exc:
                if self._on_error is None:
                    raise
                self._on_error(connector, event, exc)
