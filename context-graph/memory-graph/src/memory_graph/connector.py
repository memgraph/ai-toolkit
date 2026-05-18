"""Agent Context Graph connector for MemoryGraph.

The connector lives inside memory-graph because MemoryGraph owns the
interpretation of session events for provenance — Agent Context Graph
only routes normalized runtime events.

The connector is intentionally thin: it watches ``SessionStartEvent`` and
``SessionEndEvent`` to ensure that ``(:User)`` and ``(:Session)`` nodes
exist in the graph before any :meth:`MemoryGraph.save_memory` call tries to
wire provenance relationships.  Memory writes themselves happen through the
:class:`MemoryGraph` Python API directly, not through the event stream.

Usage::

    from memory_graph import MemoryGraph
    from memory_graph.connector import MemoryGraphConnector
    from agent_context_graph import AgentLink
    from agent_context_graph.adapters.claude import ClaudeAdapter

    graph = MemoryGraph()
    graph.setup()

    connector = MemoryGraphConnector(graph)

    link = AgentLink()
    link.add_connector(connector)
    adapter = ClaudeAdapter(link, session_id="s-1", session_kwargs={"user_id": "alice"})

    # Later, from within the agent:
    graph.save_memory(
        user_id=connector.active_user_id,
        content="User prefers concise answers",
        session_id=connector.active_session_id,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_context_graph.events import Event, EventType, SessionEndEvent, SessionStartEvent
from agent_context_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from .core import MemoryGraph


_SUPPORTED_EVENTS = {EventType.SESSION_START, EventType.SESSION_END}


class MemoryGraphConnector(GraphConnector):
    """Receives Agent Context Graph session events and prepares Memory Graph provenance.

    On ``SESSION_START``:
      - MERGEs a ``(:User {user_id})`` node so ownership relationships can be created.
      - MERGEs a ``(:Session {session_id})`` node so provenance relationships can be created.
      - Tracks ``active_user_id`` and ``active_session_id`` for convenient API access.

    On ``SESSION_END``:
      - Clears the tracked active session context.

    Args:
        graph: An initialised :class:`MemoryGraph` instance.
    """

    def __init__(self, graph: MemoryGraph) -> None:
        self._graph = graph
        self._active_user_id: str | None = None
        self._active_session_id: str | None = None

    # ------------------------------------------------------------------
    # GraphConnector interface
    # ------------------------------------------------------------------

    def supports(self, event: Event) -> bool:
        return event.event_type in _SUPPORTED_EVENTS

    def on_event(self, event: Event) -> None:
        if isinstance(event, SessionStartEvent):
            self._on_session_start(event)
        elif isinstance(event, SessionEndEvent):
            self._on_session_end(event)

    # ------------------------------------------------------------------
    # Active session context (convenience for callers)
    # ------------------------------------------------------------------

    @property
    def active_user_id(self) -> str | None:
        """The ``user_id`` from the most recent ``SessionStartEvent``, if any."""
        return self._active_user_id

    @property
    def active_session_id(self) -> str | None:
        """The ``session_id`` from the most recent ``SessionStartEvent``, if any."""
        return self._active_session_id

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_session_start(self, event: SessionStartEvent) -> None:
        user_id: str | None = getattr(event, "user_id", None)

        if user_id:
            self._active_user_id = user_id
            # Ensure the User node exists before any save_memory call
            self._graph._db.query(
                "MERGE (:User {user_id: $user_id});",
                params={"user_id": user_id},
            )

        self._active_session_id = event.session_id
        # Ensure the Session node exists for provenance wiring
        self._graph._db.query(
            "MERGE (:Session {session_id: $session_id});",
            params={"session_id": event.session_id},
        )

    def _on_session_end(self, event: SessionEndEvent) -> None:
        self._active_user_id = None
        self._active_session_id = None
