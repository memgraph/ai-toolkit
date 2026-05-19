"""Agent Context Graph connector for SessionsGraph.

The connector is intentionally thin: it watches ``SessionStartEvent`` and
``SessionEndEvent`` to MERGE ``(:User)`` and ``(:Session)`` nodes so that
provenance relationships can be wired when memories are saved.  Memory writes
themselves happen through the :class:`SessionsGraph` Python API directly,
not through the event stream.

Usage::

    from sessions_graph import SessionsGraph
    from sessions_graph.connector import SessionsGraphConnector
    from agent_context_graph import AgentLink
    from agent_context_graph.adapters.claude import ClaudeAdapter

    graph = SessionsGraph()
    graph.setup()

    connector = SessionsGraphConnector(graph)

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
    from .core import SessionsGraph


_SUPPORTED_EVENTS = {EventType.SESSION_START, EventType.SESSION_END}


class SessionsGraphConnector(GraphConnector):
    """Receives Agent Context Graph session events for memory provenance.

    On ``SESSION_START``:
      - MERGEs ``(:User {user_id})`` and ``(:Session {session_id})`` nodes.
      - Tracks ``active_user_id`` and ``active_session_id`` for use in API calls.

    On ``SESSION_END``:
      - Clears the tracked active session context.

    Args:
        graph: An initialised :class:`SessionsGraph` instance.
    """

    def __init__(self, graph: SessionsGraph) -> None:
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
            self._active_session_id = event.session_id
            # MERGE User and Session together so the relationship is always wired
            self._graph._db.query(
                """
                MERGE (u:User {user_id: $user_id})
                MERGE (s:Session {session_id: $session_id})
                MERGE (u)-[:HAD_SESSION]->(s)
                """,
                params={"user_id": user_id, "session_id": event.session_id},
            )
        else:
            self._active_session_id = event.session_id
            # No user — still ensure the Session node exists for provenance wiring
            self._graph._db.query(
                "MERGE (:Session {session_id: $session_id});",
                params={"session_id": event.session_id},
            )

    def _on_session_end(self, event: SessionEndEvent) -> None:
        self._active_user_id = None
        self._active_session_id = None
