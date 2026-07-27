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

import logging
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from agent_context_graph.events import Event, EventType, SessionEndEvent, SessionStartEvent
from agent_context_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from .core import SessionsGraph

logger = logging.getLogger(__name__)

_SUPPORTED_EVENTS = {EventType.SESSION_START, EventType.SESSION_END}

#: Read at connector construction time when auto_enrich isn't passed explicitly.
#: Note: hook-based runtimes (Claude Code, Codex) construct this connector
#: without exposing a constructor kwarg for it (see
#: agent_context_graph.adapters.claude_code/codex._add_sessions_graph_connector),
#: so this env var is currently the only way to opt in there; SDK integrations
#: that construct SessionsGraphConnector directly can pass auto_enrich=True instead.
_AUTO_ENRICH_ENV_VAR = "SESSIONS_GRAPH_AUTO_ENRICH"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class SessionsGraphConnector(GraphConnector):
    """Receives Agent Context Graph session events for memory provenance.

    On ``SESSION_START``:
      - MERGEs ``(:User {user_id})`` and ``(:Session {session_id})`` nodes.
      - Tracks ``active_user_id`` and ``active_session_id`` for use in API calls.

    On ``SESSION_END``:
      - Clears the tracked active session context.
      - Marks the Session node ``enrichment_status = 'pending'`` (cheap,
        synchronous, no LLM calls -- safe inside a hook runtime timeout).
      - If ``auto_enrich`` is enabled, best-effort spawns a **detached**
        background process to run the actual (slow, LLM-backed) enrichment,
        so this hook call itself never waits on it. The reliable path if that
        detached process dies is the ``sessions-graph enrich --pending`` CLI.

    Args:
        graph: An initialised :class:`SessionsGraph` instance.
        auto_enrich: Whether to spawn a detached enrichment process on
            SESSION_END. Defaults to the ``SESSIONS_GRAPH_AUTO_ENRICH`` env
            var (truthy: "1"/"true"/"yes"/"on") when not given explicitly.
            Off by default given LightRAG entity extraction's LLM cost.
    """

    def __init__(self, graph: SessionsGraph, *, auto_enrich: bool | None = None) -> None:
        self._graph = graph
        self._active_user_id: str | None = None
        self._active_session_id: str | None = None
        self._auto_enrich = auto_enrich if auto_enrich is not None else _env_flag(_AUTO_ENRICH_ENV_VAR)

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
        self._mark_pending_enrichment(event.session_id)
        if self._auto_enrich:
            self._spawn_enrichment(event.session_id)

    def _mark_pending_enrichment(self, session_id: str) -> None:
        self._graph._db.query(
            "MATCH (s:Session {session_id: $session_id}) SET s.enrichment_status = 'pending';",
            params={"session_id": session_id},
        )

    @staticmethod
    def _spawn_enrichment(session_id: str) -> None:
        executable = shutil.which("sessions-graph")
        command = [executable] if executable else [sys.executable, "-m", "sessions_graph.cli"]
        command += ["enrich", "--session", session_id]
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            logger.warning(f"Could not spawn detached enrichment process for session {session_id}: {e}")
