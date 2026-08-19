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

#: Read at connector construction time when auto_reconcile isn't passed explicitly.
#: Hook-based runtimes (Claude Code, Codex) no longer rely on this: since
#: agent_context_graph 0.2.0, agent_context_graph.hooks.runner._add_sessions_graph_connector
#: resolves the persistent ``reconcile.auto_reconcile`` config-file setting
#: (agent_context_graph.adapters._identity.resolve_auto_reconcile(), set via
#: ``agent-context-graph config set reconcile.auto_reconcile true``) and passes
#: it explicitly, which takes priority. This env var remains a fallback for
#: SDK integrations that construct SessionsGraphConnector directly.
_AUTO_RECONCILE_ENV_VAR = "SESSIONS_GRAPH_AUTO_RECONCILE"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class SessionsGraphConnector(GraphConnector):
    """Receives Agent Context Graph session events for memory provenance.

    On ``SESSION_START``:
      - MERGEs ``(:User {user_id})`` and ``(:Session {session_id})`` nodes.
      - Tracks ``active_user_id`` and ``active_session_id`` for use in API calls.

    On ``SESSION_END``:
      - Clears the tracked active session context.
      - Marks the Session node ``reconciliation_status = 'pending'`` (cheap,
        synchronous, no LLM calls -- safe inside a hook runtime timeout).
      - If ``auto_reconcile`` is enabled, best-effort spawns a **detached**
        background process to run the actual (slow, LLM-backed) reconciliation,
        so this hook call itself never waits on it. The reliable path if that
        detached process dies is the ``sessions-graph reconcile --pending`` CLI.

    Args:
        graph: An initialised :class:`SessionsGraph` instance.
        auto_reconcile: Whether to spawn a detached reconciliation process on
            SESSION_END. Hook-based runtimes pass this explicitly, resolved
            from the persistent ``reconcile.auto_reconcile`` config-file
            setting. Defaults to the ``SESSIONS_GRAPH_AUTO_RECONCILE`` env var
            (truthy: "1"/"true"/"yes"/"on") when not given explicitly. Off by
            default given LightRAG entity extraction's LLM cost.
    """

    def __init__(self, graph: SessionsGraph, *, auto_reconcile: bool | None = None) -> None:
        self._graph = graph
        self._active_user_id: str | None = None
        self._active_session_id: str | None = None
        self._auto_reconcile = auto_reconcile if auto_reconcile is not None else _env_flag(_AUTO_RECONCILE_ENV_VAR)

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
        self._mark_pending_reconciliation(event.session_id)
        if self._auto_reconcile:
            self._spawn_reconciliation(event.session_id)

    def _mark_pending_reconciliation(self, session_id: str) -> None:
        self._graph._db.query(
            "MATCH (s:Session {session_id: $session_id}) SET s.reconciliation_status = 'pending';",
            params={"session_id": session_id},
        )

    @staticmethod
    def _spawn_reconciliation(session_id: str) -> None:
        executable = shutil.which("sessions-graph")
        command = [executable] if executable else [sys.executable, "-m", "sessions_graph.cli"]
        command += ["reconcile", "--session", session_id]
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=_reconciliation_env(),
            )
        except OSError as e:
            logger.warning(f"Could not spawn detached reconciliation process for session {session_id}: {e}")


def _reconciliation_env() -> dict[str, str]:
    """Build the environment for the detached ``sessions-graph reconcile`` subprocess.

    This hook process resolves Memgraph connection settings from
    ``~/.config/context-graph/config.toml`` (per ADR 0002) purely as constructor
    kwargs -- never writing them into ``os.environ``. A plain ``Popen`` without an
    explicit ``env=`` would therefore leave the detached reconciliation subprocess
    with ambient ``os.environ`` only, missing both the configured Memgraph
    connection and any LLM API key LightRAG needs. Overlay the same config-file
    resolution onto a copy of the ambient environment so the child gets what this
    process would have used, without discarding real ambient values (e.g. an
    OPENAI_API_KEY already exported) when config-file values are unset.
    """
    from agent_context_graph.adapters._identity import resolve_llm_env, resolve_memgraph_env

    env = dict(os.environ)
    env.update({k: v for k, v in resolve_memgraph_env().items() if v})
    env.update({k: v for k, v in resolve_llm_env().items() if v})
    return env
