"""Memgraph Context Graph connector for SessionsGraph.

Watches session start, turn end, session end and memory-tool writes to MERGE
``(:User)`` and ``(:Session)`` nodes, link sessions to the memory files they
wrote, mark sessions for reconciliation, and spawn the detached
``sessions-graph`` processes that embed and reconcile them. Memory writes go
through the :class:`SessionsGraph` API, not the event stream.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from memgraph_context_graph.connectors import ConnectorSpec
from memgraph_context_graph.events import (
    Event,
    EventType,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    TurnEndEvent,
)
from memgraph_context_graph.protocols import GraphConnector

from .core import SessionsGraph
from .extraction import ExtractionConfig
from .recall import RecallConfig

if TYPE_CHECKING:
    from memgraph_context_graph.adapters._identity import HookConfig


_SUPPORTED_EVENTS = {EventType.SESSION_START, EventType.SESSION_END, EventType.TURN_END, EventType.TOOL_END}


class SessionsGraphConnector(GraphConnector):
    """Receives Memgraph Context Graph session events for memory provenance.

    On ``SESSION_START``:
      - MERGEs ``(:User {user_id})`` and ``(:Session {session_id})`` nodes.
      - Tracks ``active_user_id`` and ``active_session_id`` for use in API calls.

    On ``SESSION_END``:
      - Clears the tracked active session context.
      - Marks the Session node ``reconciliation_status = 'pending'`` (cheap,
        synchronous, no LLM calls -- safe inside a hook runtime timeout).
      - Always spawns a **detached** ``sessions-graph embed`` process that
        embeds the session's messages inside Memgraph for recall. It costs no
        LLM, so it doesn't wait on ``auto_reconcile``; the hook never waits on
        the model. ``sessions-graph embed --pending`` retries what fails.
      - If ``auto_reconcile`` is enabled, best-effort spawns a **detached**
        background process to run the actual (slow, CPU-bound) reconciliation,
        so this hook call itself never waits on it. The reliable path if that
        detached process dies is the ``sessions-graph reconcile --pending`` CLI.

    On ``TOOL_END`` of a successful ``memory`` tool write:
      - Records ``(:Session)-[:PRODUCED_MEMORY]->(:Memory)`` for the file it
        wrote. The MCP server that served the call doesn't know the session;
        the hook does.

    On ``TURN_END``:
      - Marks the Session node ``reconciliation_status = 'pending'``.
      - Spawns the same detached ``sessions-graph embed`` as a session end:
        it needs no LLM, so recall covers each turn as soon as it ends, and
        runtimes without a session-end hook (Codex, Antigravity,
        ``opencode run``) get embedded at all.
      - Never spawns reconciliation: the session is still open and extraction
        is slow. The ``--pending`` sweep reconciles it, and a later
        turn marks it pending again.

    Args:
        graph: An initialised :class:`SessionsGraph` instance.
        auto_reconcile: Whether to spawn a detached reconciliation process on
            SESSION_END. Off by default, since local extraction is CPU-heavy;
            hook runtimes pass the config file's ``reconcile.auto_reconcile``,
            and no environment variable is consulted (ADR 0002).
    """

    def __init__(self, graph: SessionsGraph, *, auto_reconcile: bool = False) -> None:
        self._graph = graph
        self._active_user_id: str | None = None
        self._active_session_id: str | None = None
        self._auto_reconcile = auto_reconcile

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
        elif isinstance(event, TurnEndEvent):
            self._on_turn_end(event)
        elif isinstance(event, ToolEndEvent):
            self._on_tool_end(event)

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

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        from .memory_store import is_memory_tool, memory_paths_written

        if event.is_error or not is_memory_tool(event.tool_name):
            return
        paths = memory_paths_written(event.metadata.get("tool_input"))
        if paths:
            self._graph.record_memory_provenance(event.session_id, paths)

    def _on_session_start(self, event: SessionStartEvent) -> None:
        user_id: str | None = getattr(event, "user_id", None)

        project = _project_key(event.working_directory or event.metadata.get("cwd"))
        if user_id:
            self._active_user_id = user_id
        self._active_session_id = event.session_id
        self._graph.record_session_start(event.session_id, user_id=user_id, project_key=project)

    def _on_session_end(self, event: SessionEndEvent) -> None:
        self._active_user_id = None
        self._active_session_id = None
        self._graph.reconciler.mark_pending(event.session_id)
        _spawn_detached(["embed", "--session", event.session_id], env=_child_env())
        if self._auto_reconcile:
            _spawn_detached(["reconcile", "--session", event.session_id], env=_child_env())

    def _on_turn_end(self, event: TurnEndEvent) -> None:
        self._graph.reconciler.mark_pending(event.session_id)
        _spawn_detached(["embed", "--session", event.session_id], env=_child_env())


def _project_key(cwd: object) -> str | None:
    """The project a session started in, keyed as memory folders are, so the next session can find it."""
    from .memory_index import project_key_for

    return project_key_for(cwd) if isinstance(cwd, str) and cwd else None


def _spawn_detached(args: list[str], *, env: dict[str, str]) -> None:
    """Start ``sessions-graph <args>`` in its own session, never waited on; a failure to start is logged."""
    executable = shutil.which("sessions-graph")
    command = [executable] if executable else [sys.executable, "-m", "sessions_graph.cli"]
    command += args
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    except OSError as e:
        from memgraph_context_graph.capture_log import record_failure

        session_id = args[args.index("--session") + 1] if "--session" in args else None
        record_failure("spawn", e, connector="sessions-graph", session_id=session_id)


def _child_env() -> dict[str, str]:
    """The environment for a detached ``sessions-graph`` subprocess: no config values, only which file holds them.

    The child reads its connection and provider keys from that file itself
    (``cli._fill_env_from_context_graph_config``), so a key the file leaves
    empty is never filled from this process's environment.
    """
    from memgraph_context_graph.adapters._identity import spawn_env

    return spawn_env()


def _doctor_check(config: HookConfig) -> tuple[bool, str]:
    """Memory is filed under a user, so a hook needs one to resolve."""
    from memgraph_context_graph.adapters._identity import resolve_user_id

    user_id = resolve_user_id({})
    if not user_id:
        return False, "user_id not resolved — run: memgraph-context-graph config set identity.user_id <your-name>"
    return True, f"user_id={user_id!r}"


CONNECTOR = ConnectorSpec(
    name="sessions-graph",
    graph=SessionsGraph,
    # Config file only (ADR 0002); never configured means off.
    build=lambda graph, config: SessionsGraphConnector(graph, auto_reconcile=bool(config.auto_reconcile)),
    check=_doctor_check,
    settings={"recall": RecallConfig.from_mapping, "extraction": ExtractionConfig.from_mapping},
)
