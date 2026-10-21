"""Forgetting a session: every installed component deletes its share, then the Session Node goes.

Each component owns what it wrote, so each graph class has a
``forget_session(session_id) -> dict[str, int]`` that deletes exactly its share
and reports counts. This module only decides the order and the guard around it:

1. :func:`memgraph_context_graph.session.mark_forgotten`, so no hook or
   reconciliation still running for the session creates it again;
2. sessions-graph first, since it finds the session's Chunks through Actions
   actions-graph deletes, then resources-graph, skills-graph and actions-graph;
3. the same again, for what a hook that started before step 1 wrote meanwhile;
4. the Session Node, and the session's lines in the local capture-failure log.

Deletion is final: nothing is archived, and the tombstone holds only the id.
What other sessions or the user's memory files also back (shared Entities,
Chunks, Skills, Resources, Memory files) stays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from memgraph_context_graph import capture_log, session
from memgraph_context_graph.connectors import COMPONENTS

if TYPE_CHECKING:
    from memgraph_toolbox.api.memgraph import Memgraph

#: sessions-graph reads the session's Actions, so it runs before actions-graph deletes them.
FORGET_ORDER: tuple[str, ...] = ("sessions-graph", "resources-graph", "skills-graph", "actions-graph")


@dataclass
class Forgotten:
    """What forgetting one session removed."""

    session_id: str
    #: Counts per kind of thing removed (``actions``, ``entities``, ``touches``…), summed over components.
    removed: dict[str, int] = field(default_factory=dict)
    #: Whether a Session Node was there to delete.
    existed: bool = False
    #: Components not installed here, whose share (if any) could not be reached.
    unavailable: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionSummary:
    """One session as ``forget`` shows it before deleting."""

    session_id: str
    started_at: str | None
    working_directory: str | None
    actions: int


def forget_session(memgraph: Memgraph, session_id: str) -> Forgotten:
    """Forget *session_id* across every installed component; see the module docstring for the order.

    Safe to repeat: a session already forgotten has nothing left to remove.
    """
    session.mark_forgotten(memgraph, session_id)
    graphs, unavailable = _graphs(memgraph)
    result = Forgotten(session_id=session_id, unavailable=unavailable)
    for _ in range(2):
        for graph in graphs:
            for kind, count in graph.forget_session(session_id).items():
                result.removed[kind] = result.removed.get(kind, 0) + count
    result.existed = session.delete(memgraph, session_id)
    result.removed["capture_log_lines"] = capture_log.forget_session(session_id)
    return result


def describe(memgraph: Memgraph, session_ids: list[str]) -> list[SessionSummary]:
    """The sessions *session_ids* names that exist, for showing before forgetting them."""
    rows = memgraph.query(
        """
        MATCH (s:Session) WHERE s.session_id IN $ids
        OPTIONAL MATCH (s)-[:HAS_ACTION]->(a)
        RETURN s.session_id AS id, s.started_at AS started_at, s.working_directory AS cwd, count(a) AS actions
        ORDER BY started_at
        """,
        params={"ids": session_ids},
    )
    return [SessionSummary(row["id"], _text(row["started_at"]), row["cwd"], row["actions"]) for row in rows]


def sessions_in(memgraph: Memgraph, directory: str) -> list[str]:
    """The sessions whose working directory is *directory* or inside it, oldest first."""
    root = os.path.normpath(os.path.abspath(os.path.expanduser(directory)))
    rows = memgraph.query(
        """
        MATCH (s:Session)
        WHERE s.working_directory = $root OR s.working_directory STARTS WITH $prefix
        RETURN s.session_id AS id
        ORDER BY s.started_at
        """,
        params={"root": root, "prefix": root.rstrip(os.sep) + os.sep},
    )
    return [row["id"] for row in rows]


def _graphs(memgraph: Memgraph) -> tuple[list[Any], list[str]]:
    """Each installed component's graph class over *memgraph*, in :data:`FORGET_ORDER`, and those missing."""
    graphs, unavailable = [], []
    for name in FORGET_ORDER:
        try:
            graphs.append(COMPONENTS[name].graph()(memgraph))
        except ImportError:
            unavailable.append(name)
    return graphs, unavailable


def _text(value: Any) -> str | None:
    return None if value is None else str(value)
