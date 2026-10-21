"""The Session Node: the ``(:Session {session_id})`` every Context Graph component joins on.

Any component may create it and none owns its lifecycle (sessions-graph owns
``(:User)`` and ``HAD_SESSION``), so its shape lives here rather than in one
component: the label, its key, the uniqueness constraint and the MERGE that
creates it. Hooks run one process per event, often at once, so two of them
can create a new session together; :func:`ensure` survives that race.

A component calls :func:`setup` from its own ``setup()`` and :func:`ensure`
before any query that needs the node, which then MATCHes it.

A forgotten session (``memgraph-context-graph forget``) leaves a
``(:ForgottenSession {session_id, forgotten_at})`` behind and nothing else.
:func:`ensure` never recreates a session that has one, so a hook or a
reconciliation still running for it finds no node to write to: forgetting is
permanent, even for a session the agent later resumes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memgraph_toolbox.api.memgraph import merge_with_retry


def setup(db: Any) -> None:
    """Create the Session Node's uniqueness constraint and key index, and the tombstone's; safe to repeat."""
    db.query("CREATE CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;")
    db.query("CREATE INDEX ON :Session(session_id);")
    db.query("CREATE CONSTRAINT ON (f:ForgottenSession) ASSERT f.session_id IS UNIQUE;")
    db.query("CREATE INDEX ON :ForgottenSession(session_id);")


def ensure(db: Any, session_id: str, **on_create: Any) -> None:
    """Make sure the Session Node *session_id* exists, setting *on_create* only if this call creates it.

    A forgotten session is not created again; the caller's MATCH then finds nothing.

    Args:
        db: The Memgraph client to write through.
        session_id: The session's key.
        **on_create: Properties a newly created node starts with; an existing
            node is left as it is.
    """
    merge_with_retry(
        db,
        """
        OPTIONAL MATCH (f:ForgottenSession {session_id: $session_id})
        WITH f WHERE f IS NULL
        MERGE (s:Session {session_id: $session_id}) ON CREATE SET s += $props
        """,
        params={"session_id": session_id, "props": on_create},
    )


def mark_forgotten(db: Any, session_id: str) -> None:
    """Record that *session_id* is forgotten, so nothing creates it again; safe to repeat."""
    merge_with_retry(
        db,
        "MERGE (f:ForgottenSession {session_id: $session_id}) ON CREATE SET f.forgotten_at = $now",
        params={"session_id": session_id, "now": datetime.now(timezone.utc).isoformat()},
    )


def is_forgotten(db: Any, session_id: str) -> bool:
    """Whether *session_id* has been forgotten."""
    rows = db.query(
        "MATCH (f:ForgottenSession {session_id: $session_id}) RETURN count(f) AS n", params={"session_id": session_id}
    )
    return bool(rows and rows[0]["n"])


def delete(db: Any, session_id: str) -> bool:
    """Delete the Session Node *session_id* and its relationships; whether there was one.

    Only the node: what components hung off it is theirs to delete first
    (``memgraph_context_graph.forget``).
    """
    rows = db.query(
        "MATCH (s:Session {session_id: $session_id}) DETACH DELETE s RETURN count(*) AS n",
        params={"session_id": session_id},
    )
    return bool(rows and rows[0]["n"])
