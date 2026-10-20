"""The Session Node: the ``(:Session {session_id})`` every Context Graph component joins on.

Any component may create it and none owns its lifecycle (sessions-graph owns
``(:User)`` and ``HAD_SESSION``), so its shape lives here rather than in one
component: the label, its key, the uniqueness constraint and the MERGE that
creates it. Hooks run one process per event, often at once, so two of them
can create a new session together; :func:`ensure` survives that race.

A component calls :func:`setup` from its own ``setup()`` and :func:`ensure`
before any query that needs the node, which then MATCHes it.
"""

from __future__ import annotations

from typing import Any

from memgraph_toolbox.api.memgraph import merge_with_retry


def setup(db: Any) -> None:
    """Create the Session Node's uniqueness constraint and key index; safe to repeat."""
    db.query("CREATE CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;")
    db.query("CREATE INDEX ON :Session(session_id);")


def ensure(db: Any, session_id: str, **on_create: Any) -> None:
    """Make sure the Session Node *session_id* exists, setting *on_create* only if this call creates it.

    Args:
        db: The Memgraph client to write through.
        session_id: The session's key.
        **on_create: Properties a newly created node starts with; an existing
            node is left as it is.
    """
    merge_with_retry(
        db,
        "MERGE (s:Session {session_id: $session_id}) ON CREATE SET s += $props",
        params={"session_id": session_id, "props": on_create},
    )
