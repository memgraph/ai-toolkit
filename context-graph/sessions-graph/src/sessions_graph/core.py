"""Core SessionsGraph class for storing and recalling agent memories in Memgraph.

Graph schema
------------
Nodes:
    (:User  {user_id})
    (:Memory {memory_id, user_id, content, created_at, session_id?})
    (:Session {session_id})

Relationships:
    (:User)-[:HAS_MEMORY]->(:Memory)
    (:Session)-[:PRODUCED_MEMORY]->(:Memory)   — only when session_id is provided
"""

from __future__ import annotations

import contextlib
from typing import Any

from memgraph_toolbox.api.memgraph import Memgraph

from .models import Memory, validate_content, validate_memory_id, validate_user_id

_FULLTEXT_INDEX = "memory_content_index"


class SessionsGraph:
    """Store and recall agent memories in Memgraph.

    Provides:
    - :meth:`save_memory`   — persist a new Memory for a user
    - :meth:`get_memories`  — retrieve all Memories for a user
    - :meth:`search_memories` — full-text search over Memory content
    - :meth:`update_memory` — replace the content of an existing Memory
    - :meth:`delete_memory` — remove a Memory by ID
    """

    def __init__(self, memgraph: Memgraph | None = None, **kwargs: Any) -> None:
        """Initialise SessionsGraph.

        Args:
            memgraph: An existing Memgraph client instance.  When *None* a new
                      one is created from *kwargs* / environment variables.
            **kwargs: Forwarded to :class:`Memgraph` when *memgraph* is ``None``.
        """
        self._db = memgraph or Memgraph(**kwargs)

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create constraints, indexes, and the full-text index."""
        self._db.query("CREATE CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE;")
        self._db.query("CREATE CONSTRAINT ON (m:Memory) ASSERT m.memory_id IS UNIQUE;")
        self._db.query("CREATE INDEX ON :Memory(user_id);")
        self._db.query("CREATE INDEX ON :Memory(created_at);")
        self._db.query(f"CREATE TEXT INDEX {_FULLTEXT_INDEX} ON :Memory(content);")

    def drop(self) -> None:
        """Remove all Memory-related constraints and indexes."""
        with contextlib.suppress(Exception):
            self._db.query("DROP CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE;")
        with contextlib.suppress(Exception):
            self._db.query("DROP CONSTRAINT ON (m:Memory) ASSERT m.memory_id IS UNIQUE;")
        with contextlib.suppress(Exception):
            self._db.query(f"DROP TEXT INDEX {_FULLTEXT_INDEX};")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_memory(
        self,
        user_id: str,
        content: str,
        *,
        session_id: str | None = None,
        memory_id: str | None = None,
    ) -> Memory:
        """Persist a new Memory for *user_id*.

        Args:
            user_id:    The owning user identity.
            content:    The free-form text assertion to store.
            session_id: Optional session that produced this memory (for provenance).
            memory_id:  Override the auto-generated UUID (useful in tests).

        Returns:
            The persisted :class:`Memory` instance.
        """
        memory = Memory(
            user_id=validate_user_id(user_id),
            content=validate_content(content),
            session_id=session_id,
            **({"memory_id": memory_id} if memory_id else {}),
        )

        # MERGE user, CREATE memory, wire ownership
        self._db.query(
            """
            MERGE (u:User {user_id: $user_id})
            CREATE (m:Memory {
                memory_id: $memory_id,
                user_id:   $user_id,
                content:   $content,
                created_at: $created_at
            })
            CREATE (u)-[:HAS_MEMORY]->(m)
            """,
            params={
                "user_id": memory.user_id,
                "memory_id": memory.memory_id,
                "content": memory.content,
                "created_at": memory.created_at,
            },
        )

        # Wire session provenance when a session_id is supplied
        if session_id:
            self._db.query(
                """
                MERGE (s:Session {session_id: $session_id})
                WITH s
                MATCH (m:Memory {memory_id: $memory_id})
                CREATE (s)-[:PRODUCED_MEMORY]->(m)
                """,
                params={"session_id": session_id, "memory_id": memory.memory_id},
            )

        return memory

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_memories(self, user_id: str) -> list[Memory]:
        """Return all Memories owned by *user_id*, newest first."""
        validate_user_id(user_id)
        rows = self._db.query(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            OPTIONAL MATCH (s:Session)-[:PRODUCED_MEMORY]->(m)
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.created_at AS created_at,
                   s.session_id AS session_id
            ORDER BY m.created_at DESC
            """,
            params={"user_id": user_id},
        )
        return [self._row_to_memory(r) for r in rows]

    def search_memories(self, user_id: str, query: str, *, limit: int = 10) -> list[Memory]:
        """Full-text search over Memory content for *user_id*.

        Args:
            user_id: Only return Memories owned by this user.
            query:   Full-text search query string.
            limit:   Maximum number of results to return.

        Returns:
            Matching :class:`Memory` instances ordered by relevance score.
        """
        validate_user_id(user_id)
        if not query or not query.strip():
            return []

        rows = self._db.query(
            f"""
            CALL text_search.search_all('{_FULLTEXT_INDEX}', $query)
            YIELD node AS m, score
            WHERE m.user_id = $user_id
            OPTIONAL MATCH (s:Session)-[:PRODUCED_MEMORY]->(m)
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.created_at AS created_at,
                   s.session_id AS session_id
            ORDER BY score DESC
            LIMIT {int(limit)}
            """,
            params={"user_id": user_id, "query": query.strip()},
        )
        return [self._row_to_memory(r) for r in rows]

    # ------------------------------------------------------------------
    # Update / Delete
    # ------------------------------------------------------------------

    def update_memory(self, memory_id: str, content: str) -> Memory | None:
        """Replace the content of an existing Memory.

        Returns the updated :class:`Memory`, or ``None`` if not found.
        """
        validate_memory_id(memory_id)
        validate_content(content)

        rows = self._db.query(
            """
            MATCH (m:Memory {memory_id: $memory_id})
            SET m.content = $content
            OPTIONAL MATCH (s:Session)-[:PRODUCED_MEMORY]->(m)
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.created_at AS created_at,
                   s.session_id AS session_id
            """,
            params={"memory_id": memory_id, "content": content},
        )
        if not rows:
            return None
        return self._row_to_memory(rows[0])

    def delete_memory(self, memory_id: str) -> None:
        """Remove a Memory and all its relationships by ID."""
        validate_memory_id(memory_id)
        self._db.query(
            "MATCH (m:Memory {memory_id: $memory_id}) DETACH DELETE m;",
            params={"memory_id": memory_id},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        return Memory(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            content=row["content"],
            created_at=row["created_at"],
            session_id=row.get("session_id"),
        )
