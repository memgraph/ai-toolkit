"""Core SessionsGraph class for storing and recalling agent memories in Memgraph.

Graph schema
------------
Nodes:
    (:User  {user_id})
    (:Memory {memory_id, user_id, path, content, created_at, updated_at})
    (:Project {key})                      — a folder under /memories/projects/
    (:MemoryVersion {memory_id, path, content, replaced_at, session_id, deleted})
    (:Session {session_id})

Relationships:
    (:User)-[:HAS_MEMORY]->(:Memory)
    (:Session)-[:PRODUCED_MEMORY]->(:Memory)   — every session that wrote it, when known
    (:Memory)-[:ABOUT]->(:Project)             — memories under /memories/projects/<key>/
    (:User)-[:HAS_MEMORY_VERSION]->(:MemoryVersion)  — text a memory-tool write replaced or deleted
    (:Memory)-[:PREVIOUS_VERSION]->(:MemoryVersion)  — the same, while the file still exists
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memgraph_context_graph import session as session_node
from memgraph_toolbox.api.memgraph import Memgraph, merge_with_retry

from .embeddings import DEFAULT_EMBEDDING_MODEL
from .memory_store import MemoryStore
from .models import Memory, _utc_now, project_key, validate_content, validate_memory_id, validate_user_id
from .recall import MEMORY_TEXT_INDEX, TURN_TEXT_INDEX, RecallConfig, Recalled, recall
from .reconciler import Reconciler

if TYPE_CHECKING:
    from pathlib import Path

    from .extraction import ExtractionConfig
    from .ontology import Derive, OntologyVersion

_FULLTEXT_INDEX = MEMORY_TEXT_INDEX


class SessionsGraph:
    """A user's Memories and memory files, their sessions and recall, over one Memgraph connection.

    Turning sessions and memory files into extracted entities and vectors is
    :attr:`reconciler`'s job, over the same connection.
    """

    def __init__(
        self, memgraph: Memgraph | None = None, *, extraction: ExtractionConfig | None = None, **kwargs: Any
    ) -> None:
        """Initialise SessionsGraph.

        Args:
            memgraph: An existing Memgraph client instance.  When *None* a new
                      one is created from *kwargs* / environment variables.
            extraction: The GLiNER2 settings every extraction over this graph
                uses (reconciliation, re-extraction, derivation); defaults when None.
            **kwargs: Forwarded to :class:`Memgraph` when *memgraph* is ``None``.
        """
        self._db = memgraph or Memgraph(**kwargs)
        self.reconciler = Reconciler(self._db, extraction=extraction)

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create constraints, indexes, and the full-text index."""
        self._db.query("CREATE CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE;")
        self._db.query("CREATE CONSTRAINT ON (m:Memory) ASSERT m.memory_id IS UNIQUE;")
        # A path names one file per user; the memory tool relies on this to make
        # concurrent creates of the same path converge on one node.
        self._db.query("CREATE CONSTRAINT ON (m:Memory) ASSERT m.user_id, m.path IS UNIQUE;")
        self._db.query("CREATE CONSTRAINT ON (p:Project) ASSERT p.key IS UNIQUE;")
        self._db.query("CREATE CONSTRAINT ON (v:OntologyVersion) ASSERT v.user_id, v.version IS UNIQUE;")
        self._db.query("CREATE INDEX ON :Memory(user_id);")
        self._db.query("CREATE INDEX ON :Memory(created_at);")
        self._db.query("CREATE INDEX ON :Memory(path);")
        self._db.query("CREATE INDEX ON :MemoryVersion(memory_id);")
        self._db.query("CREATE INDEX ON :Memory(extraction_status);")
        self._db.query(f"CREATE TEXT INDEX {_FULLTEXT_INDEX} ON :Memory(content);")
        session_node.setup(self._db)
        self._db.query("CREATE INDEX ON :Session(reconciliation_status);")
        self._db.query("CREATE INDEX ON :Session(embedding_status);")
        self._db.query(f"CREATE TEXT INDEX {TURN_TEXT_INDEX} ON :Action(text);")
        # Shared with unstructured2graph's Chunk.hash convention; ensured here
        # too so reconciliation works even without a prior unstructured2graph call.
        self._db.query("CREATE CONSTRAINT ON (c:Chunk) ASSERT c.hash IS UNIQUE;")

    # ------------------------------------------------------------------
    # Forgetting
    # ------------------------------------------------------------------

    def forget_session(self, session_id: str) -> dict[str, int]:
        """Delete what reconciliation derived from *session_id*, and the session's own User if it had a made-up one.

        Run while the session's Actions still exist, before actions-graph's
        forget: the Actions are how its Chunks are found. Memory files the
        session wrote stay, being the user's; the versions it wrote no longer
        name it. The Session Node itself is left for
        ``memgraph_context_graph.session.delete``.

        Returns:
            Counts of what went: ``relations``, ``mentions``, ``entities``, ``chunks``,
            ``memory_versions`` (unlinked from the session) and ``users``.
        """
        from .forgetting import chunks_of, forget_sources

        try:
            action_ids = self.reconciler._actions().owned_action_ids(session_id)
        except ImportError:
            action_ids = []  # Without actions-graph there are no Actions, so nothing was extracted from any.
        removed = forget_sources(self._db, action_ids, chunks_of(self._db, action_ids))
        versions = self._db.query(
            "MATCH (v:MemoryVersion {session_id: $session_id}) SET v.session_id = null RETURN count(*) AS n",
            params={"session_id": session_id},
        )
        # Reconciliation makes up ``anon-<session_id>`` for a session that reported no user, so it is this one's alone.
        users = self._db.query(
            """
            MATCH (u:User {user_id: $user_id})
            WHERE NOT EXISTS { MATCH (u)-[:HAD_SESSION]->(s:Session) WHERE s.session_id <> $session_id }
            DETACH DELETE u
            RETURN count(*) AS n
            """,
            params={"user_id": f"anon-{session_id}", "session_id": session_id},
        )
        return {
            **removed,
            "memory_versions": versions[0]["n"] if versions else 0,
            "users": users[0]["n"] if users else 0,
        }

    # ------------------------------------------------------------------
    # Ontology versions
    # ------------------------------------------------------------------

    def adopted_ontology(self, user_id: str) -> OntologyVersion:
        """The ontology version *user_id*'s sessions are extracted under; see ``sessions_graph.ontology``."""
        from .ontology import adopted

        return adopted(self._db, user_id)

    def supply_ontology_file(self, user_id: str, path: str | Path, *, derive: Derive = "extend") -> OntologyVersion:
        """Adopt the schema at *path* as *user_id*'s next version; see ``ontology.supply``.

        Raises:
            ValueError: if the file can't be read or parsed, *derive* is not
                "extend"/"off", or the model fails validation.
        """
        from .ontology import supply_file

        return supply_file(self._db, user_id, path, derive=derive)

    def sync_ontology_file(
        self, user_id: str, path: str | Path, *, derive: Derive = "extend"
    ) -> OntologyVersion | None:
        """Adopt *path* as a new version only if it changed since the adopted one came from it.

        Returns:
            The new version, or None when the file is unchanged.

        Raises:
            ValueError: as for :meth:`supply_ontology_file`.
        """
        from .ontology import sync_file

        return sync_file(self._db, user_id, path, derive=derive)

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
        path: str | None = None,
    ) -> Memory:
        """Persist a new Memory for *user_id*.

        Args:
            user_id:    The owning user identity.
            content:    The free-form text assertion to store.
            session_id: Optional session that produced this memory (for provenance).
            memory_id:  Override the auto-generated UUID (useful in tests).
            path:       Where the memory lives under ``/memories``; defaults to
                        ``/memories/notes/<memory_id>.md``.

        Returns:
            The persisted :class:`Memory` instance.

        Raises:
            MemoryValidationError: an invalid user, content or path.
            Exception: the database's constraint error when *path* is already taken.
        """
        memory = Memory(
            user_id=validate_user_id(user_id),
            content=validate_content(content),
            session_id=session_id,
            path=path or "",
            **({"memory_id": memory_id} if memory_id else {}),
        )

        merge_with_retry(
            self._db,
            """
            MERGE (u:User {user_id: $user_id})
            CREATE (m:Memory {
                memory_id: $memory_id,
                user_id:   $user_id,
                path:      $path,
                content:   $content,
                created_at: $created_at,
                updated_at: $created_at
            })
            CREATE (u)-[:HAS_MEMORY]->(m)
            """,
            params={
                "user_id": memory.user_id,
                "memory_id": memory.memory_id,
                "path": memory.path,
                "content": memory.content,
                "created_at": memory.created_at,
            },
        )
        if key := project_key(memory.path):
            self._db.query(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                MERGE (p:Project {key: $key})
                MERGE (m)-[:ABOUT]->(p)
                """,
                params={"memory_id": memory.memory_id, "key": key},
            )

        if session_id:
            session_node.ensure(self._db, session_id)
            self._db.query(
                """
                MATCH (s:Session {session_id: $session_id})
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
            WITH m, min(s.session_id) AS producing_session
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.path       AS path,
                   m.created_at AS created_at,
                   m.updated_at AS updated_at,
                   producing_session AS session_id
            ORDER BY m.created_at DESC
            """,
            params={"user_id": user_id},
        )
        return [self._row_to_memory(r) for r in rows]

    def get_memories_for_session(self, session_id: str) -> list[Memory]:
        """Return all Memories produced by *session_id*, newest first.

        Unlike :meth:`get_memories` (user-scoped), this follows
        ``PRODUCED_MEMORY`` provenance rather than ``HAS_MEMORY`` ownership.
        """
        rows = self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})-[:PRODUCED_MEMORY]->(m:Memory)
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.path       AS path,
                   m.created_at AS created_at,
                   m.updated_at AS updated_at,
                   $session_id  AS session_id
            ORDER BY m.created_at DESC
            """,
            params={"session_id": session_id},
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
            WITH m, score
            WHERE m.user_id = $user_id
            OPTIONAL MATCH (s:Session)-[:PRODUCED_MEMORY]->(m)
            WITH m, score, min(s.session_id) AS producing_session
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.path       AS path,
                   m.created_at AS created_at,
                   m.updated_at AS updated_at,
                   producing_session AS session_id
            ORDER BY score DESC
            LIMIT {int(limit)}
            """,
            params={"user_id": user_id, "query": query.strip()},
        )
        return [self._row_to_memory(r) for r in rows]

    def record_memory_provenance(self, session_id: str, paths: list[str]) -> None:
        """Record that *session_id* wrote the memory files at *paths*, owned by that session's user.

        For writes made where the session isn't known (the MCP server), so the
        session's hook records it afterwards.
        """
        self._db.query(
            """
            MATCH (u:User)-[:HAD_SESSION]->(s:Session {session_id: $session_id})
            MATCH (u)-[:HAS_MEMORY]->(m:Memory)
            WHERE m.path IN $paths
            MERGE (s)-[:PRODUCED_MEMORY]->(m)
            """,
            params={"session_id": session_id, "paths": paths},
        )

    def memory_store(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        author: str | None = None,
        forget_derived: bool = True,
    ) -> MemoryStore:
        """The memory tool's commands over *user_id*'s memory files; see :mod:`sessions_graph.memory_store`.

        Raises:
            MemoryValidationError: *user_id* is invalid.
        """
        return MemoryStore(self._db, user_id, session_id=session_id, author=author, forget_derived=forget_derived)

    def record_session_start(self, session_id: str, *, user_id: str | None, project_key: str | None) -> None:
        """MERGE *session_id*, owned by *user_id* when the harness reported one, in *project_key* when known."""
        session_node.ensure(self._db, session_id)
        self._db.query(
            "MATCH (s:Session {session_id: $session_id}) SET s.project_key = coalesce($project, s.project_key);",
            params={"session_id": session_id, "project": project_key},
        )
        if user_id:
            # A session's first hooks run at once, each MERGEing the same new User.
            merge_with_retry(
                self._db,
                """
                MERGE (u:User {user_id: $user_id})
                WITH u
                MATCH (s:Session {session_id: $session_id})
                MERGE (u)-[:HAD_SESSION]->(s)
                """,
                params={"user_id": user_id, "session_id": session_id},
            )

    def recall(
        self,
        user_id: str,
        question: str,
        *,
        config: RecallConfig | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> Recalled:
        """What *user_id*'s own sessions hold about *question*; see :mod:`sessions_graph.recall`.

        Needs :meth:`setup` (the message text index) and vectors from
        :meth:`Reconciler.embed` made with *model*; without MAGE the result
        comes from text search alone and says so.
        """
        return recall(self._db, validate_user_id(user_id), question, config=config, model=model)

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
            SET m.content = $content, m.updated_at = $updated_at
            WITH m
            OPTIONAL MATCH (s:Session)-[:PRODUCED_MEMORY]->(m)
            WITH m, min(s.session_id) AS producing_session
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.path       AS path,
                   m.created_at AS created_at,
                   m.updated_at AS updated_at,
                   producing_session AS session_id
            """,
            params={"memory_id": memory_id, "content": content, "updated_at": _utc_now()},
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
            path=row.get("path") or "",
            updated_at=row.get("updated_at") or "",
        )
