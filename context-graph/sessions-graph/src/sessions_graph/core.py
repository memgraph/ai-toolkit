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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memgraph_toolbox.api.memgraph import Memgraph

from .models import Memory, validate_content, validate_memory_id, validate_user_id
from .reconciliation import (
    NODE_LABELS,
    ReconciliationSource,
    ReconciliationSummary,
    build_reconciliation_sources,
    content_hash,
)

if TYPE_CHECKING:
    from actions_graph import ActionsGraph

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
        self._db.query("CREATE INDEX ON :Session(reconciliation_status);")
        # Shared with unstructured2graph's Chunk.hash convention; ensured here
        # too so reconcile_session() works even without a prior unstructured2graph call.
        self._db.query("CREATE CONSTRAINT ON (c:Chunk) ASSERT c.hash IS UNIQUE;")

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

    def get_memories_for_session(self, session_id: str) -> list[Memory]:
        """Return all Memories produced by *session_id*, newest first.

        Unlike :meth:`get_memories` (user-scoped), this follows
        ``PRODUCED_MEMORY`` provenance rather than ``HAS_MEMORY`` ownership —
        used by :meth:`reconcile_session` to gather a session's Memory content.
        """
        rows = self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})-[:PRODUCED_MEMORY]->(m:Memory)
            RETURN m.memory_id  AS memory_id,
                   m.user_id    AS user_id,
                   m.content    AS content,
                   m.created_at AS created_at,
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
            WITH m
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
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile_session(
        self,
        session_id: str,
        *,
        lightrag_wrapper: Any,
        actions_graph: ActionsGraph | None = None,
        entity_workspace: str | None = None,
    ) -> ReconciliationSummary:
        """Batch-extract entities from a session's Action + Memory content.

        Pulls all reconcilable Message/ToolCall/ToolResult text recorded for
        *session_id* in Actions Graph, plus this session's Memories, dedupes
        by content hash, and runs the result through unstructured2graph's
        chunk + LightRAG entity-extraction pipeline. Resulting Chunk nodes are
        linked back to their source Action/Memory node via ``HAS_CHUNK`` so
        entities trace back to the session that produced them.

        This is deliberately not wired to run automatically inside the
        ``SESSION_END`` hook — LightRAG extraction is LLM-backed and slow, and
        hook subprocesses run under a runtime timeout. Call this from a
        separate process (e.g. the ``sessions-graph reconcile`` CLI) instead.

        Requires the ``sessions-graph[reconciliation]`` extra (actions-graph +
        unstructured2graph).

        Args:
            session_id: Session to reconcile.
            lightrag_wrapper: An initialised ``MemgraphLightRAGWrapper``.
            actions_graph: An ``ActionsGraph`` instance sharing this graph's
                Memgraph connection. Constructed automatically if omitted.
            entity_workspace: Passed through to ``unstructured2graph.from_texts``.
                Defaults to whatever the LightRAG wrapper resolves to, so
                session-derived entities land in the same workspace as
                document-ingested ones and can merge.

        Returns:
            An :class:`ReconciliationSummary` describing what happened. Never
            raises for per-session failures — the failure is recorded on the
            Session node and returned so a sweep over many sessions can
            continue past one bad session.
        """
        if actions_graph is None:
            try:
                from actions_graph import ActionsGraph as _ActionsGraph
            except ImportError as exc:
                msg = "actions-graph is required for reconcile_session; install sessions-graph[reconciliation]"
                raise ImportError(msg) from exc
            actions_graph = _ActionsGraph(self._db)

        try:
            from unstructured2graph import from_texts
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_session; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        actions = actions_graph.get_session_actions(session_id)
        memories = self.get_memories_for_session(session_id)
        sources = build_reconciliation_sources(actions, memories)

        unique_texts: dict[str, str] = {}
        for source in sources:
            unique_texts.setdefault(content_hash(source.text), source.text)

        try:
            if unique_texts:
                grouped_chunks = await from_texts(
                    list(unique_texts.values()),
                    memgraph=self._db,
                    lightrag_wrapper=lightrag_wrapper,
                    entity_workspace=entity_workspace,
                )
                chunks_by_text_hash = dict(zip(unique_texts.keys(), grouped_chunks, strict=True))
                self._link_chunks_to_sources(sources, chunks_by_text_hash)

            self._db.query(
                """
                MATCH (s:Session {session_id: $session_id})
                SET s.reconciliation_status = 'completed', s.reconciled_at = $reconciled_at
                """,
                params={"session_id": session_id, "reconciled_at": datetime.now(timezone.utc).isoformat()},
            )
            return ReconciliationSummary(
                session_id=session_id,
                status="completed",
                texts_considered=len(sources),
                texts_deduped=len(unique_texts),
            )
        except Exception as e:
            self._db.query(
                """
                MATCH (s:Session {session_id: $session_id})
                SET s.reconciliation_status = 'failed', s.reconciliation_error = $error
                """,
                params={"session_id": session_id, "error": str(e)},
            )
            return ReconciliationSummary(
                session_id=session_id,
                status="failed",
                texts_considered=len(sources),
                texts_deduped=len(unique_texts),
                error=str(e),
            )

    def get_pending_reconciliation_sessions(self, *, limit: int = 100) -> list[str]:
        """Return session_ids marked ``reconciliation_status = 'pending'``."""
        rows = self._db.query(
            """
            MATCH (s:Session {reconciliation_status: 'pending'})
            RETURN s.session_id AS session_id
            ORDER BY s.session_id
            LIMIT $limit
            """,
            params={"limit": limit},
        )
        return [row["session_id"] for row in rows]

    def _link_chunks_to_sources(
        self,
        sources: list[ReconciliationSource],
        chunks_by_text_hash: dict[str, list[Any]],
    ) -> None:
        """Wire (:Action|:Memory)-[:HAS_CHUNK]->(:Chunk) for each source.

        Looks up each source's actual output Chunks via from_texts()'s grouped
        return value (keyed by the source text's hash) rather than
        recomputing a hash from the original text — a text long enough to be
        split by parse_text() produces multiple Chunks with hashes that don't
        match a hash of the whole original text, so the grouping is load-bearing.
        """
        rows_by_kind: dict[str, list[dict[str, str]]] = {kind: [] for kind in NODE_LABELS}
        for source in sources:
            for chunk in chunks_by_text_hash.get(content_hash(source.text), []):
                rows_by_kind[source.kind].append({"node_id": source.node_id, "hash": chunk.hash})

        for kind, rows in rows_by_kind.items():
            if not rows:
                continue
            label, id_prop = NODE_LABELS[kind]
            self._db.query(
                f"""
                UNWIND $rows AS row
                MATCH (n:{label} {{{id_prop}: row.node_id}})
                MERGE (c:Chunk {{hash: row.hash}})
                MERGE (n)-[:HAS_CHUNK]->(c)
                """,
                params={"rows": rows},
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
