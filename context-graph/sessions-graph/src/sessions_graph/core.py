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

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from memgraph_toolbox.api.memgraph import Memgraph

from .embeddings import DEFAULT_EMBEDDING_MODEL, Embedded, EmbeddingUnavailableError, embed_memory, embed_session
from .extraction import ExtractionConfig
from .memory_store import MemoryStore
from .models import Memory, _utc_now, project_key, validate_content, validate_memory_id, validate_user_id
from .passages import PASSAGE_SCHEME
from .recall import MEMORY_TEXT_INDEX, TURN_TEXT_INDEX, RecallConfig, Recalled, recall
from .reconciliation import (
    MAX_SESSION_BATCH_CHARS,
    NODE_LABELS,
    MemoryReconciliation,
    ReconciliationSource,
    ReconciliationSummary,
    build_reconciliation_sources,
    content_hash,
    normalize_timestamp,
)

if TYPE_CHECKING:
    from pathlib import Path

    from actions_graph import ActionsGraph
    from hygm import HygmModel
    from unstructured2graph import Document, ExtractionBackend, Ontology
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    from .ontology import Derive, OntologyVersion

_FULLTEXT_INDEX = MEMORY_TEXT_INDEX


@dataclass(frozen=True)
class _PreparedSession:
    """One session's reconcilable content, gathered but not yet sent anywhere."""

    session_id: str
    sources: list[ReconciliationSource]
    # Content hash -> the first source with that text.
    unique_sources: dict[str, ReconciliationSource] = field(default_factory=dict)

    @property
    def unique_texts(self) -> list[str]:
        """The session's texts, deduped by content hash, in first-seen order."""
        return [source.text for source in self.unique_sources.values()]

    @property
    def combined_text(self) -> str:
        """The session's deduped texts joined into the one document LightRAG sees."""
        return "\n\n".join(self.unique_texts)

    def document(self, user_id: str) -> Document:
        """combined_text as an unstructured2graph Document, one segment per deduped source.

        Each segment carries its source's speaker, timestamp and node id; a text
        repeated across sources is one segment, attributed to its first source.
        """
        from unstructured2graph import Document, Segment

        segments, cursor = [], 0
        for source in self.unique_sources.values():
            end = cursor + len(source.text)
            segments.append(Segment(cursor, end, source.role, source.valid_at, source.node_id))
            cursor = end + 2
        return Document(text=self.combined_text, segments=tuple(segments), user_id=user_id)


class SessionsGraph:
    """A user's Memories and memory files, session reconciliation, embeddings and recall, over one Memgraph connection."""

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
        self.extraction = extraction or ExtractionConfig()
        # One GLiNER2 backend per distinct model, all sharing the first one's loaded weights.
        self._extraction_backends: dict[str, GLiNER2Backend] = {}

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
        self._db.query("CREATE INDEX ON :Session(reconciliation_status);")
        self._db.query("CREATE INDEX ON :Session(embedding_status);")
        self._db.query(f"CREATE TEXT INDEX {TURN_TEXT_INDEX} ON :Action(text);")
        # Shared with unstructured2graph's Chunk.hash convention; ensured here
        # too so reconcile_session() works even without a prior unstructured2graph call.
        self._db.query("CREATE CONSTRAINT ON (c:Chunk) ASSERT c.hash IS UNIQUE;")

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

        self._db.query(
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
        ``PRODUCED_MEMORY`` provenance rather than ``HAS_MEMORY`` ownership —
        used by :meth:`reconcile_session` to gather a session's Memory content.
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
    # Reconciliation
    # ------------------------------------------------------------------

    def _default_actions_graph(self, actions_graph: ActionsGraph | None, caller: str) -> ActionsGraph:
        """*actions_graph*, or a new one sharing this graph's connection; *caller* names the entry point in the ImportError."""
        if actions_graph is not None:
            return actions_graph
        try:
            from actions_graph import ActionsGraph as _ActionsGraph
        except ImportError as exc:
            msg = f"actions-graph is required for {caller}; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc
        return _ActionsGraph(self._db)

    def _prepare_session(self, session_id: str, actions_graph: ActionsGraph) -> _PreparedSession:
        """Gather one session's reconcilable Action/Memory text, deduped by content hash."""
        actions = actions_graph.get_session_actions(session_id)
        # Memories aren't session content: each file is reconciled on its own
        # when it changes (reconcile_memory), since it outlives the session that wrote it.
        sources = build_reconciliation_sources(actions, [])
        unique_sources: dict[str, ReconciliationSource] = {}
        for source in sources:
            unique_sources.setdefault(content_hash(source.text), source)
        return _PreparedSession(session_id=session_id, sources=sources, unique_sources=unique_sources)

    def _write_completed(
        self,
        session_id: str,
        *,
        extraction_backend: str | None = None,
        ontology_version: int | None = None,
    ) -> str:
        """Mark *session_id* completed, stamped now, at its own completion rather than its batch's start.

        Returns that timestamp. ``extraction_backend`` is the class name of the
        backend that extracted the session's entities -- what
        ``context_graph_eval.runner._require_reconciled`` checks before trusting
        which backend built a reused graph. ``None`` means nothing was extracted
        (no reconcilable content), not an unknown backend. ``ontology_version``
        is the user's model version the session was extracted under, or
        ``None`` when the caller supplied its own backend.
        """
        reconciled_at = _utc_now()
        self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            SET s.reconciliation_status = 'completed', s.reconciled_at = $reconciled_at,
                s.extraction_backend = $extraction_backend, s.ontology_version = $ontology_version
            """,
            params={
                "session_id": session_id,
                "reconciled_at": reconciled_at,
                "extraction_backend": extraction_backend,
                "ontology_version": ontology_version,
            },
        )
        return reconciled_at

    def _write_reextracted(
        self, session_id: str, *, extraction_backend: str | None, ontology_version: int | None
    ) -> None:
        """Record a re-extraction: what extracted the session now, leaving when it was reconciled alone."""
        self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            SET s.reextracted_at = $now, s.extraction_backend = $extraction_backend,
                s.ontology_version = $ontology_version
            """,
            params={
                "session_id": session_id,
                "now": _utc_now(),
                "extraction_backend": extraction_backend,
                "ontology_version": ontology_version,
            },
        )

    def _write_failed(self, session_id: str, error: str) -> None:
        """Mark *session_id* failed, recording *error* for later inspection."""
        self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            SET s.reconciliation_status = 'failed', s.reconciliation_error = $error
            """,
            params={"session_id": session_id, "error": error},
        )

    async def reconcile_session(
        self,
        session_id: str,
        *,
        extraction_backend: ExtractionBackend | None = None,
        actions_graph: ActionsGraph | None = None,
        promote_labels: bool = False,
        enforce_ontology: bool = False,
        ontology_path: str | Path | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reextract: bool = False,
    ) -> ReconciliationSummary:
        """Extract entities from one session's content.

        The session's reconcilable Message/ToolCall/ToolResult text (memory
        files are reconciled per file by :meth:`reconcile_memory`), deduped by
        content hash, goes to ``unstructured2graph.from_documents`` as one
        document; the resulting Chunks are linked back to their sources via
        ``HAS_CHUNK``. No LLM unless *extraction_backend* brings one: recall
        answers from the turns and uses the extracted graph as an index into
        them. Slow, so it runs out of band (the ``sessions-graph reconcile``
        CLI), never inside a hook.

        Args:
            session_id: Session to reconcile.
            extraction_backend: Overrides entity extraction; the user's ontology
                version is then not consulted. Defaults to a ``GLiNER2Backend``
                over the user's adopted version (see ``sessions_graph.ontology``),
                recorded on the Session as ``ontology_version``; one per distinct
                model, all sharing one loaded GLiNER2 checkpoint.
            actions_graph: Constructed over this graph's connection if omitted.
            promote_labels: Passed through to ``from_documents``; ignored when
                ``enforce_ontology`` is set.
            enforce_ontology: Passed through to ``from_documents``: promotes only
                ``ontology_path``'s vocabulary, else the backend's, flagging the
                rest ``ontology_conformant = false``.
            ontology_path: Only consulted when ``enforce_ontology`` is set.
            embedding_model: What the session is embedded with afterwards (see
                :meth:`embed_session`); an embedding failure never fails this.
            reextract: Re-read an already reconciled session under the current
                model (derivation, #434): ``reconciled_at`` kept and
                ``reextracted_at`` set; a failure records ``reextraction_error``
                and leaves the session reconciled as it was.

        Returns:
            A :class:`ReconciliationSummary`. A per-session failure is recorded
            on the Session and returned, never raised, so a sweep continues.

        Raises:
            ImportError: the ``sessions-graph[reconciliation]`` extra is not installed.
        """
        actions_graph = self._default_actions_graph(actions_graph, "reconcile_session")

        try:
            from unstructured2graph import from_documents
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_session; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        prepared = self._prepare_session(session_id, actions_graph)

        try:
            used_backend: str | None = None
            integrity: tuple[int, int] | None = None
            ontology_version: int | None = None
            if prepared.unique_texts:
                # The whole session as ONE document, so extraction sees entities
                # and relations that span turns (coreference, a fact stated in
                # one turn and referenced in another). Handed over verbatim with
                # a segment per source rather than through `unstructured`, whose
                # partitioner rewrites text and would invalidate the turn
                # offsets the GLiNER2 backend windows on and resolves the
                # user's own mentions with.
                user_id = self._session_user(session_id)
                if extraction_backend is None:
                    from .ontology import adopted

                    version = adopted(self._db, user_id)
                    backend = self._extraction_backend_for(version.model)
                    ontology_version = version.version
                else:
                    backend = extraction_backend
                # A backend that extracts against a vocabulary (GLiNER2) is
                # enforced against that same vocabulary unless a file overrides it.
                ontology = None if ontology_path else getattr(backend, "ontology", None)
                grouped_chunks = await from_documents(
                    [prepared.document(user_id)],
                    memgraph=self._db,
                    extraction_backend=backend,
                    promote_labels=promote_labels,
                    enforce_ontology=enforce_ontology,
                    ontology_path=ontology_path,
                    ontology=ontology,
                )
                used_backend = type(backend).__name__
                session_chunks = grouped_chunks[0] if grouped_chunks else []
                self._link_chunks_to_sources(prepared.sources, session_chunks)
                if enforce_ontology and session_chunks:
                    integrity = self._integrity(backend.workspace_label, ontology_path, ontology, session_chunks)

            if not reextract:
                self._write_completed(session_id, extraction_backend=used_backend, ontology_version=ontology_version)
            else:
                self._write_reextracted(session_id, extraction_backend=used_backend, ontology_version=ontology_version)
            self._embed_after_reconcile(session_id, embedding_model)
            return ReconciliationSummary(
                session_id=session_id,
                status="completed",
                texts_considered=len(prepared.sources),
                texts_deduped=len(prepared.unique_texts),
                nonconformant_entities=integrity[0] if integrity else None,
                nonconformant_relations=integrity[1] if integrity else None,
            )
        except Exception as e:
            if reextract:
                # The session stays reconciled under what it was extracted with before.
                self._db.query(
                    "MATCH (s:Session {session_id: $session_id}) SET s.reextraction_error = $error",
                    params={"session_id": session_id, "error": str(e)},
                )
            else:
                self._write_failed(session_id, str(e))
            return ReconciliationSummary(
                session_id=session_id,
                status="failed",
                texts_considered=len(prepared.sources),
                texts_deduped=len(prepared.unique_texts),
                error=str(e),
            )

    async def reconcile_sessions_batch(
        self,
        session_ids: list[str],
        *,
        lightrag_wrapper: Any,
        actions_graph: ActionsGraph | None = None,
        promote_labels: bool = False,
        enforce_ontology: bool = False,
        ontology_path: str | Path | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[ReconciliationSummary]:
        """Reconcile many sessions with LightRAG as ONE processing pass, not one per session.

        LightRAG's pipeline holds a workspace-level "busy" lock: a concurrent
        second pass just flags itself pending and returns, so neither a loop
        nor ``asyncio.gather`` over :meth:`reconcile_session` gives its worker
        pool more than one document. This enqueues every session's document
        (``enqueue_texts``), processes them once
        (``process_enqueued_and_finalize``), then links chunks, embeds and
        records status per session.

        If the shared pass raises, every session is marked failed with that
        error. A per-document failure it reports without raising fails only
        that session, with the document's own ``error_msg``.

        Args:
            session_ids: Sessions to reconcile together.
            lightrag_wrapper: An initialised ``MemgraphLightRAGWrapper``, shared
                across the batch.
            actions_graph, promote_labels, enforce_ontology, ontology_path,
                embedding_model: As for :meth:`reconcile_session`.

        Returns:
            One :class:`ReconciliationSummary` per session, in input order.

        Raises:
            ImportError: the ``sessions-graph[reconciliation]`` extra is not installed.
        """
        actions_graph = self._default_actions_graph(actions_graph, "reconcile_sessions_batch")

        try:
            from unstructured2graph import enqueue_texts, process_enqueued_and_finalize
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_sessions_batch; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        prepared = [self._prepare_session(session_id, actions_graph) for session_id in session_ids]

        results: dict[str, ReconciliationSummary] = {}

        # Sessions with nothing to reconcile complete now and stay out of the shared enqueue.
        to_enqueue = [p for p in prepared if p.unique_texts]
        for p in prepared:
            if p.unique_texts:
                continue
            self._write_completed(p.session_id, extraction_backend=None)
            results[p.session_id] = ReconciliationSummary(
                session_id=p.session_id, status="completed", texts_considered=len(p.sources), texts_deduped=0
            )

        if not to_enqueue:
            return [results[sid] for sid in session_ids]

        try:
            grouped_chunks = await enqueue_texts(
                [p.combined_text for p in to_enqueue],
                memgraph=self._db,
                lightrag_wrapper=lightrag_wrapper,
                chunk_kwargs={"max_characters": MAX_SESSION_BATCH_CHARS},
            )
            all_chunks = [chunk for group in grouped_chunks for chunk in group]
            doc_statuses = await process_enqueued_and_finalize(
                self._db,
                lightrag_wrapper,
                all_chunks,
                promote_labels=promote_labels,
                enforce_ontology=enforce_ontology,
                ontology_path=ontology_path,
            )
        except Exception as e:
            # No per-document result to attribute this to: every session shares the one error.
            error = str(e)
            for p in to_enqueue:
                self._write_failed(p.session_id, error)
                results[p.session_id] = ReconciliationSummary(
                    session_id=p.session_id,
                    status="failed",
                    texts_considered=len(p.sources),
                    texts_deduped=len(p.unique_texts),
                    error=error,
                )
            return [results[sid] for sid in session_ids]

        def _finalize_one(prepared_session: _PreparedSession, session_chunks: list[Any]) -> None:
            # Returning is not proof this session's document succeeded: LightRAG
            # swallows per-document failures, so only the reported status decides.
            failed_chunk = next(
                (c for c in session_chunks if doc_statuses.get(c.hash, {}).get("status") != "processed"), None
            )
            if failed_chunk is not None:
                error = doc_statuses.get(failed_chunk.hash, {}).get(
                    "error_msg", f"chunk {failed_chunk.hash} did not reach 'processed'"
                )
                self._write_failed(prepared_session.session_id, error)
                results[prepared_session.session_id] = ReconciliationSummary(
                    session_id=prepared_session.session_id,
                    status="failed",
                    texts_considered=len(prepared_session.sources),
                    texts_deduped=len(prepared_session.unique_texts),
                    error=error,
                )
                return

            try:
                self._link_chunks_to_sources(prepared_session.sources, session_chunks)
                # A literal: this path only ever drives LightRAG's pipeline, with no backend instance.
                self._write_completed(prepared_session.session_id, extraction_backend="LightRAGBackend")
                self._embed_after_reconcile(prepared_session.session_id, embedding_model)
                results[prepared_session.session_id] = ReconciliationSummary(
                    session_id=prepared_session.session_id,
                    status="completed",
                    texts_considered=len(prepared_session.sources),
                    texts_deduped=len(prepared_session.unique_texts),
                )
            except Exception as e:
                # Per session: nothing here touches another session's state.
                self._write_failed(prepared_session.session_id, str(e))
                results[prepared_session.session_id] = ReconciliationSummary(
                    session_id=prepared_session.session_id,
                    status="failed",
                    texts_considered=len(prepared_session.sources),
                    texts_deduped=len(prepared_session.unique_texts),
                    error=str(e),
                )

        for i, p in enumerate(to_enqueue):
            _finalize_one(p, grouped_chunks[i] if i < len(grouped_chunks) else [])

        return [results[sid] for sid in session_ids]

    def embed_session(self, session_id: str, *, model: str = DEFAULT_EMBEDDING_MODEL) -> Embedded:
        """Embed *session_id*'s messages, entities and edges that have no vector from *model*.

        Records the outcome on the Session: ``embedding_status`` is
        ``'completed'`` with ``embedding_model``, or ``'failed'`` with
        ``embedding_error``, which :meth:`get_pending_embedding_sessions`
        picks up again.

        Raises:
            EmbeddingUnavailableError: Memgraph can't embed (no MAGE, or the model
                can't load). Recorded on the Session before it propagates.
        """
        try:
            embedded = embed_session(self._db, session_id, model)
        except EmbeddingUnavailableError as exc:
            self._db.query(
                "MATCH (s:Session {session_id: $session_id}) "
                "SET s.embedding_status = 'failed', s.embedding_error = $error",
                params={"session_id": session_id, "error": str(exc)},
            )
            raise
        self._db.query(
            "MATCH (s:Session {session_id: $session_id}) "
            "SET s.embedding_status = 'completed', s.embedding_model = $model, s.embedding_scheme = $scheme, "
            "s.embedding_error = null",
            params={"session_id": session_id, "model": model, "scheme": PASSAGE_SCHEME},
        )
        return embedded

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
        :meth:`embed_session` made with *model*; without MAGE the result
        comes from text search alone and says so.
        """
        return recall(self._db, validate_user_id(user_id), question, config=config, model=model)

    def get_pending_embedding_sessions(self, *, model: str = DEFAULT_EMBEDDING_MODEL, limit: int = 100) -> list[str]:
        """Session ids whose embedding failed, never ran, or ran with a model or passage split other than today's."""
        rows = self._db.query(
            """
            MATCH (s:Session)
            WHERE s.embedding_status IS NULL OR s.embedding_status <> 'completed' OR s.embedding_model <> $model
               OR coalesce(s.embedding_scheme, '') <> $scheme
            RETURN s.session_id AS session_id
            ORDER BY s.session_id
            LIMIT $limit
            """,
            params={"model": model, "limit": limit, "scheme": PASSAGE_SCHEME},
        )
        return [row["session_id"] for row in rows]

    def _embed_after_reconcile(self, session_id: str, model: str) -> None:
        """Embed what reconciliation just wrote, without failing the reconciliation.

        A failure is recorded on the Session by :meth:`embed_session`, and
        ``sessions-graph embed --pending`` retries it; the extracted graph is
        valid without vectors.
        """
        with contextlib.suppress(EmbeddingUnavailableError):
            self.embed_session(session_id, model=model)

    def get_pending_memory_reconciliations(self, *, limit: int = 100) -> list[str]:
        """Return memory_ids whose file changed since it was last reconciled (``extraction_status = 'pending'``)."""
        rows = self._db.query(
            """
            MATCH (m:Memory {extraction_status: 'pending'})
            RETURN m.memory_id AS memory_id
            ORDER BY m.updated_at
            LIMIT $limit
            """,
            params={"limit": limit},
        )
        return [row["memory_id"] for row in rows]

    async def reconcile_memory(
        self,
        memory_id: str,
        *,
        extraction_backend: ExtractionBackend | None = None,
        enforce_ontology: bool = True,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> MemoryReconciliation:
        """Extract entities and relations from one memory file, replacing what its last version gave.

        A memory file outlives the session that wrote it and changes across
        many, so it is reconciled on its own whenever it changes, not with a
        session. What the previous version produced is removed first
        (:func:`~sessions_graph.memory_store.forget_extraction`); the file's
        chunks link back to it via ``HAS_CHUNK`` and its relations carry its
        id as ``source_id``. No summary and no LLM: GLiNER2 over the owner's
        adopted ontology version, unless *extraction_backend* overrides it.
        The file's passages are then embedded with *embedding_model* for
        recall's memory lane.

        Returns:
            A :class:`MemoryReconciliation`. Never raises for a failure of
            this file, which is recorded on it as ``extraction_error``.

        Raises:
            ImportError: the ``reconciliation`` extra isn't installed.
        """
        try:
            from unstructured2graph import Document, Segment, from_documents
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_memory; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        rows = self._db.query(
            """
            MATCH (m:Memory {memory_id: $memory_id})
            RETURN m.user_id AS user_id, m.path AS path, m.content AS content, m.updated_at AS updated_at
            """,
            params={"memory_id": memory_id},
        )
        if not rows:
            return MemoryReconciliation(memory_id, None, "failed", error="no such memory")
        row = rows[0]
        try:
            from .memory_store import forget_extraction
            from .ontology import adopted

            forget_extraction(self._db, memory_id)
            if extraction_backend is None:
                backend = self._extraction_backend_for(adopted(self._db, row["user_id"]).model)
            else:
                backend = extraction_backend
            text = row["content"].strip()
            document = Document(
                text=text,
                segments=(Segment(0, len(text), None, normalize_timestamp(row["updated_at"]), memory_id),),
                user_id=row["user_id"],
            )
            grouped = await from_documents(
                [document],
                memgraph=self._db,
                extraction_backend=backend,
                enforce_ontology=enforce_ontology,
                ontology=getattr(backend, "ontology", None),
            )
            chunks = grouped[0] if grouped else []
            self._link_chunks_to_sources([ReconciliationSource(kind="memory", node_id=memory_id, text=text)], chunks)
            # Recall's memory lane searches these; without MAGE it falls back to text search.
            with contextlib.suppress(EmbeddingUnavailableError):
                embed_memory(self._db, memory_id, embedding_model)
            self._db.query(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                SET m.extraction_status = 'completed', m.extracted_at = $now
                REMOVE m.extraction_error
                """,
                params={"memory_id": memory_id, "now": _utc_now()},
            )
            return MemoryReconciliation(memory_id, row["path"], "completed", chunks=len(chunks))
        except Exception as exc:
            self._db.query(
                "MATCH (m:Memory {memory_id: $memory_id}) SET m.extraction_status = 'failed', m.extraction_error = $error",
                params={"memory_id": memory_id, "error": str(exc)},
            )
            return MemoryReconciliation(memory_id, row["path"], "failed", error=str(exc))

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

    def _session_user(self, session_id: str) -> str:
        """The user_id of *session_id*'s (:User), synthesizing ``anon-<session_id>`` if it has none.

        Processing, never collection, supplies the missing user (#347): every
        session needs a (:User) for the GLiNER2 backend to bind the user's own
        mentions onto, and collection records one only when the harness reports
        a user. ``anon-`` rather than ``anon:`` because ``validate_user_id``
        does not accept a colon.
        """
        rows = self._db.query(
            "MATCH (u:User)-[:HAD_SESSION]->(:Session {session_id: $session_id}) RETURN u.user_id AS user_id LIMIT 1",
            params={"session_id": session_id},
        )
        if rows:
            return rows[0]["user_id"]
        user_id = f"anon-{session_id}"
        self._db.query(
            """
            MERGE (u:User {user_id: $user_id})
            MERGE (s:Session {session_id: $session_id})
            MERGE (u)-[:HAD_SESSION]->(s)
            """,
            params={"user_id": user_id, "session_id": session_id},
        )
        return user_id

    def _extraction_backend_for(self, model: HygmModel) -> GLiNER2Backend:
        """A GLiNER2 backend over `model` with :attr:`extraction`'s settings, kept for reuse.

        Loading the checkpoint is the slow part, so it loads once.
        """
        import json

        from hygm import model_to_mapping
        from unstructured2graph import Ontology
        from unstructured2graph.gliner2_backend import GLiNER2Backend

        key = json.dumps(model_to_mapping(model), sort_keys=True)
        if key not in self._extraction_backends:
            loaded = next(iter(self._extraction_backends.values()), None)
            engine = loaded.engine if loaded is not None else None
            self._extraction_backends[key] = GLiNER2Backend(
                ontology=Ontology.from_model(model), model=engine, **self.extraction.backend_arguments()
            )
        return self._extraction_backends[key]

    def _integrity(
        self, workspace: str, ontology_path: str | Path | None, ontology: Ontology | None, chunks: list[Any]
    ) -> tuple[int, int]:
        """(non-conformant entities, non-conformant relationships) over *chunks*, for ReconciliationSummary."""
        from unstructured2graph import DEFAULT_ONTOLOGY, load_ontology, ontology_report

        enforced = ontology or (load_ontology(ontology_path) if ontology_path else DEFAULT_ONTOLOGY)
        report = ontology_report(self._db, workspace, enforced, chunk_hashes=[chunk.hash for chunk in chunks])
        return report.nonconformant_entities, report.nonconformant_relations

    def _link_chunks_to_sources(
        self,
        sources: list[ReconciliationSource],
        chunks: list[Any],
    ) -> None:
        """Wire (:Action|:Memory)-[:HAS_CHUNK]->(:Chunk) from every source to every chunk of its session.

        Exact while the session is one chunk (``MAX_SESSION_BATCH_CHARS`` keeps
        it so); a session split into several chunks gets session-level
        provenance: a source may link to a sibling chunk, never to another
        session's.
        """
        rows_by_kind: dict[str, list[dict[str, str]]] = {kind: [] for kind in NODE_LABELS}
        for source in sources:
            for chunk in chunks:
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
            path=row.get("path") or "",
            updated_at=row.get("updated_at") or "",
        )
