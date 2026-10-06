"""Core SessionsGraph class for storing and recalling agent memories in Memgraph.

Graph schema
------------
Nodes:
    (:User  {user_id})
    (:Memory {memory_id, user_id, content, created_at, session_id?})
    (:Session {session_id})
    (:Episode {summary, summarized_at})   — written by reconcile_session(), see below

Relationships:
    (:User)-[:HAS_MEMORY]->(:Memory)
    (:Session)-[:PRODUCED_MEMORY]->(:Memory)   — only when session_id is provided
    (:Session)-[:HAS_EPISODE]->(:Episode)      — at most one per session
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memgraph_toolbox.api.memgraph import Memgraph

from .embeddings import DEFAULT_EMBEDDING_MODEL, Embedded, EmbeddingUnavailableError, embed_session
from .models import Memory, validate_content, validate_memory_id, validate_user_id
from .recall import TURN_TEXT_INDEX, RecallConfig, Recalled, recall
from .reconciliation import (
    MAX_SESSION_BATCH_CHARS,
    NODE_LABELS,
    ReconciliationSource,
    ReconciliationSummary,
    build_reconciliation_sources,
    content_hash,
    summarize_session_texts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from actions_graph import ActionsGraph
    from hygm import HygmModel
    from unstructured2graph import Document, ExtractionBackend, Ontology

    from .ontology import Derive, OntologyVersion

_FULLTEXT_INDEX = "memory_content_index"


@dataclass(frozen=True)
class _PreparedSession:
    """One session's reconcilable content, gathered but not yet sent anywhere.

    Named rather than left as an anonymous tuple so ``reconcile_sessions_batch``'s
    several stages (enqueue, per-document status check, finalize) can pass it
    around by field name instead of by position.
    """

    session_id: str
    sources: list[ReconciliationSource]
    unique_texts: dict[str, str] = field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        """The session's deduped texts joined into the one document LightRAG sees."""
        return "\n\n".join(self.unique_texts.values())

    def document(self, user_id: str) -> Document:
        """combined_text as an unstructured2graph Document: one segment per deduped
        source, carrying its speaker, timestamp and node id, and the session's user.

        A text repeated across sources is one segment, attributed to its first source."""
        from unstructured2graph import Document, Segment

        first: dict[str, ReconciliationSource] = {}
        for source in self.sources:
            first.setdefault(content_hash(source.text), source)
        segments, cursor = [], 0
        for digest, text in self.unique_texts.items():
            source = first[digest]
            segments.append(Segment(cursor, cursor + len(text), source.role, source.valid_at, source.node_id))
            cursor += len(text) + 2
        return Document(text=self.combined_text, segments=tuple(segments), user_id=user_id)


class SessionsGraph:
    """Store and recall agent memories in Memgraph.

    Provides:
    - :meth:`save_memory`   — persist a new Memory for a user
    - :meth:`get_memories`  — retrieve all Memories for a user
    - :meth:`search_memories` — full-text search over Memory content
    - :meth:`update_memory` — replace the content of an existing Memory
    - :meth:`delete_memory` — remove a Memory by ID
    - :meth:`embed_session` — embed a session's messages, entities and edges for recall
    - :meth:`recall` — what a user's past sessions hold about a question
    """

    def __init__(self, memgraph: Memgraph | None = None, **kwargs: Any) -> None:
        """Initialise SessionsGraph.

        Args:
            memgraph: An existing Memgraph client instance.  When *None* a new
                      one is created from *kwargs* / environment variables.
            **kwargs: Forwarded to :class:`Memgraph` when *memgraph* is ``None``.
        """
        self._db = memgraph or Memgraph(**kwargs)
        # One GLiNER2 backend per distinct model, all sharing the first one's loaded weights.
        self._extraction_backends: dict[str, ExtractionBackend] = {}

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create constraints, indexes, and the full-text index."""
        self._db.query("CREATE CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE;")
        self._db.query("CREATE CONSTRAINT ON (m:Memory) ASSERT m.memory_id IS UNIQUE;")
        self._db.query("CREATE CONSTRAINT ON (v:OntologyVersion) ASSERT v.user_id, v.version IS UNIQUE;")
        self._db.query("CREATE INDEX ON :Memory(user_id);")
        self._db.query("CREATE INDEX ON :Memory(created_at);")
        self._db.query(f"CREATE TEXT INDEX {_FULLTEXT_INDEX} ON :Memory(content);")
        self._db.query("CREATE INDEX ON :Session(reconciliation_status);")
        self._db.query("CREATE INDEX ON :Session(embedding_status);")
        self._db.query(f"CREATE TEXT INDEX {TURN_TEXT_INDEX} ON :Action(text);")
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
    # Ontology versions
    # ------------------------------------------------------------------

    def adopted_ontology(self, user_id: str) -> OntologyVersion:
        """The ontology version *user_id*'s sessions are extracted under; see ``sessions_graph.ontology``."""
        from .ontology import adopted

        return adopted(self._db, user_id)

    def supply_ontology_file(self, user_id: str, path: str | Path, *, derive: str = "extend") -> OntologyVersion:
        """Adopt the schema at *path* as *user_id*'s next version; see ``ontology.supply``.

        Raises:
            ValueError: if the file can't be read or parsed, *derive* is not
                "extend"/"off", or the model fails validation.
        """
        from .ontology import supply_file

        return supply_file(self._db, user_id, path, derive=_derive_mode(derive))

    def sync_ontology_file(self, user_id: str, path: str | Path, *, derive: str = "extend") -> OntologyVersion | None:
        """Adopt *path* as a new version only if it changed since the adopted one came from it.

        Returns:
            The new version, or None when the file is unchanged.

        Raises:
            ValueError: as for :meth:`supply_ontology_file`.
        """
        from .ontology import sync_file

        return sync_file(self._db, user_id, path, derive=_derive_mode(derive))

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

    def _default_actions_graph(self, actions_graph: ActionsGraph | None, caller: str) -> ActionsGraph:
        """Construct an ``ActionsGraph`` sharing this graph's connection when
        the caller didn't supply one -- the same guard ``reconcile_session``
        and ``reconcile_sessions_batch`` both need, parameterised by
        ``caller`` only so the ImportError names the actual entry point."""
        if actions_graph is not None:
            return actions_graph
        try:
            from actions_graph import ActionsGraph as _ActionsGraph
        except ImportError as exc:
            msg = f"actions-graph is required for {caller}; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc
        return _ActionsGraph(self._db)

    def _prepare_session(self, session_id: str, actions_graph: ActionsGraph) -> _PreparedSession:
        """Gather one session's reconcilable Action/Memory text, deduped by
        content hash -- the read-only half of reconciliation shared by
        ``reconcile_session`` and ``reconcile_sessions_batch``."""
        actions = actions_graph.get_session_actions(session_id)
        memories = self.get_memories_for_session(session_id)
        sources = build_reconciliation_sources(actions, memories)
        unique_texts: dict[str, str] = {}
        for source in sources:
            unique_texts.setdefault(content_hash(source.text), source.text)
        return _PreparedSession(session_id=session_id, sources=sources, unique_texts=unique_texts)

    def _write_completed(
        self,
        session_id: str,
        *,
        summary_text: str | None,
        extraction_backend: str | None = None,
        ontology_version: int | None = None,
    ) -> str:
        """Mark *session_id* completed and, if a narrative summary was
        produced, MERGE its Episode -- both stamped with the SAME timestamp,
        computed here at actual completion time (not earlier, e.g. when a
        batch started), so ``reconciled_at``/``summarized_at`` reflect when
        this specific session actually finished. Returns that timestamp.

        ``extraction_backend`` records which backend's class
        (``type(backend).__name__``, e.g. ``"LightRAGBackend"``,
        ``"GLiNER2Backend"``) actually extracted this session's entities --
        ``None`` when nothing was extracted (no reconcilable content), which
        is a real, distinct state from "extracted by an unknown backend", not
        an omission. This is the ground truth a caller comparing runs across
        a reused graph needs to check before trusting which backend actually
        built it -- see ``context_graph_eval.runner._require_reconciled``,
        added after a real bug where ``--skip-reconcile
        --extraction-backend gliner2`` against a LightRAG-built graph
        recorded ``gliner2`` in ``RunMeta`` despite every entity in the graph
        coming from LightRAG.

        ``ontology_version`` is the user's model version the session was
        extracted under, or ``None`` when the caller supplied its own backend."""
        reconciled_at = datetime.now(timezone.utc).isoformat()
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
        if summary_text:
            # MERGE on the (Session)-[:HAS_EPISODE]->(Episode) pattern (not just CREATE)
            # so re-reconciling a session updates its one Episode instead of accumulating
            # duplicates -- Episode has no natural external id of its own to dedupe on.
            self._db.query(
                """
                MATCH (s:Session {session_id: $session_id})
                MERGE (s)-[:HAS_EPISODE]->(e:Episode)
                SET e.summary = $summary, e.summarized_at = $summarized_at
                """,
                params={"session_id": session_id, "summary": summary_text, "summarized_at": reconciled_at},
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
                "now": datetime.now(timezone.utc).isoformat(),
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
        lightrag_wrapper: Any,
        extraction_backend: ExtractionBackend | None = None,
        actions_graph: ActionsGraph | None = None,
        entity_workspace: str | None = None,
        promote_labels: bool = False,
        enforce_ontology: bool = False,
        ontology_path: str | Path | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        summarize: bool = True,
        reextract: bool = False,
    ) -> ReconciliationSummary:
        """Batch-extract entities and a narrative summary from a session's content.

        Pulls all reconcilable Message/ToolCall/ToolResult text recorded for
        *session_id* in Actions Graph, plus this session's Memories, dedupes
        by content hash, and runs the result through unstructured2graph's
        chunk + entity-extraction pipeline -- GLiNER2 over the session user's
        adopted ontology version by default (see ``sessions_graph.ontology``;
        ``hygm.default_model()`` until they have one), or whatever
        ``extraction_backend`` overrides it to. The version used is recorded
        on the Session as ``ontology_version``.
        Resulting Chunk nodes are linked back to their source Action/Memory
        node via ``HAS_CHUNK`` so entities trace back to the session that
        produced them.

        This same pass also produces the session's episodic memory: an
        ``(:Episode {summary, summarized_at})`` node linked from the session
        via ``HAS_EPISODE``, via a second, dedicated LLM call (entity
        extraction and narrative summarization are different task shapes, so
        this doesn't piggyback on LightRAG's own extraction prompt) -- but
        it's still one trigger, one fetch/dedupe of session text, no separate
        schedule. Re-reconciling a session updates its one Episode rather
        than creating another.

        This is deliberately not wired to run automatically inside the
        ``SESSION_END`` hook — LightRAG extraction is LLM-backed and slow, and
        hook subprocesses run under a runtime timeout. Call this from a
        separate process (e.g. the ``sessions-graph reconcile`` CLI) instead.

        Requires the ``sessions-graph[reconciliation]`` extra (actions-graph +
        unstructured2graph).

        Args:
            session_id: Session to reconcile.
            lightrag_wrapper: An initialised ``MemgraphLightRAGWrapper``. Always
                required, even when ``extraction_backend`` overrides entity
                extraction to a different backend: the narrative summary above
                is always produced via this wrapper's own LLM
                (``summarize_session_texts``), since summarization is a
                generative task no non-LLM backend (e.g. GLiNER2) can do.
            extraction_backend: Overrides what runs entity extraction (e.g.
                ``LightRAGBackend(lightrag_wrapper)``); the user's ontology
                version is then not consulted. Defaults to a ``GLiNER2Backend``
                over the user's adopted model, one per distinct model, all
                sharing one loaded GLiNER2 checkpoint.
            actions_graph: An ``ActionsGraph`` instance sharing this graph's
                Memgraph connection. Constructed automatically if omitted.
            entity_workspace: Passed through to ``unstructured2graph.from_texts``.
                Defaults to whatever the LightRAG wrapper resolves to, so
                session-derived entities land in the same workspace as
                document-ingested ones and can merge.
            promote_labels: Passed through to ``unstructured2graph.from_texts``.
                Promotes every entity_type to a real Memgraph label with no fixed
                vocabulary and no ontology_conformant flagging. Ignored if
                enforce_ontology is also True. Both default to False, matching
                unstructured2graph's own default: no label promotion unless
                explicitly requested.
            enforce_ontology: Passed through to ``unstructured2graph.from_documents``.
                Restricts entity_type promotion to ontology_path's vocabulary (or,
                without one, the vocabulary the backend extracted against, else
                unstructured2graph's bundled default), flagging anything outside it
                ontology_conformant=false instead of promoting a label. Takes
                precedence over promote_labels.
            ontology_path: Passed through to ``unstructured2graph.from_texts``. Only
                consulted when enforce_ontology=True.
            embedding_model: The model the session's messages, entities and
                edges are embedded with once extraction completes (see
                :meth:`embed_session`). An embedding failure is recorded on the
                Session and never fails the reconciliation.
            summarize: False skips the summary LLM call, so no Episode is
                written and `lightrag_wrapper` may be None. Recall never reads
                Episodes, so a benchmark build can skip them.
            reextract: True re-reads an already reconciled session: no summary,
                and the Session keeps its ``reconciled_at``, recording
                ``reextracted_at`` instead -- how derivation re-reads its delta
                under a new model (#434) without the delta counting as new
                sessions again.

        Returns:
            An :class:`ReconciliationSummary` describing what happened. Never
            raises for per-session failures — the failure is recorded on the
            Session node and returned so a sweep over many sessions can
            continue past one bad session.

        Raises:
            ImportError: if ``actions-graph`` or ``unstructured2graph`` (the
                ``sessions-graph[reconciliation]`` extra) is not installed.
                Only these dependency-availability errors propagate; any
                failure from actually reconciling *session_id* is caught and
                reported through the returned summary instead.
        """
        actions_graph = self._default_actions_graph(actions_graph, "reconcile_session")

        try:
            from unstructured2graph import from_documents
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_session; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        prepared = self._prepare_session(session_id, actions_graph)

        try:
            summary_text: str | None = None
            used_backend: str | None = None
            integrity: tuple[int, int] | None = None
            ontology_version: int | None = None
            if prepared.unique_texts:
                # The whole session's deduped texts as ONE document, not one
                # per turn. Each turn used to be extracted in total isolation
                # from every other turn in the same session, one independent
                # LightRAG document (and therefore two LLM calls) each. That
                # undercounts the real unit worth extracting from: a session's entities and
                # relations often span turns (coreference, a fact stated in
                # one turn and referenced in another), invisible to an
                # extractor that never sees more than one turn at a time.
                #
                # Batched, LightRAG's own chunking decides the real extraction
                # granularity from actual content size instead of forcing
                # per-turn calls regardless of size. The real ceiling on that
                # granularity turned out to be the local embedder's
                # max_token_size (256, all-MiniLM-L6-v2's trained sequence
                # length -- LightRAG re-splits any chunk down to that before
                # embedding, and extraction runs on the re-split pieces), not
                # LightRAG's own larger CHUNK_SIZE default. Since an average
                # turn here is already ~245 tokens, close to that 256 ceiling,
                # the win is real but modest -- measured on 5 real sessions,
                # 106 -> 70 extraction+gleaning calls (1.51x), not the 4x+ a
                # naive CHUNK_SIZE=1200 assumption would predict.
                #
                # Handed over verbatim, as one Document with a segment per
                # source, rather than re-chunked through `unstructured`, whose
                # partitioner rewrites text and would invalidate the turn
                # offsets. The segments are what the GLiNER2 backend windows on
                # (one turn each, #352) and resolves the user's own mentions
                # with (#358); LightRAG reads the text alone.
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
                    entity_workspace=entity_workspace,
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
                if summarize and not reextract:
                    summary_text = await summarize_session_texts(lightrag_wrapper, list(prepared.unique_texts.values()))

            if not reextract:
                self._write_completed(
                    session_id,
                    summary_text=summary_text,
                    extraction_backend=used_backend,
                    ontology_version=ontology_version,
                )
            else:
                self._write_reextracted(session_id, extraction_backend=used_backend, ontology_version=ontology_version)
            self._embed_after_reconcile(session_id, embedding_model)
            return ReconciliationSummary(
                session_id=session_id,
                status="completed",
                texts_considered=len(prepared.sources),
                texts_deduped=len(prepared.unique_texts),
                summary_written=summary_text is not None,
                nonconformant_entities=integrity[0] if integrity else None,
                nonconformant_relations=integrity[1] if integrity else None,
            )
        except Exception as e:
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
        entity_workspace: str | None = None,
        promote_labels: bool = False,
        enforce_ontology: bool = False,
        ontology_path: str | Path | None = None,
        summary_concurrency: int = 4,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[ReconciliationSummary]:
        """Reconcile many sessions as ONE LightRAG processing pass, not one per session.

        ``reconcile_session`` calls ``unstructured2graph.from_texts``, which
        calls LightRAG's ``ainsert`` -- enqueue *and* immediately process, in
        one call. Looping that per session (as callers of ``reconcile_session``
        must) never gives LightRAG's own worker pool (``MAX_PARALLEL_INSERT``
        documents at once, gated by the separate ``MAX_ASYNC_LLM`` LLM-call
        semaphore) more than one document to actually parallelize over.
        Worse, ``apipeline_process_enqueue_documents`` holds a workspace-level
        "busy" lock: a second concurrent caller doesn't run its own pass in
        parallel, it just sets a pending flag and returns having done no
        work -- so wrapping ``reconcile_session`` calls in ``asyncio.gather``
        would not parallelize anything either.

        This stages every session's combined document first
        (``unstructured2graph.enqueue_texts``), then triggers LightRAG's
        processing pass exactly once (``process_enqueued_and_finalize``) for
        the whole group, so its worker pool has something real to
        parallelize. Per-session bookkeeping that has no shared-state
        contention -- the episode-summary LLM call, HAS_CHUNK linking,
        ``reconciliation_status`` -- happens after, fanned out with its own
        bounded concurrency (``summary_concurrency``); it doesn't go through
        LightRAG's pipeline at all, so the "busy" lock above doesn't apply to
        it.

        Known limitation: if the shared processing pass raises outright (e.g.
        a Memgraph connection error), every session in this call is marked
        failed with the same error -- coarser than ``reconcile_session``'s
        per-session isolation, since there is no per-document result to
        attribute it to. A per-document extraction failure that does *not*
        raise is handled precisely, not lumped into this case: this method
        reads each session's own chunk(s) back from
        ``process_enqueued_and_finalize``'s returned status map and marks
        only the sessions whose chunks did not reach ``"processed"`` as
        failed, with that chunk's real ``error_msg``.

        Measured against a real, dedicated eval instance (MAX_PARALLEL_INSERT
        and MAX_ASYNC_LLM both raised to 16, gpt-4o-mini extraction, local
        bge-m3 embeddings), 20 real sessions, one call each way: 458s / 86
        extraction+gleaning calls sequentially (``reconcile_session`` x20)
        versus 201s / 42 calls as one batch here -- 2.28x faster and 2.05x
        fewer calls on identical content. The call-count drop is larger than
        parallelism alone predicts; confirmed cause (not the response-cache
        hypothesis this docstring originally recorded): ``enqueue_texts``
        ids chunks by content hash, and upstream's real distractor-session
        reuse means several of a batch's chunks are byte-identical, so they
        collapse into fewer LightRAG documents than the same content
        submitted one ``ainsert`` at a time ever could.

        Args:
            session_ids: Sessions to reconcile together as one batch. Choosing
                how many to group here is the caller's call: LightRAG's own
                concurrency knobs bound how many actually run at once
                regardless of batch size, but a very large batch delays any
                progress signal until the whole group finishes.
            lightrag_wrapper: An initialised ``MemgraphLightRAGWrapper``,
                shared across the whole batch.
            actions_graph: An ``ActionsGraph`` instance sharing this graph's
                Memgraph connection. Constructed automatically if omitted.
            entity_workspace: Passed through to
                ``unstructured2graph.process_enqueued_and_finalize``. Defaults
                to whatever the LightRAG wrapper resolves to.
            promote_labels: Passed through; see ``reconcile_session``.
            enforce_ontology: Passed through; see ``reconcile_session``.
                Takes precedence over ``promote_labels``.
            ontology_path: Only consulted when ``enforce_ontology=True``.
            summary_concurrency: Bound on concurrent episode-summary LLM
                calls during finalize. Independent of LightRAG's own
                ``MAX_ASYNC_LLM``, since this call never enters its pipeline.
                Must be at least 1.
            embedding_model: Passed through; see ``reconcile_session``.

        Returns:
            One :class:`ReconciliationSummary` per input session_id, in the
            same order as ``session_ids``.

        Raises:
            ValueError: if ``summary_concurrency`` is less than 1 -- checked
                up front, before any paid extraction work, since
                ``asyncio.Semaphore(0)`` would otherwise deadlock the
                finalize step forever *after* the batch has already been
                billed.
            ImportError: if ``actions-graph`` or ``unstructured2graph`` (the
                ``sessions-graph[reconciliation]`` extra) is not installed.
        """
        if summary_concurrency < 1:
            raise ValueError(f"summary_concurrency must be >= 1, got {summary_concurrency}")

        actions_graph = self._default_actions_graph(actions_graph, "reconcile_sessions_batch")

        try:
            from unstructured2graph import enqueue_texts, process_enqueued_and_finalize
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_sessions_batch; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        prepared = [self._prepare_session(session_id, actions_graph) for session_id in session_ids]

        results: dict[str, ReconciliationSummary] = {}

        # Sessions with nothing to reconcile complete immediately -- same as
        # reconcile_session's own empty-content branch -- and must not be
        # included in the shared enqueue below (an empty text would just
        # waste a slot in the batch).
        to_enqueue = [p for p in prepared if p.unique_texts]
        for p in prepared:
            if p.unique_texts:
                continue
            self._write_completed(p.session_id, summary_text=None, extraction_backend=None)
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
                entity_workspace=entity_workspace,
                promote_labels=promote_labels,
                enforce_ontology=enforce_ontology,
                ontology_path=ontology_path,
            )
        except Exception as e:
            # The shared pass itself failed outright (not a per-document
            # extraction error, which process_enqueued_and_finalize reports
            # through doc_statuses without raising) -- no per-document result
            # exists to attribute this to, so every session in the batch
            # shares the one error.
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

        semaphore = asyncio.Semaphore(summary_concurrency)

        async def _finalize_one(prepared_session: _PreparedSession, session_chunks: list[Any]) -> None:
            # process_enqueued_and_finalize returning is not proof this
            # session's own document succeeded -- it reports every document's
            # real, final status (including per-document failures LightRAG
            # itself swallows rather than raises), and only that status
            # decides completed vs failed here.
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
                async with semaphore:
                    summary_text = await summarize_session_texts(
                        lightrag_wrapper, list(prepared_session.unique_texts.values())
                    )
                # Literal, not type(...).__name__: this whole method only ever
                # drives LightRAG's enqueue/process pipeline (map #322), so
                # there is no backend instance here to introspect -- unlike
                # reconcile_session, which accepts an arbitrary one.
                self._write_completed(
                    prepared_session.session_id, summary_text=summary_text, extraction_backend="LightRAGBackend"
                )
                self._embed_after_reconcile(prepared_session.session_id, embedding_model)
                results[prepared_session.session_id] = ReconciliationSummary(
                    session_id=prepared_session.session_id,
                    status="completed",
                    texts_considered=len(prepared_session.sources),
                    texts_deduped=len(prepared_session.unique_texts),
                    summary_written=summary_text is not None,
                )
            except Exception as e:
                # Isolated per session, unlike the shared pass above: nothing
                # here touches another session's state, so one failure must
                # not cost the rest of the batch its result.
                self._write_failed(prepared_session.session_id, str(e))
                results[prepared_session.session_id] = ReconciliationSummary(
                    session_id=prepared_session.session_id,
                    status="failed",
                    texts_considered=len(prepared_session.sources),
                    texts_deduped=len(prepared_session.unique_texts),
                    error=str(e),
                )

        await asyncio.gather(
            *(_finalize_one(p, grouped_chunks[i] if i < len(grouped_chunks) else []) for i, p in enumerate(to_enqueue))
        )

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
            "SET s.embedding_status = 'completed', s.embedding_model = $model, s.embedding_error = null",
            params={"session_id": session_id, "model": model},
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
        """Session ids whose embedding failed, never ran, or ran with a model other than *model*."""
        rows = self._db.query(
            """
            MATCH (s:Session)
            WHERE s.embedding_status IS NULL OR s.embedding_status <> 'completed' OR s.embedding_model <> $model
            RETURN s.session_id AS session_id
            ORDER BY s.session_id
            LIMIT $limit
            """,
            params={"model": model, "limit": limit},
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
        mentions onto. Collection records one only when the harness reports a
        user, and the eval injector never does (#354). ``anon-`` rather than
        ``anon:`` because ``validate_user_id`` does not accept a colon.
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

    def _extraction_backend_for(self, model: HygmModel) -> ExtractionBackend:
        """A GLiNER2 backend over `model`, kept for reuse: loading the checkpoint is the slow part, so it loads once."""
        import json

        from hygm import model_to_mapping
        from unstructured2graph import Ontology
        from unstructured2graph.gliner2_backend import GLiNER2Backend

        key = json.dumps(model_to_mapping(model), sort_keys=True)
        if key not in self._extraction_backends:
            loaded = next(iter(self._extraction_backends.values()), None)
            engine = loaded.engine if isinstance(loaded, GLiNER2Backend) else None
            self._extraction_backends[key] = GLiNER2Backend(ontology=Ontology.from_model(model), model=engine)
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
        """Wire (:Action|:Memory)-[:HAS_CHUNK]->(:Chunk) for every source.

        Every source in the session links to every chunk the session's one
        combined document produced. In the overwhelmingly common case that is
        exact, not an approximation: MAX_SESSION_BATCH_CHARS keeps the whole
        session as a single chunk, which genuinely does contain every
        source's text verbatim. Only a session large enough to make
        unstructured2graph split the combined document into more than one
        chunk trades that precision for a session-level (rather than
        per-source) provenance signal -- a source may link to a chunk its own
        text isn't actually inside, but never to another session's chunk.
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
        )


def _derive_mode(derive: str) -> Derive:
    """*derive* as a mode, so a bad config value fails as ValueError like every other schema problem."""
    if derive not in ("extend", "off"):
        raise ValueError(f"derive must be 'extend' or 'off', got {derive!r}")
    return "extend" if derive == "extend" else "off"
