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

from .models import Memory, validate_content, validate_memory_id, validate_user_id
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

    def _write_completed(self, session_id: str, *, summary_text: str | None) -> str:
        """Mark *session_id* completed and, if a narrative summary was
        produced, MERGE its Episode -- both stamped with the SAME timestamp,
        computed here at actual completion time (not earlier, e.g. when a
        batch started), so ``reconciled_at``/``summarized_at`` reflect when
        this specific session actually finished. Returns that timestamp."""
        reconciled_at = datetime.now(timezone.utc).isoformat()
        self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            SET s.reconciliation_status = 'completed', s.reconciled_at = $reconciled_at
            """,
            params={"session_id": session_id, "reconciled_at": reconciled_at},
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
        actions_graph: ActionsGraph | None = None,
        entity_workspace: str | None = None,
        promote_labels: bool = False,
        enforce_ontology: bool = False,
        ontology_path: str | Path | None = None,
    ) -> ReconciliationSummary:
        """Batch-extract entities and a narrative summary from a session's content.

        Pulls all reconcilable Message/ToolCall/ToolResult text recorded for
        *session_id* in Actions Graph, plus this session's Memories, dedupes
        by content hash, and runs the result through unstructured2graph's
        chunk + LightRAG entity-extraction pipeline. Resulting Chunk nodes are
        linked back to their source Action/Memory node via ``HAS_CHUNK`` so
        entities trace back to the session that produced them.

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
            lightrag_wrapper: An initialised ``MemgraphLightRAGWrapper``.
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
            enforce_ontology: Passed through to ``unstructured2graph.from_texts``.
                Restricts entity_type promotion to ontology_path's vocabulary (or
                unstructured2graph's bundled default), flagging anything outside it
                ontology_conformant=false instead of promoting a label. Takes
                precedence over promote_labels.
            ontology_path: Passed through to ``unstructured2graph.from_texts``. Only
                consulted when enforce_ontology=True.

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
            from unstructured2graph import LightRAGBackend, from_texts
        except ImportError as exc:
            msg = "unstructured2graph is required for reconcile_session; install sessions-graph[reconciliation]"
            raise ImportError(msg) from exc

        prepared = self._prepare_session(session_id, actions_graph)

        try:
            summary_text: str | None = None
            if prepared.unique_texts:
                # The whole session's deduped texts as ONE document, not one
                # per turn. A turn is still never split mid-utterance (that's
                # what #327 fixed, and MAX_SESSION_BATCH_CHARS stays well
                # above MAX_RECONCILABLE_CHARS so a turn's own truncation
                # bound is always the tighter one) -- but today each turn was
                # also extracted in total isolation from every other turn in
                # the same session, one independent LightRAG document (and
                # therefore two LLM calls) each. That undercounts the real
                # unit worth extracting from: a session's entities and
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
                grouped_chunks = await from_texts(
                    [prepared.combined_text],
                    memgraph=self._db,
                    extraction_backend=LightRAGBackend(lightrag_wrapper),
                    entity_workspace=entity_workspace,
                    promote_labels=promote_labels,
                    enforce_ontology=enforce_ontology,
                    ontology_path=ontology_path,
                    chunk_kwargs={"max_characters": MAX_SESSION_BATCH_CHARS},
                )
                session_chunks = grouped_chunks[0] if grouped_chunks else []
                self._link_chunks_to_sources(prepared.sources, session_chunks)
                summary_text = await summarize_session_texts(lightrag_wrapper, list(prepared.unique_texts.values()))

            self._write_completed(session_id, summary_text=summary_text)
            return ReconciliationSummary(
                session_id=session_id,
                status="completed",
                texts_considered=len(prepared.sources),
                texts_deduped=len(prepared.unique_texts),
                summary_written=summary_text is not None,
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
            self._write_completed(p.session_id, summary_text=None)
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
                self._write_completed(prepared_session.session_id, summary_text=summary_text)
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
