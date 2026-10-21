"""Session Reconciliation: a session's or memory file's text into extracted entities, and the status that tracks it.

:class:`Reconciler` is the one place that writes a Session's processing
state -- ``reconciliation_status``/``reconciled_at``/``reconciliation_error``,
``reextracted_at``/``reextraction_error``, ``extraction_backend``/
``ontology_version`` and ``embedding_*`` -- and a Memory's
``extraction_status``. Collection only ever marks a session pending
(:meth:`Reconciler.mark_pending`); everything after that runs out of band.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from memgraph_context_graph import session as session_node
from memgraph_toolbox.api.memgraph import merge_with_retry

from .embeddings import DEFAULT_EMBEDDING_MODEL, Embedded, EmbeddingUnavailableError, embed_memory, embed_session
from .extraction import ExtractionConfig
from .models import _utc_now
from .passages import PASSAGE_SCHEME
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

    def summary(
        self, status: str, *, error: str | None = None, integrity: tuple[int, int] | None = None
    ) -> ReconciliationSummary:
        """This session's :class:`ReconciliationSummary` with *status*."""
        return ReconciliationSummary(
            session_id=self.session_id,
            status=status,
            texts_considered=len(self.sources),
            texts_deduped=len(self.unique_texts),
            nonconformant_entities=integrity[0] if integrity else None,
            nonconformant_relations=integrity[1] if integrity else None,
            error=error,
        )


class Reconciler:
    """Extracts entities from sessions and memory files over one Memgraph connection, and records how that went."""

    def __init__(
        self, db: Any, *, extraction: ExtractionConfig | None = None, actions_graph: ActionsGraph | None = None
    ) -> None:
        """Initialise a Reconciler.

        Args:
            db: The Memgraph client every read and write goes through.
            extraction: The GLiNER2 settings every extraction uses (reconciliation,
                re-extraction, derivation); defaults when None.
            actions_graph: Where sessions' actions are read from; one over *db*
                is built on first use when None.
        """
        self.db = db
        self.extraction = extraction or ExtractionConfig()
        self._actions_graph = actions_graph
        # One GLiNER2 backend per distinct model, all sharing the first one's loaded weights.
        self._extraction_backends: dict[str, GLiNER2Backend] = {}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def mark_pending(self, session_id: str) -> None:
        """Queue *session_id* for the next ``reconcile --pending`` sweep; cheap enough for a hook."""
        self.db.query(
            "MATCH (s:Session {session_id: $session_id}) SET s.reconciliation_status = 'pending';",
            params={"session_id": session_id},
        )

    def pending_sessions(self, *, limit: int = 100) -> list[str]:
        """Return session_ids marked ``reconciliation_status = 'pending'``."""
        rows = self.db.query(
            """
            MATCH (s:Session {reconciliation_status: 'pending'})
            RETURN s.session_id AS session_id
            ORDER BY s.session_id
            LIMIT $limit
            """,
            params={"limit": limit},
        )
        return [row["session_id"] for row in rows]

    def pending_memories(self, *, limit: int = 100) -> list[str]:
        """Return memory_ids whose file changed since it was last reconciled (``extraction_status = 'pending'``)."""
        rows = self.db.query(
            """
            MATCH (m:Memory {extraction_status: 'pending'})
            RETURN m.memory_id AS memory_id
            ORDER BY m.updated_at
            LIMIT $limit
            """,
            params={"limit": limit},
        )
        return [row["memory_id"] for row in rows]

    def pending_embeddings(self, *, model: str = DEFAULT_EMBEDDING_MODEL, limit: int = 100) -> list[str]:
        """Session ids whose embedding failed, never ran, or ran with a model or passage split other than today's."""
        rows = self.db.query(
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

    def _complete(
        self, session_id: str, *, extraction_backend: str | None, ontology_version: int | None, reextract: bool
    ) -> None:
        """Record a successful pass over *session_id*: what extracted it, under which ontology version.

        ``extraction_backend`` is the class name of the backend that extracted
        the session's entities -- what ``context_graph_eval.runner._require_reconciled``
        checks before trusting which backend built a reused graph. ``None``
        means nothing was extracted (no reconcilable content), not an unknown
        backend. ``ontology_version`` is ``None`` when the caller supplied its
        own backend. A re-extraction keeps when the session was reconciled and
        stamps ``reextracted_at`` instead; a first pass is stamped at its own
        completion rather than its batch's start.
        """
        stamp = "s.reextracted_at" if reextract else "s.reconciliation_status = 'completed', s.reconciled_at"
        self.db.query(
            f"""
            MATCH (s:Session {{session_id: $session_id}})
            SET {stamp} = $now, s.extraction_backend = $extraction_backend, s.ontology_version = $ontology_version
            """,
            params={
                "session_id": session_id,
                "now": _utc_now(),
                "extraction_backend": extraction_backend,
                "ontology_version": ontology_version,
            },
        )

    def _fail(self, session_id: str, error: str, *, reextract: bool = False) -> None:
        """Record *error* on *session_id*; a failed re-extraction leaves it reconciled as it was."""
        stamp = "s.reextraction_error" if reextract else "s.reconciliation_status = 'failed', s.reconciliation_error"
        self.db.query(
            f"MATCH (s:Session {{session_id: $session_id}}) SET {stamp} = $error",
            params={"session_id": session_id, "error": error},
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        session_id: str,
        *,
        extraction_backend: ExtractionBackend | None = None,
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
            promote_labels: Passed through to ``from_documents``; ignored when
                ``enforce_ontology`` is set.
            enforce_ontology: Passed through to ``from_documents``: promotes only
                ``ontology_path``'s vocabulary, else the backend's, flagging the
                rest ``ontology_conformant = false``.
            ontology_path: Only consulted when ``enforce_ontology`` is set.
            embedding_model: What the session is embedded with afterwards (see
                :meth:`embed`); an embedding failure never fails this.
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
        from_documents = _require("from_documents", "reconcile")
        prepared = self._prepare(session_id)

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
                user_id = self.session_user(session_id)
                if extraction_backend is None:
                    from .ontology import adopted

                    version = adopted(self.db, user_id)
                    backend = self.backend_for(version.model)
                    ontology_version = version.version
                else:
                    backend = extraction_backend
                # A backend that extracts against a vocabulary (GLiNER2) is
                # enforced against that same vocabulary unless a file overrides it.
                ontology = None if ontology_path else getattr(backend, "ontology", None)
                grouped_chunks = await from_documents(
                    [prepared.document(user_id)],
                    memgraph=self.db,
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

            self._complete(
                session_id, extraction_backend=used_backend, ontology_version=ontology_version, reextract=reextract
            )
            self._embed_quietly(session_id, embedding_model)
            return prepared.summary("completed", integrity=integrity)
        except Exception as e:
            self._fail(session_id, str(e), reextract=reextract)
            return prepared.summary("failed", error=str(e))

    async def reconcile_batch(
        self,
        session_ids: list[str],
        *,
        lightrag_wrapper: Any,
        promote_labels: bool = False,
        enforce_ontology: bool = False,
        ontology_path: str | Path | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[ReconciliationSummary]:
        """Reconcile many sessions with LightRAG as ONE processing pass, not one per session.

        LightRAG's pipeline holds a workspace-level "busy" lock: a concurrent
        second pass just flags itself pending and returns, so neither a loop
        nor ``asyncio.gather`` over :meth:`reconcile` gives its worker
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
            promote_labels, enforce_ontology, ontology_path, embedding_model:
                As for :meth:`reconcile`.

        Returns:
            One :class:`ReconciliationSummary` per session, in input order.

        Raises:
            ImportError: the ``sessions-graph[reconciliation]`` extra is not installed.
        """
        enqueue_texts = _require("enqueue_texts", "reconcile_batch")
        process_enqueued_and_finalize = _require("process_enqueued_and_finalize", "reconcile_batch")

        prepared = [self._prepare(session_id) for session_id in session_ids]
        results: dict[str, ReconciliationSummary] = {}

        # Sessions with nothing to reconcile complete now and stay out of the shared enqueue.
        to_enqueue = [p for p in prepared if p.unique_texts]
        for p in prepared:
            if not p.unique_texts:
                self._complete(p.session_id, extraction_backend=None, ontology_version=None, reextract=False)
                results[p.session_id] = p.summary("completed")

        if not to_enqueue:
            return [results[sid] for sid in session_ids]

        try:
            grouped_chunks = await enqueue_texts(
                [p.combined_text for p in to_enqueue],
                memgraph=self.db,
                lightrag_wrapper=lightrag_wrapper,
                chunk_kwargs={"max_characters": MAX_SESSION_BATCH_CHARS},
            )
            all_chunks = [chunk for group in grouped_chunks for chunk in group]
            doc_statuses = await process_enqueued_and_finalize(
                self.db,
                lightrag_wrapper,
                all_chunks,
                promote_labels=promote_labels,
                enforce_ontology=enforce_ontology,
                ontology_path=ontology_path,
            )
        except Exception as e:
            # No per-document result to attribute this to: every session shares the one error.
            for p in to_enqueue:
                self._fail(p.session_id, str(e))
                results[p.session_id] = p.summary("failed", error=str(e))
            return [results[sid] for sid in session_ids]

        for i, p in enumerate(to_enqueue):
            session_chunks = grouped_chunks[i] if i < len(grouped_chunks) else []
            # Returning is not proof this session's document succeeded: LightRAG
            # swallows per-document failures, so only the reported status decides.
            failed_chunk = next(
                (c for c in session_chunks if doc_statuses.get(c.hash, {}).get("status") != "processed"), None
            )
            try:
                if failed_chunk is not None:
                    raise RuntimeError(
                        doc_statuses.get(failed_chunk.hash, {}).get(
                            "error_msg", f"chunk {failed_chunk.hash} did not reach 'processed'"
                        )
                    )
                self._link_chunks_to_sources(p.sources, session_chunks)
                # A literal: this path only ever drives LightRAG's pipeline, with no backend instance.
                self._complete(
                    p.session_id, extraction_backend="LightRAGBackend", ontology_version=None, reextract=False
                )
                self._embed_quietly(p.session_id, embedding_model)
                results[p.session_id] = p.summary("completed")
            except Exception as e:
                # Per session: nothing here touches another session's state.
                self._fail(p.session_id, str(e))
                results[p.session_id] = p.summary("failed", error=str(e))

        return [results[sid] for sid in session_ids]

    def session_document(self, session_id: str, user_id: str) -> Document:
        """*session_id*'s reconcilable text as the one Document extraction reads, attributed to *user_id*."""
        return self._prepare(session_id).document(user_id)

    def session_user(self, session_id: str) -> str:
        """The user_id of *session_id*'s (:User), synthesizing ``anon-<session_id>`` if it has none.

        Processing, never collection, supplies the missing user (#347): every
        session needs a (:User) for the GLiNER2 backend to bind the user's own
        mentions onto, and collection records one only when the harness reports
        a user. ``anon-`` rather than ``anon:`` because ``validate_user_id``
        does not accept a colon.
        """
        rows = self.db.query(
            "MATCH (u:User)-[:HAD_SESSION]->(:Session {session_id: $session_id}) RETURN u.user_id AS user_id LIMIT 1",
            params={"session_id": session_id},
        )
        if rows:
            return rows[0]["user_id"]
        user_id = f"anon-{session_id}"
        session_node.ensure(self.db, session_id)
        merge_with_retry(
            self.db,
            """
            MERGE (u:User {user_id: $user_id})
            WITH u
            MATCH (s:Session {session_id: $session_id})
            MERGE (u)-[:HAD_SESSION]->(s)
            """,
            params={"user_id": user_id, "session_id": session_id},
        )
        return user_id

    def backend_for(self, model: HygmModel) -> GLiNER2Backend:
        """A GLiNER2 backend over *model* with :attr:`extraction`'s settings, kept for reuse.

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

    def _prepare(self, session_id: str) -> _PreparedSession:
        """Gather one session's reconcilable Action text, deduped by content hash."""
        actions = self._actions().get_session_actions(session_id)
        # Memories aren't session content: each file is reconciled on its own
        # when it changes (reconcile_memory), since it outlives the session that wrote it.
        sources = build_reconciliation_sources(actions, [])
        unique_sources: dict[str, ReconciliationSource] = {}
        for source in sources:
            unique_sources.setdefault(content_hash(source.text), source)
        return _PreparedSession(session_id=session_id, sources=sources, unique_sources=unique_sources)

    def _actions(self) -> ActionsGraph:
        if self._actions_graph is None:
            try:
                from actions_graph import ActionsGraph
            except ImportError as exc:
                msg = "actions-graph is required for reconciliation; install sessions-graph[reconciliation]"
                raise ImportError(msg) from exc
            self._actions_graph = ActionsGraph(self.db)
        return self._actions_graph

    # ------------------------------------------------------------------
    # Memory files
    # ------------------------------------------------------------------

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
        from_documents = _require("from_documents", "reconcile_memory")
        from unstructured2graph import Document, Segment

        rows = self.db.query(
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

            forget_extraction(self.db, memory_id)
            if extraction_backend is None:
                backend = self.backend_for(adopted(self.db, row["user_id"]).model)
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
                memgraph=self.db,
                extraction_backend=backend,
                enforce_ontology=enforce_ontology,
                ontology=getattr(backend, "ontology", None),
            )
            chunks = grouped[0] if grouped else []
            self._link_chunks_to_sources([ReconciliationSource(kind="memory", node_id=memory_id, text=text)], chunks)
            # Recall's memory lane searches these; without MAGE it falls back to text search.
            with contextlib.suppress(EmbeddingUnavailableError):
                embed_memory(self.db, memory_id, embedding_model)
            self.db.query(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                SET m.extraction_status = 'completed', m.extracted_at = $now
                REMOVE m.extraction_error
                """,
                params={"memory_id": memory_id, "now": _utc_now()},
            )
            return MemoryReconciliation(memory_id, row["path"], "completed", chunks=len(chunks))
        except Exception as exc:
            self.db.query(
                "MATCH (m:Memory {memory_id: $memory_id}) SET m.extraction_status = 'failed', m.extraction_error = $error",
                params={"memory_id": memory_id, "error": str(exc)},
            )
            return MemoryReconciliation(memory_id, row["path"], "failed", error=str(exc))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, session_id: str, *, model: str = DEFAULT_EMBEDDING_MODEL) -> Embedded:
        """Embed *session_id*'s messages, entities and edges that have no vector from *model*.

        Records the outcome on the Session: ``embedding_status`` is
        ``'completed'`` with ``embedding_model``, or ``'failed'`` with
        ``embedding_error``, which :meth:`pending_embeddings` picks up again.

        Raises:
            EmbeddingUnavailableError: Memgraph can't embed (no MAGE, or the model
                can't load). Recorded on the Session before it propagates.
        """
        try:
            embedded = embed_session(self.db, session_id, model)
        except EmbeddingUnavailableError as exc:
            self.db.query(
                "MATCH (s:Session {session_id: $session_id}) "
                "SET s.embedding_status = 'failed', s.embedding_error = $error",
                params={"session_id": session_id, "error": str(exc)},
            )
            raise
        self.db.query(
            "MATCH (s:Session {session_id: $session_id}) "
            "SET s.embedding_status = 'completed', s.embedding_model = $model, s.embedding_scheme = $scheme, "
            "s.embedding_error = null",
            params={"session_id": session_id, "model": model, "scheme": PASSAGE_SCHEME},
        )
        return embedded

    def _embed_quietly(self, session_id: str, model: str) -> None:
        """Embed what reconciliation just wrote, without failing the reconciliation.

        A failure is recorded on the Session by :meth:`embed`, and
        ``sessions-graph embed --pending`` retries it; the extracted graph is
        valid without vectors.
        """
        with contextlib.suppress(EmbeddingUnavailableError):
            self.embed(session_id, model=model)

    # ------------------------------------------------------------------
    # Graph writes shared by both kinds of source
    # ------------------------------------------------------------------

    def _integrity(
        self, workspace: str, ontology_path: str | Path | None, ontology: Ontology | None, chunks: list[Any]
    ) -> tuple[int, int]:
        """(non-conformant entities, non-conformant relationships) over *chunks*, for ReconciliationSummary."""
        from unstructured2graph import DEFAULT_ONTOLOGY, load_ontology, ontology_report

        enforced = ontology or (load_ontology(ontology_path) if ontology_path else DEFAULT_ONTOLOGY)
        report = ontology_report(self.db, workspace, enforced, chunk_hashes=[chunk.hash for chunk in chunks])
        return report.nonconformant_entities, report.nonconformant_relations

    def _link_chunks_to_sources(self, sources: list[ReconciliationSource], chunks: list[Any]) -> None:
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
            # Label and key come from NODE_LABELS, a fixed table, never from input.
            label, id_prop = NODE_LABELS[kind]
            self.db.query(
                f"""
                UNWIND $rows AS row
                MATCH (n:{label} {{{id_prop}: row.node_id}})
                MERGE (c:Chunk {{hash: row.hash}})
                MERGE (n)-[:HAS_CHUNK]->(c)
                """,
                params={"rows": rows},
            )


def _require(name: str, caller: str) -> Any:
    """unstructured2graph's *name*, or an ImportError naming the extra *caller* needs."""
    try:
        import unstructured2graph
    except ImportError as exc:
        msg = f"unstructured2graph is required for {caller}; install sessions-graph[reconciliation]"
        raise ImportError(msg) from exc
    return getattr(unstructured2graph, name)
