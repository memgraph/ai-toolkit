"""Reconcile an injected eval batch: the same pass a real session gets, with eval-scoped cost tuning.

Runs the same entity-extraction pass a real session gets, so retrieval is
scored against the genuine emerged graph. Separate from injection because
extraction is slow.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from memgraph_toolbox.api.memgraph import Memgraph

from .inject import PENDING

#: LightRAG's fallback store, as `sessions-graph reconcile --working-dir`
#: defaults it. Never None: LightRAG checks the key's presence, not its value.
DEFAULT_WORKING_DIR = "./lightrag_storage"


@dataclass(frozen=True)
class Reconciled:
    """Outcome of reconciling a batch."""

    reconciled: int
    failed: int
    errors: tuple[str, ...] = ()


def _resolve_llm_credentials() -> None:
    """Fill LLM env vars from context-graph's config file if unset.

    Per ADR 0002, the config file is the canonical source; env vars are a
    write-time convenience. Eval runs standalone, so nothing has overlaid the
    config for us. Existing env always wins, so an explicit key stays in charge.
    memgraph-context-graph is an optional extra -- skip quietly without it.
    """
    import os

    try:
        from memgraph_context_graph.adapters._identity import resolve_llm_env
    except ImportError:
        return
    for key, value in resolve_llm_env().items():
        if value:
            os.environ.setdefault(key, value)


#: LightRAG's default (8) re-pays an LLM merge-summary on every later chunk
#: touching an entity past 8 mentions; in a batch-wide workspace recurring
#: entities pay it all run. Raised so only the hottest few do, at the cost of
#: plain concatenation below it.
_EVAL_FORCE_LLM_SUMMARY_ON_MERGE = "30"

#: Raised together, matched: MAX_ASYNC_LLM is a global semaphore over every
#: MAX_PARALLEL_INSERT worker's LLM calls, so raising one alone buys nothing.
#: 16 is a starting point, not a measured figure.
_EVAL_MAX_PARALLEL_INSERT = "16"
_EVAL_MAX_ASYNC_LLM = "16"

#: Lowered, unlike the LLM knobs: every embedding call hits the same local,
#: CPU-bound bge-m3, where concurrency contends rather than parallelizes (9 of
#: 10 sessions timed out at LightRAG's default 8). The timeout is headroom.
_EVAL_EMBEDDING_FUNC_MAX_ASYNC = "2"
_EVAL_EMBEDDING_TIMEOUT = "120"


def _resolve_reconciliation_tuning() -> None:
    """Set eval-scoped LightRAG cost/concurrency env vars, ``setdefault`` only.

    LightRAG reads them once, as dataclass defaults when ``lightrag.lightrag``
    is first imported, so this must run before that import.
    """
    import os

    os.environ.setdefault("FORCE_LLM_SUMMARY_ON_MERGE", _EVAL_FORCE_LLM_SUMMARY_ON_MERGE)
    os.environ.setdefault("MAX_PARALLEL_INSERT", _EVAL_MAX_PARALLEL_INSERT)
    os.environ.setdefault("MAX_ASYNC_LLM", _EVAL_MAX_ASYNC_LLM)
    os.environ.setdefault("EMBEDDING_FUNC_MAX_ASYNC", _EVAL_EMBEDDING_FUNC_MAX_ASYNC)
    os.environ.setdefault("EMBEDDING_TIMEOUT", _EVAL_EMBEDDING_TIMEOUT)


#: LightRAG re-splits chunks to the embedder's max_token_size before
#: extraction, so the wrapper's default all-MiniLM (256 tokens) capped chunk
#: size. bge-m3 (local, 8192 tokens) cut extraction chunks from 8.77 to ~1.0
#: per session (map #297).
_EVAL_EMBEDDING_MODEL = "BAAI/bge-m3"
_EVAL_EMBEDDING_DIM = 1024
_EVAL_EMBEDDING_MAX_TOKENS = 8192


def _eval_embedding_func() -> Any:
    """The eval-scoped bge-m3 embedding function, passed to the wrapper rather than changing its default."""
    from lightrag_memgraph.embeddings import build_memgraph_sentence_embed

    return build_memgraph_sentence_embed(
        model_name=_EVAL_EMBEDDING_MODEL,
        embedding_dim=_EVAL_EMBEDDING_DIM,
        max_token_size=_EVAL_EMBEDDING_MAX_TOKENS,
    )


def pending_sessions(db: "Memgraph", limit: int | None = None) -> list[str]:
    """Session ids awaiting reconciliation, oldest first.

    Ordered so a limited run makes deterministic progress through the batch
    rather than revisiting whichever sessions the planner happens to return.
    """
    query = (
        "MATCH (s:Session) WHERE s.reconciliation_status = $status "
        "RETURN s.session_id AS session_id ORDER BY s.session_id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [row["session_id"] for row in db.query(query, {"status": PENDING})]


#: Extraction backends reconcile_batch knows how to build: "gliner2" (local,
#: LLM-free) or "lightrag" (LLM-based). A closed set because each needs a
#: different reconciling loop, not just a different object.
ExtractionBackendName = Literal["lightrag", "gliner2"]
EXTRACTION_BACKENDS: tuple[ExtractionBackendName, ...] = ("lightrag", "gliner2")

#: The class name sessions-graph persists as ``Session.extraction_backend``,
#: which runner._require_reconciled checks a reused graph against.
BACKEND_CLASS_NAMES: dict[ExtractionBackendName, str] = {
    "lightrag": "LightRAGBackend",
    "gliner2": "GLiNER2Backend",
}

#: The vocabulary GLiNER2 extracts against on this corpus (#361's hand-written
#: one), until a derived vocabulary replaces it.
GLINER2_ONTOLOGY_PATH = Path(__file__).parent / "ontologies" / "longmemeval.yaml"


#: What reconcile extracts under: "fixed", the LongMemEval vocabulary; "default",
#: each user's adopted version (hygm's default model, nothing derived); "learned",
#: the same, then each user's own model derived and adopted (learned.py).
ONTOLOGIES = ("fixed", "default", "learned")


async def reconcile_batch(
    db: "Memgraph",
    *,
    limit: int | None = None,
    memgraph_url: str | None = None,
    working_dir: str = DEFAULT_WORKING_DIR,
    lightrag_wrapper: Any = None,
    extraction_backend: ExtractionBackendName = "gliner2",
    progress: bool = True,
    sessions_per_call: int = 20,
    gliner2_concurrency: int = 4,
    ontology: str = "fixed",
) -> Reconciled:
    """Reconcile pending sessions in the eval graph, reporting failures in the result rather than raising.

    ``memgraph_url`` must name the instance ``db`` is connected to: LightRAG's
    stores read ``MEMGRAPH_URL`` from the environment, not the client, and
    would otherwise write to whatever graph it names (#309).

    "lightrag" sessions go through ``reconcile_sessions_batch``,
    ``sessions_per_call`` at a time, so LightRAG's worker pool has documents
    to parallelize over (map #322). "gliner2" sessions go through
    ``reconcile_session``, ``gliner2_concurrency`` at a time, against
    :data:`GLINER2_ONTOLOGY_PATH`; non-conformant output is reported as an
    error. No LightRAG wrapper is built when nothing is pending or GLiNER2
    extracts, so neither needs LLM credentials.

    ``ontology="default"`` or ``"learned"`` (GLiNER2 only) extracts each
    session under its user's adopted ontology version instead of
    :data:`GLINER2_ONTOLOGY_PATH` -- hygm's default model until
    ``learned.derive_users`` adopts one, which only ``"learned"`` runs.

    Raises:
        ValueError: ``sessions_per_call`` or ``gliner2_concurrency`` below 1,
            an unknown ``extraction_backend``, ``ontology`` "default" or
            "learned" without GLiNER2, or no Memgraph URL to point LightRAG at.
    """
    if sessions_per_call < 1:
        raise ValueError(f"sessions_per_call must be >= 1, got {sessions_per_call}")
    if gliner2_concurrency < 1:
        raise ValueError(f"gliner2_concurrency must be >= 1, got {gliner2_concurrency}")
    if extraction_backend not in EXTRACTION_BACKENDS:
        raise ValueError(f"extraction_backend must be one of {EXTRACTION_BACKENDS}, got {extraction_backend!r}")
    if ontology not in ONTOLOGIES or (ontology != "fixed" and extraction_backend != "gliner2"):
        raise ValueError(f"ontology must be one of {ONTOLOGIES}, the last two with gliner2; got {ontology!r}")

    import os

    from sessions_graph import SessionsGraph

    session_ids = pending_sessions(db, limit=limit)
    if not session_ids:
        return Reconciled(reconciled=0, failed=0)

    _resolve_llm_credentials()
    _resolve_reconciliation_tuning()

    owns_wrapper = lightrag_wrapper is None and extraction_backend != "gliner2"

    if memgraph_url:
        # Set, not defaulted: this is what LightRAG's stores actually follow.
        os.environ["MEMGRAPH_URL"] = memgraph_url
        os.environ.setdefault("MEMGRAPH_USER", "")
        os.environ.setdefault("MEMGRAPH_PASSWORD", "")
        os.environ.setdefault("MEMGRAPH_DATABASE", "memgraph")
    elif owns_wrapper and not os.environ.get("MEMGRAPH_URL"):
        # A caller-supplied wrapper has already decided where its stores point.
        raise ValueError(
            "reconcile_batch needs memgraph_url (or MEMGRAPH_URL in the environment): "
            "LightRAG's storage backends read the environment rather than the client "
            "passed in, so without it reconciliation cannot start -- and with a wrong "
            "one it would write to a different graph than the one being evaluated."
        )

    graph = SessionsGraph(memgraph=db)
    graph.setup()

    if owns_wrapper:
        from lightrag_memgraph import MemgraphLightRAGWrapper

        lightrag_wrapper = MemgraphLightRAGWrapper()
        await lightrag_wrapper.initialize(working_dir=working_dir, embedding_func=_eval_embedding_func())

    gliner2_backend = None
    if extraction_backend == "gliner2" and ontology == "fixed":
        from unstructured2graph import load_ontology
        from unstructured2graph.gliner2_backend import GLiNER2Backend

        gliner2_backend = GLiNER2Backend(ontology=load_ontology(GLINER2_ONTOLOGY_PATH))

    reconciled = 0
    errors: list[str] = []

    def _tally(summaries: list) -> None:
        """Count each summary as reconciled or as an error."""
        nonlocal reconciled
        for summary in summaries:
            if summary.status == "completed":
                reconciled += 1
            else:
                errors.append(f"{summary.session_id}: {summary.error}")

    def _report(done: int) -> None:
        # Silent until done is indistinguishable from hung.
        if progress:
            print(f"  reconciled {done}/{len(session_ids)} ({reconciled} ok, {len(errors)} failed)", flush=True)

    try:
        if extraction_backend == "gliner2":
            # Extraction runs in a worker thread per session, so sessions
            # overlap; returns flatten past 4 since torch already uses several
            # threads. Memgraph writes stay on the event loop thread, so no two
            # MERGEs race.
            semaphore = asyncio.Semaphore(gliner2_concurrency)
            done = 0

            async def _one(session_id: str) -> None:
                nonlocal done
                async with semaphore:
                    summary = await graph.reconciler.reconcile(
                        session_id,
                        # None under a learned ontology: the user's adopted version.
                        extraction_backend=gliner2_backend,
                        enforce_ontology=True,
                        # The same file the backend extracts against, so label
                        # promotion and the domain/range check can't drift from it.
                        ontology_path=GLINER2_ONTOLOGY_PATH if gliner2_backend is not None else None,
                    )
                if summary.nonconformant_entities or summary.nonconformant_relations:
                    # Zero by construction on GLiNER2 output (#355): nonzero is a bug.
                    errors.append(
                        f"{session_id}: {summary.nonconformant_entities} non-conformant entities, "
                        f"{summary.nonconformant_relations} non-conformant relationships"
                    )
                _tally([summary])
                done += 1
                _report(done)

            await asyncio.gather(*(_one(session_id) for session_id in session_ids))
        else:
            for start in range(0, len(session_ids), sessions_per_call):
                chunk = session_ids[start : start + sessions_per_call]
                outcomes = await graph.reconciler.reconcile_batch(
                    chunk,
                    lightrag_wrapper=lightrag_wrapper,
                    enforce_ontology=True,
                )
                _tally(outcomes)
                _report(start + len(chunk))
    finally:
        if owns_wrapper:
            finalize = getattr(lightrag_wrapper, "finalize", None)
            if finalize is not None:
                await finalize()

    return Reconciled(reconciled=reconciled, failed=len(errors), errors=tuple(errors))
