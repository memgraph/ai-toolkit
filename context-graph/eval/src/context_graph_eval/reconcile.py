"""Trigger reconciliation over an injected eval batch.

Injection stages raw turns; reconciliation is what turns them into memory. It
runs the same pass a real harness session would get -- one LLM call extracting
entities into Chunks (semantic), and a second producing the session's Episode
(episodic) -- so what retrieval is later scored against is the genuine emerged
graph, not a shortcut built for eval.

Deliberately a separate step from injection: reconciliation is LLM-backed and
slow, and running it per-injection would make staging a batch cost as much as
scoring one.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from memgraph_toolbox.api.memgraph import Memgraph

from .inject import PENDING

#: LightRAG's fallback store for anything not backed by Memgraph. Matches
#: `sessions-graph reconcile --working-dir`'s default, so both entry points
#: behave the same.
#:
#: Must never be None. LightRAG guards on the *key* being present
#: (``if "working_dir" in lightrag_kwargs``), not on its value, so a None
#: reaches ``os.path.exists(None)`` and raises TypeError before any
#: reconciliation happens -- which is exactly how this was found.
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
    agent-context-graph is an optional extra -- skip quietly without it.
    """
    import os

    try:
        from agent_context_graph.adapters._identity import resolve_llm_env
    except ImportError:
        return
    for key, value in resolve_llm_env().items():
        if value:
            os.environ.setdefault(key, value)


#: LightRAG's own default (8) re-triggers a paid merge-summary call on *every*
#: later chunk that touches an entity/relation once it has crossed 8 raw
#: mentions -- confirmed by reading operate.py's merge path: the description
#: list handed to the merge is rebuilt from all historical per-chunk mentions
#: (capped at max_source_ids_per_relation/entity, default 200) on every merge
#: event, not just the first. In a batch-wide eval workspace shared across
#: thousands of sessions, any recurring entity (the user's own name, a
#: recurring topic) keeps re-paying this cost for the rest of the run. Raised
#: well past what most entities in a single eval batch will realistically
#: reach, so the (already-existing, always-on) LLM merge only fires for the
#: genuinely hot few -- at the cost of a plain concatenation instead of an
#: LLM-written summary for the entities below this line.
_EVAL_FORCE_LLM_SUMMARY_ON_MERGE = "30"


def _resolve_reconciliation_tuning() -> None:
    """Set eval-scoped LightRAG cost knobs, without touching anyone else's default.

    LightRAG reads ``FORCE_LLM_SUMMARY_ON_MERGE`` once, as a dataclass field
    default evaluated when ``lightrag.lightrag`` is first imported -- so this
    must run before that happens (reconciliation's own ``unstructured2graph``/
    ``lightrag`` imports are lazy, deferred until reconciliation actually
    starts, so calling this from ``run``'s setup is early enough).

    ``setdefault`` only: an operator's own exported value, or production
    sessions-graph reconciliation (which never sets this at all and keeps
    LightRAG's default of 8), are both left alone.
    """
    import os

    os.environ.setdefault("FORCE_LLM_SUMMARY_ON_MERGE", _EVAL_FORCE_LLM_SUMMARY_ON_MERGE)


#: `MemgraphLightRAGWrapper`'s own default (all-MiniLM-L6-v2, max_token_size
#: 256) is what turned reconciliation's session-batching (map #297) into a
#: much smaller win than it should be: LightRAG re-splits any chunk down to
#: the embedder's own max_token_size *before* embedding, and extraction runs
#: on those re-split pieces -- so 256, not LightRAG's larger CHUNK_SIZE, was
#: the real ceiling. BAAI/bge-m3 raises that ceiling to 8192 tokens, above
#: all but the largest sessions in the eval corpus (max observed: ~17k
#: tokens), while staying local -- confirmed directly against the running
#: eval Memgraph instance: dimension=1024, max_sequence_length=8192 (via
#: `embeddings.model_info()`), no external cost. Measured effect on the real
#: corpus: extraction+gleaning chunks drop from 8.77/session (the current
#: 256-token ceiling) to ~1.00/session.
_EVAL_EMBEDDING_MODEL = "BAAI/bge-m3"
_EVAL_EMBEDDING_DIM = 1024
_EVAL_EMBEDDING_MAX_TOKENS = 8192


def _eval_embedding_func() -> Any:
    """The eval-scoped embedding function, built fresh so nothing outside
    context-graph-eval shares or is affected by this override.

    Passed explicitly to ``MemgraphLightRAGWrapper.initialize()`` rather than
    changed as lightrag-memgraph's own default: bge-m3 is a much heavier
    model (~568M params vs all-MiniLM's ~22M) with a different vector
    dimension, and production sessions-graph reconciliation never asked for
    that trade -- this stays scoped to the eval batches deciding whether it's
    worth making the default everywhere.
    """
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


async def reconcile_batch(
    db: "Memgraph",
    *,
    limit: int | None = None,
    memgraph_url: str | None = None,
    working_dir: str = DEFAULT_WORKING_DIR,
    lightrag_wrapper: Any = None,
    progress: bool = True,
) -> Reconciled:
    """Reconcile pending sessions in the eval graph.

    Returns counts rather than raising on the first failure: one session that
    cannot be distilled should not abandon the rest of a batch, and a caller
    needs to know how much of the graph is actually populated before trusting a
    score computed against it.

    No LightRAG wrapper is constructed when nothing is pending, so an empty
    batch costs nothing and needs no LLM credentials.

    ``memgraph_url`` must name the same instance ``db`` is connected to, and is
    not optional in practice. LightRAG's Memgraph storage backends resolve their
    connection from the **environment**, not from the client passed in here --
    they raise outright if ``MEMGRAPH_URL`` is unset, and worse, if it is set to
    something else they will happily write reconciliation output to *that*
    graph instead. Left unset while an ambient ``MEMGRAPH_URL`` points at a
    development instance, an eval batch would distil straight into it: the exact
    pollution #309's dedicated-instance decision exists to prevent, with nothing
    to indicate it happened.
    """
    import os

    from sessions_graph import SessionsGraph

    session_ids = pending_sessions(db, limit=limit)
    if not session_ids:
        return Reconciled(reconciled=0, failed=0)

    _resolve_llm_credentials()

    owns_wrapper = lightrag_wrapper is None

    if memgraph_url:
        # Set, not defaulted: this is what LightRAG's stores actually follow.
        os.environ["MEMGRAPH_URL"] = memgraph_url
        os.environ.setdefault("MEMGRAPH_USER", "")
        os.environ.setdefault("MEMGRAPH_PASSWORD", "")
        os.environ.setdefault("MEMGRAPH_DATABASE", "memgraph")
    elif owns_wrapper and not os.environ.get("MEMGRAPH_URL"):
        # Only when we build the wrapper ourselves. A caller supplying one has
        # already decided where its stores point, and demanding the variable
        # anyway would refuse to run a batch that needs no LightRAG at all.
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

    reconciled = 0
    errors: list[str] = []
    try:
        for index, session_id in enumerate(session_ids, start=1):
            summary = await graph.reconcile_session(
                session_id,
                lightrag_wrapper=lightrag_wrapper,
                enforce_ontology=True,
            )
            if summary.status == "completed":
                reconciled += 1
            else:
                errors.append(f"{session_id}: {summary.error}")

            # Printed per session, because this loop is sequential and each
            # session costs two LLM calls -- so a modest batch runs for many
            # minutes. Silent until done is indistinguishable from hung, which
            # is how two runs were abandoned without knowing whether they were
            # progressing.
            if progress:
                print(f"  reconciled {index}/{len(session_ids)} ({reconciled} ok, {len(errors)} failed)", flush=True)
    finally:
        if owns_wrapper:
            finalize = getattr(lightrag_wrapper, "finalize", None)
            if finalize is not None:
                await finalize()

    return Reconciled(reconciled=reconciled, failed=len(errors), errors=tuple(errors))
