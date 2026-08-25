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
    working_dir: str | None = None,
    lightrag_wrapper: Any = None,
) -> Reconciled:
    """Reconcile pending sessions in the eval graph.

    Returns counts rather than raising on the first failure: one session that
    cannot be distilled should not abandon the rest of a batch, and a caller
    needs to know how much of the graph is actually populated before trusting a
    score computed against it.

    No LightRAG wrapper is constructed when nothing is pending, so an empty
    batch costs nothing and needs no LLM credentials.
    """
    from sessions_graph import SessionsGraph

    session_ids = pending_sessions(db, limit=limit)
    if not session_ids:
        return Reconciled(reconciled=0, failed=0)

    _resolve_llm_credentials()

    graph = SessionsGraph(memgraph=db)
    graph.setup()

    owns_wrapper = lightrag_wrapper is None
    if owns_wrapper:
        from lightrag_memgraph import MemgraphLightRAGWrapper

        lightrag_wrapper = MemgraphLightRAGWrapper()
        await lightrag_wrapper.initialize(working_dir=working_dir)

    reconciled = 0
    errors: list[str] = []
    try:
        for session_id in session_ids:
            summary = await graph.reconcile_session(
                session_id,
                lightrag_wrapper=lightrag_wrapper,
                enforce_ontology=True,
            )
            if summary.status == "completed":
                reconciled += 1
            else:
                errors.append(f"{session_id}: {summary.error}")
    finally:
        if owns_wrapper:
            finalize = getattr(lightrag_wrapper, "finalize", None)
            if finalize is not None:
                await finalize()

    return Reconciled(reconciled=reconciled, failed=len(errors), errors=tuple(errors))
