"""Hybrid retrieval: the product's own recall, answered by the eval's answerer.

Retrieval is ``sessions_graph.recall`` -- the code the read path ships --
so this strategy benchmarks it directly: turns found by vector and text
search, the typed graph's facts and the turns they lead to, all scoped to
the asking user. Only the answer is the eval's own: the rows go through the
same ``answer_prompt`` every strategy uses.

Recall reads vectors stored in Memgraph (``sessions_graph.embeddings``), so
:func:`ensure_recall_ready` embeds whatever the graph is missing before the
first question, and fails the run when Memgraph can't embed: without the
vector lanes every question would be answered from text search alone and
the run would report that as the strategy's score.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Protocol

from neo4j.exceptions import TransientError
from sessions_graph import RecallConfig, SessionsGraph
from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL
from sessions_graph.recall import LANES, recall

from .retrieval import Retrieved, answer_prompt

if TYPE_CHECKING:
    from memgraph_toolbox.api.memgraph import Memgraph

    from .retrieval import LLM, ReadOnlyGraph

__all__ = ["LANES", "RecallConfig", "ensure_recall_ready", "retrieve_hybrid"]


class _Setup(Protocol):
    def setup(self) -> None: ...


def _setup_with_retry(graph: _Setup, *, attempts: int = 6, wait: float = 20.0) -> None:
    """``graph.setup()``, retried while Memgraph reports a transient storage error.

    A periodic snapshot holds the storage's unique access, and a DDL statement
    arriving then fails with ``TransientError``. On the full LongMemEval-S run
    that landed on the first statement after a 7.7 hour build and discarded it.
    """
    for attempt in range(1, attempts + 1):
        try:
            graph.setup()
            return
        except TransientError:
            if attempt == attempts:
                raise
            time.sleep(wait)


def ensure_recall_ready(db: Memgraph, *, model: str = DEFAULT_EMBEDDING_MODEL, batch: int = 100) -> int:
    """Create recall's text index and embed every session not yet embedded with ``model``.

    Returns:
        How many sessions were embedded.

    Raises:
        sessions_graph.embeddings.EmbeddingUnavailableError: Memgraph can't
            embed (no MAGE, or the model can't load).
    """
    graph = SessionsGraph(db)
    _setup_with_retry(graph)
    embedded = 0
    while pending := graph.reconciler.pending_embeddings(model=model, limit=batch):
        for session_id in pending:
            graph.reconciler.embed(session_id, model=model)
        embedded += len(pending)
    return embedded


async def retrieve_hybrid(
    question: str,
    *,
    graph: ReadOnlyGraph,
    llm: LLM,
    user_id: str,
    config: RecallConfig | None = None,
    today: str | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> Retrieved:
    """Answer ``question`` from what ``sessions_graph.recall`` finds in ``user_id``'s history."""
    started = time.monotonic()
    # recall is several blocking Memgraph round-trips; a worker thread lets
    # concurrent questions overlap them. The neo4j driver is thread-safe and
    # ReadOnlyGraph holds no per-call state.
    recall_started = time.monotonic()
    recalled = await asyncio.to_thread(recall, graph, user_id, question, config=config, model=model)
    recall_seconds = time.monotonic() - recall_started
    seen = recalled.lines()
    answer = await llm.complete(answer_prompt(question, seen, today))
    errors = [f"vector lanes off: {recalled.vector_lanes_off}"] if recalled.vector_lanes_off else []
    return Retrieved(
        answer=answer.strip(),
        retrieval_context=seen,
        errors=errors,
        latency_seconds=time.monotonic() - started,
        recall_seconds=recall_seconds,
    )
