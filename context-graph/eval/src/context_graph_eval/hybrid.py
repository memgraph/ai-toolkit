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

import time
from typing import TYPE_CHECKING

from sessions_graph import RecallConfig, SessionsGraph
from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL
from sessions_graph.recall import LANES, recall

from .retrieval import Retrieved, answer_prompt

if TYPE_CHECKING:
    from memgraph_toolbox.api.memgraph import Memgraph

    from .retrieval import LLM, ReadOnlyGraph

__all__ = ["LANES", "RecallConfig", "ensure_recall_ready", "retrieve_hybrid"]


def ensure_recall_ready(db: Memgraph, *, model: str = DEFAULT_EMBEDDING_MODEL, batch: int = 100) -> int:
    """Create recall's text index and embed every session not yet embedded with ``model``.

    Returns:
        How many sessions were embedded.

    Raises:
        sessions_graph.embeddings.EmbeddingUnavailableError: Memgraph can't
            embed (no MAGE, or the model can't load).
    """
    graph = SessionsGraph(db)
    graph.setup()
    embedded = 0
    while pending := graph.get_pending_embedding_sessions(model=model, limit=batch):
        for session_id in pending:
            graph.embed_session(session_id, model=model)
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
    recalled = recall(graph, user_id, question, config=config, model=model)
    seen = recalled.lines()
    answer = await llm.complete(answer_prompt(question, seen, today))
    errors = [f"vector lanes off: {recalled.vector_lanes_off}"] if recalled.vector_lanes_off else []
    return Retrieved(
        answer=answer.strip(), retrieval_context=seen, errors=errors, latency_seconds=time.monotonic() - started
    )
