"""Answer a question the way the product does: ``sessions_graph.recall``, then the eval's answering model.

Retrieval is the read path the product ships, scoped to the asking user:
turns found by vector and text search, the typed graph's facts and the turns
they lead to. Only the answer is the eval's own -- every benchmark answers
through the same ``answer_prompt``, so a score difference is down to what
recall found.

Recall reads vectors stored in Memgraph (``sessions_graph.embeddings``), so
:func:`ensure_recall_ready` embeds whatever the graph is missing before the
first question, and fails the run when Memgraph can't embed: without the
vector lanes every question would be answered from text search alone and
the run would report that as recall's score.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from neo4j.exceptions import TransientError
from sessions_graph import RecallConfig, SessionsGraph
from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL
from sessions_graph.recall import LANES, recall

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from memgraph_toolbox.api.memgraph import Memgraph

__all__ = [
    "LANES",
    "LLM",
    "DeepEvalLLM",
    "ReadOnlyGraph",
    "RecallConfig",
    "Retrieved",
    "answer_all",
    "answer_prompt",
    "ensure_recall_ready",
    "retrieve",
    "setup_with_retry",
]

T = TypeVar("T")


class LLM(Protocol):
    """Minimal completion interface, so answering can be exercised with a stub."""

    async def complete(self, prompt: str) -> str: ...


class DeepEvalLLM:
    """Adapts a deepeval model, the answering model under test, to the :class:`LLM` protocol."""

    def __init__(self, model: Any):
        self._model = model

    async def complete(self, prompt: str) -> str:
        result = await self._model.a_generate(prompt)
        # Some deepeval models return (text, cost).
        if isinstance(result, tuple):
            result = result[0]
        return str(result)


#: Mirrors the guard in `mcp_memgraph.servers.server.is_write_query`. Duplicated
#: rather than imported: that module binds a client registry to global env
#: config at import time, which would point retrieval at whatever Memgraph the
#: environment names instead of the eval instance.
_WRITE_PATTERNS = (
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bCALL\b.*\bapoc\.(create|merge|refactor)\b",
)


class WriteRefusedError(RuntimeError):
    """Raised when retrieval attempts to modify the graph under test."""


def is_write_query(cypher: str) -> bool:
    """Whether ``cypher`` matches any write pattern, case-insensitively (the apoc pattern is lowercase)."""
    return any(re.search(pattern, cypher, re.IGNORECASE) for pattern in _WRITE_PATTERNS)


class ReadOnlyGraph:
    """Read-only Cypher access to the graph being scored.

    Retrieval must not be able to alter what it is measured against -- the
    same reasoning that keeps the corpus in git rather than in Memgraph (#302).
    """

    def __init__(self, db: Memgraph):
        self._db = db

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if is_write_query(cypher):
            raise WriteRefusedError(f"retrieval may not write to the graph under test: {cypher!r}")
        return self._db.query(cypher, params)


@dataclass(frozen=True)
class Retrieved:
    """What recall found for one question, and the answer built from it."""

    answer: str
    #: Recall's rendered lines: what is scored and token-counted (#309).
    retrieval_context: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Wall-clock seconds for recall plus the answering call.
    latency_seconds: float = 0.0
    #: Wall-clock seconds of recall alone -- embedding the question and the
    #: Memgraph queries, no answering model. None when recall raised.
    recall_seconds: float | None = None


def answer_prompt(question: str, seen: list[str], today: str | None = None) -> str:
    """The final-answer prompt every benchmark shares, so a quality difference is down to what was retrieved."""
    rows = "\n".join(seen[:200]) if seen else "(nothing was retrieved)"
    asked = f"The question is being asked on {today}.\n" if today else ""
    return (
        "Answer the user's question from their memory: the rows below, retrieved from their past "
        "conversations. Each row carries the date it was said or became true.\n"
        '- "Today", "yesterday", "last week" inside a row are relative to that row\'s date, not to now.\n'
        "- For a count, a total, or the time between events: list each matching item or value with its "
        "date, count an item mentioned in several rows once, then compute.\n"
        "- When rows disagree about the same thing, the most recent one is current.\n"
        "- When the question asks for a recommendation or suggestion, always recommend: use the "
        "preferences, interests and possessions the rows show, add your own knowledge where they name "
        'nothing specific, and say which preferences you used. Never answer one with "not in memory".\n'
        "- Any other question about something the rows never mention, or that assumes something they do "
        'not show: say exactly "not in memory" -- do not answer a related question instead.\n'
        '- Otherwise, if the rows do not contain the answer, say exactly "not in memory".\n'
        "Keep the working short and put the final answer last.\n\n"
        f"Rows:\n{rows}\n\n{asked}Question: {question}\nAnswer:"
    )


class _Setup(Protocol):
    def setup(self) -> None: ...


def setup_with_retry(graph: _Setup, *, attempts: int = 6, wait: float = 20.0) -> None:
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
    setup_with_retry(graph)
    embedded = 0
    while pending := graph.reconciler.pending_embeddings(model=model, limit=batch):
        for session_id in pending:
            graph.reconciler.embed(session_id, model=model)
        embedded += len(pending)
    return embedded


async def retrieve(
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
    recalled = await asyncio.to_thread(recall, graph, user_id, question, config=config, model=model)
    recall_seconds = time.monotonic() - started
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


async def answer_all(
    items: Sequence[T], answer_one: Callable[[T], Awaitable[Retrieved]], *, max_concurrent: int
) -> list[Retrieved]:
    """Run ``answer_one`` over ``items``, at most ``max_concurrent`` at once, in input order.

    An item whose retrieval raises becomes an empty, timed result -- a
    reported miss rather than a silently shorter run.
    """
    limiter = asyncio.Semaphore(max_concurrent)

    async def one(item: T) -> Retrieved:
        async with limiter:
            started = time.monotonic()
            try:
                return await answer_one(item)
            except Exception as exc:
                return Retrieved(answer="", errors=[str(exc)], latency_seconds=time.monotonic() - started)

    return list(await asyncio.gather(*(one(item) for item in items)))
