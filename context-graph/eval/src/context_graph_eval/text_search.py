"""The text-search baseline: Memgraph's own full-text index over raw, unreconciled turns.

Deliberately untuned -- no ranking beyond the index's score, no query
expansion: it measures what the zero-effort tool gives, at no reconciliation cost.
"""

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .retrieval import Retrieved, answer_prompt

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from actions_graph import ActionsGraph

    from .retrieval import LLM, ReadOnlyGraph


class _Queryable(Protocol):
    """Anything that runs parameterized Cypher: a raw client or a ReadOnlyGraph."""

    def query(self, cypher: str, params: dict[str, Any] | None = None, /) -> list[dict[str, Any]]: ...


TEXT_INDEX_NAME = "eval_turn_text_index"

#: How many matching turns to hand the answering LLM. Deliberately untuned.
DEFAULT_LIMIT = 10


@dataclass(frozen=True)
class Indexed:
    """What indexing covered."""

    turns: int


def ensure_turn_text_index(graph: "ActionsGraph") -> Indexed:
    """Index every turn's ``text``, and prove search runs.

    ``actions-graph`` writes ``text`` on user and assistant messages when it
    records them, so tool calls and other actions are never indexed.

    Ends with a probe through the same :func:`_search` call retrieval makes,
    so a broken index fails the run rather than reading as 0% coverage.

    Raises:
        RuntimeError: the index is missing after creation, or text search
            rejects the query retrieval would send.
    """
    db = graph.db
    turns = db.query("MATCH (a:Action) WHERE a.text IS NOT NULL RETURN count(a) AS n")[0]["n"]

    if not _index_exists(db):
        db.query(f"CREATE TEXT INDEX {TEXT_INDEX_NAME} ON :Action(text);")
    if not _index_exists(db):
        raise RuntimeError(f"text index {TEXT_INDEX_NAME} is missing after CREATE TEXT INDEX")

    try:
        _search(db, "probe", limit=1)
    except Exception as exc:
        raise RuntimeError(f"text search cannot run on this Memgraph: {exc}") from exc
    return Indexed(turns=turns)


def _index_exists(db: _Queryable) -> bool:
    marker = f"(name: {TEXT_INDEX_NAME})"
    return any(marker in str(row.get("index type", "")) for row in db.query("SHOW INDEX INFO"))


def _search(graph: _Queryable, query: str, limit: int) -> list[dict[str, Any]]:
    # The index name is a module constant, never input; query and limit are
    # parameters. The limit is applied in Cypher because search_all's third
    # argument changed type across Memgraph releases.
    return graph.query(
        f"CALL text_search.search_all('{TEXT_INDEX_NAME}', $query) YIELD node, score "
        "WITH node, score ORDER BY score DESC LIMIT $limit "
        "OPTIONAL MATCH (s:Session)-[:HAS_ACTION]->(node) "
        "RETURN s.session_id AS session_id, node.text AS content, score",
        {"query": query, "limit": limit},
    )


#: Bare word tokens, so Tantivy's query parser never sees its special characters.
_QUERY_TOKEN = re.compile(r"\w+")

#: search_all ORs its terms, so a near-universal word alone can make an
#: unrelated turn outrank the one holding the real keyword.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "i",
        "me",
        "my",
        "you",
        "your",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "as",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "to",
        "with",
        "and",
        "or",
        "but",
    ]
)


def _safe_query(question: str) -> str:
    tokens = [t for t in _QUERY_TOKEN.findall(question) if t.lower() not in _STOPWORDS]
    return " ".join(tokens)


async def retrieve_by_text_search(
    question: str,
    *,
    graph: "ReadOnlyGraph",
    llm: "LLM",
    limit: int = DEFAULT_LIMIT,
    today: str | None = None,
) -> Retrieved:
    """Answer ``question`` from Memgraph's full-text search over raw turns, via the shared ``answer_prompt``.

    Raises:
        Exception: whatever ``text_search`` raises; the runner records it as a miss.
    """
    started = time.monotonic()
    query = _safe_query(question)
    rows = _search(graph, query, limit) if query else []
    seen = [f"session={row['session_id']} content={row['content']}" for row in rows]
    errors = [f"text_search.search_all({query!r}): returned 0 rows"] if query and not rows else []

    answer = await llm.complete(answer_prompt(question, seen, today))
    return Retrieved(
        answer=answer.strip(),
        retrieval_context=seen,
        queries=[query] if query else [],
        errors=errors,
        latency_seconds=time.monotonic() - started,
    )
