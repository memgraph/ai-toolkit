"""The text-search baseline: Memgraph's own full-text index over raw turns.

Where ``retrieval.py`` asks "how well does the *memory* pipeline (reconciled
graph, agent-written Cypher) answer this question", this module asks the
cheaper question a real harness would ask first: "how well does just
full-text-searching the raw session transcript do, with no distillation at
all?" Reconciliation is the dominant cost of a run (see ``RunPlan.reconcile``'s
docstring) -- this baseline skips it entirely, so it is a lower bound on cost
as much as it is a comparison point on quality.

Nothing here is retrieval strategy in the sense ``retrieval.py`` defers (#300):
there is no ranking beyond Memgraph's own text-index score, no query
expansion, no chunking. That is deliberate -- the question this baseline
answers is "what does the *existing*, zero-effort tool give you", not "what is
the best possible text-search baseline".
"""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .retrieval import Retrieved, answer_prompt

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from actions_graph import ActionsGraph

    from .retrieval import LLM, ReadOnlyGraph

#: One index, scoped to the eval instance only -- never the real harness's
#: Action nodes, which this package must never touch (see inject.py's module
#: docstring on why the eval instance is dedicated).
TEXT_INDEX_NAME = "eval_turn_text_index"

#: How many matching turns to hand the answering LLM. Unranked beyond
#: Memgraph's own text-search score, deliberately: tuning this number is
#: retrieval-strategy design, which #300's reasoning defers for the graph-agent
#: baseline too -- this baseline's whole point is the untuned, zero-effort
#: number.
DEFAULT_LIMIT = 10


@dataclass(frozen=True)
class Indexed:
    """What indexing wrote."""

    turns: int


def ensure_turn_text_index(graph: "ActionsGraph") -> Indexed:
    """Materialize a plain-text ``text`` property on every turn, then index it.

    Turn text lives inside ``Action.properties`` as a JSON string (see
    ``retrieval.graph_schema``'s docstring) -- Memgraph has no APOC, so that
    JSON cannot be unpacked in Cypher. Extracted in Python instead: read every
    ``Action``'s ``properties``, and for the ones that carry a ``content``
    string (turns -- not every Action does; a tool call's properties has no
    such key) write it back as a plain string property a text index can
    actually index.

    ``CREATE TEXT INDEX`` on an index name that already exists is a verified
    no-op (checked directly against a live instance), and the index stays
    correct across ``inject_batch``'s wipe-and-reload (also verified directly:
    old nodes gone, new nodes' content searchable, nothing stale) -- so this
    runs unconditionally every batch rather than checking first.
    """
    db = graph.db
    rows = db.query("MATCH (a:Action) WHERE a.text IS NULL RETURN a.action_id AS action_id, a.properties AS properties")

    materialized = []
    for row in rows:
        try:
            content = json.loads(row["properties"] or "{}").get("content")
        except ValueError:
            content = None
        if isinstance(content, str) and content:
            materialized.append({"action_id": row["action_id"], "text": content})

    if materialized:
        db.query(
            "UNWIND $rows AS row MATCH (a:Action {action_id: row.action_id}) SET a.text = row.text",
            {"rows": materialized},
        )

    db.query(f"CREATE TEXT INDEX {TEXT_INDEX_NAME} ON :Action(text);")
    return Indexed(turns=len(materialized))


#: Tantivy's query parser treats characters like : ( ) " * ~ ^ specially, and a
#: natural-language question is full of the ones that are not (mostly '?').
#: Reduced to bare word tokens rather than escaped: this is making the query
#: parseable, not a search-quality choice -- #300 defers those.
_QUERY_TOKEN = re.compile(r"\w+")

#: text_search.search_all is OR-across-terms, not AND (verified directly): a
#: turn only needs to share SOME query word to score, not all of them. These
#: are near-universal in English regardless of topic, so keeping them in the
#: query does not add precision -- it adds a term with almost no discriminating
#: power that can still contribute a nonzero score. Verified directly: on a
#: two-document corpus, a query sharing only "my" with an unrelated turn
#: outranked the turn actually containing the query's real keyword (documented
#: in the README's Known Limitations). Dropping them is not the search-strategy
#: tuning #300 defers -- it is the same kind of thing as stripping Tantivy's
#: special characters above: removing terms that were never going to
#: discriminate rather than adding any ranking sophistication.
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
) -> Retrieved:
    """Answer ``question`` from Memgraph's own full-text search over raw turns.

    The final-answer call is the exact same ``answer_prompt`` the graph-agent
    baseline uses (see ``retrieval.retrieve``) -- deliberately, so a quality
    difference between the two baselines is attributable to what was
    *retrieved*, not to two different answering prompts.
    """
    query = _safe_query(question)
    seen: list[str] = []
    errors: list[str] = []

    if query:
        try:
            rows = graph.query(
                f"CALL text_search.search_all('{TEXT_INDEX_NAME}', $query) YIELD node, score "
                "WITH node, score ORDER BY score DESC LIMIT $limit "
                "OPTIONAL MATCH (s:Session)-[:HAS_ACTION]->(node) "
                "RETURN s.session_id AS session_id, node.text AS content, score",
                {"query": query, "limit": limit},
            )
        except Exception as exc:
            rows = []
            errors.append(f"text_search.search_all({query!r}): {exc}")
        seen = [f"session={row['session_id']} content={row['content']}" for row in rows]
        if not rows:
            errors.append(f"text_search.search_all({query!r}): returned 0 rows")

    answer = await llm.complete(answer_prompt(question, seen))
    return Retrieved(
        answer=answer.strip(),
        retrieval_context=seen,
        queries=[query] if query else [],
        errors=errors,
    )
