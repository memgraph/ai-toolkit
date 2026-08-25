"""The v1 retrieval baseline: an agent writing its own Cypher.

#300 fixed this deliberately as the *existing* query surface rather than a
purpose-built retrieval layer. Ranking, query templates, and vector search are
all deferred until this baseline's failures say what they should actually be --
building them first would be designing against a score nobody has seen.

What lives here is plumbing, not retrieval strategy: execute read-only Cypher,
describe the graph's shape, and loop an LLM over the two.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from memgraph_toolbox.api.memgraph import Memgraph

#: Mirrors the guard in `mcp_memgraph.servers.server.is_write_query`. Duplicated
#: rather than imported: that module binds a client registry to global env
#: config at import time, which would point retrieval at whatever Memgraph the
#: environment names instead of the dedicated eval instance. Kept in sync by
#: hand -- if that guard gains a pattern, this needs it too.
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

_CYPHER_FENCE = re.compile(r"```(?:cypher)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)

#: How many query/observe rounds an agent gets before it must answer. Bounded
#: because retrieval cost is itself scored (#309): an agent allowed to query
#: indefinitely could buy coverage with an unbounded payload.
DEFAULT_MAX_STEPS = 4


class WriteRefusedError(RuntimeError):
    """Raised when retrieval attempts to modify the graph under test."""


class LLM(Protocol):
    """Minimal completion interface, so the loop can be exercised with a stub."""

    async def complete(self, prompt: str) -> str: ...


class DeepEvalLLM:
    """Adapts a deepeval model to the :class:`LLM` protocol.

    Reuses deepeval's provider wrappers rather than adding an API client, so the
    retrieval agent and the judge (#304) are configured the same way.

    Note this is the *agent under test*, not the judge -- #304 deliberately runs
    the judge on a different provider from the pipeline so their blind spots do
    not correlate. Keep them distinct when configuring a run.
    """

    def __init__(self, model: Any):
        self._model = model

    async def complete(self, prompt: str) -> str:
        result = await self._model.a_generate(prompt)
        # Some deepeval models return (text, cost).
        if isinstance(result, tuple):
            result = result[0]
        return str(result)


class ReadOnlyGraph:
    """Read-only Cypher access to the graph being scored.

    Retrieval must not be able to alter what it is measured against -- the same
    reasoning that keeps the corpus in git rather than in Memgraph (#302).
    """

    def __init__(self, db: "Memgraph"):
        self._db = db

    def query(self, cypher: str) -> list[dict[str, Any]]:
        if is_write_query(cypher):
            raise WriteRefusedError(f"retrieval may not write to the graph under test: {cypher!r}")
        return self._db.query(cypher)


def is_write_query(cypher: str) -> bool:
    upper = cypher.upper()
    return any(re.search(pattern, upper) for pattern in _WRITE_PATTERNS)


def graph_schema(graph: ReadOnlyGraph) -> str:
    """A compact description of what is in the graph.

    The agent writes its own Cypher, so without this it can only guess at the
    labels and relationship types available.
    """
    labels = graph.query("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label ORDER BY label")
    rel_types = graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS type ORDER BY type")
    lines = [
        "Node labels: " + ", ".join(row["label"] for row in labels),
        "Relationship types: " + ", ".join(row["type"] for row in rel_types),
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class Retrieved:
    """What retrieval produced for one question."""

    answer: str
    #: Rows the graph actually returned, rendered as text. This -- not the
    #: model's prose about it -- is what ContextualRecall scores and what the
    #: efficiency metric counts tokens over (#309).
    retrieval_context: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def retrieve(
    question: str,
    *,
    graph: ReadOnlyGraph,
    llm: LLM,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Retrieved:
    """Answer ``question`` by letting ``llm`` query ``graph``.

    A failed query is recorded and the loop continues: one malformed Cypher
    statement should cost that question its answer, not abort a whole batch
    mid-run.
    """
    schema = graph_schema(graph)
    seen: list[str] = []
    queries: list[str] = []
    errors: list[str] = []

    for _ in range(max_steps):
        reply = await llm.complete(_query_prompt(question, schema, seen, errors))
        cypher = _extract_cypher(reply)
        if cypher is None:
            break

        queries.append(cypher)
        try:
            rows = graph.query(cypher)
        except Exception as exc:
            errors.append(f"{cypher}: {exc}")
            continue
        seen.extend(_render(row) for row in rows)

    answer = await llm.complete(_answer_prompt(question, seen))
    return Retrieved(answer=answer.strip(), retrieval_context=seen, queries=queries, errors=errors)


def _extract_cypher(reply: str) -> str | None:
    """Pull a Cypher statement out of a model reply, fenced or bare."""
    fenced = _CYPHER_FENCE.search(reply)
    candidate = (fenced.group(1) if fenced else reply).strip()
    # A reply with no clause keyword is the model talking, not querying.
    if not re.search(r"\b(MATCH|RETURN|WITH|UNWIND|CALL)\b", candidate, re.IGNORECASE):
        return None
    return candidate


def _render(row: dict[str, Any]) -> str:
    return " | ".join(f"{key}={value}" for key, value in row.items())


def _query_prompt(question: str, schema: str, seen: list[str], errors: list[str]) -> str:
    parts = [
        "You are answering a question using only a Memgraph graph database.",
        "Write ONE read-only Cypher query to gather what you still need.",
        "Return only the query. Reply with prose instead if you already have enough.",
        "",
        f"Graph schema:\n{schema}",
        "",
        f"Question: {question}",
    ]
    if seen:
        parts += ["", "Rows so far:", *seen[:50]]
    if errors:
        parts += ["", "Queries that failed (do not repeat them):", *errors]
    return "\n".join(parts)


def _answer_prompt(question: str, seen: list[str]) -> str:
    rows = "\n".join(seen[:200]) if seen else "(nothing was retrieved)"
    return (
        "Answer the question using only the rows below. Be concise. "
        'If the rows do not contain the answer, say exactly "not in memory".\n\n'
        f"Rows:\n{rows}\n\nQuestion: {question}\nAnswer:"
    )
