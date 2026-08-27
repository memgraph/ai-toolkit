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

#: Ceiling on the payload handed back for one question.
#:
#: Efficiency is a scored axis (#309), so an unbounded payload is not merely
#: untidy -- and it is not hypothetical: measured at scale, the median payload
#: reached ~19k tokens and one question returned 1,067,650. Beyond the score, a
#: payload that size risks exhausting the judge's context window and costs real
#: money per question.
#:
#: Set above the observed median rather than below it: the job here is to clip
#: the pathological tail, not to squeeze the typical case, which belongs to
#: retrieval v2 and should be driven by measurement rather than a guess.
MAX_PAYLOAD_TOKENS = 20_000

#: Conservative characters-per-token ratio for the cheap size estimate in
#: :func:`_fit_payload`. English averages nearer 4; under-estimating keeps the
#: real payload below the cap rather than just around it.
_CHARS_PER_TOKEN = 3


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
    # Case-insensitive matching rather than upper-casing the query first. That
    # earlier form silently disabled any pattern containing lowercase letters:
    # the apoc rule could never fire against an upper-cased string, so
    # `CALL apoc.refactor.rename.label(...)` passed the guard -- `refactor` is
    # matched by no other pattern.
    return any(re.search(pattern, cypher, re.IGNORECASE) for pattern in _WRITE_PATTERNS)


def graph_schema(graph: ReadOnlyGraph) -> str:
    """A description of what is in the graph, for the agent to write Cypher against.

    Uses ``memgraph_toolbox``'s ``SearchSchemaTool`` -- the same implementation
    the MCP server's ``search_schema`` tool wraps -- so the baseline really is
    the existing query surface (#300) rather than a reimplementation of it,
    while still being pointed at the dedicated eval instance.

    Property-level detail matters here beyond tidiness: injected turn text lives
    inside ``Action.properties`` as JSON rather than in a ``content`` property,
    so an agent given only label names would have to guess that.

    Falls back to labels and relationship types when Memgraph was started
    without ``--schema-info-enabled``, so a contributor whose instance lacks the
    flag gets a weaker prompt rather than a crash.
    """
    detailed = _detailed_schema(graph)
    if detailed:
        return detailed

    labels = graph.query("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label ORDER BY label")
    rel_types = graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS type ORDER BY type")
    return "\n".join(
        [
            "Node labels: " + ", ".join(row["label"] for row in labels),
            "Relationship types: " + ", ".join(row["type"] for row in rel_types),
            "(property-level schema unavailable: start Memgraph with --schema-info-enabled)",
        ]
    )


#: A property is treated as *schema* -- and its values shown -- only when it has
#: few distinct values AND those values are short. Cardinality alone is not
#: safe: in a small graph a free-text field can easily have only two distinct
#: values, and printing those would put the content being asked about straight
#: into the agent's prompt. Length is what separates an enum from a sentence.
_MAX_DISTINCT_VALUES = 6
_MAX_SCHEMA_VALUE_LENGTH = 40

#: Bounds the introspection itself, so describing the graph cannot become more
#: expensive than querying it.
_MAX_LABELS = 25


def _detailed_schema(graph: ReadOnlyGraph) -> str | None:
    """Property-level schema, or None when Memgraph has schema info disabled."""
    try:
        from memgraph_toolbox.tools.schema import SearchSchemaTool

        rows = SearchSchemaTool(db=graph._db).call({"pattern": ".*"})
    except Exception:
        return None

    if not rows or any("error" in row for row in rows if isinstance(row, dict)):
        return None

    lines = ["Node labels and their properties:"]
    for labels in _label_sets(graph):
        lines.append(f"  (:{':'.join(labels)})")
        lines.extend(_describe_properties(graph, labels))

    edges = [row for row in rows if isinstance(row, dict) and row.get("type") == "edge-match"]
    if edges:
        lines.append("")
        lines.append("Relationships:")
        for edge in edges:
            start = ":".join(edge.get("start_labels") or ["?"])
            end = ":".join(edge.get("end_labels") or ["?"])
            lines.append(f"  (:{start})-[:{edge.get('edge_type')}]->(:{end})")
    return "\n".join(lines)


def _label_sets(graph: ReadOnlyGraph) -> list[list[str]]:
    """Distinct label combinations present in the graph.

    Combinations, not individual labels: a node carrying ``:Action:Message`` is
    one thing with one property set, and splitting it would suggest two.
    """
    # Sorted in Python: Memgraph refuses ORDER BY on a list value.
    rows = graph.query(f"MATCH (n) WITH labels(n) AS labels RETURN DISTINCT labels LIMIT {_MAX_LABELS}")
    return sorted((row["labels"] for row in rows if row["labels"]), key=lambda labels: sorted(labels))


def _describe_properties(graph: ReadOnlyGraph, labels: list[str]) -> list[str]:
    """One line per property: its values when they are schema, its shape when not."""
    match = f"MATCH (n:{':'.join(labels)})"
    keys = graph.query(f"{match} UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key")

    described: list[str] = []
    for row in keys:
        key = row["key"]
        values = graph.query(
            f"{match} WHERE n.`{key}` IS NOT NULL RETURN DISTINCT n.`{key}` AS value LIMIT {_MAX_DISTINCT_VALUES + 1}"
        )
        sample = [v["value"] for v in values]

        json_keys = _json_keys(sample)
        if json_keys:
            # The blob that made this whole change necessary: turn text lives
            # inside Action.properties as JSON, so an agent told only that
            # "properties" exists still cannot find any content. Inner keys are
            # structure and safe; inner values are data and stay hidden.
            described.append(f"    {key}: JSON string, keys: {', '.join(json_keys)} (values are free text)")
            continue

        if _is_enumerable(sample):
            described.append(f"    {key}: {', '.join(sorted(str(v) for v in sample))}")
        else:
            described.append(f"    {key}: free text (search it, do not match it exactly)")
    return described


def _is_enumerable(sample: list) -> bool:
    """Whether these values describe the schema rather than carry its content."""
    if not sample or len(sample) > _MAX_DISTINCT_VALUES:
        return False
    return all(len(str(value)) <= _MAX_SCHEMA_VALUE_LENGTH for value in sample)


def _json_keys(sample: list) -> list[str]:
    """Keys of a JSON-object-valued property, if that is what this is."""
    import json

    for value in sample:
        if not isinstance(value, str) or not value.lstrip().startswith("{"):
            continue
        try:
            parsed = json.loads(value)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return sorted(parsed)
    return []


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
        reply = await llm.complete(_query_prompt(question, schema, seen, errors, insist=not seen))
        cypher = _extract_cypher(reply)
        if cypher is None:
            # A reply with no query means "I have enough" -- but only once
            # something has actually been retrieved. Breaking unconditionally
            # meant an opening line of prose ended the loop before it began:
            # measured at scale, 4 of 20 questions issued zero queries and
            # answered from an empty context, one of them a question whose
            # answer was sitting in the graph untouched.
            if seen:
                break
            continue

        queries.append(cypher)
        try:
            rows = graph.query(cypher)
        except Exception as exc:
            errors.append(f"{cypher}: {exc}")
            continue

        rendered, dropped = _fit_payload(seen, rows)
        if dropped:
            # Recorded, never silent. Truncating without telling the agent
            # would cost coverage for a reason nothing captures -- it would
            # believe it had seen everything its query matched, and stop.
            errors.append(
                f"{cypher}: matched too much -- {dropped} rows dropped at the "
                f"{MAX_PAYLOAD_TOKENS}-token payload cap. Narrow the query or add LIMIT."
            )

        if not rows:
            # A query that is valid Cypher but matches nothing used to be
            # invisible: the loop recorded errors only, so the agent saw no
            # difference between "I have not queried yet" and "my assumption
            # was wrong". Observed against a real model -- it invented an
            # action_type from the question's wording, got zero rows four times
            # running with nothing to contradict it, and concluded the fact was
            # not in memory. Reporting the empty result is what lets it revise.
            errors.append(f"{cypher}: returned 0 rows")
            continue
        seen.extend(rendered)

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


def _fit_payload(seen: list[str], rows: list[dict[str, Any]]) -> tuple[list[str], int]:
    """Rows that fit under the payload cap, and how many were dropped.

    Counted in characters rather than tokens: this runs on every row of every
    query, and an approximation that never under-estimates is worth more here
    than an exact count. ``_CHARS_PER_TOKEN`` is deliberately conservative, so
    the real token payload lands under the cap rather than near it.
    """
    budget = MAX_PAYLOAD_TOKENS * _CHARS_PER_TOKEN - sum(len(row) for row in seen)
    fitted: list[str] = []
    for index, row in enumerate(rows):
        rendered = _render(row)
        if len(rendered) > budget:
            return fitted, len(rows) - index
        budget -= len(rendered)
        fitted.append(rendered)
    return fitted, 0


def _query_prompt(question: str, schema: str, seen: list[str], errors: list[str], *, insist: bool = False) -> str:
    parts = [
        "You are answering a question using only a Memgraph graph database.",
        "Write ONE read-only Cypher query to gather what you still need.",
        (
            # Nothing retrieved yet, so there is nothing to answer from and no
            # legitimate reason to reply in prose.
            "Reply with the query and NOTHING else -- no preamble, no explanation."
            if insist
            # A named exit token, offered up front. "Reply with prose if you
            # have enough" was too weak against a model primed to emit queries:
            # it spent the whole step budget every run, long after the answer
            # was already in hand, and every extra query enlarges the payload
            # the efficiency metric counts (#309).
            else "If the rows below already answer the question, reply with exactly STOP "
            "and nothing else. Otherwise return only the next query."
        ),
        "",
        # Both learned from watching a real model fail against this graph.
        "This is Memgraph, not Neo4j: APOC is NOT available, so JSON-valued "
        "properties cannot be parsed in Cypher. Match substrings inside them "
        "with CONTAINS instead.",
        # The illustration is deliberately drawn from a domain no corpus
        # question touches. An earlier version used the same words as a test
        # question's answer, which put that answer into the prompt -- the agent
        # could then "retrieve" what it had just been told, exactly the leak the
        # schema description is careful to avoid.
        "Stored wording rarely matches the question's wording, so prefer ONE broad "
        "term over several ANDed together: someone asking about a vehicle may have "
        "written 'motorcycle', and CONTAINS 'vehicle' would then find nothing.",
        # The counterweight to the line above. Broadening the match is what made
        # queries land at all, and also what made payloads explode; a LIMIT keeps
        # the first from paying for the second.
        "Always add a LIMIT (50 or fewer). Everything returned counts against a "
        "payload budget, and rows past the cap are dropped rather than shown.",
        "",
        f"Graph schema:\n{schema}",
        "",
        f"Question: {question}",
    ]
    if seen:
        parts += [
            "",
            f"Rows retrieved so far ({len(seen)}). Reply STOP if these answer the question:",
            *seen[:50],
        ]
    if errors:
        parts += [
            "",
            "Queries that failed or matched nothing (do not repeat them, and revise your "
            "assumptions about the schema rather than rephrasing the same idea):",
            *errors,
        ]
    return "\n".join(parts)


def _answer_prompt(question: str, seen: list[str]) -> str:
    rows = "\n".join(seen[:200]) if seen else "(nothing was retrieved)"
    return (
        "Answer the question using only the rows below. Be concise. "
        'If the rows do not contain the answer, say exactly "not in memory".\n\n'
        f"Rows:\n{rows}\n\nQuestion: {question}\nAnswer:"
    )
