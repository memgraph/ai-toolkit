"""The graph-agent strategy: an agent writing its own read-only Cypher (#300), plus shared retrieval types.

``Retrieved``, the ``LLM`` protocol, ``ReadOnlyGraph`` and ``answer_prompt``
are shared by every strategy.
"""

import re
import time
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

#: Ceiling on one question's payload: clips the pathological tail (one question
#: once returned ~1M tokens), set above the observed ~19k median.
MAX_PAYLOAD_TOKENS = 20_000

#: Conservative characters-per-token ratio for the cheap size estimate in
#: :func:`_fit_payload`. English averages nearer 4; under-estimating keeps the
#: real payload below the cap rather than just around it.
_CHARS_PER_TOKEN = 3


class QueryRefusedError(RuntimeError):
    """Raised when retrieval attempts a query it is not allowed to run."""


class WriteRefusedError(QueryRefusedError):
    """Raised when retrieval attempts to modify the graph under test."""


class InternalLabelRefusedError(QueryRefusedError):
    """Raised when retrieval reaches into LightRAG's own storage (#321)."""


#: LightRAG persists its KV, vector, doc-status and LLM-response-cache stores
#: into the same graph. Refused rather than hidden: the response cache holds
#: the answers, so querying it is a false pass. Matched by prefix, deliberately
#: not LightRAG's `base` label, which is stamped on the extracted entities.
_INTERNAL_LABEL_PREFIX = "LightRAG"

#: Label positions in Cypher: a colon, optional whitespace, an optional
#: backtick, then the name. Matches labels and relationship types, and not map
#: keys (`{name: 'x'}` puts the identifier BEFORE the colon) -- so a property
#: value like 'database' cannot be mistaken for a label.
_LABEL_IN_CYPHER = re.compile(r":\s*`?([A-Za-z_][A-Za-z_0-9]*)`?")


def is_internal_label(label: str) -> bool:
    """Whether a label belongs to LightRAG's storage rather than the model."""
    return label.startswith(_INTERNAL_LABEL_PREFIX)


def internal_labels_in(cypher: str) -> set[str]:
    """Internal storage labels a query references, if any."""
    return {label for label in _LABEL_IN_CYPHER.findall(cypher) if is_internal_label(label)}


class LLM(Protocol):
    """Minimal completion interface, so the loop can be exercised with a stub."""

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


class ReadOnlyGraph:
    """Read-only Cypher access to the graph being scored.

    Retrieval must not be able to alter what it is measured against -- the same
    reasoning that keeps the corpus in git rather than in Memgraph (#302).
    """

    def __init__(self, db: "Memgraph"):
        self._db = db

    @property
    def db(self) -> "Memgraph":
        """The unguarded client, for tool introspection such as ``SearchSchemaTool`` -- never for retrieval."""
        return self._db

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if is_write_query(cypher):
            raise WriteRefusedError(f"retrieval may not write to the graph under test: {cypher!r}")
        internal = internal_labels_in(cypher)
        if internal:
            raise InternalLabelRefusedError(
                f"retrieval may not query internal storage {sorted(internal)}: these hold "
                "LightRAG's own bookkeeping, including a cache of LLM responses that contains "
                "the answers. Query the model instead (Session, Action, Chunk, and the "
                "entity labels)."
            )
        return self._db.query(cypher, params)


def is_write_query(cypher: str) -> bool:
    """Whether ``cypher`` matches any write pattern, case-insensitively (the apoc pattern is lowercase)."""
    return any(re.search(pattern, cypher, re.IGNORECASE) for pattern in _WRITE_PATTERNS)


def graph_schema(graph: ReadOnlyGraph) -> str:
    """A description of what is in the graph, for the agent to write Cypher against.

    Uses ``memgraph_toolbox``'s ``SearchSchemaTool``, the MCP server's own
    schema tool (#300). Falls back to bare labels and relationship types when
    Memgraph runs without ``--schema-info-enabled``.
    """
    detailed = _detailed_schema(graph)
    if detailed:
        return detailed

    labels = graph.query(
        "MATCH (n) UNWIND labels(n) AS label WITH DISTINCT label "
        f"WHERE NOT label STARTS WITH '{_INTERNAL_LABEL_PREFIX}' RETURN label ORDER BY label"
    )
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
#: Properties that hold content by definition: a turn's message text, an entity's
#: name. Never sampled, however short and few their values are.
_CONTENT_KEYS = frozenset({"text"})

#: Bounds the introspection itself, so describing the graph cannot become more
#: expensive than querying it.
_MAX_LABELS = 25


def _detailed_schema(graph: ReadOnlyGraph) -> str | None:
    """Property-level schema, or None when Memgraph has schema info disabled."""
    try:
        from memgraph_toolbox.tools.schema import SearchSchemaTool

        rows = SearchSchemaTool(db=graph.db).call({"pattern": ".*"})
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
        edge_types = sorted({str(e["edge_type"]) for e in edges if e.get("edge_type")})
        properties = _describe_relationship_properties(graph, edge_types)
        if properties:
            lines.append("")
            lines.append("Relationship properties:")
            lines.extend(properties)
    return "\n".join(lines)


#: Memgraph's valueType() names for temporal values, rendered as what they are.
_TEMPORAL_TYPES = {"ZONED_DATE_TIME": "datetime", "LOCAL_DATE_TIME": "datetime", "DATE": "date", "DURATION": "duration"}


def _describe_relationship_properties(graph: ReadOnlyGraph, edge_types: list[str]) -> list[str]:
    """One line per relationship type that carries properties: each key and its value type, never values."""
    described: list[str] = []
    for edge_type in edge_types:
        rows = graph.query(
            f"MATCH ()-[r:`{edge_type}`]->() WITH r LIMIT 200 "
            "UNWIND keys(r) AS key RETURN DISTINCT key, valueType(r[key]) AS type ORDER BY key"
        )
        if not rows:
            continue
        fields = ", ".join(
            f"{row['key']} ({_TEMPORAL_TYPES.get(row['type'], str(row['type']).lower())})" for row in rows
        )
        described.append(f"  :{edge_type} -- {fields}")
    return described


def _label_sets(graph: ReadOnlyGraph) -> list[list[str]]:
    """Distinct label combinations present in the graph.

    Combinations, not individual labels: a node carrying ``:Action:Message`` is
    one thing with one property set, and splitting it would suggest two.
    """
    # Sorted in Python: Memgraph refuses ORDER BY on a list value. Filtered in
    # the query so storage labels cannot use up the LIMIT.
    rows = graph.query(
        "MATCH (n) WITH labels(n) AS labels "
        f"WHERE none(l IN labels WHERE l STARTS WITH '{_INTERNAL_LABEL_PREFIX}') "
        f"RETURN DISTINCT labels LIMIT {_MAX_LABELS}"
    )
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
            # Turn text lives inside Action.properties as JSON: keys are structure, values stay hidden.
            described.append(f"    {key}: JSON string, keys: {', '.join(json_keys)} (values are free text)")
            continue

        if key not in _CONTENT_KEYS and _is_enumerable(sample):
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
    #: Rows the graph returned, rendered as text: what is scored and token-counted (#309).
    retrieval_context: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Wall-clock seconds for retrieval plus the final answer call.
    latency_seconds: float = 0.0


async def retrieve(
    question: str,
    *,
    graph: ReadOnlyGraph,
    llm: LLM,
    max_steps: int = DEFAULT_MAX_STEPS,
    today: str | None = None,
) -> Retrieved:
    """Answer ``question`` by letting ``llm`` query ``graph``.

    A failed query is recorded and the loop continues: one malformed Cypher
    statement should cost that question its answer, not abort a whole batch
    mid-run.
    """
    started = time.monotonic()
    schema = graph_schema(graph)
    seen: list[str] = []
    queries: list[str] = []
    errors: list[str] = []

    for _ in range(max_steps):
        reply = await llm.complete(_query_prompt(question, schema, seen, errors, insist=not seen))
        cypher = _extract_cypher(reply)
        if cypher is None:
            # No query means "I have enough" -- but only once something was retrieved.
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
            # Told to the agent, or it would believe it had seen everything its query matched.
            errors.append(
                f"{cypher}: matched too much -- {dropped} rows dropped at the "
                f"{MAX_PAYLOAD_TOKENS}-token payload cap. Narrow the query or add LIMIT."
            )

        if not rows:
            # Reported so the agent revises its assumptions instead of concluding "not in memory".
            errors.append(f"{cypher}: returned 0 rows")
            continue
        seen.extend(rendered)

    answer = await llm.complete(answer_prompt(question, seen, today))
    return Retrieved(
        answer=answer.strip(),
        retrieval_context=seen,
        queries=queries,
        errors=errors,
        latency_seconds=time.monotonic() - started,
    )


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
            # A named exit token: without one the model spent its whole step budget.
            else "If the rows below already answer the question, reply with exactly STOP "
            "and nothing else. Otherwise return only the next query."
        ),
        "",
        # A fact about the environment, deliberately not search strategy (#300).
        "This is Memgraph, not Neo4j: APOC is NOT available, so JSON-valued "
        "properties cannot be parsed in Cypher. Match substrings inside them "
        "with CONTAINS instead.",
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


def answer_prompt(question: str, seen: list[str], today: str | None = None) -> str:
    """The final-answer prompt, shared by every strategy so a quality difference is down to what was retrieved."""
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
