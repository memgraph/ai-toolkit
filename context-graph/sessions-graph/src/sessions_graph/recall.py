"""Recall: what one user's past sessions say about a question (#394).

Turns are the answer store and the extracted graph is an index into them.
Six lanes, each over the user's own history only:

    memories    memory files the user's agents saved deliberately, by vector and full text
    turns       message passages nearest the question by vector
    text        messages matching it by full-text search
    entities    entities nearest it by vector, and each one's facts
    facts       extracted facts nearest it by vector
    user_facts  every fact of the relation types nearest it, from the user's turns

then the turns the facts were read from. Memory files come first: they are
curated, where turns are everything that was said. A user with no memory
files gets exactly what the five conversation lanes give, header included. A turn too long to show whole is
shown as the passages that matched (:mod:`.passages`): the one a vector
hit, the one a fact was read from, the one sharing most words with the
question. The result is the evidence, not an
answer: the caller's model answers from it. This is the hybrid retrieval the
eval benchmarks (``context-graph-eval run --retrieval-strategy hybrid``),
which calls this same code. Default exchange packing groups related
source messages, uses facts only for ranking, and budgets whole groups together
with memory notes. Supplemental discovery is enabled by default; turn/fact
packing remains available with ``packing="turns"``.

Vectors come from :mod:`.embeddings`, so similarity is computed in Memgraph
over the user's own rows (``vector_search.search`` has no filter, and a
global top-k could crowd one user's rows out entirely). Without MAGE the
vector lanes are off: the text lane runs alone, with the facts read from the
turns it found, and :attr:`Recalled.vector_lanes_off` says why.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal, cast

from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingUnavailableError, embed_texts
from .passages import best_passage, excerpt, split_passages

LANES = ("memories", "turns", "text", "entities", "facts", "user_facts")

#: bge's English retrieval instruction, prepended to the question only, never to what is searched.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
#: The models trained with :data:`QUERY_INSTRUCTION`: bge's English checkpoints. Another model gets no
#: prefix unless ``RecallConfig.query_instruction`` gives its own; bge's text would only add noise to it.
_BGE_ENGLISH = re.compile(r"bge-(small|base|large)-en", re.IGNORECASE)


def query_instruction(model: str) -> str:
    """The prefix *model* expects on a search query: :data:`QUERY_INSTRUCTION` for bge's English models, else none."""
    return QUERY_INSTRUCTION if _BGE_ENGLISH.search(model) else ""


#: Full-text index over message text; created by ``SessionsGraph.setup``.
TURN_TEXT_INDEX = "recall_turn_text_index"


@dataclass(frozen=True)
class RecallConfig:
    """Which recall lanes run and how wide each is."""

    lanes: tuple[str, ...] = LANES
    #: Memory files shown at most, found by vector and full text together.
    memories_k: int = 5
    #: At most this many vector-selected turns, preserving the original top passage hits.
    turns_k: int = 8
    text_k: int = 8
    entities_k: int = 15
    edges_per_entity: int = 6
    facts_k: int = 15
    #: How many turns the facts lead to are added.
    fact_turns_k: int = 8
    #: user_facts: how many relation types to pick, and facts to keep across them.
    user_fact_types: int = 2
    user_facts_k: int = 30
    #: Characters of each turn shown: a longer turn shows its matched passages instead of its opening.
    turn_chars: int = 1500
    #: Prepended to the question before embedding it; None means :func:`query_instruction` for the model.
    query_instruction: str | None = None

    #: Coherent, source-only exchanges; "turns" selects separate turn/fact rows.
    packing: Literal["turns", "exchanges"] = "exchanges"
    #: Row budget in cl100k_base tokens, including exchange headers and separators.
    token_budget: int = 6000
    #: Discover problem-resolution episodes in vector-selected sessions (exchange packing only).
    supplemental_search: bool = True

    def __post_init__(self) -> None:
        if self.packing not in ("turns", "exchanges"):
            raise ValueError("recall packing must be 'turns' or 'exchanges'")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> RecallConfig:
        """Defaults overridden by ``values`` (e.g. a config file's ``[recall]``); unknown keys are ignored.

        Raises:
            ValueError: a width that isn't a non-negative integer, or a lane that doesn't exist.
        """
        known = {f.name for f in fields(cls)}
        overrides: dict[str, Any] = {}
        for key, value in values.items():
            if key not in known:
                continue
            if key == "lanes":
                lanes = tuple(lane.strip() for lane in (value.split(",") if isinstance(value, str) else value))
                unknown = [lane for lane in lanes if lane not in LANES]
                if unknown:
                    raise ValueError(f"unknown recall lanes {unknown}; choose from {', '.join(LANES)}")
                overrides[key] = lanes
            elif key == "packing":
                overrides[key] = cast("Literal['turns', 'exchanges']", value)
            elif key == "supplemental_search":
                if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                    value = value.strip().lower() == "true"
                if not isinstance(value, bool):
                    raise ValueError("recall supplemental_search must be a boolean")
                overrides[key] = value
            elif key == "query_instruction":
                overrides[key] = None if value is None else str(value)
            else:
                number = int(value)
                if number < 0:
                    raise ValueError(f"recall {key} must be >= 0, got {number}")
                overrides[key] = number
        return cls(**overrides)


@dataclass(frozen=True)
class Turn:
    """One message, as recall shows it."""

    action_id: str
    session_id: str
    timestamp: str
    speaker: str
    text: str
    #: Indices of the passages that matched, most relevant first; what a turn too long to show whole shows.
    passages: tuple[int, ...] = ()

    def line(self, chars: int) -> str:
        shown = excerpt(self.text, list(self.passages), chars)
        return f"TURN [session {self.session_id}, {self.timestamp[:16]}, {self.speaker}]: {shown}"


@dataclass(frozen=True)
class MemoryNote:
    """One memory file, as recall shows it."""

    path: str
    updated_at: str
    text: str
    #: Indices of the passages that matched, most relevant first; what a file too long to show whole shows.
    passages: tuple[int, ...] = ()

    def line(self, chars: int) -> str:
        shown = excerpt(self.text, list(self.passages), chars)
        return f"MEMORY [{self.path}, updated {self.updated_at[:10]}]: {shown}"


@dataclass(frozen=True)
class Fact:
    """One extracted edge: what it says, when it became true, and the sentence it was read from."""

    head: str
    type: str
    tail: str
    valid_at: str | None
    speaker: str | None
    sentence: str | None
    turn: str | None

    def line(self) -> str:
        when = f" @ {self.valid_at[:10]}" if self.valid_at else ""
        speaker = f"{self.speaker}: " if self.speaker else ""
        said = f' -- {speaker}"{self.sentence}"' if self.sentence else ""
        return f"FACT: {self.head} -[{self.type}{when}]-> {self.tail}{said}"


#: How to read recall's rows: the rules the benchmark's answerer followed.
READING_RULES = (
    "These rows are from the user's past conversations. Each row carries the date it was said or became true.\n"
    '- "Today", "yesterday", "last week" inside a row are relative to that row\'s date, not to now.\n'
    "- For a count, a total, or the time between events: list each matching item or value with its date, "
    "count an item mentioned in several rows once, then compute.\n"
    "- When rows disagree about the same thing, the most recent one is current.\n"
    "- When asked for a recommendation, recommend: use the preferences, interests and possessions the rows "
    "show, add your own knowledge where they name nothing specific.\n"
    "- When the rows never mention what is asked, or don't show what the question assumes, say it is not "
    "in memory rather than answering a related question."
)
#: Added only when memory files are shown, so recall over conversations alone reads exactly as benchmarked.
MEMORY_RULE = (
    "- MEMORY rows are notes saved deliberately to remember; trust them over a TURN about the same thing, "
    "unless the TURN is more recent."
)


@dataclass(frozen=True)
class Recalled:
    """Selected memory notes and sources, displayed as turn/fact rows or whole exchanges."""

    question: str
    turns: list[Turn] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    memories: list[MemoryNote] = field(default_factory=list)
    #: Why the vector lanes didn't run, or None when they did.
    vector_lanes_off: str | None = None
    turn_chars: int = RecallConfig.turn_chars
    #: Spaces that couldn't be searched, each with why; their rows are missing.
    missing_spaces: tuple[str, ...] = ()

    #: Selected source IDs grouped into complete exchanges, in display order.
    exchanges: tuple[tuple[str, ...], ...] = ()

    #: Original results retained when fusion includes exchanges. IDs are local to each space.
    space_results: tuple[Recalled, ...] = field(default=(), repr=False)

    def lines(self) -> list[str]:
        """Render selected rows; fused exchange groups retain their source-space boundaries."""
        if self.space_results:
            rows = [note.line(self.turn_chars) for note in self.memories]
            # Keep each space's groups intact, including mixed turn/exchange configurations.
            for result in self.space_results:
                rows.extend(replace(result, memories=[]).lines())
            return rows
        if self.exchanges:
            turns = {turn.action_id: turn for turn in self.turns}
            rows = [note.line(self.turn_chars) for note in self.memories]
            for group in self.exchanges:
                first = turns[group[0]]
                rows.append(f"EXCHANGE [session {first.session_id}, {first.timestamp[:19]}]:")
                rows.extend(turns[source].line(self.turn_chars) for source in group)
            return rows + [fact.line() for fact in self.facts]
        return (
            [note.line(self.turn_chars) for note in self.memories]
            + [turn.line(self.turn_chars) for turn in self.turns]
            + [fact.line() for fact in self.facts]
        )

    def render(self, today: str | None = None) -> str:
        """The rows behind a header with the reading rules, for a model to answer from."""
        header = [f"Memory recall for: {self.question}"]
        if today:
            header.append(f"Today is {today}.")
        header.append(READING_RULES + ("\n" + MEMORY_RULE if self.memories else ""))
        if self.vector_lanes_off:
            header.append(f"Vector search is off ({self.vector_lanes_off}); these rows come from text search only.")
        if self.missing_spaces:
            header.append(f"Not searched, so rows from them are missing: {'; '.join(self.missing_spaces)}.")
        rows = "\n".join(self.lines()) or "(nothing in memory matched)"
        return "\n".join(header) + "\n\nRows:\n" + rows

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "turns": [asdict(turn) for turn in self.turns],
            "facts": [asdict(fact) for fact in self.facts],
            "memories": [asdict(note) for note in self.memories],
            "vector_lanes_off": self.vector_lanes_off,
            **({"exchanges": [list(group) for group in self.exchanges]} if self.exchanges else {}),
            **({"spaces": [result.to_json() for result in self.space_results]} if self.space_results else {}),
            **({"missing_spaces": list(self.missing_spaces)} if self.missing_spaces else {}),
        }


# Every lane starts from the user's own messages: their sessions' actions,
# directly or through an Agent, that carry text.
_OWN_TURNS = (
    "MATCH (:User {user_id: $user})-[:HAD_SESSION]->(:Session)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action) "
    "WHERE a.text IS NOT NULL "
)
_SCORE = "vector_search.cosine_similarity({x}.embedding, $query)"

# Preserve the top k passages, then fill distinct-source slots from a bounded
# candidate pool. Existing hits keep their passages; new sources get one best
# passage. Collection stays bounded even for users with very long histories.
_TURNS = (
    _OWN_TURNS + "AND a.embedding_model = $model AND a.passage_embeddings IS NOT NULL "
    "WITH DISTINCT a, a.passage_embeddings AS vectors UNWIND range(0, size(vectors) - 1) AS i "
    "WITH a, i, vector_search.cosine_similarity(vectors[i], $query) AS score "
    "ORDER BY score DESC, a.action_id, i LIMIT $k * 8 "
    "WITH collect({id: a.action_id, passage: i}) AS hits "
    "UNWIND range(0, size(hits) - 1) AS rank "
    "WITH hits[rank] AS hit, rank ORDER BY rank "
    "WITH hit.id AS id, collect({passage: hit.passage, rank: rank}) AS passages "
    "ORDER BY passages[0].rank LIMIT $k "
    "WITH id, passages[0] AS best, [h IN passages WHERE h.rank < $k] AS original "
    "UNWIND CASE WHEN size(original) > 0 THEN original ELSE [best] END AS kept "
    "RETURN id, kept.passage AS passage ORDER BY kept.rank"
)
# A session's messages share chunks. Expand each chunk once, retaining the
# exact source IDs: sharing a chunk does not grant ownership of all its facts.
_OWN_CHUNKS = _OWN_TURNS + "MATCH (a)-[:HAS_CHUNK]->(c:Chunk) WITH c, collect(DISTINCT a.action_id) AS own "
# An entity is the user's when one of its mentions is from the user's turn:
# chunks are content-addressed, so another user's identical text shares the chunk.
_ENTITIES = (
    _OWN_CHUNKS + "MATCH (c)<-[m:MENTIONED_IN]-(n) "
    "WHERE n.embedding_model = $model AND any(source IN coalesce(m.sources, []) WHERE source IN own) "
    "WITH DISTINCT n WITH n, " + _SCORE.format(x="n") + " AS score "
    "ORDER BY score DESC, id(n) LIMIT $k RETURN id(n) AS id"
)
# A fact is the user's when it was read from the user's turn.
_OWN_FACTS = _OWN_CHUNKS + "MATCH (c)<-[:MENTIONED_IN]-(n)-[r]-() WHERE r.source_id IN own "
_FACT_FIELDS = (
    "RETURN coalesce(h.text, 'user') AS head, type(r) AS type, coalesce(t.text, 'user') AS tail, "
    "toString(r.valid_at) AS valid_at, r.role AS speaker, r.text AS sentence, r.source_id AS turn"
)
_FACTS = (
    _OWN_FACTS + "AND r.embedding_model = $model WITH DISTINCT r "
    "WITH r, " + _SCORE.format(x="r") + " AS score ORDER BY score DESC, id(r) LIMIT $k "
    "WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)
# Check ownership of adjacent fact sources in one batch before confidence
# cutoffs. Per-edge Action lookups are expensive when its ID index is absent.
# Keep candidates outside the ownership aggregation to avoid grouping by a
# potentially large list of edges.
_ENTITY_FACTS = (
    "UNWIND range(0, size($ids) - 1) AS rank MATCH (n) WHERE id(n) = $ids[rank] "
    "MATCH (n)-[r]-() WHERE r.source_id IS NOT NULL "
    "WITH collect({rank: rank, n: n, r: r}) AS candidates, collect(DISTINCT r.source_id) AS sources "
    "CALL { WITH sources MATCH (src:Action) WHERE src.action_id IN sources "
    "MATCH (src)<-[:HAS_ACTION|HAS_AGENT*1..2]-(:Session)<-[:HAD_SESSION]-(:User {user_id: $user}) "
    "RETURN collect(DISTINCT src.action_id) AS own } UNWIND candidates AS candidate "
    "WITH candidate.rank AS rank, candidate.n AS n, candidate.r AS r, own WHERE r.source_id IN own "
    "WITH rank, n, r ORDER BY rank, r.confidence DESC, id(r) "
    "WITH rank, n, collect(DISTINCT r) AS rs UNWIND rs[0..$per] AS r "
    "WITH r, min(rank) AS entity_rank ORDER BY entity_rank, r.confidence DESC, id(r) "
    "WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)
# Every relation type on a fact read from the user's turns, whatever its head:
# a learned domain relation (Project depends_on Library) needn't start at the
# User (#439). Types come from the data, not the model, so facts under a type
# a later version retired stay reachable.
_USER_FACT_TYPES = _OWN_FACTS + "RETURN DISTINCT type(r) AS type"
_USER_FACTS = (
    _OWN_FACTS + "AND type(r) IN $types AND r.embedding_model = $model WITH DISTINCT r "
    "WITH r, " + _SCORE.format(x="r") + " AS score ORDER BY score DESC, id(r) LIMIT $k "
    "WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)
_TYPE_DESCRIPTIONS = (
    "MATCH (:User {user_id: $user})-[:ADOPTED]->(v:OntologyVersion) RETURN v.model AS model, v.pool AS pool"
)
_TEXT = (
    f"CALL text_search.search_all('{TURN_TEXT_INDEX}', $text, {{limit: $pool}}) YIELD node, score "
    "WITH node, score WHERE node.action_id IN $own "
    "WITH node, score ORDER BY score DESC, node.action_id LIMIT $k RETURN node.action_id AS id"
)
_OWN_TURN_IDS = _OWN_TURNS + "RETURN DISTINCT a.action_id AS id"
_TURN_ROWS = (
    "MATCH (a:Action) WHERE a.action_id IN $ids "
    "MATCH (s:Session)-[:HAS_ACTION|HAS_AGENT*1..2]->(a) "
    "RETURN DISTINCT a.action_id AS id, s.session_id AS session, toString(a.timestamp) AS ts, "
    "a.action_type AS kind, a.text AS text"
)
# Without vectors: the facts read from the turns text search found.
_FACTS_OF_TURNS = (
    "UNWIND $ids AS id MATCH (a:Action {action_id: id})-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(n)-[r]-() "
    "WHERE r.source_id = a.action_id WITH DISTINCT r ORDER BY r.confidence DESC, id(r) LIMIT $k "
    "WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)

_MEMORY_VECTORS = (
    "MATCH (m:Memory {user_id: $user}) WHERE m.embedding_model = $model AND m.passage_embeddings IS NOT NULL "
    "WITH m, m.passage_embeddings AS vectors UNWIND range(0, size(vectors) - 1) AS i "
    "WITH m, i, vector_search.cosine_similarity(vectors[i], $query) AS score "
    "ORDER BY score DESC, m.memory_id, i LIMIT $k RETURN m.memory_id AS id, i AS passage"
)
_MEMORY_TEXT = (
    "CALL text_search.search_all($index, $text, {limit: $pool}) YIELD node, score "
    "WITH node, score WHERE node.user_id = $user "
    "WITH node, score ORDER BY score DESC, node.memory_id LIMIT $k RETURN node.memory_id AS id"
)
_MEMORY_ROWS = (
    "UNWIND $ids AS id MATCH (m:Memory {memory_id: id}) "
    "RETURN m.memory_id AS id, m.path AS path, m.updated_at AS updated_at, m.content AS text"
)
#: Full-text index over memory content; created by ``SessionsGraph.setup``.
MEMORY_TEXT_INDEX = "memory_content_index"

#: Relation-type vectors, per model and text embedded: types are few and reused across questions.
_type_vectors: dict[tuple[str, str], list[float]] = {}


def recall(
    db: Any,
    user_id: str,
    question: str,
    *,
    config: RecallConfig | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> Recalled:
    """What ``user_id``'s own sessions hold about ``question``.

    Args:
        db: Anything with ``query(cypher, params)``, e.g. a ``Memgraph`` client.
        user_id: Whose memory; nothing from another user is read.
        question: Searched as written.
        config: Lanes, widths and packing; budgeted exchanges when omitted.
        model: The embedding model the stored vectors were made with.

    Returns:
        Selected memory notes and budgeted source exchanges by default, or
        chronological turn/fact rows with packing="turns". Extracted facts rank
        exchanges but are not rendered.
    """
    config = config or RecallConfig()
    if config.packing == "exchanges":
        from .recall_exchanges import prepare

        config = prepare(config)
    try:
        prefix = config.query_instruction if config.query_instruction is not None else query_instruction(model)
        query = embed_texts(db, [prefix + question], model)[0]
    except EmbeddingUnavailableError as exc:
        return _text_only(db, user_id, question, config, reason=str(exc))

    params = {"user": user_id, "query": query, "model": model}
    memories = _memories(db, user_id, question, config, query=query, model=model)
    turn_ids: list[str] = []
    vector_hits: dict[str, list[int]] = {}
    facts: list[dict[str, Any]] = []
    lanes: dict[str, list[str]] = {}
    text_state: dict[str, int] = {}
    if "turns" in config.lanes and config.turns_k:
        for row in db.query(_TURNS, {**params, "k": config.turns_k}):
            turn_ids.append(row["id"])
            vector_hits.setdefault(row["id"], []).append(row["passage"])
        lanes["turns"] = list(dict.fromkeys(turn_ids))
    if "text" in config.lanes and config.text_k:
        lanes["text"] = _text_lane(db, user_id, question, config.text_k, state=text_state)
        turn_ids += lanes["text"]
    if "entities" in config.lanes:
        entity_ids = [row["id"] for row in db.query(_ENTITIES, {**params, "k": config.entities_k})]
        if entity_ids:
            rows = db.query(_ENTITY_FACTS, {"ids": entity_ids, "user": user_id, "per": config.edges_per_entity})
            facts += rows
            lanes["entities"] = list(dict.fromkeys(row["turn"] for row in rows if row.get("turn")))
    if "facts" in config.lanes:
        rows = db.query(_FACTS, {**params, "k": config.facts_k})
        facts += rows
        lanes["facts"] = list(dict.fromkeys(row["turn"] for row in rows if row.get("turn")))
    if "user_facts" in config.lanes:
        rows = _user_facts(db, user_id, query, model, config)
        facts += rows
        lanes["user_facts"] = list(dict.fromkeys(row["turn"] for row in rows if row.get("turn")))
    result = replace(_assemble(db, question, turn_ids, facts, config, vector_hits), memories=memories)
    if config.packing == "exchanges":
        from .recall_exchanges import assemble

        result = assemble(db, user_id, result, lanes, config, text_state.get("count"))
    return result


def _memories(
    db: Any, user_id: str, question: str, config: RecallConfig, *, query: list[float] | None, model: str
) -> list[MemoryNote]:
    """The user's memory files nearest the question: vector hits first, then full-text ones."""
    if "memories" not in config.lanes or config.memories_k == 0:
        return []
    hits: dict[str, list[int]] = {}
    if query is not None:
        for row in db.query(_MEMORY_VECTORS, {"user": user_id, "query": query, "model": model, "k": config.memories_k}):
            hits.setdefault(row["id"], []).append(row["passage"])
    text = _safe_query(question)
    if text:
        pool = db.query("MATCH (m:Memory) RETURN count(m) AS n")[0]["n"]
        if pool:
            for row in db.query(
                _MEMORY_TEXT,
                {"index": MEMORY_TEXT_INDEX, "text": text, "pool": pool, "user": user_id, "k": config.memories_k},
            ):
                hits.setdefault(row["id"], [])
    ids = list(hits)[: config.memories_k]
    if not ids:
        return []
    rows = {row["id"]: row for row in db.query(_MEMORY_ROWS, {"ids": ids})}
    notes = []
    for memory_id in ids:
        row = rows.get(memory_id)
        if row is None:
            continue
        note = MemoryNote(path=row["path"], updated_at=row["updated_at"] or "", text=row["text"] or "")
        if len(note.text) > config.turn_chars:
            passages = split_passages(note.text)
            found = [*hits[memory_id], best_passage(passages, question, ignore=_STOPWORDS)]
            note = replace(note, passages=tuple(dict.fromkeys(found)))
        notes.append(note)
    return notes


def _text_lane(db: Any, user_id: str, question: str, k: int, *, state: dict[str, int] | None = None) -> list[str]:
    text = _safe_query(question)
    if not k or not text:
        return []
    own = [row["id"] for row in db.query(_OWN_TURN_IDS, {"user": user_id})]
    if not own:
        return []
    # search_all stops at 1,000 hits unless given a limit, which would drop
    # this user's turns before the ownership filter sees them.
    pool = db.query("MATCH (a:Action) WHERE a.text IS NOT NULL RETURN count(a) AS n")[0]["n"]
    if state is not None:
        state["count"] = pool
    return [row["id"] for row in db.query(_TEXT, {"text": text, "pool": max(pool, 1), "own": own, "k": k})]


def _user_facts(db: Any, user_id: str, query: list[float], model: str, config: RecallConfig) -> list[dict[str, Any]]:
    """Facts from the user's turns of the relation types nearest the question.

    What "how many weddings did I attend" needs and top-k similarity over
    turns can't gather: every fact of the right type, across all sessions.
    A type is ranked by its name and description (#439): a learned name can
    be terse (`pins`), and the description is what a question matches.
    """
    types = sorted(row["type"] for row in db.query(_USER_FACT_TYPES, {"user": user_id}))
    if not types:
        return []
    descriptions = _type_descriptions(db, user_id)
    texts = {t: _type_text(t, descriptions.get(t, "")) for t in types}
    missing = sorted({text for text in texts.values() if (model, text) not in _type_vectors})
    if missing:
        vectors = embed_texts(db, missing, model)
        _type_vectors.update({(model, text): v for text, v in zip(missing, vectors, strict=True)})
    ranked = sorted(types, key=lambda t: -_dot(_type_vectors[(model, texts[t])], query))
    wanted = ranked[: config.user_fact_types]
    return db.query(
        _USER_FACTS, {"user": user_id, "types": wanted, "query": query, "model": model, "k": config.user_facts_k}
    )


def _type_descriptions(db: Any, user_id: str) -> dict[str, str]:
    """Relation descriptions from the user's adopted ontology version, then its pool of retired types.

    A user with no stored version is on hygm's default model. Recall can run
    without hygm (the MCP server needs only sessions-graph), and then a type
    ranks by its name alone.
    """
    rows = db.query(_TYPE_DESCRIPTIONS, {"user": user_id})
    if rows:
        mappings = [json.loads(rows[0][key]) for key in ("model", "pool") if rows[0].get(key)]
    else:
        try:
            from hygm import default_model, model_to_mapping
        except ImportError:
            return {}
        mappings = [model_to_mapping(default_model())]
    descriptions: dict[str, str] = {}
    for mapping in mappings:
        for relation in mapping.get("relation_types", []):
            if relation.get("description"):
                descriptions.setdefault(relation["label"], relation["description"])
    return descriptions


def _type_text(name: str, description: str) -> str:
    words = name.replace("_", " ")
    return f"{words}: {description}" if description else words


def _text_only(db: Any, user_id: str, question: str, config: RecallConfig, *, reason: str) -> Recalled:
    turn_ids = _text_lane(db, user_id, question, config.text_k) if "text" in config.lanes else []
    facts = db.query(_FACTS_OF_TURNS, {"ids": turn_ids, "k": config.facts_k}) if turn_ids else []
    result = replace(
        _assemble(db, question, turn_ids, facts, config),
        memories=_memories(db, user_id, question, config, query=None, model=""),
        vector_lanes_off=reason,
    )
    if config.packing == "exchanges":
        from .recall_exchanges import assemble

        result = assemble(db, user_id, result, {"text": turn_ids}, config, None)
    return result


def _assemble(
    db: Any,
    question: str,
    turn_ids: list[str],
    fact_rows: list[dict[str, Any]],
    config: RecallConfig,
    vector_hits: dict[str, list[int]] | None = None,
) -> Recalled:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in fact_rows:
        unique.setdefault((row["head"], row["type"], row["tail"], row["turn"], row["sentence"]), row)
    # The turns the facts were read from, in the order the lanes found them.
    turn_ids += [row["turn"] for row in unique.values() if row.get("turn")][: config.fact_turns_k]
    sentences: dict[str, list[str]] = {}
    for row in unique.values():
        if row.get("turn") and row.get("sentence"):
            sentences.setdefault(row["turn"], []).append(row["sentence"])
    turns = [
        _with_passages(
            turn, question, (vector_hits or {}).get(turn.action_id, []), sentences.get(turn.action_id, []), config
        )
        for turn in _turns(db, list(dict.fromkeys(turn_ids)))
    ]
    # Both in time order: a knowledge-update question wants the latest value, a temporal one the sequence.
    facts = sorted((Fact(**row) for row in unique.values()), key=lambda f: f.valid_at or "")
    return Recalled(question=question, turns=turns, facts=facts, turn_chars=config.turn_chars)


def _with_passages(
    turn: Turn, question: str, vector_hits: list[int], sentences: list[str], config: RecallConfig
) -> Turn:
    """``turn`` with the passages worth showing: vector hits, then each fact's sentence, then the question's words.

    Only a turn too long to show whole needs them.
    """
    if len(turn.text) <= config.turn_chars:
        return turn
    passages = split_passages(turn.text)
    found = [*vector_hits, *(best_passage(passages, sentence) for sentence in sentences)]
    found.append(best_passage(passages, question, ignore=_STOPWORDS))
    return replace(turn, passages=tuple(dict.fromkeys(found)))


def _turns(db: Any, turn_ids: list[str]) -> list[Turn]:
    if not turn_ids:
        return []
    rows = db.query(_TURN_ROWS, {"ids": turn_ids})
    return [
        Turn(
            action_id=row["id"],
            session_id=row["session"],
            timestamp=row["ts"] or "",
            speaker="user" if row["kind"] == "user_message" else "assistant",
            text=row["text"] or "",
        )
        for row in sorted(rows, key=lambda r: (r["ts"] or "", r["id"]))
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


#: Near-universal words: text search is OR-across-terms, so keeping them lets
#: an unrelated turn outrank the one holding the question's real keyword.
_STOPWORDS = frozenset(
    [
        "a",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "had",
        "has",
        "have",
        "he",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "she",
        "should",
        "that",
        "the",
        "their",
        "them",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)
_QUERY_TOKEN = re.compile(r"\w+")


def _safe_query(question: str) -> str:
    """The question as literal word tokens for the full-text index, minus near-universal words.

    Strip query punctuation and quote each remaining word: a word such as
    ``not`` must be searched literally, never interpreted as an operator.
    """
    return " ".join(f'"{t}"' for t in _QUERY_TOKEN.findall(question) if t.lower() not in _STOPWORDS)


#: Reciprocal rank fusion's damping constant: the usual 60, so a list's top few ranks don't drown the rest.
RRF_K = 60


def fuse(question: str, results: list[Recalled], *, memories_k: int, missing: tuple[str, ...] = ()) -> Recalled:
    """One recall result from several spaces' results, the personal space's first.

    Memories compete for one budget, so they are merged by reciprocal rank fusion
    (each space ranks its own; raw scores from different databases aren't
    comparable) and cut to *memories_k*. A note whose text another space already
    holds is dropped, keeping the later space's copy: shared spaces come after
    the personal one, and a promoted file belongs where it was shared. Turns and
    facts, each already chosen by its space's own widths, are kept whole: turns
    in time order, as recall shows them. When any space uses exchanges, rendering
    preserves each space's original grouping and source IDs through ``space_results``;
    JSON exposes these scoped results under ``spaces``. Budgets remain per space.
    """
    if len(results) == 1 and not missing:
        return results[0]
    scores: dict[int, float] = {}
    notes: dict[int, MemoryNote] = {}
    owner: dict[str, int] = {}
    for space_index, result in enumerate(results):
        for rank, note in enumerate(result.memories):
            key = space_index * 1_000_000 + rank
            scores[key] = 1.0 / (RRF_K + rank + 1)
            notes[key] = note
            if (previous := owner.get(note.text)) is not None:
                scores[key] = max(scores[key], scores.pop(previous))
                notes.pop(previous)
            owner[note.text] = key
    memories = [notes[key] for key in sorted(scores, key=lambda key: (-scores[key], key))][:memories_k]
    turns = sorted((turn for result in results for turn in result.turns), key=lambda turn: turn.timestamp)
    facts = [fact for result in results for fact in result.facts]
    off = next((result.vector_lanes_off for result in results if result.vector_lanes_off), None)
    return Recalled(
        question=question,
        turns=turns,
        facts=facts,
        memories=memories,
        vector_lanes_off=off,
        turn_chars=results[0].turn_chars if results else RecallConfig.turn_chars,
        missing_spaces=missing,
        space_results=tuple(results) if any(result.exchanges or result.space_results for result in results) else (),
    )
