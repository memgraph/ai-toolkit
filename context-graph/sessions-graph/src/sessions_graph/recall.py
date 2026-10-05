"""Recall: what one user's past sessions say about a question (#394).

Turns are the answer store and the extracted graph is an index into them.
Five lanes, each over the user's own history only:

    turns       messages nearest the question by vector
    text        messages matching it by full-text search
    entities    entities nearest it by vector, and each one's facts
    facts       extracted facts nearest it by vector
    user_facts  the user's own facts of the relation types nearest it

then the turns the facts were read from. The result is the evidence, not an
answer: the caller's model answers from it. This is the hybrid retrieval the
eval benchmarks (``context-graph-eval run --retrieval-strategy hybrid``),
which calls this same code.

Vectors come from :mod:`.embeddings`, so similarity is computed in Memgraph
over the user's own rows (``vector_search.search`` has no filter, and a
global top-k could crowd one user's rows out entirely). Without MAGE the
vector lanes are off: the text lane runs alone, with the facts read from the
turns it found, and :attr:`Recalled.vector_lanes_off` says why.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingUnavailableError, embed_texts

LANES = ("turns", "text", "entities", "facts", "user_facts")

#: bge's retrieval instruction, prepended to the question only, never to what is searched.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

#: Full-text index over message text; created by ``SessionsGraph.setup``.
TURN_TEXT_INDEX = "recall_turn_text_index"


@dataclass(frozen=True)
class RecallConfig:
    """Which lanes run and how wide each is: the setup the hybrid benchmark measured."""

    lanes: tuple[str, ...] = LANES
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
    #: Characters of each turn shown.
    turn_chars: int = 1500

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

    def line(self, chars: int) -> str:
        return f"TURN [session {self.session_id}, {self.timestamp[:16]}, {self.speaker}]: {self.text[:chars]}"


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


@dataclass(frozen=True)
class Recalled:
    """What recall found for one question: turns first, then facts, each in time order."""

    question: str
    turns: list[Turn] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    #: Why the vector lanes didn't run, or None when they did.
    vector_lanes_off: str | None = None
    turn_chars: int = RecallConfig.turn_chars

    def lines(self) -> list[str]:
        """The rows as text: exactly what the benchmark's answerer was shown."""
        return [turn.line(self.turn_chars) for turn in self.turns] + [fact.line() for fact in self.facts]

    def render(self, today: str | None = None) -> str:
        """The rows behind a header with the reading rules, for a model to answer from."""
        header = [f"Memory recall for: {self.question}"]
        if today:
            header.append(f"Today is {today}.")
        header.append(READING_RULES)
        if self.vector_lanes_off:
            header.append(f"Vector search is off ({self.vector_lanes_off}); these rows come from text search only.")
        rows = "\n".join(self.lines()) or "(nothing in memory matched)"
        return "\n".join(header) + "\n\nRows:\n" + rows

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "turns": [asdict(turn) for turn in self.turns],
            "facts": [asdict(fact) for fact in self.facts],
            "vector_lanes_off": self.vector_lanes_off,
        }


# Every lane starts from the user's own messages: their sessions' actions,
# directly or through an Agent, that carry text.
_OWN_TURNS = (
    "MATCH (:User {user_id: $user})-[:HAD_SESSION]->(:Session)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action) "
    "WHERE a.text IS NOT NULL "
)
_SCORE = "vector_search.cosine_similarity({x}.embedding, $query)"

_TURNS = (
    _OWN_TURNS + "AND a.embedding_model = $model WITH DISTINCT a "
    "WITH a, " + _SCORE.format(x="a") + " AS score ORDER BY score DESC LIMIT $k RETURN a.action_id AS id"
)
# An entity is the user's when one of its mentions is from the user's turn:
# chunks are content-addressed, so another user's identical text shares the chunk.
_ENTITIES = (
    _OWN_TURNS + "MATCH (a)-[:HAS_CHUNK]->(:Chunk)<-[m:MENTIONED_IN]-(n) "
    "WHERE n.embedding_model = $model AND a.action_id IN coalesce(m.sources, []) WITH DISTINCT n "
    "WITH n, " + _SCORE.format(x="n") + " AS score ORDER BY score DESC LIMIT $k RETURN id(n) AS id"
)
# A fact is the user's when it was read from the user's turn.
_OWN_FACTS = (
    _OWN_TURNS + "MATCH (a)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(n)-[r]-() WHERE r.source_id = a.action_id "
)
_FACT_FIELDS = (
    "RETURN coalesce(h.text, 'user') AS head, type(r) AS type, coalesce(t.text, 'user') AS tail, "
    "toString(r.valid_at) AS valid_at, r.role AS speaker, r.text AS sentence, r.source_id AS turn"
)
_FACTS = (
    _OWN_FACTS + "AND r.embedding_model = $model WITH DISTINCT r "
    "WITH r, " + _SCORE.format(x="r") + " AS score ORDER BY score DESC LIMIT $k "
    "WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)
# Each entity's most confident facts, the user's own only: the source turn is
# looked up by its indexed id and walked back to the user.
_ENTITY_FACTS = (
    "UNWIND $ids AS i MATCH (n) WHERE id(n) = i MATCH (n)-[r]-() WHERE r.source_id IS NOT NULL "
    "MATCH (src:Action {action_id: r.source_id})<-[:HAS_ACTION|HAS_AGENT*1..2]-(:Session)"
    "<-[:HAD_SESSION]-(:User {user_id: $user}) "
    "WITH n, r ORDER BY r.confidence DESC WITH n, collect(DISTINCT r) AS rs UNWIND rs[0..$per] AS r "
    "WITH DISTINCT r WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)
_USER_FACT_TYPES = (
    "MATCH (:User {user_id: $user})-[r]->() WHERE r.source_id IS NOT NULL RETURN DISTINCT type(r) AS type"
)
_USER_FACTS = (
    "MATCH (h:User {user_id: $user})-[r]->(t) WHERE r.source_id IS NOT NULL AND type(r) IN $types "
    "AND r.embedding_model = $model "
    "WITH h, r, t, " + _SCORE.format(x="r") + " AS score ORDER BY score DESC LIMIT $k " + _FACT_FIELDS
)
_TEXT = (
    f"CALL text_search.search_all('{TURN_TEXT_INDEX}', $text, {{limit: $pool}}) YIELD node, score "
    "WITH node, score WHERE node.action_id IN $own "
    "WITH node, score ORDER BY score DESC LIMIT $k RETURN node.action_id AS id"
)
_OWN_TURN_IDS = _OWN_TURNS + "RETURN DISTINCT a.action_id AS id"
_TURN_ROWS = (
    "UNWIND $ids AS id MATCH (a:Action {action_id: id}) "
    "MATCH (s:Session)-[:HAS_ACTION|HAS_AGENT*1..2]->(a) "
    "RETURN DISTINCT a.action_id AS id, s.session_id AS session, toString(a.timestamp) AS ts, "
    "a.action_type AS kind, a.text AS text"
)
# Without vectors: the facts read from the turns text search found.
_FACTS_OF_TURNS = (
    "UNWIND $ids AS id MATCH (a:Action {action_id: id})-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(n)-[r]-() "
    "WHERE r.source_id = a.action_id WITH DISTINCT r ORDER BY r.confidence DESC LIMIT $k "
    "WITH startNode(r) AS h, r, endNode(r) AS t " + _FACT_FIELDS
)

#: Relation-type label vectors, per model: types are few and reused across questions.
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
        config: Lanes and widths; the benchmarked defaults when omitted.
        model: The embedding model the stored vectors were made with.

    Returns:
        The turns and facts found, turns first, each in time order.
    """
    config = config or RecallConfig()
    try:
        query = embed_texts(db, [QUERY_INSTRUCTION + question], model)[0]
    except EmbeddingUnavailableError as exc:
        return _text_only(db, user_id, question, config, reason=str(exc))

    params = {"user": user_id, "query": query, "model": model}
    turn_ids: list[str] = []
    facts: list[dict[str, Any]] = []
    if "turns" in config.lanes:
        turn_ids += [row["id"] for row in db.query(_TURNS, {**params, "k": config.turns_k})]
    if "text" in config.lanes:
        turn_ids += _text_lane(db, user_id, question, config.text_k)
    if "entities" in config.lanes:
        entity_ids = [row["id"] for row in db.query(_ENTITIES, {**params, "k": config.entities_k})]
        if entity_ids:
            facts += db.query(_ENTITY_FACTS, {"ids": entity_ids, "user": user_id, "per": config.edges_per_entity})
    if "facts" in config.lanes:
        facts += db.query(_FACTS, {**params, "k": config.facts_k})
    if "user_facts" in config.lanes:
        facts += _user_facts(db, user_id, query, model, config)
    return _assemble(db, question, turn_ids, facts, config)


def _text_lane(db: Any, user_id: str, question: str, k: int) -> list[str]:
    text = _safe_query(question)
    if not text:
        return []
    own = [row["id"] for row in db.query(_OWN_TURN_IDS, {"user": user_id})]
    if not own:
        return []
    # search_all stops at 1,000 hits unless given a limit, which would drop
    # this user's turns before the ownership filter sees them.
    pool = db.query("MATCH (a:Action) WHERE a.text IS NOT NULL RETURN count(a) AS n")[0]["n"]
    return [row["id"] for row in db.query(_TEXT, {"text": text, "pool": max(pool, 1), "own": own, "k": k})]


def _user_facts(db: Any, user_id: str, query: list[float], model: str, config: RecallConfig) -> list[dict[str, Any]]:
    """The user's facts of the relation types whose names are nearest the question.

    What "how many weddings did I attend" needs and top-k similarity over
    turns can't gather: every fact of the right type, across all sessions.
    """
    types = sorted(row["type"] for row in db.query(_USER_FACT_TYPES, {"user": user_id}))
    if not types:
        return []
    missing = [t for t in types if (model, t) not in _type_vectors]
    if missing:
        vectors = embed_texts(db, [t.replace("_", " ") for t in missing], model)
        _type_vectors.update({(model, t): v for t, v in zip(missing, vectors, strict=True)})
    ranked = sorted(types, key=lambda t: -_dot(_type_vectors[(model, t)], query))
    wanted = ranked[: config.user_fact_types]
    return db.query(
        _USER_FACTS, {"user": user_id, "types": wanted, "query": query, "model": model, "k": config.user_facts_k}
    )


def _text_only(db: Any, user_id: str, question: str, config: RecallConfig, *, reason: str) -> Recalled:
    turn_ids = _text_lane(db, user_id, question, config.text_k) if "text" in config.lanes else []
    facts = db.query(_FACTS_OF_TURNS, {"ids": turn_ids, "k": config.facts_k}) if turn_ids else []
    recalled = _assemble(db, question, turn_ids, facts, config)
    return Recalled(
        question=question,
        turns=recalled.turns,
        facts=recalled.facts,
        vector_lanes_off=reason,
        turn_chars=config.turn_chars,
    )


def _assemble(
    db: Any, question: str, turn_ids: list[str], fact_rows: list[dict[str, Any]], config: RecallConfig
) -> Recalled:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in fact_rows:
        unique.setdefault((row["head"], row["type"], row["tail"], row["turn"], row["sentence"]), row)
    # The turns the facts were read from, in the order the lanes found them.
    turn_ids += [row["turn"] for row in unique.values() if row.get("turn")][: config.fact_turns_k]
    turns = _turns(db, list(dict.fromkeys(turn_ids)))
    # Both in time order: a knowledge-update question wants the latest value, a temporal one the sequence.
    facts = sorted((Fact(**row) for row in unique.values()), key=lambda f: f.valid_at or "")
    return Recalled(question=question, turns=turns, facts=facts, turn_chars=config.turn_chars)


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
    """The question as bare word tokens for the full-text index, minus near-universal words.

    Tantivy's query parser treats characters like ``: ( ) " * ~ ^`` specially;
    reducing to words makes any question parseable.
    """
    return " ".join(t for t in _QUERY_TOKEN.findall(question) if t.lower() not in _STOPWORDS)
