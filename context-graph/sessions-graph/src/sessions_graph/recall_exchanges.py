"""Opt-in bounded exchange retrieval over existing sources; no graph writes or extraction."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

# Package-private helpers keep ownership, literal text search and source excerpts identical in both policies.
from .recall import _STOPWORDS, _TEXT, RecallConfig, Recalled, Turn, _safe_query, _turns, _with_passages

if TYPE_CHECKING:
    from tiktoken import Encoding

# Experimental policy selected during BEAM 100K development; see eval/recall-exchanges.md.
MIN_DIRECT_SOURCES = 24
MIN_FACT_SOURCES = 16
PASSAGE_NEIGHBOR_HOPS = 3
EXCHANGE_RRF_K = 20
GRAPH_LANE_WEIGHT = 1 / 3
SPLIT_TEXT_WEIGHT = 0.5
SUPPLEMENTAL_BUDGET_DIVISOR = 4
SUPPLEMENTAL_SESSIONS = 4
SUPPLEMENTAL_SOURCES = 24
TOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> Encoding:
    import tiktoken

    try:
        return tiktoken.get_encoding(TOKEN_ENCODING)
    except Exception as exc:
        raise RuntimeError(
            "Exchange recall could not load cl100k_base. Preload it with "
            "tiktoken.get_encoding('cl100k_base') while online, then preserve "
            "TIKTOKEN_CACHE_DIR for offline use."
        ) from exc


_OWN_SOURCE = (
    "MATCH (a:Action) WHERE a.action_id IN $ids WITH DISTINCT a "
    "MATCH (:User {user_id: $user})-[:HAD_SESSION]->(s:Session) "
    "MATCH (s)-[:HAS_AGENT*0..1]->(container)-[:HAS_ACTION]->(a) WITH DISTINCT a, container "
)
_NEAREST = (
    "WITH a, b, min(size(relationships(p))) AS distance ORDER BY a.action_id, distance, b.action_id "
    "WITH a, collect(b)[0] AS b RETURN a.action_id AS source, b.action_id AS neighbor"
)
_NEIGHBORS = (
    _OWN_SOURCE
    + f"MATCH p=(a)-[:FOLLOWED_BY*1..{PASSAGE_NEIGHBOR_HOPS}]-(b:Action) MATCH (container)-[:HAS_ACTION]->(b) "
    "WITH a, b, p, nodes(p) AS ns, relationships(p) AS rs "
    "WHERE ((a.action_type = 'user_message' AND b.action_type = 'assistant_message' "
    "AND all(i IN range(0, size(rs)-1) WHERE startNode(rs[i]) = ns[i])) "
    "OR (a.action_type = 'assistant_message' AND b.action_type = 'user_message' "
    "AND all(i IN range(0, size(rs)-1) WHERE endNode(rs[i]) = ns[i]))) "
    "AND all(n IN ns[1..size(ns)-1] WHERE coalesce(n.action_type, '') <> 'user_message') " + _NEAREST
)


def prepare(config: RecallConfig) -> RecallConfig:
    """Check optional dependencies and widen enabled source lanes for exchange ranking."""
    try:
        import tiktoken  # noqa: F401 -- fail early before any graph or model work

        if config.supplemental_search:
            import nltk  # noqa: F401 -- the local BM25 stemmer requires no downloaded corpus
    except ImportError as exc:
        raise ImportError("Exchange recall requires sessions-graph[compact]") from exc
    _encoding()
    if config.token_budget < 0:
        raise ValueError("recall token_budget must be >= 0")
    return replace(
        config,
        turns_k=max(MIN_DIRECT_SOURCES, config.turns_k) if config.turns_k else 0,
        text_k=max(MIN_DIRECT_SOURCES, config.text_k) if config.text_k else 0,
        fact_turns_k=max(MIN_FACT_SOURCES, config.fact_turns_k) if config.fact_turns_k else 0,
    )


def assemble(
    db: Any, user: str, result: Recalled, lanes: dict[str, list[str]], config: RecallConfig, count: int | None
) -> Recalled:
    """Group owned adjacent turns, optionally discover extra episodes, and pack whole exchanges.

    Missing vector/text lanes disable supplemental discovery. Local search exceeding
    its work cap preserves the primary pool. Graph/query failures propagate to callers.
    """
    turns = {turn.action_id: turn for turn in result.turns}
    pairs = db.query(_NEIGHBORS, {"user": user, "ids": list(turns)}) if turns else []
    _materialize(db, turns, [pair["neighbor"] for pair in pairs], result.question, config)
    weights = {name: (1.0 if name in ("turns", "text") else GRAPH_LANE_WEIGHT) for name in lanes}
    supplemental: list[str] = []
    if config.supplemental_search and lanes.get("turns") and "text" in lanes:
        ids = _discover(db, user, result.question, turns, lanes["turns"], count)
        if ids is not None:
            extra_pairs = db.query(_NEIGHBORS, {"user": user, "ids": ids}) if ids else []
            supplemental = list(
                dict.fromkeys(source for source in ids + [p["neighbor"] for p in extra_pairs] if source not in turns)
            )
            _materialize(db, turns, supplemental, result.question, config)
            pairs.extend(extra_pairs)
            weights["text"] = SPLIT_TEXT_WEIGHT
            weights["supplemental"] = SPLIT_TEXT_WEIGHT
            lanes = {**lanes, "supplemental": ids}
    return _pack(result, turns, pairs, lanes, weights, supplemental, config.token_budget)


def _materialize(db: Any, turns: dict[str, Turn], ids: list[str], question: str, config: RecallConfig) -> None:
    missing = list(dict.fromkeys(source for source in ids if source not in turns))
    for turn in _turns(db, missing):
        turns[turn.action_id] = _with_passages(turn, question, [], [], config)


def _pack(
    result: Recalled,
    turns: dict[str, Turn],
    pairs: list[dict[str, Any]],
    lanes: dict[str, list[str]],
    weights: dict[str, float],
    supplemental: list[str],
    budget: int,
) -> Recalled:
    encoding = _encoding()

    def cost(line: str) -> int:
        return len(encoding.encode(line, disallowed_special=())) + 1

    parent = {source: source for source in turns}

    def root(source: str) -> str:
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    for pair in pairs:
        a, b = pair["source"], pair["neighbor"]
        if a in parent and b in parent:
            left, right = sorted((root(a), root(b)))
            parent[right] = left
    groups: dict[str, list[str]] = {}
    for source in turns:
        groups.setdefault(root(source), []).append(source)
    scores = dict.fromkeys(groups, 0.0)
    for name, sources in lanes.items():
        ranks: dict[str, int] = {}
        for rank, source in enumerate(sources):
            if source in parent:
                ranks.setdefault(root(source), rank)
        for group, rank in ranks.items():
            scores[group] += weights[name] / (EXCHANGE_RRF_K + rank + 1)

    def timestamp(source: str) -> tuple[str, str]:
        return turns[source].timestamp, source

    costs: dict[str, int] = {}
    for group, sources in groups.items():
        sources.sort(key=timestamp)
        first = turns[sources[0]]
        header = f"EXCHANGE [session {first.session_id}, {first.timestamp[:19]}]:"
        costs[group] = cost(header) + sum(cost(turns[source].line(result.turn_chars)) for source in sources)
    used = 0
    memories = []
    for note in result.memories:
        amount = cost(note.line(result.turn_chars))
        if used + amount <= budget:
            memories.append(note)
            used += amount
    ordered = sorted(groups, key=lambda group: (-scores[group], group))
    selected = []
    extra = {root(source) for source in supplemental if source in parent}
    allowance = (budget - used) // SUPPLEMENTAL_BUDGET_DIVISOR
    for group in ordered:
        if group in extra and costs[group] <= allowance:
            selected.append(group)
            used += costs[group]
            allowance -= costs[group]
    for group in ordered:
        if group not in selected and used + costs[group] <= budget:
            selected.append(group)
            used += costs[group]
    exchanges = tuple(tuple(groups[group]) for group in sorted(selected, key=lambda group: timestamp(groups[group][0])))
    selected_ids = {source for group in exchanges for source in group}
    return replace(
        result,
        turns=sorted((turns[source] for source in selected_ids), key=lambda turn: timestamp(turn.action_id)),
        facts=[],
        memories=memories,
        exchanges=exchanges,
    )


PROBLEM_RESOLUTION_CUES = (
    "issue issues problem problems error errors fix fixed resolve resolved debug debugging failure broken troubleshoot"
)
_SESSION_ACTIONS = (
    "MATCH (:User {user_id: $user})-[:HAD_SESSION]->(s:Session) "
    "WHERE s.session_id IN $sessions "
    "MATCH (s)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action) "
    "WHERE a.text IS NOT NULL RETURN DISTINCT a.action_id AS id"
)
_SESSION_DOCS = (
    # Stop enumeration before ordering; prefixes are ranking features only.
    # Original source text is fetched separately for every selected turn.
    "MATCH (:User {user_id: $user})-[:HAD_SESSION]->(s:Session) "
    "WHERE s.session_id IN $sessions "
    "MATCH (s)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action) WHERE a.text IS NOT NULL "
    "WITH DISTINCT a LIMIT $limit RETURN a.action_id AS id, "
    "left(a.text, $chars) AS text ORDER BY id"
)
MAX_DOCS = 512
MAX_CHARS = 8192
MAX_TOTAL_CHARS = 2 * 1024 * 1024
MAX_GLOBAL_CUE_DOCS = 10000


PROBLEM_RESOLUTION_TRIGGER = re.compile(
    r"\b(issues?|problems?|resolv\w*|fix\w*|debug\w*|errors?|troubleshoot\w*)\b", re.I
)


def _discover(
    db: Any, user: str, question: str, turns: dict[str, Turn], vector_sources: list[str], count: int | None
) -> list[str] | None:
    if not PROBLEM_RESOLUTION_TRIGGER.search(question):
        return None
    sessions = list(dict.fromkeys(turns[source].session_id for source in vector_sources if source in turns))[
        :SUPPLEMENTAL_SESSIONS
    ]
    if not sessions:
        return None
    if count is None or count > MAX_GLOBAL_CUE_DOCS:
        rows = db.query(_SESSION_DOCS, {"user": user, "sessions": sessions, "limit": MAX_DOCS + 1, "chars": MAX_CHARS})
        if len(rows) > MAX_DOCS or sum(len(row["text"]) for row in rows) > MAX_TOTAL_CHARS:
            return None
        if not rows:
            return None
        scores = _bm25(PROBLEM_RESOLUTION_CUES, {row["id"]: row["text"] for row in rows})
        return sorted(
            (source for source in scores if scores[source] > 0), key=lambda source: (-scores[source], source)
        )[:SUPPLEMENTAL_SOURCES]
    own = [row["id"] for row in db.query(_SESSION_ACTIONS, {"user": user, "sessions": sessions})]
    if not own:
        return None
    return [
        row["id"]
        for row in db.query(
            _TEXT,
            {
                "text": _safe_query(PROBLEM_RESOLUTION_CUES),
                "pool": max(count, 1),
                "own": own,
                "k": SUPPLEMENTAL_SOURCES,
            },
        )
    ]


_FORMAT = re.compile(r"\b(?:mention|return|provide|give)\s+only\b.*", re.IGNORECASE)
_SCAFFOLD = frozenset(
    [
        "can",
        "could",
        "please",
        "question",
        "describe",
        "summarize",
        "summary",
        "quick",
        "complete",
        "what",
        "how",
        "which",
        "why",
        "list",
        "only",
        "items",
        "mention",
        "conversations",
        "conversation",
        "chats",
        "chat",
        "throughout",
        "across",
        "different",
        "aspects",
        "brought",
        "discussed",
        "give",
        "walk",
        "through",
        "talked",
        "tell",
        "order",
    ]
)


_IGNORED = _STOPWORDS | _SCAFFOLD


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


@lru_cache(maxsize=32768)
def _stem(word: str) -> str:
    return _stemmer().stem(word)


def _tokens(text: str) -> list[str]:
    return [_stem(word) for word in re.findall(r"\w+", text.lower()) if word not in _IGNORED]


def _bm25(question: str, documents: dict[str, str]) -> dict[str, float]:
    """Score displayed exchange text with BM25; no embeddings or reference inputs.

    Stemming uses NLTK's local algorithm and requires no downloaded corpus.
    BM25 parameters are fixed at k1=1.2 and b=0.75.
    """
    query = set(_tokens(_FORMAT.sub("", question)))
    counts = {key: Counter(_tokens(text)) for key, text in documents.items()}
    lengths = {key: sum(count.values()) for key, count in counts.items()}
    average = sum(lengths.values()) / max(len(lengths), 1)
    frequency = {word: sum(word in count for count in counts.values()) for word in query}
    scores = dict.fromkeys(documents, 0.0)
    if not query or not average:
        return scores
    for key, count in counts.items():
        normalizer = 1.2 * (0.25 + 0.75 * lengths[key] / average)
        for word in query:
            tf = count[word]
            if tf:
                idf = math.log(1 + (len(counts) - frequency[word] + 0.5) / (frequency[word] + 0.5))
                scores[key] += idf * tf * 2.2 / (tf + normalizer)
    return scores
