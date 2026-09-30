"""Hybrid retrieval: find turns by text and vector, use the typed graph to find more.

The graph-agent baseline makes typed edges the answer store and asks an LLM to
write Cypher over them. The diagnosis on the 100-question, 5-session batch
showed why that caps low: answering from every typed edge of the right
sessions scores 21/100, from their full text 48/100. This strategy keeps text
as the answer store and uses the graph as an index into it:

    turns     vector search and full-text search over turns
    entities  entity names by vector; each entity's extracted edges
    facts     extracted edges by vector over their source sentence

then pulls the source turns of the facts it found. The answering LLM gets
facts (with ``valid_at``) and turn text, through the same ``answer_prompt``
every strategy uses.

Indexing (``ensure_hybrid_index``) derives what extraction did not store:
which turn each entity and edge came from, and each edge's source sentence.
An edge's source turn is exact -- its ``valid_at`` is that turn's timestamp
(#364) -- while entity-to-turn links match mention text within the session.
Embeddings are computed on the host and cached beside the run, not written
into Memgraph.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .retrieval import Retrieved, answer_prompt
from .text_search import TEXT_INDEX_NAME, _safe_query, ensure_turn_text_index

if TYPE_CHECKING:
    from pathlib import Path

    from actions_graph import ActionsGraph

    from .retrieval import LLM, ReadOnlyGraph

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
#: bge's retrieval instruction, prepended to queries only (not to passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
LANES = ("turns", "text", "entities", "facts", "user_facts")
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass(frozen=True)
class HybridConfig:
    """Which lanes run and how wide each is. Defaults are a starting point, not tuned."""

    lanes: tuple[str, ...] = LANES
    turns_k: int = 8
    text_k: int = 8
    entities_k: int = 15
    edges_per_entity: int = 6
    facts_k: int = 15
    fact_turns_k: int = 8
    #: user_facts: how many relation types to pick, and facts to keep across them.
    user_fact_types: int = 2
    user_facts_k: int = 30
    #: How user_facts picks its relation types: "facts" (the types of the
    #: user's facts nearest the question) or "names" (type labels by embedding).
    user_fact_types_from: str = "names"
    turn_chars: int = 1500


class Embedder:
    """A small local sentence embedder (CLS pooling, L2-normalized), loaded once."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self._model: Any = AutoModel.from_pretrained(model_name).eval()

    def __call__(self, texts: list[str], batch_size: int = 64) -> Any:
        import numpy as np

        out = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self._tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                cls = self._model(**batch).last_hidden_state[:, 0]
                out.append(self._torch.nn.functional.normalize(cls, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype="float32")


@dataclass
class HybridIndex:
    """What retrieval searches: ids and texts per lane, with their embedding matrices."""

    turn_ids: list[str]
    turn_vecs: Any
    entity_ids: list[str]
    entity_vecs: Any
    edge_ids: list[int]
    edge_vecs: Any
    embedder: Any = field(repr=False, default=None)
    #: Per edge, aligned with edge_ids: its relationship type, and whether it starts at a (:User).
    edge_types: list[str] = field(default_factory=list)
    edge_from_user: list[bool] = field(default_factory=list)
    _type_vecs: dict[str, Any] = field(default_factory=dict, repr=False)

    def relation_types(self, query: Any, k: int) -> list[str]:
        """The k relationship types whose names are closest to the query."""
        import numpy as np

        names = sorted(set(self.edge_types))
        missing = [n for n in names if n not in self._type_vecs]
        if missing:
            for name, vec in zip(missing, self.embedder([n.replace("_", " ") for n in missing]), strict=True):
                self._type_vecs[name] = vec
        scores = np.array([self._type_vecs[n] @ query for n in names])
        return [names[i] for i in np.argsort(-scores)[:k]]


def _sentence_with(text: str, needle: str) -> str:
    lowered = needle.lower()
    for sentence in _SENTENCE.findall(text):
        if lowered in sentence.lower():
            return sentence.strip()[:300]
    return ""


def link_turns(db: Any) -> dict[str, int]:
    """Write entity -[:MENTIONED_IN_TURN]-> turn links, and each extracted edge's
    source turn (``r.turn``) and source sentence (``r.text``). Idempotent."""
    turns_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for row in db.query(
        "MATCH (a:Action)-[:HAS_CHUNK]->(c:Chunk) WHERE a.text IS NOT NULL "
        "RETURN c.hash AS chunk, a.action_id AS id, a.text AS text, a.timestamp AS ts"
    ):
        turns_by_chunk.setdefault(row["chunk"], []).append(row)

    links = []
    for row in db.query(
        "MATCH (n:gliner2)-[:MENTIONED_IN]->(c:Chunk) RETURN c.hash AS chunk, n.entity_id AS id, n.text AS text"
    ):
        needle = (row["text"] or "").lower()
        if len(needle) < 2:
            continue
        for turn in turns_by_chunk.get(row["chunk"], []):
            if needle in turn["text"].lower():
                links.append({"entity": row["id"], "turn": turn["id"]})
    for start in range(0, len(links), 5000):
        db.query(
            "UNWIND $rows AS row MATCH (n:gliner2 {entity_id: row.entity}) MATCH (a:Action {action_id: row.turn}) "
            "MERGE (n)-[:MENTIONED_IN_TURN]->(a)",
            {"rows": links[start : start + 5000]},
        )

    sourced = []
    for row in db.query(
        "MATCH (a)-[r]->(b) WHERE r.chunk IS NOT NULL "
        "RETURN id(r) AS id, r.chunk AS chunk, toString(r.valid_at) AS valid_at, b.text AS tail"
    ):
        candidates = turns_by_chunk.get(row["chunk"], [])
        # valid_at is the source turn's timestamp; Memgraph renders it with
        # microseconds, the stored turn timestamp without.
        stamp = (row["valid_at"] or "").replace(".000000", "")
        turn = next((t for t in candidates if t["ts"] and t["ts"] == stamp), None)
        if turn is None:
            turn = next((t for t in candidates if (row["tail"] or "").lower() in t["text"].lower()), None)
        if turn is not None:
            sourced.append(
                {"id": row["id"], "turn": turn["id"], "text": _sentence_with(turn["text"], row["tail"] or "")}
            )
    for start in range(0, len(sourced), 5000):
        db.query(
            "UNWIND $rows AS row MATCH ()-[r]->() WHERE id(r) = row.id SET r.turn = row.turn, r.text = row.text",
            {"rows": sourced[start : start + 5000]},
        )
    return {"entity_turn_links": len(links), "edges_sourced": len(sourced)}


def _fact(row: dict[str, Any]) -> str:
    when = f" @ {row['valid_at'][:10]}" if row.get("valid_at") else ""
    said = f' -- "{row["text"]}"' if row.get("text") else ""
    return f"FACT: {row['head']} -[{row['type']}{when}]-> {row['tail']}{said}"


def build_index(db: Any, cache: Path, embedder: Any) -> HybridIndex:
    """Embed turns, entity names and edge texts, caching the matrices at ``cache`` (an .npz)."""
    import numpy as np

    turns = db.query("MATCH (a:Action) WHERE a.text IS NOT NULL RETURN a.action_id AS id, a.text AS text ORDER BY id")
    entities = db.query("MATCH (n:gliner2) RETURN n.entity_id AS id, n.text AS text ORDER BY id")
    edges = db.query(
        "MATCH (a)-[r]->(b) WHERE r.chunk IS NOT NULL "
        "RETURN id(r) AS id, type(r) AS type, coalesce(a.text, 'user') AS head, b.text AS tail, r.text AS text, "
        "'User' IN labels(a) AS from_user ORDER BY id"
    )
    key = hashlib.sha256(
        json.dumps([len(turns), len(entities), len(edges), EMBEDDING_MODEL, "v2"]).encode()
    ).hexdigest()[:16]
    edge_types = [e["type"] for e in edges]
    edge_from_user = [bool(e["from_user"]) for e in edges]
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if str(stored["key"]) == key:
            return HybridIndex(
                list(stored["turn_ids"]),
                stored["turn_vecs"],
                list(stored["entity_ids"]),
                stored["entity_vecs"],
                [int(i) for i in stored["edge_ids"]],
                stored["edge_vecs"],
                embedder,
                edge_types,
                edge_from_user,
            )
    turn_vecs = embedder([t["text"][:2000] for t in turns])
    entity_vecs = embedder([e["text"] or "" for e in entities])
    edge_vecs = embedder([f"{e['head']} {e['type'].replace('_', ' ')} {e['tail']}. {e['text'] or ''}" for e in edges])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        key=key,
        turn_ids=[t["id"] for t in turns],
        turn_vecs=turn_vecs,
        entity_ids=[e["id"] for e in entities],
        entity_vecs=entity_vecs,
        edge_ids=[e["id"] for e in edges],
        edge_vecs=edge_vecs,
    )
    return HybridIndex(
        [t["id"] for t in turns],
        turn_vecs,
        [e["id"] for e in entities],
        entity_vecs,
        [e["id"] for e in edges],
        edge_vecs,
        embedder,
        edge_types,
        edge_from_user,
    )


def ensure_hybrid_index(graph: ActionsGraph, cache: Path, embedder: Any | None = None) -> HybridIndex:
    """Materialize turn text (as text search does), link entities and edges to turns, and embed."""
    ensure_turn_text_index(graph)
    link_turns(graph.db)
    return build_index(graph.db, cache, embedder or Embedder())


def _top(vecs: Any, query: Any, k: int) -> list[int]:
    import numpy as np

    if k <= 0 or len(vecs) == 0:
        return []
    scores = vecs @ query
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in top[np.argsort(-scores[top])]]


def _turn_rows(graph: ReadOnlyGraph, turn_ids: list[str], chars: int) -> list[str]:
    if not turn_ids:
        return []
    rows = graph.query(
        "MATCH (s:Session)-[:HAS_ACTION]->(a:Action) WHERE a.action_id IN $ids "
        "RETURN a.action_id AS id, s.session_id AS session, a.timestamp AS ts, a.action_type AS kind, a.text AS text",
        {"ids": turn_ids},
    )
    by_id = {r["id"]: r for r in rows}
    out = []
    for turn_id in turn_ids:
        r = by_id.get(turn_id)
        if r:
            speaker = "user" if r["kind"] == "user_message" else "assistant"
            out.append(f"TURN [session {r['session']}, {str(r['ts'])[:16]}, {speaker}]: {r['text'][:chars]}")
    return out


_EDGE_RETURN = (
    "RETURN id(r) AS id, type(r) AS type, coalesce(a.text, 'user') AS head, coalesce(b.text, 'user') AS tail, "
    "toString(r.valid_at) AS valid_at, r.text AS text, r.turn AS turn"
)


async def retrieve_hybrid(
    question: str,
    *,
    graph: ReadOnlyGraph,
    llm: LLM,
    index: HybridIndex,
    config: HybridConfig | None = None,
    today: str | None = None,
) -> Retrieved:
    """Answer ``question`` from turns found by text and vector search, plus the typed facts and
    source turns the graph leads to. Same ``answer_prompt`` as every other strategy."""
    config = config or HybridConfig()
    started = time.monotonic()
    query = index.embedder([QUERY_INSTRUCTION + question])[0]
    turn_ids: list[str] = []
    facts: dict[int, dict[str, Any]] = {}
    queries: list[str] = []

    if "turns" in config.lanes:
        turn_ids += [index.turn_ids[i] for i in _top(index.turn_vecs, query, config.turns_k)]
    if "text" in config.lanes and (text := _safe_query(question)):
        queries.append(text)
        hits = graph.query(
            f"CALL text_search.search_all('{TEXT_INDEX_NAME}', $query) YIELD node, score "
            "WITH node, score ORDER BY score DESC LIMIT $limit RETURN node.action_id AS id",
            {"query": text, "limit": config.text_k},
        )
        turn_ids += [h["id"] for h in hits]
    if "entities" in config.lanes:
        entity_ids = list(dict.fromkeys(index.entity_ids[i] for i in _top(index.entity_vecs, query, config.entities_k)))
        for row in graph.query(
            "MATCH (n:gliner2) WHERE n.entity_id IN $ids MATCH (n)-[r]-() WHERE r.chunk IS NOT NULL "
            "WITH n, r ORDER BY r.confidence DESC WITH n, collect(r) AS rs UNWIND rs[0..$per] AS r "
            "WITH DISTINCT r WITH startNode(r) AS a, r, endNode(r) AS b " + _EDGE_RETURN,
            {"ids": entity_ids, "per": config.edges_per_entity},
        ):
            facts.setdefault(row["id"], row)
    if "facts" in config.lanes:
        edge_ids = [index.edge_ids[i] for i in _top(index.edge_vecs, query, config.facts_k)]
        for row in graph.query("MATCH (a)-[r]->(b) WHERE id(r) IN $ids " + _EDGE_RETURN, {"ids": edge_ids}):
            facts.setdefault(row["id"], row)
    if "user_facts" in config.lanes and index.edge_types:
        # The graph-native lane: every fact the user holds of the relation types
        # the question is about, across all sessions -- what "how many
        # weddings did I attend" needs and top-k similarity over turns cannot
        # gather. Ranked by similarity to keep the context bounded.
        import numpy as np

        user_edges = [i for i, u in enumerate(index.edge_from_user) if u]
        if config.user_fact_types_from == "names":
            wanted = set(index.relation_types(query, config.user_fact_types))
        else:
            # The types of the user's facts most similar to the question: naming
            # the type by embedding its label picked generic ones (spent_time)
            # for almost every question.
            nearest = np.argsort(-(index.edge_vecs[user_edges] @ query))[:10] if user_edges else []
            counts: dict[str, int] = {}
            for i in nearest:
                counts[index.edge_types[user_edges[int(i)]]] = counts.get(index.edge_types[user_edges[int(i)]], 0) + 1
            wanted = set(sorted(counts, key=lambda t: -counts[t])[: config.user_fact_types])
        candidates = [i for i in user_edges if index.edge_types[i] in wanted]
        if candidates:
            scores = index.edge_vecs[candidates] @ query
            chosen = [index.edge_ids[candidates[i]] for i in np.argsort(-scores)[: config.user_facts_k]]
            queries.append(f"user facts: {sorted(wanted)}")
            for row in graph.query("MATCH (a)-[r]->(b) WHERE id(r) IN $ids " + _EDGE_RETURN, {"ids": chosen}):
                facts.setdefault(row["id"], row)
    if facts:
        turn_ids += [f["turn"] for f in facts.values() if f.get("turn")][: config.fact_turns_k]

    # Facts in time order: a knowledge-update question wants the latest value,
    # a temporal one the sequence.
    ordered = sorted(facts.values(), key=lambda f: f.get("valid_at") or "")
    seen = [_fact(f) for f in ordered] + _turn_rows(graph, list(dict.fromkeys(turn_ids)), config.turn_chars)
    answer = await llm.complete(answer_prompt(question, seen, today))
    return Retrieved(
        answer=answer.strip(), retrieval_context=seen, queries=queries, latency_seconds=time.monotonic() - started
    )
