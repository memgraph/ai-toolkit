"""Vectors for recall, computed inside Memgraph by MAGE's ``embeddings`` module (#393).

Three units carry an ``embedding``:

- user and assistant messages: ``Action.text``, which actions-graph writes;
- entities: the ``text`` of a node mentioned in one of the session's chunks;
- extracted edges: the fact and the sentence it was read from, as
  ``"<head> <type> <tail>. <r.text>"`` -- what the hybrid retrieval
  benchmark embedded, so a question can match the relation as well as the words.

Each also carries ``embedding_model``, the model that produced it. A vector
from another model is found by the same query as a missing one and replaced,
so vectors from two models never sit side by side.

Embedding runs inside Memgraph, not on the host: the model loads once in the
database process instead of in every hook or CLI process that needs it.
Without MAGE (plain ``memgraph/memgraph``) or without the model, every call
raises :class:`EmbeddingUnavailableError`; the session records why, and
``sessions-graph embed --pending`` retries later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What the hybrid retrieval benchmark was measured with (map #390).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: Texts per ``embeddings.text`` call: bounds one query's payload, not a tuning knob.
BATCH_SIZE = 64

# A session's actions hang off the Session directly or off one of its Agents,
# the same two paths actions-graph's get_session_actions follows.
_SESSION_ACTIONS = "MATCH (s:Session {session_id: $session_id})-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action) "
_STALE = "(x.embedding IS NULL OR x.embedding_model IS NULL OR x.embedding_model <> $model)"

_MESSAGES = (
    _SESSION_ACTIONS
    + "WITH DISTINCT a AS x WHERE x.text IS NOT NULL AND "
    + _STALE
    + " RETURN id(x) AS id, x.text AS text"
)
_ENTITIES = (
    _SESSION_ACTIONS + "MATCH (a)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(n) "
    "WITH DISTINCT n AS x WHERE x.text IS NOT NULL AND " + _STALE + " RETURN id(x) AS id, x.text AS text"
)
# Through the session's own entities rather than a scan of every edge's
# `chunk`: relationship properties carry no index.
_EDGES = (
    _SESSION_ACTIONS + "MATCH (a)-[:HAS_CHUNK]->(c:Chunk)<-[:MENTIONED_IN]-(n)-[x]-() "
    "WHERE x.chunk = c.hash AND x.text IS NOT NULL AND " + _STALE + " "
    "WITH DISTINCT x RETURN id(startNode(x)) AS head, id(x) AS id, "
    "coalesce(startNode(x).text, 'user') + ' ' + replace(type(x), '_', ' ') + ' ' "
    "+ coalesce(endNode(x).text, 'user') + '. ' + x.text AS text"
)

_SET_NODES = (
    "UNWIND $rows AS row MATCH (x) WHERE id(x) = row.id SET x.embedding = row.vector, x.embedding_model = $model"
)
# Anchored on the start node so Memgraph looks it up by id instead of scanning every edge.
_SET_EDGES = (
    "UNWIND $rows AS row MATCH (h)-[x]->() WHERE id(h) = row.head AND id(x) = row.id "
    "SET x.embedding = row.vector, x.embedding_model = $model"
)


class EmbeddingUnavailableError(RuntimeError):
    """Memgraph can't embed: no MAGE ``embeddings`` module, or the model failed to load or run."""


@dataclass(frozen=True)
class Embedded:
    """How many of each unit one call embedded."""

    messages: int = 0
    entities: int = 0
    edges: int = 0


def embed_texts(db: Any, texts: list[str], model: str) -> list[list[float]]:
    """``texts``' vectors from ``model``, computed by Memgraph.

    Raises:
        EmbeddingUnavailableError: the module is missing, or it reported failure
            (a model that can't be downloaded or loaded does this).
    """
    try:
        rows = db.query(
            "CALL embeddings.text($texts, {model_name: $model}) YIELD success, embeddings RETURN success, embeddings",
            {"texts": texts, "model": model},
        )
    except Exception as exc:
        raise EmbeddingUnavailableError(f"embeddings.text failed: {exc}") from exc
    if not rows or not rows[0]["success"] or rows[0]["embeddings"] is None:
        raise EmbeddingUnavailableError(f"embeddings.text could not embed with model {model!r}")
    vectors = rows[0]["embeddings"]
    if len(vectors) != len(texts):
        raise EmbeddingUnavailableError(f"embeddings.text returned {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def check_available(db: Any, model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """The model's dimension, after embedding one probe text; loads the model if it isn't yet.

    Raises:
        EmbeddingUnavailableError: as :func:`embed_texts`.
    """
    return len(embed_texts(db, ["probe"], model)[0])


def embed_session(db: Any, session_id: str, model: str = DEFAULT_EMBEDDING_MODEL) -> Embedded:
    """Embed every unit of ``session_id`` that has no vector from ``model``.

    Idempotent: what is already embedded with ``model`` is skipped, so the
    session-end step and reconciliation can both call it -- the first finds
    only messages, the second the entities and edges extraction just wrote.

    Raises:
        EmbeddingUnavailableError: as :func:`embed_texts`. Units embedded before
            the failure keep their vectors.
    """
    params = {"session_id": session_id, "model": model}
    messages = _embed_rows(db, db.query(_MESSAGES, params), model, _SET_NODES)
    entities = _embed_rows(db, db.query(_ENTITIES, params), model, _SET_NODES)
    edges = _embed_rows(db, db.query(_EDGES, params), model, _SET_EDGES)
    return Embedded(messages=messages, entities=entities, edges=edges)


def _embed_rows(db: Any, rows: list[dict[str, Any]], model: str, write: str) -> int:
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        vectors = embed_texts(db, [row["text"] for row in batch], model)
        db.query(
            write,
            {"rows": [{**row, "vector": vector} for row, vector in zip(batch, vectors, strict=True)], "model": model},
        )
    return len(rows)
