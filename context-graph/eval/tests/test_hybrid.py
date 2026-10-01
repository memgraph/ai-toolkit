"""Tests for the hybrid retrieval strategy, against the real eval Memgraph.

The embedder is a deterministic bag-of-words stand-in, so no model is
downloaded; everything the strategy reads from and writes to the graph is
real.
"""

import hashlib
import re

import numpy as np
import pytest
from context_graph_eval.hybrid import HybridConfig, ensure_hybrid_index, retrieve_hybrid
from context_graph_eval.retrieval import ReadOnlyGraph

from actions_graph import ActionsGraph, MessageRole, Session


class _BagOfWords:
    def __call__(self, texts):
        out = np.zeros((len(texts), 64), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class _EchoLLM:
    async def complete(self, prompt: str) -> str:
        return prompt


def _plant(graph: ActionsGraph) -> None:
    """One reconciled-looking session: two turns, their chunk, an entity and a typed fact
    carrying the provenance extraction writes (source turn, speaker, sentence)."""
    graph.ensure_session(Session(session_id="s1", started_at="2023-05-30T17:27:00+00:00"))
    graph.record_message(
        session_id="s1",
        role=MessageRole.USER,
        content="I went to the Museum of Modern Art. It was wonderful.",
        timestamp="2023-05-30T17:27:00+00:00",
    )
    graph.record_message(
        session_id="s1",
        role=MessageRole.ASSISTANT,
        content="Glad you enjoyed it!",
        timestamp="2023-05-30T17:27:01+00:00",
    )
    db = graph.db
    user_turn = db.query("MATCH (a:UserMessage) RETURN a.action_id AS id")[0]["id"]
    db.query("MATCH (a:Action) MERGE (c:Chunk {hash: 'c1', text: 'session'}) MERGE (a)-[:HAS_CHUNK]->(c)")
    db.query(
        "MATCH (c:Chunk {hash: 'c1'}) "
        "CREATE (n:gliner2:Location {entity_id: 'moma', entity_type: 'Location', text: 'Museum of Modern Art'})"
        "-[:MENTIONED_IN {sources: [$turn]}]->(c) "
        "CREATE (u:User {user_id: 'u1'}) "
        "CREATE (u)-[:visited {chunk: 'c1', source_id: $turn, role: 'user', confidence: 0.9, "
        "valid_at: datetime('2023-05-30T17:27:00+00:00'), text: 'I went to the Museum of Modern Art.'}]->(n)",
        {"turn": user_turn},
    )


@pytest.mark.asyncio
async def test_hybrid_hands_the_answerer_facts_and_their_source_turn(eval_graph: ActionsGraph, tmp_path):
    """The facts lane alone reaches the turn: through the edge's source_id, not a turn search."""
    _plant(eval_graph)
    index = ensure_hybrid_index(eval_graph, tmp_path / "index.npz", embedder=_BagOfWords())

    result = await retrieve_hybrid(
        "When did I visit the Museum of Modern Art?",
        graph=ReadOnlyGraph(eval_graph.db),
        llm=_EchoLLM(),
        index=index,
        config=HybridConfig(lanes=("facts",)),
    )

    assert (
        'FACT: user -[visited @ 2023-05-30]-> Museum of Modern Art -- user: "I went to the Museum of Modern Art."'
        in result.retrieval_context
    )
    assert any(row.startswith("TURN [session s1, 2023-05-30T17:27, user]") for row in result.retrieval_context)


@pytest.mark.asyncio
async def test_dropping_the_graph_lanes_leaves_turns_only(eval_graph: ActionsGraph, tmp_path):
    _plant(eval_graph)
    index = ensure_hybrid_index(eval_graph, tmp_path / "index.npz", embedder=_BagOfWords())

    result = await retrieve_hybrid(
        "Museum of Modern Art",
        graph=ReadOnlyGraph(eval_graph.db),
        llm=_EchoLLM(),
        index=index,
        config=HybridConfig(lanes=("turns", "text")),
    )

    assert result.retrieval_context
    assert not any(row.startswith("FACT:") for row in result.retrieval_context)


def test_index_is_cached_between_runs_over_the_same_graph(eval_graph: ActionsGraph, tmp_path):
    _plant(eval_graph)
    calls = []

    class _Counting(_BagOfWords):
        def __call__(self, texts):
            calls.append(len(texts))
            return super().__call__(texts)

    ensure_hybrid_index(eval_graph, tmp_path / "index.npz", embedder=_Counting())
    embedded = len(calls)
    ensure_hybrid_index(eval_graph, tmp_path / "index.npz", embedder=_Counting())
    assert len(calls) == embedded


@pytest.mark.asyncio
async def test_user_facts_gathers_the_users_facts_of_a_relation_type_across_sessions(
    eval_graph: ActionsGraph, tmp_path
):
    """The aggregation a counting question needs: every wedding the user attended, whichever session it was in."""
    db = eval_graph.db
    db.query("CREATE (:User {user_id: 'u1'})")
    for session, couple in (("s1", "Rachel and Mike"), ("s2", "Emily and Sarah"), ("s3", "Jen and Tom")):
        eval_graph.ensure_session(Session(session_id=session, started_at="2023-05-30T17:27:00+00:00"))
        eval_graph.record_message(
            session_id=session, role=MessageRole.USER, content=f"I attended the wedding of {couple}."
        )
        db.query(
            "MATCH (a:Action) WHERE NOT (a)-[:HAS_CHUNK]->() MERGE (c:Chunk {hash: $s, text: 'x'}) MERGE (a)-[:HAS_CHUNK]->(c) "
            "WITH c MATCH (u:User {user_id: 'u1'}) "
            "CREATE (n:gliner2:Event {entity_id: $s, entity_type: 'Event', text: $w})-[:MENTIONED_IN]->(c) "
            "CREATE (u)-[:attended {chunk: $s, confidence: 0.9}]->(n)",
            {"s": session, "w": f"wedding of {couple}"},
        )
    index = ensure_hybrid_index(eval_graph, tmp_path / "index.npz", embedder=_BagOfWords())

    result = await retrieve_hybrid(
        "How many weddings have I attended?",
        graph=ReadOnlyGraph(db),
        llm=_EchoLLM(),
        index=index,
        config=HybridConfig(lanes=("user_facts",), user_fact_types=1),
    )

    facts = [row for row in result.retrieval_context if row.startswith("FACT:")]
    assert len(facts) == 3
    assert all(row.startswith("FACT: user -[attended]-> wedding of") for row in facts)
