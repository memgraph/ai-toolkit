"""Tests for answering's own part: preparing the graph for recall, answering from it, and staying read-only.

What recall retrieves is tested where it lives, in sessions-graph's
``test_e2e_recall.py``. These run against the real eval Memgraph, which has
MAGE in CI, so the vectors are real.
"""

import pytest
from context_graph_eval.answering import (
    ReadOnlyGraph,
    RecallConfig,
    Retrieved,
    WriteRefusedError,
    answer_all,
    ensure_recall_ready,
    retrieve,
)
from sessions_graph.embeddings import EmbeddingUnavailableError, check_available

from actions_graph import ActionsGraph, MessageRole, Session


class _EchoLLM:
    async def complete(self, prompt: str) -> str:
        return prompt


@pytest.fixture
def owned_session(eval_graph: ActionsGraph):
    try:
        check_available(eval_graph.db)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"eval Memgraph can't embed (needs MAGE): {exc}")
    eval_graph.ensure_session(Session(session_id="s1", started_at="2023-05-30T17:27:00+00:00"))
    eval_graph.record_message(
        session_id="s1",
        role=MessageRole.USER,
        content="I went to the Museum of Modern Art. It was wonderful.",
        timestamp="2023-05-30T17:27:00+00:00",
    )
    eval_graph.db.query(
        "MERGE (u:User {user_id: 'u1'}) WITH u MATCH (s:Session {session_id: 's1'}) MERGE (u)-[:HAD_SESSION]->(s)"
    )
    return eval_graph


def test_ensure_recall_ready_embeds_what_is_missing_once(owned_session: ActionsGraph):
    assert ensure_recall_ready(owned_session.db) == 1
    assert ensure_recall_ready(owned_session.db) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("packing", [None, "turns"])
async def test_the_answerer_reads_exactly_the_rows_recall_found(owned_session: ActionsGraph, packing):
    ensure_recall_ready(owned_session.db)

    result = await retrieve(
        "When did I go to the Museum of Modern Art?",
        graph=ReadOnlyGraph(owned_session.db),
        llm=_EchoLLM(),
        user_id="u1",
        config=None if packing is None else RecallConfig(lanes=("turns", "text"), packing=packing),
        today="2023-06-01",
    )

    expected = ["TURN [session s1, 2023-05-30T17:27, user]: I went to the Museum of Modern Art. It was wonderful."]
    if packing is None:
        expected.insert(0, "EXCHANGE [session s1, 2023-05-30T17:27:00]:")
    assert result.retrieval_context == expected
    assert "\n".join(expected) in result.answer
    assert "The question is being asked on 2023-06-01." in result.answer
    assert not result.errors


def test_setup_is_retried_while_memgraph_holds_the_storage(monkeypatch):
    from context_graph_eval import answering
    from neo4j.exceptions import TransientError

    class Graph:
        calls = 0

        def setup(self):
            self.calls += 1
            if self.calls < 3:
                raise TransientError("Cannot get unique access to the storage")

    monkeypatch.setattr(answering.time, "sleep", lambda _: None)
    graph = Graph()

    answering.setup_with_retry(graph)

    assert graph.calls == 3


def test_retrieval_cannot_write_to_the_graph_it_is_scored_against(eval_graph: ActionsGraph):
    graph = ReadOnlyGraph(eval_graph.db)

    with pytest.raises(WriteRefusedError):
        graph.query("CREATE (:Planted {answer: 'yes'})")

    assert eval_graph.db.query("MATCH (n:Planted) RETURN count(n) AS n")[0]["n"] == 0


@pytest.mark.asyncio
async def test_a_question_whose_retrieval_raises_is_an_empty_timed_answer_not_a_shorter_run():
    async def answer(question: str) -> Retrieved:
        if question == "bad":
            raise RuntimeError("recall exploded")
        return Retrieved(answer=question)

    results = await answer_all(["good", "bad", "also good"], answer, max_concurrent=2)

    assert [r.answer for r in results] == ["good", "", "also good"]
    assert results[1].errors == ["recall exploded"]
    assert results[1].recall_seconds is None
