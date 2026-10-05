"""Tests for the hybrid strategy's own part: preparing the graph for recall and answering from it.

What recall retrieves is tested where it lives, in sessions-graph's
``test_e2e_recall.py``. These run against the real eval Memgraph, which has
MAGE in CI, so the vectors are real.
"""

import pytest
from context_graph_eval.hybrid import RecallConfig, ensure_recall_ready, retrieve_hybrid
from context_graph_eval.retrieval import ReadOnlyGraph
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
async def test_the_answerer_reads_exactly_the_rows_recall_found(owned_session: ActionsGraph):
    ensure_recall_ready(owned_session.db)

    result = await retrieve_hybrid(
        "When did I go to the Museum of Modern Art?",
        graph=ReadOnlyGraph(owned_session.db),
        llm=_EchoLLM(),
        user_id="u1",
        config=RecallConfig(lanes=("turns", "text")),
        today="2023-06-01",
    )

    assert result.retrieval_context == [
        "TURN [session s1, 2023-05-30T17:27, user]: I went to the Museum of Modern Art. It was wonderful."
    ]
    assert result.retrieval_context[0] in result.answer
    assert "The question is being asked on 2023-06-01." in result.answer
    assert not result.errors
