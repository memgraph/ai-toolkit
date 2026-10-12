"""End-to-end test that exercises real session reconciliation.

Requires:
- A live Memgraph reachable at MEMGRAPH_URL (see conftest.py's `memgraph`/
  `graph` fixtures) -- skips if unreachable.
- The GLiNER2 checkpoint, downloaded on first use: extraction is the default
  GLiNER2 backend, run for real.
- The `sessions-graph[reconciliation]` extra (actions-graph + unstructured2graph).

No LLM key: reconciliation makes no LLM call, and the test unsets both
provider keys to prove it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("actions_graph", reason="actions-graph not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from actions_graph import MessageRole, Session


@pytest.mark.asyncio
async def test_reconcile_session_extracts_real_entities_without_an_llm_key(graph, memgraph, actions_graph, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    session_id = "s-e2e-reconciliation"
    actions_graph.create_session(Session(session_id=session_id))
    actions_graph.record_message(
        session_id=session_id,
        role=MessageRole.USER,
        content="What team does Bob Smith work on?",
    )
    actions_graph.record_message(
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content="Bob Smith works on the Payments team at Northwind Traders.",
    )

    summary = await graph.reconcile_session(session_id, actions_graph=actions_graph)

    assert summary.status == "completed", summary.error
    assert summary.texts_considered == 2
    assert summary.texts_deduped == 2

    rows = memgraph.query(
        "MATCH (s:Session {session_id: $session_id}) RETURN s.reconciliation_status AS status",
        params={"session_id": session_id},
    )
    assert rows[0]["status"] == "completed"

    has_chunk_rows = memgraph.query(
        """
        MATCH (:Session {session_id: $session_id})-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)
        RETURN count(*) AS count
        """,
        params={"session_id": session_id},
    )
    assert has_chunk_rows[0]["count"] > 0

    # GLiNER2, the default backend, writes entities under its own workspace label.
    entity_rows = memgraph.query("MATCH (:gliner2)-[:MENTIONED_IN]->(:Chunk) RETURN count(*) AS count")
    assert entity_rows[0]["count"] > 0
