"""End-to-end test that exercises real session reconciliation.

Requires:
- A live Memgraph reachable at MEMGRAPH_URL (see conftest.py's `memgraph`/
  `graph` fixtures) -- skips if unreachable.
- MEMGRAPH_URL (and MEMGRAPH_USER/MEMGRAPH_PASSWORD/MEMGRAPH_DATABASE if
  non-default) set as literal environment variables, not just left at
  memgraph-toolbox's Python-level defaults: LightRAG's own storage backends
  read these directly and raise (not skip) if MEMGRAPH_URL is unset, even
  when a real Memgraph is reachable at the default bolt://localhost:7687.
- OPENAI_API_KEY -- skips if unset (mirrors skills-graph's
  tests/test_connector_e2e.py `requires_openai_key` convention).
- The `sessions-graph[reconciliation]` extra (actions-graph + unstructured2graph).

Embedding stays at MemgraphLightRAGWrapper's default (Memgraph's own local
sentence-transformer via the mage image's `embeddings` module), so only the
LLM completion call costs anything.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

pytest.importorskip("actions_graph", reason="actions-graph not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from actions_graph import MessageRole, Session
from lightrag_memgraph import MemgraphLightRAGWrapper

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest_asyncio.fixture
async def lightrag_wrapper(memgraph, tmp_path):
    # Skip here too, not just via the test's `requires_openai_key` marker --
    # LightRAG initialization must never run before the marker's skip takes
    # effect.
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(working_dir=str(tmp_path / "lightrag_storage"))
    yield wrapper
    await wrapper.afinalize()


@requires_openai_key
@pytest.mark.asyncio
async def test_reconcile_session_extracts_real_entities_from_session_content(
    graph, memgraph, actions_graph, lightrag_wrapper
):
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
    graph.save_memory(
        user_id="alice",
        content="User is investigating the Payments team's on-call rotation.",
        session_id=session_id,
    )

    summary = await graph.reconcile_session(session_id, lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "completed"
    assert summary.texts_considered == 3
    assert summary.texts_deduped == 3
    assert summary.summary_written is True

    rows = memgraph.query(
        "MATCH (s:Session {session_id: $session_id}) RETURN s.reconciliation_status AS status",
        params={"session_id": session_id},
    )
    assert rows[0]["status"] == "completed"

    episode_rows = memgraph.query(
        "MATCH (:Session {session_id: $session_id})-[:HAS_EPISODE]->(e:Episode) "
        "RETURN e.summary AS summary, e.summarized_at AS summarized_at",
        params={"session_id": session_id},
    )
    assert len(episode_rows) == 1
    assert episode_rows[0]["summary"]
    assert episode_rows[0]["summarized_at"]

    has_chunk_rows = memgraph.query(
        """
        MATCH (:Session {session_id: $session_id})-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)
        RETURN count(*) AS count
        """,
        params={"session_id": session_id},
    )
    assert has_chunk_rows[0]["count"] > 0

    workspace = lightrag_wrapper.get_lightrag().chunk_entity_relation_graph.workspace
    entity_rows = memgraph.query(f"MATCH (:{workspace})-[:MENTIONED_IN]->(:Chunk) RETURN count(*) AS count")
    assert entity_rows[0]["count"] > 0

    # Re-reconciling the same session must update the one Episode, not accumulate another.
    second_summary = await graph.reconcile_session(
        session_id, lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
    )
    assert second_summary.summary_written is True
    episode_rows_after_rerun = memgraph.query(
        "MATCH (:Session {session_id: $session_id})-[:HAS_EPISODE]->(e:Episode) RETURN count(e) AS count",
        params={"session_id": session_id},
    )
    assert episode_rows_after_rerun[0]["count"] == 1
