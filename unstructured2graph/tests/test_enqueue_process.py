"""Tests for enqueue_texts / process_enqueued_and_finalize against a REAL
LightRAG instance -- real chunking, real doc_status tracking, real Memgraph
writes. Only the LLM completion function is stubbed, so most of these run
free and deterministic without OPENAI_API_KEY, unlike test_e2e_lightrag.py
(which pays for real extraction quality).

A MagicMock standing in for the whole lightrag_wrapper -- the pattern this
file replaces for the enqueue/process integration specifically -- proves
nothing about whether the real apipeline_enqueue_documents /
apipeline_process_enqueue_documents / doc_status contract is actually being
used correctly. It already hid one real defect: apipeline_process_enqueue_
documents swallows a per-document extraction failure and returns normally,
which a mock that just returns None either way cannot surface.

Requires a live Memgraph reachable at MEMGRAPH_URL -- skips cleanly if
unreachable (see conftest.py's `memgraph` fixture).
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from lightrag.base import DocStatus

from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import enqueue_texts, process_enqueued_and_finalize

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


async def _stub_no_entities(prompt, system_prompt=None, history_messages=None, **kwargs):
    """A minimal, valid LightRAG extraction response meaning "nothing found"
    -- real enough for the real pipeline to parse successfully (confirmed:
    doc_status reaches PROCESSED, not FAILED), with no real LLM call."""
    return "<|COMPLETE|>"


def _failing_for(*needles: str):
    """A stub LLM that raises for any prompt containing one of `needles`,
    completing normally otherwise -- simulates a real per-document
    extraction failure without needing a real, flaky one."""

    async def _stub(prompt, system_prompt=None, history_messages=None, **kwargs):
        if any(needle in prompt for needle in needles):
            raise RuntimeError(f"simulated extraction failure ({needles})")
        return "<|COMPLETE|>"

    return _stub


@pytest_asyncio.fixture
async def lightrag_wrapper(memgraph, tmp_path):
    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(
        working_dir=str(tmp_path / "lightrag_storage"),
        llm_model_func=_stub_no_entities,
    )
    yield wrapper
    await wrapper.afinalize()


@pytest.mark.asyncio
async def test_enqueue_texts_stages_without_processing(memgraph, lightrag_wrapper):
    """Real assertion a MagicMock cannot make: doc_status genuinely shows
    the document still pending after enqueue_texts, proving processing
    has not happened yet -- not just that some mock wasn't awaited."""
    grouped = await enqueue_texts(["Alice works at Acme."], memgraph, lightrag_wrapper)
    chunk = grouped[0][0]

    records = await lightrag_wrapper.get_lightrag().doc_status.get_by_ids([chunk.hash])

    assert records[0]["status"] == DocStatus.PENDING.value


@pytest.mark.asyncio
async def test_enqueue_then_process_reaches_processed_status(memgraph, lightrag_wrapper):
    grouped = await enqueue_texts(["Alice works at Acme.", "Bob likes hiking."], memgraph, lightrag_wrapper)
    chunks = [c for group in grouped for c in group]

    statuses = await process_enqueued_and_finalize(memgraph, lightrag_wrapper, chunks)

    assert len(statuses) == len(chunks)
    assert all(record["status"] == DocStatus.PROCESSED.value for record in statuses.values())


@pytest.mark.asyncio
async def test_one_failing_document_is_reported_failed_others_processed(memgraph, tmp_path):
    """The exact defect this pair of functions exists to surface correctly:
    apipeline_process_enqueue_documents swallows a per-document extraction
    error and returns normally regardless -- proven for real here, with a
    genuinely raising LLM function, not assumed from a mock's silence."""
    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(working_dir=str(tmp_path / "lightrag_storage"), llm_model_func=_failing_for("Bob"))
    try:
        grouped = await enqueue_texts(["Alice works at Acme.", "Bob likes hiking."], memgraph, wrapper)
        chunks = [c for group in grouped for c in group]

        statuses = await process_enqueued_and_finalize(memgraph, wrapper, chunks)
    finally:
        await wrapper.afinalize()

    alice_hash, bob_hash = chunks[0].hash, chunks[1].hash
    assert statuses[alice_hash]["status"] == DocStatus.PROCESSED.value
    assert statuses[bob_hash]["status"] == DocStatus.FAILED.value
    assert statuses[bob_hash]["error_msg"]


@pytest.mark.asyncio
async def test_duplicate_chunk_content_collapses_to_one_document(memgraph, lightrag_wrapper):
    """enqueue_texts ids chunks by content hash, so two chunks with
    byte-identical text share an id -- and LightRAG's own enqueue-time
    dedup then treats them as the same document. Verified for real: this is
    what actually explained an earlier, surprising drop in extraction call
    counts when batching many sessions together (real distractor-session
    reuse in the eval corpus), not an unconfirmed cache-hit theory."""
    grouped = await enqueue_texts(["Same text here.", "Same text here."], memgraph, lightrag_wrapper)

    assert grouped[0][0].hash == grouped[1][0].hash

    chunks = [c for group in grouped for c in group]
    statuses = await process_enqueued_and_finalize(memgraph, lightrag_wrapper, chunks)

    assert len(statuses) == 1


@requires_openai_key
@pytest.mark.asyncio
async def test_process_enqueued_and_finalize_links_real_entities_from_a_batch(memgraph, tmp_path):
    """connect_chunks_to_entities now runs once per batch instead of once
    per chunk (see the function's docstring) -- this proves that
    consolidation still links entities from a real multi-document batch,
    not just from a single document the way the old per-chunk call was
    exercised in test_e2e_lightrag.py."""
    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(working_dir=str(tmp_path / "lightrag_storage"))
    try:
        grouped = await enqueue_texts(
            [
                "Alice Johnson works at Acme Corp on the graph database engine.",
                "Bob Smith works at Widget Inc on the mobile app.",
            ],
            memgraph,
            wrapper,
        )
        chunks = [c for group in grouped for c in group]

        await process_enqueued_and_finalize(memgraph, wrapper, chunks)

        workspace = wrapper.get_lightrag().chunk_entity_relation_graph.workspace
        rows = memgraph.query(f"MATCH (e:{workspace})-[:MENTIONED_IN]->(c:Chunk) RETURN count(*) AS count")
    finally:
        await wrapper.afinalize()

    assert rows[0]["count"] > 0
