"""End-to-end tests for triggering reconciliation over an eval batch.

Reconciliation is LLM-backed, so the tests that actually distil content are
gated on OPENAI_API_KEY -- the family's existing `requires_openai_key`
convention. The tests that are *not* about LLM behaviour (which sessions get
picked up, what happens when none are pending) run without a key, so a
contributor without one still gets meaningful coverage.
"""

import os

import pytest

# Resolve from context-graph's config file before deciding to skip, so a
# contributor whose key lives in config.toml (ADR 0002) rather than the
# environment still runs these instead of silently skipping them.
from conftest import EVAL_MEMGRAPH_URL
from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.inject import PENDING, inject_batch
from context_graph_eval.reconcile import (
    _resolve_llm_credentials,
    pending_sessions,
    reconcile_batch,
)

from actions_graph import ActionsGraph

_resolve_llm_credentials()

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="no OPENAI_API_KEY in env or context-graph config",
)


def _fixture(session_id: str) -> SessionFixture:
    return SessionFixture(
        session_id=session_id,
        date="2023/05/20 (Sat) 14:03",
        turns=[
            Turn(role="user", content=f"I adopted a beagle named Max, in {session_id}"),
            Turn(role="assistant", content="Congratulations on Max!"),
        ],
        holds_evidence=True,
    )


def test_every_injected_session_is_pending(eval_graph: ActionsGraph):
    inject_batch([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert set(pending_sessions(eval_graph._db)) == {"s1", "s2"}


def test_nothing_is_pending_on_an_empty_graph(eval_graph: ActionsGraph):
    assert pending_sessions(eval_graph._db) == []


def test_pending_can_be_limited(eval_graph: ActionsGraph):
    """A batch is reconciled in bounded chunks: an LLM-backed pass over every
    session of a full run is slow and costly, so a caller needs to be able to
    take a slice."""
    inject_batch([_fixture(f"s{i}") for i in range(5)], graph=eval_graph)

    assert len(pending_sessions(eval_graph._db, limit=2)) == 2


@requires_openai_key
async def test_reconciling_a_batch_clears_its_pending_flag(eval_graph: ActionsGraph):
    inject_batch([_fixture("s1")], graph=eval_graph)

    await reconcile_batch(eval_graph._db, limit=1, memgraph_url=EVAL_MEMGRAPH_URL)

    rows = eval_graph._db.query("MATCH (s:Session {session_id: 's1'}) RETURN s.reconciliation_status AS status")
    assert rows[0]["status"] != PENDING


@requires_openai_key
async def test_reconciling_produces_chunks_linked_to_their_source_action(eval_graph: ActionsGraph):
    """The semantic-memory half: injected turns become Chunks traceable back to
    the Action that produced them."""
    inject_batch([_fixture("s1")], graph=eval_graph)

    await reconcile_batch(eval_graph._db, limit=1, memgraph_url=EVAL_MEMGRAPH_URL)

    rows = eval_graph._db.query(
        "MATCH (:Session {session_id: 's1'})-[:HAS_ACTION]->(a)-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n"
    )
    assert rows[0]["n"] > 0


@requires_openai_key
async def test_reconciling_produces_an_episode_for_the_session(eval_graph: ActionsGraph):
    """The episodic-memory half of the same pass."""
    inject_batch([_fixture("s1")], graph=eval_graph)

    await reconcile_batch(eval_graph._db, limit=1, memgraph_url=EVAL_MEMGRAPH_URL)

    rows = eval_graph._db.query(
        "MATCH (:Session {session_id: 's1'})-[:HAS_EPISODE]->(e:Episode) RETURN e.summary AS summary"
    )
    assert rows and rows[0]["summary"]


@requires_openai_key
async def test_reconcile_reports_per_session_outcomes(eval_graph: ActionsGraph):
    inject_batch([_fixture("s1")], graph=eval_graph)

    result = await reconcile_batch(eval_graph._db, limit=1, memgraph_url=EVAL_MEMGRAPH_URL)

    assert result.reconciled == 1
    assert result.failed == 0


async def test_one_failing_session_does_not_abandon_the_batch(eval_graph: ActionsGraph):
    """Runs without an LLM key by injecting a wrapper that always fails -- this
    is about batch resilience, not extraction quality, so the LLM boundary is
    legitimately stubbed here (see CONTEXT-MAP's testing tiers).

    A score is only meaningful if you know how much of the graph is actually
    populated, so a batch must report partial failure rather than raising on
    the first bad session and leaving the rest silently unattempted.
    """

    class AlwaysFails:
        async def initialize(self, **_kwargs):
            return self

        def __getattr__(self, name):
            raise RuntimeError(f"extraction unavailable ({name})")

    inject_batch([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    result = await reconcile_batch(eval_graph._db, lightrag_wrapper=AlwaysFails())

    assert result.reconciled == 0
    assert result.failed == 2
    assert len(result.errors) == 2


async def test_reconciling_an_empty_batch_is_a_no_op(eval_graph: ActionsGraph):
    """Runs without an LLM key: nothing pending means no wrapper is ever
    initialised, so this must not require one."""
    result = await reconcile_batch(eval_graph._db)

    assert result.reconciled == 0
    assert result.failed == 0
