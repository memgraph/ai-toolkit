"""End-to-end tests for reconciling an eval graph's pending sessions.

Reconciliation extracts with local GLiNER2, so none of these needs an LLM key.
"""

from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.inject import PENDING, inject
from context_graph_eval.reconcile import pending_sessions, reconcile_pending

from actions_graph import ActionsGraph


def _fixture(session_id: str) -> SessionFixture:
    return SessionFixture(
        session_id=session_id,
        date="2023/05/20 (Sat) 14:03",
        turns=[
            Turn(role="user", content=f"I adopted a beagle named Max, in {session_id}"),
            Turn(role="assistant", content="Congratulations on Max!"),
        ],
        holds_evidence=True,
        user_id="u1",
    )


def test_every_injected_session_is_pending(eval_graph: ActionsGraph):
    inject([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert set(pending_sessions(eval_graph._db)) == {"s1", "s2"}


def test_nothing_is_pending_on_an_empty_graph(eval_graph: ActionsGraph):
    assert pending_sessions(eval_graph._db) == []


def test_pending_can_be_limited(eval_graph: ActionsGraph):
    inject([_fixture(f"s{i}") for i in range(5)], graph=eval_graph)

    assert len(pending_sessions(eval_graph._db, limit=2)) == 2


async def test_reconciling_clears_the_pending_flag_and_reports_each_session(eval_graph: ActionsGraph):
    inject([_fixture("s1")], graph=eval_graph)

    result = await reconcile_pending(eval_graph._db, progress=False)

    rows = eval_graph._db.query("MATCH (s:Session {session_id: 's1'}) RETURN s.reconciliation_status AS status")
    assert rows[0]["status"] != PENDING
    assert (result.reconciled, result.failed) == (1, 0)


async def test_reconciling_produces_chunks_linked_to_their_source_action(eval_graph: ActionsGraph):
    """Injected turns become Chunks traceable back to the Action that produced them."""
    inject([_fixture("s1")], graph=eval_graph)

    await reconcile_pending(eval_graph._db, progress=False)

    rows = eval_graph._db.query(
        "MATCH (:Session {session_id: 's1'})-[:HAS_ACTION]->(a)-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n"
    )
    assert rows[0]["n"] > 0


async def test_one_failing_session_does_not_abandon_the_build(eval_graph: ActionsGraph, monkeypatch):
    """A build takes hours; a session that raises is reported, and the rest
    are still reconciled. The extraction boundary is stubbed: a real
    Memgraph can't be made to raise inside one session's extraction."""
    from sessions_graph.reconciler import Reconciler

    real = Reconciler.reconcile

    async def flaky(self, session_id, **kwargs):
        if session_id == "s1":
            raise RuntimeError("extraction unavailable")
        return await real(self, session_id, **kwargs)

    monkeypatch.setattr(Reconciler, "reconcile", flaky)
    inject([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    result = await reconcile_pending(eval_graph._db, progress=False)

    assert (result.reconciled, result.failed) == (1, 1)
    assert result.errors == ("s1: extraction unavailable",)
    # Left pending, so a resumed build tries it again.
    assert pending_sessions(eval_graph._db) == ["s1"]


async def test_reconciling_nothing_is_a_no_op(eval_graph: ActionsGraph):
    result = await reconcile_pending(eval_graph._db, progress=False)

    assert (result.reconciled, result.failed) == (0, 0)
