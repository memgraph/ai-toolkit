"""End-to-end tests for injecting fixtures into an eval database.

Real Memgraph, not a mocked client: these assert that Cypher actually executes
and produces the intended graph shape. Per `context-graph/CONTEXT-MAP.md`, a
test that asserts on a query *string* passed to a mock proves nothing about
whether real Memgraph accepts it.

Requires a dedicated eval instance -- see EVAL_MEMGRAPH_URL in conftest.
"""

import pytest
from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.inject import inject_batch

from actions_graph import ActionsGraph


def _fixture(session_id: str, *, holds_evidence: bool = False) -> SessionFixture:
    return SessionFixture(
        session_id=session_id,
        date="2023/05/20 (Sat) 14:03",
        turns=[
            Turn(role="user", content=f"a user turn in {session_id}"),
            Turn(role="assistant", content=f"an assistant reply in {session_id}"),
        ],
        holds_evidence=holds_evidence,
    )


def test_every_fixture_session_lands_in_the_graph(eval_graph: ActionsGraph):
    inject_batch([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert eval_graph.get_session("s1") is not None
    assert eval_graph.get_session("s2") is not None


def test_turns_land_as_actions_under_their_session(eval_graph: ActionsGraph):
    inject_batch([_fixture("s1")], graph=eval_graph)

    actions = eval_graph.get_session_actions("s1")

    assert [a.content for a in actions] == [
        "a user turn in s1",
        "an assistant reply in s1",
    ]


def test_a_batch_starts_from_a_clean_graph(eval_graph: ActionsGraph):
    """#309: each batch runs against known, fixed state. A previous batch's
    sessions surviving would let a question be answered from the last run's
    data rather than this batch's fixtures."""
    inject_batch([_fixture("stale")], graph=eval_graph)

    inject_batch([_fixture("fresh")], graph=eval_graph)

    assert eval_graph.get_session("stale") is None
    assert eval_graph.get_session("fresh") is not None


def test_a_session_shared_by_two_questions_is_written_once(eval_graph: ActionsGraph):
    """Upstream reuses distractor sessions across questions, and repeated ids
    carry identical content -- so a repeat is the same session, not a new one.
    Writing it per occurrence would append its turns again on every reuse,
    duplicating content in the graph and paying to reconcile each copy."""
    shared = _fixture("shared")

    inject_batch([shared, _fixture("unique"), shared], graph=eval_graph)

    actions = eval_graph.get_session_actions("shared")
    assert [a.content for a in actions] == [
        "a user turn in shared",
        "an assistant reply in shared",
    ]


def test_deduplicated_sessions_are_not_counted_twice(eval_graph: ActionsGraph):
    shared = _fixture("shared")

    written = inject_batch([shared, shared, _fixture("unique")], graph=eval_graph)

    assert written.sessions == 2
    assert written.turns == 4


def test_a_batch_also_clears_memory_tier_nodes_from_the_previous_run(eval_graph: ActionsGraph):
    """ActionsGraph.clear() only removes Session|Agent|Action|Tool -- it leaves
    Chunk, Entity, Episode and Memory standing. Those are exactly what
    reconciliation produces, so relying on it alone would let the previous
    batch's *distilled memory* survive into this one, and a question could be
    answered from the last run rather than from this batch's fixtures. That is
    the leak #309 exists to prevent.
    """
    eval_graph._db.query("CREATE (:Chunk {text: 'from a previous batch'})")
    eval_graph._db.query("CREATE (:Episode {summary: 'from a previous batch'})")

    inject_batch([_fixture("fresh")], graph=eval_graph)

    survivors = eval_graph._db.query("MATCH (n) WHERE n:Chunk OR n:Episode RETURN count(n) AS n")
    assert survivors[0]["n"] == 0


def test_injection_reports_what_it_wrote(eval_graph: ActionsGraph):
    written = inject_batch([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert written.sessions == 2
    assert written.turns == 4


def test_evidence_marking_never_reaches_the_graph(eval_graph: ActionsGraph):
    """holds_evidence is corpus-side bookkeeping. Writing it would hand
    retrieval the answer's location -- the thing under test getting told where
    to look."""
    inject_batch([_fixture("s1", holds_evidence=True)], graph=eval_graph)

    rows = eval_graph._db.query("MATCH (n) UNWIND keys(n) AS key RETURN collect(DISTINCT key) AS keys")
    written_keys = set(rows[0]["keys"])

    assert "holds_evidence" not in written_keys


def test_reconciliation_is_left_pending_for_each_session(eval_graph: ActionsGraph):
    """Injection stages content; it does not distil it. Reconciliation is a
    separate, LLM-backed pass that the runner triggers afterwards."""
    inject_batch([_fixture("s1")], graph=eval_graph)

    rows = eval_graph._db.query("MATCH (s:Session {session_id: 's1'}) RETURN s.reconciliation_status AS status")

    assert rows[0]["status"] == "pending"


@pytest.mark.parametrize("bad_id", ["", None])
def test_a_fixture_without_a_session_id_is_refused(eval_graph: ActionsGraph, bad_id):
    """A blank id would collapse every fixture onto one node, silently merging
    distinct sessions and destroying the haystack."""
    with pytest.raises(ValueError, match="session_id"):
        inject_batch([_fixture("ok"), _fixture(bad_id)], graph=eval_graph)
