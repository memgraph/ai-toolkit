"""End-to-end tests for injecting fixtures into an eval database.

Real Memgraph, not a mocked client: these assert that Cypher actually executes
and produces the intended graph shape. Per `context-graph/CONTEXT-MAP.md`, a
test that asserts on a query *string* passed to a mock proves nothing about
whether real Memgraph accepts it.

Requires a dedicated eval instance -- see EVAL_MEMGRAPH_URL in conftest.
"""

import pytest
from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.inject import inject, wipe

from actions_graph import ActionsGraph
from actions_graph.models import Message


def _fixture(session_id: str, *, holds_evidence: bool = False, user_id: str = "u1") -> SessionFixture:
    return SessionFixture(
        session_id=session_id,
        date="2023/05/20 (Sat) 14:03",
        turns=[
            Turn(role="user", content=f"a user turn in {session_id}"),
            Turn(role="assistant", content=f"an assistant reply in {session_id}"),
        ],
        holds_evidence=holds_evidence,
        user_id=user_id,
    )


def test_every_fixture_session_lands_in_the_graph(eval_graph: ActionsGraph):
    inject([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert eval_graph.get_session("s1") is not None
    assert eval_graph.get_session("s2") is not None


def test_turns_land_as_actions_under_their_session(eval_graph: ActionsGraph):
    inject([_fixture("s1")], graph=eval_graph)

    actions = eval_graph.get_session_actions("s1")

    messages = [a for a in actions if isinstance(a, Message)]
    assert len(messages) == len(actions)
    assert [m.content for m in messages] == [
        "a user turn in s1",
        "an assistant reply in s1",
    ]


def test_injection_keeps_what_an_earlier_injection_wrote(eval_graph: ActionsGraph):
    """A build resumes rather than starting over: injection never clears, and a
    session already written whole is neither duplicated nor rewritten."""
    inject([_fixture("s1")], graph=eval_graph)

    written = inject([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert (written.sessions, written.kept) == (1, 1)
    assert len(eval_graph.get_session_actions("s1")) == 2
    assert eval_graph.get_session("s2") is not None


def test_a_session_cut_off_mid_write_is_written_again(eval_graph: ActionsGraph):
    """A session without its pending mark was interrupted after some of its
    turns landed; resuming must not leave it short, nor append to it."""
    from actions_graph import MessageRole, Session

    eval_graph.ensure_session(Session(session_id="s1", started_at="2023-05-20T14:03:00+00:00"))
    eval_graph.record_message(session_id="s1", role=MessageRole.USER, content="a user turn in s1")

    written = inject([_fixture("s1")], graph=eval_graph)

    assert (written.sessions, written.kept) == (1, 0)
    assert [a.content for a in eval_graph.get_session_actions("s1") if isinstance(a, Message)] == [
        "a user turn in s1",
        "an assistant reply in s1",
    ]


def test_a_session_shared_by_two_questions_is_written_once(eval_graph: ActionsGraph):
    """Upstream reuses distractor sessions across questions, and repeated ids
    carry identical content -- so a repeat is the same session, not a new one.
    Writing it per occurrence would append its turns again on every reuse,
    duplicating content in the graph and paying to reconcile each copy."""
    shared = _fixture("shared")

    inject([shared, _fixture("unique"), shared], graph=eval_graph)

    actions = eval_graph.get_session_actions("shared")
    messages = [a for a in actions if isinstance(a, Message)]
    assert len(messages) == len(actions)
    assert [m.content for m in messages] == [
        "a user turn in shared",
        "an assistant reply in shared",
    ]


def test_deduplicated_sessions_are_not_counted_twice(eval_graph: ActionsGraph):
    shared = _fixture("shared")

    written = inject([shared, shared, _fixture("unique")], graph=eval_graph)

    assert written.sessions == 2
    assert written.turns == 4


def test_wipe_also_clears_memory_tier_nodes(eval_graph: ActionsGraph):
    """ActionsGraph.clear() only removes Session|Agent|Action|Tool -- it leaves
    Chunk, Entity and Memory standing. Those are exactly what reconciliation
    produces or reads, so relying on it alone would let the previous build's
    index survive into a rebuild (#309)."""
    eval_graph._db.query("CREATE (:Chunk {text: 'from a previous build'})")
    eval_graph._db.query("CREATE (:Memory {content: 'from a previous build'})")

    wipe(eval_graph)

    survivors = eval_graph._db.query("MATCH (n) RETURN count(n) AS n")
    assert survivors[0]["n"] == 0


def test_injection_reports_what_it_wrote(eval_graph: ActionsGraph):
    written = inject([_fixture("s1"), _fixture("s2")], graph=eval_graph)

    assert written.sessions == 2
    assert written.turns == 4


def test_evidence_marking_never_reaches_the_graph(eval_graph: ActionsGraph):
    """holds_evidence is corpus-side bookkeeping. Writing it would hand
    retrieval the answer's location -- the thing under test getting told where
    to look."""
    inject([_fixture("s1", holds_evidence=True)], graph=eval_graph)

    rows = eval_graph._db.query("MATCH (n) UNWIND keys(n) AS key RETURN collect(DISTINCT key) AS keys")
    written_keys = set(rows[0]["keys"])

    assert "holds_evidence" not in written_keys


def test_reconciliation_is_left_pending_for_each_session(eval_graph: ActionsGraph):
    """Injection stages content; it does not distil it. Reconciliation is a
    separate pass the build runs afterwards."""
    inject([_fixture("s1")], graph=eval_graph)

    rows = eval_graph._db.query("MATCH (s:Session {session_id: 's1'}) RETURN s.reconciliation_status AS status")

    assert rows[0]["status"] == "pending"


@pytest.mark.parametrize("bad_id", ["", None])
def test_a_fixture_without_a_session_id_is_refused(eval_graph: ActionsGraph, bad_id):
    """A blank id would collapse every fixture onto one node, silently merging
    distinct sessions and destroying the haystack."""
    with pytest.raises(ValueError, match="session_id"):
        inject([_fixture("ok"), _fixture(bad_id)], graph=eval_graph)


def test_a_stale_vector_index_does_not_survive_a_wipe(eval_graph: ActionsGraph):
    """Deleting nodes leaves an index definition standing, so a rebuild with
    another embedding model would inherit the old index at the wrong
    dimension; wipe must drop it too."""
    eval_graph._db.query(
        "CREATE VECTOR INDEX stale_test_index ON :StaleTestLabel(embedding) "
        'WITH CONFIG {"dimension": 384, "capacity": 1000, "metric": "cos"}'
    )

    wipe(eval_graph)

    names = {row["index_name"] for row in eval_graph._db.query("SHOW VECTOR INDEX INFO")}
    assert "stale_test_index" not in names


def test_turns_are_stamped_with_the_session_date_in_order(eval_graph: ActionsGraph):
    """An extracted fact's valid_at is its source turn's timestamp (#364), so a
    turn must carry its session's date, not the time the eval ran."""
    inject([_fixture("s1")], graph=eval_graph)

    session = eval_graph.get_session("s1")
    assert session is not None
    assert session.started_at == "2023-05-20T14:03:00+00:00"
    assert [a.timestamp for a in eval_graph.get_session_actions("s1")] == [
        "2023-05-20T14:03:00+00:00",
        "2023-05-20T14:03:01+00:00",
    ]


def test_every_session_belongs_to_its_questions_user(eval_graph: ActionsGraph):
    """A haystack is one person's history: one question's sessions share a
    (:User), so its facts are gatherable across them, and another question's
    sessions belong to someone else."""
    inject(
        [_fixture("s1", user_id="q1"), _fixture("s2", user_id="q1"), _fixture("s3", user_id="q2")],
        graph=eval_graph,
    )

    rows = eval_graph.db.query(
        "MATCH (u:User)-[:HAD_SESSION]->(s:Session) RETURN u.user_id AS user, count(s) AS sessions ORDER BY user"
    )
    assert rows == [{"user": "q1", "sessions": 2}, {"user": "q2", "sessions": 1}]
