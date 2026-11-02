"""The Session Node module against a live Memgraph; skips when none is reachable.

Each test works on its own uniquely named session and removes only that, so
pointing it at a Memgraph that holds real data leaves the data alone.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from memgraph_context_graph import session
from memgraph_toolbox.api.memgraph import Memgraph


@pytest.fixture
def db():
    try:
        client = Memgraph(user_agent="memgraph-context-graph-tests")
    except Exception as exc:
        pytest.skip(f"No live Memgraph reachable: {exc}")
    session.setup(client)
    yield client
    client.close()


@pytest.fixture
def session_id(db):
    sid = f"test-session-{uuid.uuid4()}"
    yield sid
    db.query("MATCH (s:Session {session_id: $sid}) DETACH DELETE s", {"sid": sid})
    db.query("MATCH (f:ForgottenSession {session_id: $sid}) DELETE f", {"sid": sid})


def _sessions(db, session_id: str) -> list[dict]:
    return db.query("MATCH (s:Session {session_id: $sid}) RETURN s.model AS model", {"sid": session_id})


def test_setup_can_run_again(db):
    session.setup(db)


def test_ensure_creates_the_session_with_its_on_create_properties(db, session_id):
    session.ensure(db, session_id, model="opus")

    assert _sessions(db, session_id) == [{"model": "opus"}]


def test_ensure_leaves_an_existing_session_as_it_is(db, session_id):
    session.ensure(db, session_id, model="opus")
    session.ensure(db, session_id, model="sonnet")

    assert _sessions(db, session_id) == [{"model": "opus"}]


def test_concurrent_ensures_of_a_new_session_create_one_node(db, session_id):
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: session.ensure(db, session_id), range(64)))

    assert _sessions(db, session_id) == [{"model": None}]


def test_a_forgotten_session_is_never_created_again(db, session_id):
    session.ensure(db, session_id, model="opus")
    session.mark_forgotten(db, session_id)
    session.mark_forgotten(db, session_id)

    assert session.delete(db, session_id)
    session.ensure(db, session_id, model="opus")

    assert _sessions(db, session_id) == []
    assert session.is_forgotten(db, session_id)
    assert not session.delete(db, session_id)


def test_other_sessions_are_not_forgotten(db, session_id):
    session.mark_forgotten(db, f"{session_id}-other")
    try:
        session.ensure(db, session_id)
        assert not session.is_forgotten(db, session_id)
        assert _sessions(db, session_id) == [{"model": None}]
    finally:
        db.query("MATCH (f:ForgottenSession {session_id: $sid}) DELETE f", {"sid": f"{session_id}-other"})


def test_a_new_session_records_the_release_that_created_it(db, session_id):
    session.ensure(db, session_id, captured_by="a caller's value is overridden")
    session.ensure(db, session_id)

    rows = db.query("MATCH (s:Session {session_id: $sid}) RETURN s.captured_by AS by", {"sid": session_id})
    assert rows == [{"by": session.captured_by()}]
    assert rows[0]["by"].startswith("memgraph-context-graph ")
