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
