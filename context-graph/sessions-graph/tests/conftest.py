"""Shared fixtures for sessions-graph end-to-end tests.

These tests run against a real Memgraph instance. sessions-graph is a
standalone pip-installable package, so `pytest` must still work without
Docker: tests that need a live server skip cleanly rather than fail when one
isn't reachable.
"""

from __future__ import annotations

import contextlib
import threading

import pytest
from sessions_graph import SessionsGraph

from memgraph_toolbox.api.memgraph import Memgraph


@pytest.fixture(autouse=True)
def _isolated_context_graph_config(monkeypatch, tmp_path):
    """Point context-graph's config lookup at an empty file, never the host's.

    The CLI fills its environment from that file (Memgraph connection, LLM
    keys) and reads the embedding model from it, so without this a test run
    would pick up whatever the developer's machine has configured.
    """
    try:
        from memgraph_context_graph.adapters import _identity
    except ImportError:
        yield
        return
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "context-graph" / "config.toml"))
    _identity._reset_cache()
    yield
    _identity._reset_cache()


@pytest.fixture
def requires_mage(memgraph):
    """Skip the test unless this Memgraph can embed (MAGE with its embeddings module)."""
    from sessions_graph.embeddings import EmbeddingUnavailableError, check_available

    try:
        check_available(memgraph)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"Memgraph can't embed (needs MAGE): {exc}")


def link_user(memgraph, user_id, session_id):
    """Attach an existing session to ``user_id``, as the connector does when a harness reports the user."""
    memgraph.query(
        "MERGE (u:User {user_id: $u}) WITH u MATCH (s:Session {session_id: $s}) MERGE (u)-[:HAD_SESSION]->(s)",
        {"u": user_id, "s": session_id},
    )


@pytest.fixture
def busy_storage(memgraph):
    """Hold another client's read transaction open, as a long-running query does, for ``with busy_storage(...)``.

    Schema statements need the storage's unique access, so while it is held
    they wait Memgraph's storage-access timeout (1 s by default) and fail with
    ``TransientError``; ordinary reads and writes go ahead. Pass
    ``release_after`` to commit it that many seconds in, else it is held to the
    end of the block.
    """

    @contextlib.contextmanager
    def hold(release_after: float | None = None):
        with memgraph.driver.session() as session:
            tx = session.begin_transaction()
            tx.run("MATCH (n) RETURN count(n)").consume()
            timer = threading.Timer(release_after, tx.commit) if release_after is not None else None
            if timer is not None:
                timer.start()
            try:
                yield
            finally:
                if timer is not None:
                    timer.join()
                if not tx.closed():
                    tx.commit()

    return hold


@pytest.fixture
def memgraph():
    """A real Memgraph client, wiped clean before and after each test.

    Memgraph.__init__ already verifies connectivity, so constructing it is
    the reachability probe -- skip the test if that fails.
    """
    try:
        client = Memgraph(user_agent="sessions-graph-tests")
    except Exception as e:
        pytest.skip(f"No live Memgraph reachable: {e}")

    client.query("MATCH (n) DETACH DELETE n")
    yield client
    client.query("MATCH (n) DETACH DELETE n")
    client.close()


@pytest.fixture
def graph(memgraph):
    """A SessionsGraph wired to the same live Memgraph as the `memgraph` fixture."""
    g = SessionsGraph(memgraph)
    with contextlib.suppress(Exception):
        # CREATE CONSTRAINT isn't safely repeatable across test runs against
        # a long-lived server -- same defensive style as actions-graph's own
        # e2e fixture.
        g.setup()
    return g


@pytest.fixture
def actions_graph(memgraph):
    """A real ActionsGraph wired to the same live Memgraph as `memgraph`/`graph`.

    Shared by test_reconciliation.py and test_e2e_reconciliation.py so both
    exercise Reconciler.reconcile against real recorded actions rather than a
    hand-rolled fake that can silently drift from actions-graph's real shape.
    Requires the sessions-graph[reconciliation] extra; skip is handled by
    each test module's own pytest.importorskip("actions_graph").
    """
    from actions_graph import ActionsGraph

    ag = ActionsGraph(memgraph)
    with contextlib.suppress(Exception):
        ag.setup()
    return ag
