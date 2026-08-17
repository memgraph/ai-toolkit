"""Shared fixtures for sessions-graph end-to-end tests.

These tests run against a real Memgraph instance. sessions-graph is a
standalone pip-installable package, so `pytest` must still work without
Docker: tests that need a live server skip cleanly rather than fail when one
isn't reachable.
"""

from __future__ import annotations

import contextlib

import pytest
from sessions_graph import SessionsGraph

from memgraph_toolbox.api.memgraph import Memgraph


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
    exercise reconcile_session against real recorded actions rather than a
    hand-rolled fake that can silently drift from actions-graph's real shape.
    Requires the sessions-graph[reconciliation] extra; skip is handled by
    each test module's own pytest.importorskip("actions_graph").
    """
    from actions_graph import ActionsGraph

    ag = ActionsGraph(memgraph)
    with contextlib.suppress(Exception):
        ag.setup()
    return ag
