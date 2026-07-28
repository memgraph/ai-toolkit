"""Shared fixtures for unstructured2graph end-to-end tests.

These tests run against a real Memgraph instance. unstructured2graph is a
standalone pip-installable package, so `pytest` must still work without
Docker: tests that need a live server skip cleanly rather than fail when one
isn't reachable.
"""

from __future__ import annotations

import pytest

from memgraph_toolbox.api.memgraph import Memgraph


@pytest.fixture
def memgraph():
    """A real Memgraph client, wiped clean before and after each test.

    Connects via the canonical MEMGRAPH_URL/MEMGRAPH_USER/MEMGRAPH_PASSWORD/
    MEMGRAPH_DATABASE env vars (default: bolt://localhost:7687, no auth).
    Memgraph.__init__ already verifies connectivity, so constructing it is
    the reachability probe -- skip the test if that fails.
    """
    try:
        client = Memgraph(user_agent="unstructured2graph-tests")
    except Exception as e:
        pytest.skip(f"No live Memgraph reachable: {e}")

    client.query("MATCH (n) DETACH DELETE n")
    yield client
    client.query("MATCH (n) DETACH DELETE n")
    client.close()
