"""Shared fixtures. Graph tests run against a real Memgraph (MEMGRAPH_URL, default bolt://localhost:7687)."""

from __future__ import annotations

import pytest

from resources_graph import ResourcesGraph

from .fixtures.replay import PAGE_SIZE, Replay


@pytest.fixture()
def graph():
    """A ResourcesGraph on a wiped Memgraph, with the schema in place."""
    graph = ResourcesGraph()
    graph._db.query("MATCH (n) DETACH DELETE n")
    graph.setup()
    yield graph
    graph._db.query("MATCH (n) DETACH DELETE n")


@pytest.fixture()
def replay():
    """Recorded GitHub responses; edit them per test to stage what GitHub can't be made to return."""
    return Replay()


@pytest.fixture()
def page_size():
    return PAGE_SIZE
