"""Shared fixtures. Graph tests run against a real Memgraph (MEMGRAPH_URL, default bolt://localhost:7687)."""

from __future__ import annotations

import pytest
from sessions_graph import SessionsGraph
from sessions_graph.connector import SessionsGraphConnector

from resources_graph import ResourcesGraph
from resources_graph.connector import ResourcesGraphConnector
from resources_graph.github import GitHubSource

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


@pytest.fixture()
def harness(graph):
    """Both connectors, as a hook with ``--connector sessions-graph --connector resources-graph`` runs them."""
    sessions = SessionsGraphConnector(SessionsGraph())
    resources = ResourcesGraphConnector(graph)

    def emit(event):
        for connector in (sessions, resources):
            if connector.supports(event):
                connector.on_event(event)

    return emit


@pytest.fixture()
def source(replay, page_size):
    """The GraphQL source over recorded responses."""
    return GitHubSource(replay, page_size=page_size)
