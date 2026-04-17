import os

import pytest
from neo4j import GraphDatabase

MEMGRAPH_URL = os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")


@pytest.fixture(scope="session")
def memgraph_url():
    return MEMGRAPH_URL


@pytest.fixture(scope="session")
def memgraph_username():
    return MEMGRAPH_USERNAME


@pytest.fixture(scope="session")
def memgraph_password():
    return MEMGRAPH_PASSWORD


@pytest.fixture(scope="session")
def memgraph_driver():
    """Shared neo4j driver for direct Cypher assertions."""
    driver = GraphDatabase.driver(
        MEMGRAPH_URL, auth=(MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)
    )
    yield driver
    driver.close()


def _node_count(driver) -> int:
    with driver.session() as session:
        return session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]


def _relationship_count(driver) -> int:
    with driver.session() as session:
        return session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]


@pytest.fixture()
def assert_graph_not_empty(memgraph_driver):
    """Return a callable that asserts Memgraph contains nodes after the framework writes."""

    def _check(*, min_nodes: int = 1, min_relationships: int = 0):
        nodes = _node_count(memgraph_driver)
        rels = _relationship_count(memgraph_driver)
        assert (
            nodes >= min_nodes
        ), f"Expected at least {min_nodes} node(s) in Memgraph, found {nodes}"
        assert (
            rels >= min_relationships
        ), f"Expected at least {min_relationships} relationship(s) in Memgraph, found {rels}"

    return _check


@pytest.fixture()
def run_cypher(memgraph_driver):
    """Return a callable that executes an arbitrary Cypher query and returns records."""

    def _run(query: str, **params):
        with memgraph_driver.session() as session:
            return list(session.run(query, **params))

    return _run


@pytest.fixture(autouse=True)
def _clean_memgraph(memgraph_driver):
    """Wipe all data before each test so tests are isolated."""
    with memgraph_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield
