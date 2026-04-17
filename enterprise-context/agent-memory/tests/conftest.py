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


@pytest.fixture(autouse=True)
def _clean_memgraph():
    """Wipe all data before each test so tests are isolated."""
    driver = GraphDatabase.driver(
        MEMGRAPH_URL, auth=(MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)
    )
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    yield
