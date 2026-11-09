"""A real Memgraph for the plan's end-to-end tests; the rest of the suite needs none.

Connects through MEMGRAPH_URL/MEMGRAPH_USER/MEMGRAPH_PASSWORD, as
`scripts/dev-memgraph.sh test hygm` and CI set them. Unlike other packages'
fixtures there is no default URL: the plan creates enums, which Memgraph can't
drop yet, so these tests must never fall through to whatever Memgraph a
developer runs on the default port. Without MEMGRAPH_URL, or with no Memgraph
reachable there, they skip, so the pure suite still runs anywhere.
"""

import os
from collections.abc import Iterator

import pytest


@pytest.fixture
def memgraph_run() -> Iterator:
    """`run(cypher) -> rows` against a real Memgraph, each statement in its own auto-commit transaction."""
    neo4j = pytest.importorskip("neo4j")
    url = os.environ.get("MEMGRAPH_URL")
    if not url:
        pytest.skip("MEMGRAPH_URL is not set; run through scripts/dev-memgraph.sh test hygm")
    auth = (os.environ.get("MEMGRAPH_USER", ""), os.environ.get("MEMGRAPH_PASSWORD", ""))
    driver = neo4j.GraphDatabase.driver(url, auth=auth)
    try:
        driver.verify_connectivity()
    except Exception as error:  # any connection failure means "no Memgraph here", not a test failure
        driver.close()
        pytest.skip(f"No live Memgraph reachable at {url}: {error}")

    def run(cypher: str, **parameters) -> list[dict]:
        with driver.session() as session:
            return [record.data() for record in session.run(cypher, parameters)]

    yield run
    driver.close()
