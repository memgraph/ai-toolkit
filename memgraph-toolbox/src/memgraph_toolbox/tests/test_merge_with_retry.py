"""Tests for merge_with_retry against a live Memgraph; they skip when none is reachable."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from neo4j.exceptions import ClientError

from ..api.memgraph import Memgraph, merge_with_retry

_LABEL = "MergeWithRetryProbe"


@pytest.fixture
def db():
    try:
        client = Memgraph()
    except Exception as exc:
        pytest.skip(f"No Memgraph server reachable at the configured MEMGRAPH_URL: {exc}")
    client.query(f"MATCH (n:{_LABEL}) DETACH DELETE n")
    client.query(f"CREATE CONSTRAINT ON (n:{_LABEL}) ASSERT n.key IS UNIQUE;")
    yield client
    client.query(f"MATCH (n:{_LABEL}) DETACH DELETE n")
    client.query(f"DROP CONSTRAINT ON (n:{_LABEL}) ASSERT n.key IS UNIQUE;")
    client.close()


def test_concurrent_merges_of_new_nodes_converge_on_one_node_each(db):
    keys = [f"k{i}" for i in range(10)]

    def merge(key: str) -> None:
        merge_with_retry(db, f"MERGE (n:{_LABEL} {{key: $key}})", {"key": key})

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(merge, keys * 16))

    rows = db.query(f"MATCH (n:{_LABEL}) RETURN n.key AS key, count(*) AS c ORDER BY key")
    assert rows == [{"key": key, "c": 1} for key in sorted(keys)]


def test_returns_the_rows_of_the_run_that_committed(db):
    rows = merge_with_retry(db, f"MERGE (n:{_LABEL} {{key: 'a'}}) RETURN n.key AS key")
    assert rows == [{"key": "a"}]


def test_any_other_client_error_is_raised_without_retrying(db):
    calls = []

    class Counting:
        def query(self, query, params=None):
            calls.append(query)
            return db.query(query, params)

    with pytest.raises(ClientError):
        merge_with_retry(Counting(), "MERGE (n:")
    assert len(calls) == 1


def test_a_conflict_on_every_attempt_is_raised_after_the_last():
    """A conflict that never clears can't be produced on demand by a real server."""
    calls = []

    class AlwaysConflicting:
        def query(self, query, params=None):
            calls.append(query)
            raise ClientError("Unable to commit due to unique constraint violation on :X(key)")

    with pytest.raises(ClientError, match="unique constraint"):
        merge_with_retry(AlwaysConflicting(), "MERGE (n:X {key: 1})", attempts=2)
    assert len(calls) == 2
