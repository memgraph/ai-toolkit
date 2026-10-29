"""Setting up sessions-graph's schema while another client holds the storage, on a real Memgraph."""

from __future__ import annotations

import pytest
from neo4j.exceptions import TransientError
from sessions_graph import cli


def test_setup_waits_out_another_clients_transaction(graph, busy_storage):
    with busy_storage(release_after=2.5):
        # The contention itself: one attempt fails, and auto-committed DDL is never retried by the driver.
        with pytest.raises(TransientError):
            graph.setup()

        cli._setup_graph()
