"""Forgetting a session's Touches against a real Memgraph; another session's stay."""

from __future__ import annotations

from resources_graph.address import parse_address
from resources_graph.models import FETCHED


def test_forget_deletes_only_the_sessions_touches(graph):
    address = parse_address("https://github.com/memgraph/memgraph/issues/1")
    assert address is not None
    graph.record_touch("s-1", address, FETCHED, discriminator="t-1")
    graph.record_touch("s-2", address, FETCHED, discriminator="t-2")

    assert graph.forget_session("s-1") == {"touches": 1}

    rows = graph._db.query("MATCH (t:Touch)-[:IN_SESSION]->(s:Session) RETURN s.session_id AS session")
    assert rows == [{"session": "s-2"}]
