"""Components racing to create one new Session Node converge on a single node.

Hooks run one process per event, so a session's first events reach
actions-graph and sessions-graph at once; each MERGEs the Session through
memgraph_context_graph.session.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("actions_graph")


def test_actions_and_sessions_graph_racing_on_a_new_session_create_one_node(graph, memgraph, actions_graph):
    from actions_graph import Session

    writes = [
        lambda: actions_graph.ensure_session(Session(session_id="s-1", model="opus")),
        lambda: actions_graph.record_session_start(Session(session_id="s-1", model="opus")),
        lambda: graph.record_session_start("s-1", user_id="alice", project_key="github.com_memgraph_ai-toolkit"),
        lambda: graph.reconciler.session_user("s-1"),
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in [pool.submit(write) for write in writes * 8]:
            future.result()

    rows = memgraph.query(
        "MATCH (s:Session {session_id: 's-1'}) "
        "RETURN count(s) AS sessions, collect(DISTINCT s.model) AS models, collect(DISTINCT s.project_key) AS projects"
    )
    assert rows == [{"sessions": 1, "models": ["opus"], "projects": ["github.com_memgraph_ai-toolkit"]}]


def test_a_new_users_first_sessions_starting_at_once_share_one_user(graph, memgraph):
    """Each session start MERGEs the same new User, which races on its uniqueness constraint too."""
    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in [
            pool.submit(graph.record_session_start, f"s-{i}", user_id="alice", project_key=None) for i in range(32)
        ]:
            future.result()

    rows = memgraph.query("MATCH (u:User)-[:HAD_SESSION]->(s:Session) RETURN u.user_id AS user, count(s) AS sessions")
    assert rows == [{"user": "alice", "sessions": 32}]
