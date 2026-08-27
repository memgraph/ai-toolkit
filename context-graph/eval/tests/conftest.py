"""Shared fixtures for eval tests.

The eval instance is deliberately separate from the dev Memgraph. Batches clear
the graph before loading fixtures (#309), and clearing is only safe because
nothing else lives there.

Multi-database isolation was the original plan, but that is Memgraph
multi-tenancy and needs an Enterprise licence -- a dedicated instance gives the
same known-fixed-state guarantee on a community licence.
"""

import contextlib
import os

import pytest

EVAL_MEMGRAPH_URL = os.environ.get("EVAL_MEMGRAPH_URL", "bolt://localhost:7689")


@pytest.fixture
def eval_graph():
    """An ActionsGraph on the dedicated eval instance, empty before and after."""
    from actions_graph import ActionsGraph
    from memgraph_toolbox.api.memgraph import Memgraph

    try:
        db = Memgraph(url=EVAL_MEMGRAPH_URL, username="", password="")
        db.query("RETURN 1;")
    except Exception as exc:
        pytest.skip(f"no eval Memgraph at {EVAL_MEMGRAPH_URL}: {exc}")

    graph = ActionsGraph(memgraph=db)
    with contextlib.suppress(Exception):
        graph.setup()  # constraints may already exist

    # Deliberately not ActionsGraph.clear(), which removes only
    # Session|Agent|Action|Tool and leaves Chunk, Entity, Episode and Memory
    # standing. The reconciliation tests create exactly those, so clear() alone
    # let a test's distilled memory survive into the next one -- the same leak
    # inject._wipe exists to prevent, reintroduced here in the fixture.
    _wipe(graph)
    yield graph
    _wipe(graph)


def _wipe(graph) -> None:
    """Empty the eval instance. Safe only because it is dedicated to eval."""
    graph._db.query("MATCH (n) DETACH DELETE n")
