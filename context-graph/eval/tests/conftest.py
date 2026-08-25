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
    graph.clear()
    yield graph
    graph.clear()
