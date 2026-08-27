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
from context_graph_eval.reconcile import _resolve_llm_credentials

EVAL_MEMGRAPH_URL = os.environ.get("EVAL_MEMGRAPH_URL", "bolt://localhost:7689")


# Resolved from context-graph's config file before deciding to skip, so a
# contributor whose key lives in config.toml (ADR 0002) rather than the
# environment still runs these instead of silently skipping them.
_resolve_llm_credentials()

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="no OPENAI_API_KEY in env or context-graph config",
)


class ScriptedLLM:
    """An LLM that replies with a fixed script, one entry per call.

    Replaces seven near-identical stubs that each counted calls and returned
    "cypher, then prose". The shape was always the same; only the script
    differed, so the script is the parameter.

    The last entry repeats once exhausted -- the loop asks once more for a final
    answer after it stops querying, and a test should not have to pad for that.
    """

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[index]

    @property
    def calls(self) -> int:
        return len(self.prompts)


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
