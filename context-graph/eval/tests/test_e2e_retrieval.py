"""End-to-end tests for the retrieval baseline.

#300 fixed the v1 baseline as the existing `run_cypher_query` surface: an agent
writes its own Cypher, with no ranking, templates, or vector search. Those are
deferred until this baseline's failures say what they should be.
"""

import os

import pytest
from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.inject import inject_batch
from context_graph_eval.reconcile import _resolve_llm_credentials
from context_graph_eval.retrieval import (
    DeepEvalLLM,
    ReadOnlyGraph,
    WriteRefusedError,
    graph_schema,
    retrieve,
)

from actions_graph import ActionsGraph

_resolve_llm_credentials()

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="no OPENAI_API_KEY in env or context-graph config",
)


def _fixture(session_id: str, content: str) -> SessionFixture:
    return SessionFixture(
        session_id=session_id,
        date="2023/05/20 (Sat) 14:03",
        turns=[Turn(role="user", content=content)],
        holds_evidence=True,
    )


@pytest.fixture
def populated(eval_graph: ActionsGraph):
    inject_batch(
        [
            _fixture("s1", "I adopted a beagle named Max"),
            _fixture("s2", "The hardware store closes at 6pm"),
        ],
        graph=eval_graph,
    )
    return ReadOnlyGraph(eval_graph._db)


def test_a_read_query_returns_rows(populated: ReadOnlyGraph):
    rows = populated.query("MATCH (s:Session) RETURN s.session_id AS id ORDER BY id")

    assert [r["id"] for r in rows] == ["s1", "s2"]


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (n:Sneaky) RETURN n",
        "MATCH (s:Session) SET s.tampered = true",
        "MATCH (s:Session) DETACH DELETE s",
        "MERGE (n:Sneaky {id: 1})",
    ],
)
def test_writes_are_refused(populated: ReadOnlyGraph, cypher: str):
    """Retrieval must not be able to alter the graph it is being scored
    against -- the same "keep the ruler outside the thing measured" reasoning
    that put the corpus in git rather than in Memgraph."""
    with pytest.raises(WriteRefusedError):
        populated.query(cypher)


def test_a_refused_write_does_not_reach_the_graph(populated: ReadOnlyGraph):
    with pytest.raises(WriteRefusedError):
        populated.query("MATCH (s:Session {session_id: 's1'}) DETACH DELETE s")

    assert populated.query("MATCH (s:Session {session_id: 's1'}) RETURN count(s) AS n")[0]["n"] == 1


def test_schema_describes_what_was_injected(populated: ReadOnlyGraph):
    """The agent writes its own Cypher, so it needs to know what labels exist --
    without this it can only guess at the graph's shape."""
    schema = graph_schema(populated)

    assert "Session" in schema


async def test_retrieval_returns_the_context_it_actually_saw(populated: ReadOnlyGraph):
    """retrieval_context is what the efficiency metric counts tokens over
    (#309), so it must be what came back from the graph -- not the model's
    prose about it."""

    class StubLLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                # Turn text lives inside the Action's JSON `properties` string,
                # not a `content` property -- see the note in this module.
                return "```cypher\nMATCH (a:Action) RETURN a.properties AS props ORDER BY props\n```"
            return "A beagle."

    result = await retrieve("What breed is the dog?", graph=populated, llm=StubLLM())

    assert any("beagle" in row for row in result.retrieval_context)
    assert result.answer == "A beagle."


async def test_retrieval_records_the_queries_it_ran(populated: ReadOnlyGraph):
    """A score is not diagnosable without knowing what retrieval actually
    asked for."""

    class StubLLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "MATCH (s:Session) RETURN s.session_id AS id"
            return "done"

    result = await retrieve("anything?", graph=populated, llm=StubLLM())

    assert result.queries and "MATCH (s:Session)" in result.queries[0]


async def test_a_bad_query_is_reported_not_raised(populated: ReadOnlyGraph):
    """One malformed query should cost that question its answer, not abort the
    whole batch mid-run."""

    class StubLLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                # Reads as a query (has MATCH/RETURN) so it is executed, but is
                # not valid Cypher -- the case that must be caught, not the
                # model simply replying in prose.
                return "MATCH (((( RETURN nonsense"
            return "no answer"

    result = await retrieve("anything?", graph=populated, llm=StubLLM())

    assert result.errors


@requires_openai_key
async def test_a_real_model_can_find_an_injected_fact(populated: ReadOnlyGraph):
    """#300's actual premise: given only schema and read-only Cypher, can a real
    model reach an injected fact?

    Asserts it succeeds at least once in three attempts, not every time --
    because measured against a real model it does **not** succeed every time.
    Observed success rate on this, the easiest possible case (two sessions, one
    fact, one distractor): roughly two runs in three.

    The failure mode is consistent: the model invents an ``action_type`` from
    the question's wording ("adopt_dog"), and the schema dump gives it property
    *keys* without values or samples, so nothing contradicts the guess. Adding
    zero-row feedback to the loop made it recoverable but not reliable.

    That unreliability is a finding about the baseline, not a flaky test to be
    silenced: at ~2/3 on the easy case, a Tier 1 batch's coverage score would
    largely measure whether the agent guessed workable Cypher rather than
    whether the memory is any good. It is the concrete evidence #300 wanted
    before designing retrieval v2, and it says a schema description carrying
    real property *values* is the first thing to try.
    """
    from deepeval.models import GPTModel

    attempts = [
        await retrieve("What breed of dog was adopted?", graph=populated, llm=DeepEvalLLM(GPTModel())) for _ in range(3)
    ]

    assert any("beagle" in a.answer.lower() for a in attempts), (
        f"the baseline could not reach the fact in three attempts: {[a.answer for a in attempts]}"
    )
    assert all(a.queries for a in attempts)
