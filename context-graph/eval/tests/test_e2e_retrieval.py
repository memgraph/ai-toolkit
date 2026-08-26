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


def test_schema_never_leaks_the_content_being_asked_about(populated: ReadOnlyGraph):
    """The load-bearing constraint. Sampling values out of the graph under test
    could hand the agent the answer directly, and retrieval would then 'succeed'
    without having retrieved anything -- the eval measuring its own prompt."""
    schema = graph_schema(populated)

    assert "beagle" not in schema.lower()
    assert "hardware store" not in schema.lower()


def test_schema_names_low_cardinality_values(populated: ReadOnlyGraph):
    """A property with a handful of distinct values *is* schema, not data.
    Without this the model cannot know action_type holds 'message' rather than
    a domain verb -- the exact hallucination observed against a real model,
    which invented action_type='adopt_dog' and queried it four times."""
    schema = graph_schema(populated)

    assert "message" in schema.lower()


def test_schema_reveals_the_keys_inside_the_properties_blob(populated: ReadOnlyGraph):
    """Turn text lives inside Action.properties as a JSON string, so an agent
    told only that 'properties' exists still cannot find content. Its inner
    keys are structure and safe to show; its values are data and are not."""
    schema = graph_schema(populated)

    assert "content" in schema
    assert "role" in schema


def test_schema_marks_free_text_properties_as_searchable(populated: ReadOnlyGraph):
    """High-cardinality text cannot be enumerated, so the agent needs telling
    that it is free text to be searched rather than matched exactly."""
    schema = graph_schema(populated)

    assert "free text" in schema.lower()


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

    Asserts a majority of three attempts rather than all three: this is a real
    model, and treating it as deterministic would make the test flaky rather
    than make the model reliable.

    History, because the number moved a long way and the reason is the
    interesting part. Originally this failed outright -- the model invented an
    ``action_type`` from the question's wording ("adopt_dog"), queried that same
    idea four times, and answered "not in memory". Two fixes moved it:

    1. Reporting zero-row results back, so a valid-but-empty query stopped being
       indistinguishable from not having queried at all. Took it to roughly two
       runs in three.
    2. Describing property *values* rather than only keys -- so ``action_type``
       is visibly ``user_message``, and ``properties`` is visibly a JSON string
       whose keys include ``content``. Took it to 6/6 across six measured runs.

    The second mattered more, and its lesson generalises: the agent was not
    reasoning badly, it was reasoning correctly from a schema that said nothing
    about where content actually lives.
    """
    from deepeval.models import GPTModel

    attempts = [
        await retrieve("What breed of dog was adopted?", graph=populated, llm=DeepEvalLLM(GPTModel())) for _ in range(3)
    ]
    hits = [a for a in attempts if "beagle" in a.answer.lower()]

    assert len(hits) >= 2, f"the baseline reached the fact {len(hits)}/3 times: {[a.answer for a in attempts]}"
    assert all(a.queries for a in attempts)
