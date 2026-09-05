"""End-to-end tests for the retrieval baseline.

#300 fixed the v1 baseline as the existing `run_cypher_query` surface: an agent
writes its own Cypher, with no ranking, templates, or vector search. Those are
deferred until this baseline's failures say what they should be.
"""

import pytest
from conftest import ScriptedLLM, requires_openai_key
from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.inject import inject_batch
from context_graph_eval.retrieval import (
    MAX_PAYLOAD_TOKENS,
    DeepEvalLLM,
    QueryRefusedError,
    ReadOnlyGraph,
    WriteRefusedError,
    graph_schema,
    retrieve,
)
from context_graph_eval.scoring import efficiency_tokens

from actions_graph import ActionsGraph


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
        # Lower-case, and matched by the apoc rule alone -- 'refactor' appears
        # in no other pattern. The guard used to upper-case the query before
        # matching, which disabled every pattern containing lowercase letters
        # and let this through.
        'CALL apoc.refactor.rename.label("A", "B")',
        "create (n:Sneaky) return n",
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

    # Turn text lives inside the Action's JSON `properties` string, not a
    # `content` property -- see the note in this module.
    llm = ScriptedLLM(
        "```cypher\nMATCH (a:Action) RETURN a.properties AS props ORDER BY props\n```",
        "A beagle.",
    )

    result = await retrieve("What breed is the dog?", graph=populated, llm=llm)

    assert any("beagle" in row for row in result.retrieval_context)
    assert result.answer == "A beagle."


async def test_retrieval_records_the_queries_it_ran(populated: ReadOnlyGraph):
    """A score is not diagnosable without knowing what retrieval actually
    asked for."""

    llm = ScriptedLLM("MATCH (s:Session) RETURN s.session_id AS id", "done")

    result = await retrieve("anything?", graph=populated, llm=llm)

    assert result.queries and "MATCH (s:Session)" in result.queries[0]


async def test_prose_before_the_first_query_does_not_end_the_loop(populated: ReadOnlyGraph):
    """Observed at scale: 4 of 20 questions issued ZERO queries. The model
    opened with prose, the loop read that as "I have enough", and answered from
    an empty context -- one of them a question whose answer was sitting in the
    graph untouched.

    A reply with no query means "done" only once something has actually been
    retrieved. Before that it means the model needs asking again.
    """

    llm = ScriptedLLM(
        "Sure! Let me look that up for you.",
        "MATCH (a:Action) RETURN a.properties AS props",
        "A beagle.",
    )

    result = await retrieve("What breed is the dog?", graph=populated, llm=llm)

    assert result.queries, "the loop gave up before issuing a single query"
    assert any("beagle" in row for row in result.retrieval_context)


@pytest.fixture
def overflowing(eval_graph: ActionsGraph):
    """A graph whose Cartesian product comfortably exceeds the payload cap."""
    inject_batch(
        [_fixture(f"s{i}", f"session {i} discussing an unremarkable topic at length") for i in range(40)],
        graph=eval_graph,
    )
    return ReadOnlyGraph(eval_graph._db)


async def test_an_enormous_result_set_is_capped(overflowing: ReadOnlyGraph):
    """Measured at scale: median payload rose to ~19k tokens and one question
    returned 1,067,650. Efficiency is a scored axis (#309), and an unbounded
    payload also risks exhausting the judge's context window and costs real
    money per question."""
    result = await retrieve(
        "anything?",
        graph=overflowing,
        llm=ScriptedLLM("MATCH (a:Action), (b:Action) RETURN a.properties AS x, b.properties AS y", "STOP"),
    )

    assert efficiency_tokens(result) <= MAX_PAYLOAD_TOKENS


async def test_a_capped_payload_tells_the_agent_it_was_truncated(overflowing: ReadOnlyGraph):
    """Truncating silently would cost coverage for a reason nothing records --
    the agent would believe it had seen everything its query matched, and stop
    looking."""
    result = await retrieve(
        "anything?",
        graph=overflowing,
        llm=ScriptedLLM("MATCH (a:Action), (b:Action) RETURN a.properties AS x, b.properties AS y", "STOP"),
    )

    assert any("dropped" in e.lower() for e in result.errors)


async def test_a_small_result_set_is_untouched(populated: ReadOnlyGraph):
    llm = ScriptedLLM("MATCH (a:Action) RETURN a.properties AS props", "STOP")

    result = await retrieve("anything?", graph=populated, llm=llm)

    assert result.retrieval_context
    assert not any("dropped" in e.lower() for e in result.errors)


async def test_the_agent_is_offered_a_named_way_to_stop(populated: ReadOnlyGraph):
    """Once rows are in hand the prompt must name an explicit exit. "Reply with
    prose if you have enough" was too weak against a model primed to emit
    queries: it spent the whole step budget every run, long after the answer was
    in hand, and each extra query enlarges the payload efficiency counts."""

    llm = ScriptedLLM("MATCH (a:Action) RETURN a.properties AS props", "STOP")
    await retrieve("What breed is the dog?", graph=populated, llm=llm)

    # The second prompt is the one issued with rows already retrieved.
    assert "STOP" in llm.prompts[1]


async def test_prose_after_retrieving_does_end_the_loop(populated: ReadOnlyGraph):
    """The other half of the same rule: once rows are in hand, a reply without a
    query is the model saying it has enough, and spending more steps on it would
    inflate the payload the efficiency metric counts."""

    llm = ScriptedLLM("MATCH (a:Action) RETURN a.properties AS props", "I have enough now.")

    result = await retrieve("What breed is the dog?", graph=populated, llm=llm)

    assert len(result.queries) == 1


async def test_a_bad_query_is_reported_not_raised(populated: ReadOnlyGraph):
    """One malformed query should cost that question its answer, not abort the
    whole batch mid-run."""

    # Reads as a query (has MATCH/RETURN) so it is executed, but is not valid
    # Cypher -- the case that must be caught, not the model replying in prose.
    llm = ScriptedLLM("MATCH (((( RETURN nonsense", "no answer")

    result = await retrieve("anything?", graph=populated, llm=llm)

    assert result.errors


@requires_openai_key
async def test_a_real_model_can_find_an_injected_fact(populated: ReadOnlyGraph):
    """#300's actual premise: given only schema and read-only Cypher, can a real
    model reach an injected fact?

    Asserts it succeeds at least once in three, not a majority and not always.

    That bound is deliberately weak, and the history of it is the point. An
    earlier version asserted a majority on the strength of a single 6/6
    measurement; a later run of the same question scored 1/3. One favourable
    sample was read as a settled improvement -- the exact mistake this whole
    eval exists to make harder. The honest claim the evidence supports is that
    the baseline *can* reach the fact, not that it reliably does.

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

    assert hits, f"the baseline reached the fact {len(hits)}/3 times: {[a.answer for a in attempts]}"
    assert all(a.queries for a in attempts)


# --- LightRAG's storage must not be part of the query surface (#321) ---


@pytest.fixture
def with_lightrag_storage(populated: ReadOnlyGraph):
    """Adds the storage nodes LightRAG persists into the same graph.

    Not a hypothetical shape: reconciling two sessions produced eleven such
    labels alongside the domain model, all of them queryable.
    """
    populated._db.query(
        "CREATE (:LightRAGKV_base_llm_response_cache {id: 'cache-1', data: 'the personal best time is 25:50'})"
    )
    populated._db.query("CREATE (:LightRAGVector_base_entities {id: 'vec-1', content: 'beagle'})")
    populated._db.query("CREATE (:LightRAGDocStatus_base {id: 'doc-1', status: 'processed'})")
    return populated


def test_the_llm_response_cache_cannot_be_read(with_lightrag_storage: ReadOnlyGraph):
    """The extraction LLM's cache sits in the graph under test and contains the
    answers. An agent querying it reads the expected output straight out of the
    cache without exercising the graph model at all -- scoring coverage while
    measuring nothing.

    Every other harness bug so far manufactured a false ZERO. This one
    manufactures a false PASS, which is worse: a zero gets investigated, a pass
    gets believed.
    """
    with pytest.raises(QueryRefusedError, match="internal storage"):
        with_lightrag_storage.query("MATCH (n:LightRAGKV_base_llm_response_cache) RETURN n.data AS data")


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n:LightRAGVector_base_entities) RETURN n",
        "MATCH (n:LightRAGDocStatus_base) RETURN n",
        "MATCH (n:`LightRAGKV_base_text_chunks`) RETURN n",
        "MATCH (a:Action)--(n:LightRAGVector_base_chunks) RETURN n",
    ],
)
def test_every_internal_store_is_refused(with_lightrag_storage: ReadOnlyGraph, cypher: str):
    with pytest.raises(QueryRefusedError):
        with_lightrag_storage.query(cypher)


def test_the_schema_does_not_advertise_internal_stores(with_lightrag_storage: ReadOnlyGraph):
    """Beyond the leak, these cost real budget: the rendered schema was 12,739
    characters with roughly a third of it storage internals, spent out of the
    agent's payload before it issues a single query -- and #309 scores payload
    size."""
    schema = graph_schema(with_lightrag_storage)

    assert "LightRAG" not in schema


def test_domain_entities_carrying_lightrags_base_label_stay_queryable(populated: ReadOnlyGraph):
    """`base` is NOT storage -- LightRAG stamps it on the extracted entities
    themselves, 309 of them in a two-session run, mixed in with Concept, Person,
    Event and the rest. Guarding on it would refuse most of the memory tier,
    which is the half of the graph retrieval most needs."""
    populated._db.query("CREATE (:base:Concept {entity_id: 'Personal Best Time', description: 'a running time'})")

    rows = populated.query("MATCH (n:base:Concept) RETURN n.entity_id AS id")

    assert [r["id"] for r in rows] == ["Personal Best Time"]


def test_the_schema_still_describes_domain_entities(populated: ReadOnlyGraph):
    populated._db.query("CREATE (:base:Concept {entity_id: 'Personal Best Time', description: 'a running time'})")

    schema = graph_schema(populated)

    assert "Concept" in schema
