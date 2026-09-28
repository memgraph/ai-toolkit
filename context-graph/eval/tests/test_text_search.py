"""Tests for the text-search baseline.

Runs against the real eval Memgraph (no stubbing the index or the search
procedure): the whole point of this baseline is Memgraph's own text index, so
a test that mocked it would verify nothing about what actually happens. The
one exception is forcing ``_search`` to raise -- a real Memgraph that can
index but not search is not something a test can set up on demand.
"""

import pytest
from context_graph_eval import text_search
from context_graph_eval.retrieval import ReadOnlyGraph
from context_graph_eval.text_search import _safe_query, ensure_turn_text_index, retrieve_by_text_search

from actions_graph import ActionsGraph, MessageRole, Session


def _plant(graph: ActionsGraph, session_id: str, *, role: MessageRole, content: str) -> None:
    graph.ensure_session(Session(session_id=session_id, started_at="2023-01-01T00:00:00"))
    graph.record_message(session_id=session_id, role=role, content=content)


class _EchoLLM:
    """Answers with whatever it was told to answer from -- so a test can
    assert on the answer without depending on real model behaviour."""

    def __init__(self):
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return prompt


def test_indexing_skips_actions_with_no_content(eval_graph: ActionsGraph):
    """A tool-call Action's properties has no 'content' key -- materializing a
    'text' property for it would index noise the question was never about."""
    _plant(eval_graph, "s1", role=MessageRole.USER, content="I adopted a beagle named Max")
    eval_graph.record_tool_call(session_id="s1", tool_name="Read", tool_input={"file_path": "notes.md"})

    indexed = ensure_turn_text_index(eval_graph)

    assert indexed.turns == 1
    untexted = eval_graph.db.query("MATCH (a:Action) WHERE a.text IS NULL RETURN count(a) AS n")[0]["n"]
    assert untexted == 1


def test_indexing_fails_the_run_when_search_cannot_run(eval_graph: ActionsGraph, monkeypatch):
    """A search that cannot run fails every question identically; left to the
    per-question path, each becomes a recorded miss and the run reports ~0%
    as if it were a measurement. It must fail before anything is scored."""

    def _broken(*_args, **_kwargs):
        raise RuntimeError("argument named 'config' at position 2 must be of type MAP")

    monkeypatch.setattr(text_search, "_search", _broken)

    with pytest.raises(RuntimeError, match="text search cannot run"):
        ensure_turn_text_index(eval_graph)


def test_indexing_twice_is_harmless(eval_graph: ActionsGraph):
    """Runs every batch, against an instance whose index may already exist."""
    _plant(eval_graph, "s1", role=MessageRole.USER, content="I adopted a beagle named Max")

    ensure_turn_text_index(eval_graph)
    ensure_turn_text_index(eval_graph)


def test_indexed_turns_are_searchable_by_content(eval_graph: ActionsGraph):
    _plant(eval_graph, "s1", role=MessageRole.USER, content="I adopted a beagle named Max")
    _plant(eval_graph, "s2", role=MessageRole.USER, content="My favorite color is teal")
    ensure_turn_text_index(eval_graph)

    graph = ReadOnlyGraph(eval_graph.db)
    rows = graph.query(
        "CALL text_search.search_all('eval_turn_text_index', $query) YIELD node, score RETURN node.text AS text",
        {"query": "beagle"},
    )

    assert any("beagle" in row["text"] for row in rows)
    assert all("teal" not in row["text"] for row in rows)


async def test_retrieve_by_text_search_hands_matching_turns_to_the_answering_llm(eval_graph: ActionsGraph):
    _plant(eval_graph, "s1", role=MessageRole.USER, content="I adopted a beagle named Max")
    _plant(eval_graph, "s2", role=MessageRole.USER, content="The weather today is sunny and warm")
    ensure_turn_text_index(eval_graph)

    # Shares a literal word ("beagle") with the planted content -- this is a
    # lexical baseline, not a semantic one, so the question has to overlap in
    # wording for text_search to find anything at all. "me"/"my" are dropped
    # as stopwords now, so this no longer risks the unrelated turn outranking
    # the real match the way it could before that fix.
    llm = _EchoLLM()
    result = await retrieve_by_text_search("Tell me about my beagle.", graph=ReadOnlyGraph(eval_graph.db), llm=llm)

    assert result.retrieval_context == ["session=s1 content=I adopted a beagle named Max"]
    # The final-answer prompt is what got echoed back, so the retrieved rows
    # must have actually reached the LLM call, not just been computed and
    # discarded.
    assert "beagle" in result.answer


async def test_retrieve_by_text_search_caps_hits_at_the_configured_limit(eval_graph: ActionsGraph):
    """search_all itself returns every matching turn, uncapped -- more turns
    than the limit leaking through would hand the answering LLM a bigger,
    costlier payload than the run configured."""
    for i in range(5):
        _plant(eval_graph, f"s{i}", role=MessageRole.USER, content=f"I adopted a beagle turn number {i}")
    ensure_turn_text_index(eval_graph)

    result = await retrieve_by_text_search(
        "Tell me about my beagle.", graph=ReadOnlyGraph(eval_graph.db), llm=_EchoLLM(), limit=2
    )

    assert len(result.retrieval_context) == 2


async def test_retrieve_by_text_search_records_its_latency(eval_graph: ActionsGraph):
    """Without it every successful text-search question reported 0.0s, making
    this baseline's mean latency near-zero next to graph-agent's -- the one
    axis this baseline exists to be compared on."""
    _plant(eval_graph, "s1", role=MessageRole.USER, content="I adopted a beagle named Max")
    ensure_turn_text_index(eval_graph)

    result = await retrieve_by_text_search(
        "Tell me about my beagle.", graph=ReadOnlyGraph(eval_graph.db), llm=_EchoLLM()
    )

    assert result.latency_seconds > 0.0


async def test_a_search_that_raises_is_not_turned_into_an_empty_answer(eval_graph: ActionsGraph, monkeypatch):
    """Swallowing the error here meant the LLM answered from nothing and the
    question scored as an ordinary "not in memory" -- indistinguishable from a
    real miss. It must reach the runner, which records it as an error."""

    def _broken(*_args, **_kwargs):
        raise RuntimeError("search failed")

    monkeypatch.setattr(text_search, "_search", _broken)
    llm = _EchoLLM()

    with pytest.raises(RuntimeError, match="search failed"):
        await retrieve_by_text_search("Tell me about my beagle.", graph=ReadOnlyGraph(eval_graph.db), llm=llm)
    assert llm.prompts == []


async def test_retrieve_by_text_search_reports_no_rows_rather_than_silence(eval_graph: ActionsGraph):
    """An empty index (nothing indexed yet, or nothing matches) must be
    visible in errors, the same way retrieval.retrieve() records a query that
    matched nothing rather than leaving the agent to guess why."""
    ensure_turn_text_index(eval_graph)

    result = await retrieve_by_text_search("What breed is my dog?", graph=ReadOnlyGraph(eval_graph.db), llm=_EchoLLM())

    assert result.retrieval_context == []
    assert any("returned 0 rows" in e for e in result.errors)


def test_safe_query_strips_tantivy_special_characters():
    """Tantivy's query parser treats ':', '(', ')', '"' specially -- a natural
    question is full of characters that are not that, mostly '?'. This is
    making the query parseable, not a search-quality choice (#300)."""
    assert _safe_query("What breed dog?") == "What breed dog"


def test_safe_query_is_empty_for_a_question_with_no_word_characters():
    assert _safe_query("???") == ""


def test_safe_query_drops_near_universal_words():
    """search_all is OR-across-terms (verified directly against a live
    index): a turn only needs to share SOME query word to score, so a word
    with almost no discriminating power (is/the/a/my/...) can still
    contribute a nonzero score to something unrelated. Dropped for the same
    reason Tantivy's special characters are stripped above -- removing a term
    that was never going to discriminate, not adding ranking sophistication."""
    assert _safe_query("Is my mom using the same grocery list method as me?") == "mom using same grocery list method"


def test_safe_query_keeps_meaningful_short_words():
    """The stopword list must not be so broad it eats real content -- a bare
    number is exactly the kind of short term this baseline depends on when
    the expected answer IS a number. Must not be dropped as if it were a
    near-universal word just because it is short."""
    assert _safe_query("Was it 5 or 6 days?") == "5 6 days"
