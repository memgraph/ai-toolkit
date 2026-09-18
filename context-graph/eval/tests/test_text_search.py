"""Tests for the text-search baseline.

Runs against the real eval Memgraph (no stubbing the index or the search
procedure): the whole point of this baseline is Memgraph's own text index, so
a test that mocked it would verify nothing about what actually happens.
"""

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

    indexed = ensure_turn_text_index(eval_graph)

    assert indexed.turns == 1


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
    # wording for text_search to find anything at all. Not asserting the
    # unrelated turn is absent: search_all ranks by score rather than
    # filtering to term matches, and on a two-document corpus a stray
    # stopword can outrank real overlap (verified directly) -- exactly the
    # kind of weak precision this baseline exists to expose, not something to
    # engineer around here (#300's reasoning again: tuning this is retrieval
    # strategy).
    llm = _EchoLLM()
    result = await retrieve_by_text_search("Tell me about my beagle.", graph=ReadOnlyGraph(eval_graph.db), llm=llm)

    assert any("beagle" in row for row in result.retrieval_context)
    # The final-answer prompt is what got echoed back, so the retrieved rows
    # must have actually reached the LLM call, not just been computed and
    # discarded.
    assert "beagle" in result.answer


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
    assert _safe_query("What breed is the dog?") == "What breed is the dog"


def test_safe_query_is_empty_for_a_question_with_no_word_characters():
    assert _safe_query("???") == ""
