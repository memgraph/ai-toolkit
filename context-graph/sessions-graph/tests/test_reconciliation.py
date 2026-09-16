"""Tests for Sessions Graph reconciliation (extract_reconcilable_text,
build_reconciliation_sources, SessionsGraph.reconcile_session).

extract_reconcilable_text/build_reconciliation_sources/summarize_session_texts
are pure-logic unit tests, no I/O. reconcile_session's own tests use a real
Memgraph and a real ActionsGraph (via conftest.py's `graph`/`memgraph`/
`actions_graph` fixtures, which skip cleanly if unreachable) -- only the LLM
boundary (unstructured2graph.from_texts, the LightRAG wrapper's
llm_model_func) is mocked, so schema drift between sessions-graph and
actions-graph gets caught without needing OPENAI_API_KEY.

These require the sessions-graph[reconciliation] extra (actions-graph +
unstructured2graph); tests skip cleanly if it isn't installed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("actions_graph", reason="actions-graph not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from sessions_graph.models import Memory
from sessions_graph.reconciliation import (
    MAX_RECONCILABLE_CHARS,
    MAX_SESSION_BATCH_CHARS,
    ReconciliationSource,
    build_reconciliation_sources,
    build_session_summary_prompt,
    content_hash,
    extract_reconcilable_text,
    summarize_session_texts,
)

from actions_graph.models import ErrorEvent, Message, MessageRole, ToolCall, ToolResult
from unstructured2graph import Chunk

# ---------------------------------------------------------------------------
# extract_reconcilable_text
# ---------------------------------------------------------------------------


class TestExtractReconcilableText:
    def test_message_with_string_content(self):
        action = Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Hello there")
        assert extract_reconcilable_text(action) == "assistant: Hello there"

    def test_message_with_content_blocks_joins_text(self):
        action = Message(
            session_id="s-1",
            role=MessageRole.ASSISTANT,
            content=[{"type": "text", "text": "Part one"}, {"type": "text", "text": "Part two"}],
        )
        assert extract_reconcilable_text(action) == "assistant: Part one\nPart two"

    def test_tool_call_stringifies_tool_input(self):
        action = ToolCall(session_id="s-1", tool_name="Read", tool_input={"file_path": "/tmp/x.py"})
        result = extract_reconcilable_text(action)
        assert result is not None
        assert "/tmp/x.py" in result

    def test_tool_result_with_string_content(self):
        action = ToolResult(session_id="s-1", tool_use_id="t-1", tool_name="Bash", content="output text")
        assert extract_reconcilable_text(action) == "output text"

    def test_empty_content_returns_none(self):
        action = Message(session_id="s-1", role=MessageRole.USER, content="")
        assert extract_reconcilable_text(action) is None

    def test_whitespace_only_content_returns_none(self):
        action = Message(session_id="s-1", role=MessageRole.USER, content="   \n  ")
        assert extract_reconcilable_text(action) is None

    def test_unsupported_action_type_returns_none(self):
        action = ErrorEvent(session_id="s-1", error_type="Timeout", error_message="took too long")
        assert extract_reconcilable_text(action) is None

    def test_long_content_is_truncated(self):
        long_text = "x" * (MAX_RECONCILABLE_CHARS + 500)
        action = Message(session_id="s-1", role=MessageRole.ASSISTANT, content=long_text)
        result = extract_reconcilable_text(action)
        assert result is not None
        assert len(result) == MAX_RECONCILABLE_CHARS


# ---------------------------------------------------------------------------
# build_reconciliation_sources
# ---------------------------------------------------------------------------


class TestBuildReconciliationSources:
    def test_combines_actions_and_memories(self):
        actions = [
            Message(session_id="s-1", role=MessageRole.USER, content="Question"),
            Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Answer"),
        ]
        memories = [Memory(user_id="alice", content="User prefers concise answers", memory_id="m-1")]

        sources = build_reconciliation_sources(actions, memories)

        assert len(sources) == 3
        assert sources[0] == ReconciliationSource(kind="action", node_id=actions[0].action_id, text="user: Question")
        assert sources[1] == ReconciliationSource(kind="action", node_id=actions[1].action_id, text="assistant: Answer")
        assert sources[2] == ReconciliationSource(kind="memory", node_id="m-1", text="User prefers concise answers")

    def test_skips_actions_with_no_reconcilable_text(self):
        actions = [
            Message(session_id="s-1", role=MessageRole.USER, content=""),
            ErrorEvent(session_id="s-1", error_type="x", error_message="y"),
        ]
        sources = build_reconciliation_sources(actions, [])
        assert sources == []


# ---------------------------------------------------------------------------
# build_session_summary_prompt / summarize_session_texts
# ---------------------------------------------------------------------------


class TestSummarizeSessionTexts:
    def test_prompt_includes_all_texts(self):
        prompt = build_session_summary_prompt(["Alice asked about the graph engine.", "Bob replied with a plan."])
        assert "Alice asked about the graph engine." in prompt
        assert "Bob replied with a plan." in prompt

    @pytest.mark.asyncio
    async def test_calls_lightrag_wrappers_llm_model_func_and_strips_result(self):
        lightrag_wrapper = MagicMock()
        lightrag_wrapper.get_lightrag.return_value.llm_model_func = AsyncMock(return_value="  A tidy summary.  ")

        result = await summarize_session_texts(lightrag_wrapper, ["Some session text."])

        assert result == "A tidy summary."
        lightrag_wrapper.get_lightrag.return_value.llm_model_func.assert_awaited_once()
        prompt_arg = lightrag_wrapper.get_lightrag.return_value.llm_model_func.call_args.args[0]
        assert "Some session text." in prompt_arg


# ---------------------------------------------------------------------------
# SessionsGraph.reconcile_session (real Memgraph + real ActionsGraph, mocked LLM)
# ---------------------------------------------------------------------------
#
# ActionsGraph is real here -- a hand-rolled fake let this file drift from
# actions-graph's real shape unnoticed. Only the LLM boundary
# (unstructured2graph.from_texts, the LightRAG wrapper's llm_model_func)
# stays mocked: real, but cost-free and deterministic, protection against
# schema drift without needing OPENAI_API_KEY the way test_e2e_reconciliation.py's
# fully-real version does.


def _stub_db():
    db = MagicMock()
    db.query.return_value = []
    return db


def _graph(db=None):
    from sessions_graph.core import SessionsGraph

    g = SessionsGraph.__new__(SessionsGraph)
    g._db = db or _stub_db()
    return g


def _fake_lightrag_wrapper(summary_text: str = "A narrative summary of the session."):
    wrapper = MagicMock()
    wrapper.get_lightrag.return_value.llm_model_func = AsyncMock(return_value=summary_text)
    return wrapper


def _all_processed(grouped_chunks: list[list[Chunk]]) -> dict[str, dict[str, str]]:
    """process_enqueued_and_finalize's real return shape: every chunk's hash
    mapped to its doc_status record. AsyncMock()'s own default return value
    is itself an AsyncMock (a well-known gotcha), so every patch of this
    function must supply an explicit, real dict -- otherwise
    reconcile_sessions_batch's status.get(...) calls silently operate on a
    coroutine instead of failing loudly."""
    return {chunk.hash: {"status": "processed"} for group in grouped_chunks for chunk in group}


@pytest.mark.asyncio
async def test_reconcile_session_success_marks_completed_and_links_chunks(graph, actions_graph):
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(
        session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine."
    )
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        summary = await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "completed"
    assert summary.texts_considered == 1
    assert summary.texts_deduped == 1
    mock_from_texts.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_session_extraction_backend_override_replaces_the_lightrag_default(graph, actions_graph):
    """A non-LightRAG backend (e.g. GLiNER2Backend) must reach from_texts as-is,
    not get wrapped in LightRAGBackend(lightrag_wrapper) -- the default that
    applies only when extraction_backend is omitted."""
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(
        session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine."
    )
    lightrag_wrapper = _fake_lightrag_wrapper()
    fake_backend = MagicMock()

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        summary = await graph.reconcile_session(
            "s-1",
            lightrag_wrapper=lightrag_wrapper,
            extraction_backend=fake_backend,
            actions_graph=actions_graph,
        )

    assert summary.status == "completed"
    assert mock_from_texts.call_args.kwargs["extraction_backend"] is fake_backend
    # Narrative summarization still runs via lightrag_wrapper's own LLM --
    # entity extraction and summarization are decoupled, not both replaced.
    lightrag_wrapper.get_lightrag.return_value.llm_model_func.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_session_writes_episode_from_dedicated_llm_call(graph, memgraph, actions_graph):
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(
        session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine."
    )
    lightrag_wrapper = _fake_lightrag_wrapper("Alice was discussed working on the graph engine.")

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])):
        summary = await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.summary_written is True
    lightrag_wrapper.get_lightrag.return_value.llm_model_func.assert_awaited_once()
    episode_rows = memgraph.query(
        "MATCH (:Session {session_id: $session_id})-[:HAS_EPISODE]->(e:Episode) RETURN e.summary AS summary",
        params={"session_id": "s-1"},
    )
    assert len(episode_rows) == 1
    assert episode_rows[0]["summary"] == "Alice was discussed working on the graph engine."


@pytest.mark.asyncio
async def test_reconcile_session_passes_promotion_and_ontology_kwargs_through_to_from_texts(graph, actions_graph):
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(
        session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine."
    )
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        await graph.reconcile_session(
            "s-1",
            lightrag_wrapper=lightrag_wrapper,
            actions_graph=actions_graph,
            promote_labels=True,
            enforce_ontology=True,
            ontology_path="/some/ontology.yaml",
        )

    mock_from_texts.assert_awaited_once()
    call_kwargs = mock_from_texts.call_args.kwargs
    assert call_kwargs["promote_labels"] is True
    assert call_kwargs["enforce_ontology"] is True
    assert call_kwargs["ontology_path"] == "/some/ontology.yaml"


@pytest.mark.asyncio
async def test_reconcile_session_dedupes_identical_text_before_calling_lightrag(graph, actions_graph):
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    # Same speaker twice: genuinely the same source text, so it should collapse.
    # A user line and an assistant line with identical words are NOT the same
    # source -- that case is covered in TestSpeakerAttribution (#328).
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="Same question")
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="Same question")
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="user: Same question", hash=content_hash("user: Same question"))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        summary = await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.texts_considered == 2
    assert summary.texts_deduped == 1
    mock_from_texts.assert_awaited_once()
    called_texts = mock_from_texts.call_args.args[0]
    assert called_texts == ["user: Same question"]


@pytest.mark.asyncio
async def test_reconcile_session_joins_distinct_turns_into_one_document(graph, actions_graph):
    """A session's turns are extracted together, not one independent LightRAG
    document per turn -- each turn was previously invisible to every other
    turn's extraction call, hiding cross-turn facts (coreference, a fact
    stated in one turn and referenced in another). from_texts should see one
    combined document per session, not one entry per turn."""
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="Alice joined the graph team.")
    actions_graph.record_message(
        session_id="s-1", role=MessageRole.ASSISTANT, content="Noted, she'll need repo access."
    )
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="combined", hash=content_hash("combined"))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    called_texts = mock_from_texts.call_args.args[0]
    assert len(called_texts) == 1
    assert "user: Alice joined the graph team." in called_texts[0]
    assert "assistant: Noted, she'll need repo access." in called_texts[0]

    chunk_kwargs = mock_from_texts.call_args.kwargs["chunk_kwargs"]
    assert chunk_kwargs["max_characters"] == MAX_SESSION_BATCH_CHARS


@pytest.mark.asyncio
async def test_reconcile_session_links_every_source_to_the_shared_session_chunk(graph, actions_graph, memgraph):
    """Every source in the session -- including one whose exact-duplicate text
    was deduped away before ever reaching from_texts -- must still get its own
    HAS_CHUNK edge to whatever chunk(s) the session's one combined document
    produced. Provenance is source-level even though extraction is now
    session-level."""
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="Same question")
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="Same question")
    actions_graph.record_message(session_id="s-1", role=MessageRole.ASSISTANT, content="A different reply.")
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="combined", hash=content_hash("combined"))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])):
        await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    rows = memgraph.query(
        """
        MATCH (:Session {session_id: $session_id})-[:HAS_ACTION]->(a:Action)-[:HAS_CHUNK]->(c:Chunk {hash: $hash})
        RETURN count(a) AS count
        """,
        params={"session_id": "s-1", "hash": fake_chunk.hash},
    )
    # All 3 recorded actions, not just the 2 distinct texts that survived dedup.
    assert rows[0]["count"] == 3


@pytest.mark.asyncio
async def test_reconcile_session_no_reconcilable_content_skips_lightrag_but_still_completes(graph, actions_graph):
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    lightrag_wrapper = MagicMock()

    with patch("unstructured2graph.from_texts", new=AsyncMock()) as mock_from_texts:
        summary = await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "completed"
    assert summary.texts_considered == 0
    assert summary.summary_written is False
    mock_from_texts.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_session_failure_marks_failed_and_returns_error(graph, actions_graph):
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(session_id="s-1", role=MessageRole.ASSISTANT, content="Some content")
    lightrag_wrapper = MagicMock()

    with patch("unstructured2graph.from_texts", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        summary = await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "failed"
    assert "LLM down" in summary.error


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_enqueues_and_processes_once_for_the_whole_group(graph, actions_graph):
    """The whole point: one enqueue_texts call and one
    process_enqueued_and_finalize call for N sessions, not N of each --
    that's what gives LightRAG's worker pool more than one document to
    parallelize over."""
    from actions_graph import Session

    for sid in ("s-1", "s-2", "s-3"):
        actions_graph.create_session(Session(session_id=sid))
        actions_graph.record_message(session_id=sid, role=MessageRole.USER, content=f"content for {sid}")
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunks = [[Chunk(text=f"content for s-{i + 1}", hash=f"h{i + 1}")] for i in range(3)]
    with (
        patch("unstructured2graph.enqueue_texts", new=AsyncMock(return_value=fake_chunks)) as mock_enqueue,
        patch(
            "unstructured2graph.process_enqueued_and_finalize", new=AsyncMock(return_value=_all_processed(fake_chunks))
        ) as mock_process,
    ):
        summaries = await graph.reconcile_sessions_batch(
            ["s-1", "s-2", "s-3"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    mock_enqueue.assert_awaited_once()
    assert len(mock_enqueue.call_args.args[0]) == 3
    mock_process.assert_awaited_once()
    assert [s.status for s in summaries] == ["completed", "completed", "completed"]


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_returns_results_in_input_order(graph, actions_graph):
    """Finalize is fanned out concurrently, so completion order is not
    guaranteed -- the returned list must still match session_ids order."""
    from actions_graph import Session

    for sid in ("s-a", "s-b", "s-c"):
        actions_graph.create_session(Session(session_id=sid))
        actions_graph.record_message(session_id=sid, role=MessageRole.USER, content=f"content {sid}")
    lightrag_wrapper = _fake_lightrag_wrapper()
    fake_chunks = [[Chunk(text="c", hash=f"h{i}")] for i in range(3)]

    with (
        patch("unstructured2graph.enqueue_texts", new=AsyncMock(return_value=fake_chunks)),
        patch(
            "unstructured2graph.process_enqueued_and_finalize", new=AsyncMock(return_value=_all_processed(fake_chunks))
        ),
    ):
        summaries = await graph.reconcile_sessions_batch(
            ["s-a", "s-b", "s-c"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    assert [s.session_id for s in summaries] == ["s-a", "s-b", "s-c"]


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_links_chunks_per_session_from_shared_batch(graph, actions_graph, memgraph):
    """Each session must link to its OWN group's chunk, not another
    session's -- the shared batch call must not blur per-session
    provenance."""
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="about s-1")
    actions_graph.create_session(Session(session_id="s-2"))
    actions_graph.record_message(session_id="s-2", role=MessageRole.USER, content="about s-2")
    lightrag_wrapper = _fake_lightrag_wrapper()

    chunk_1 = Chunk(text="about s-1", hash="hash-for-s1")
    chunk_2 = Chunk(text="about s-2", hash="hash-for-s2")
    with (
        patch("unstructured2graph.enqueue_texts", new=AsyncMock(return_value=[[chunk_1], [chunk_2]])),
        patch(
            "unstructured2graph.process_enqueued_and_finalize",
            new=AsyncMock(return_value=_all_processed([[chunk_1], [chunk_2]])),
        ),
    ):
        await graph.reconcile_sessions_batch(
            ["s-1", "s-2"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    rows = memgraph.query(
        "MATCH (:Session {session_id: $sid})-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk) RETURN c.hash AS hash",
        params={"sid": "s-1"},
    )
    assert [r["hash"] for r in rows] == ["hash-for-s1"]

    rows = memgraph.query(
        "MATCH (:Session {session_id: $sid})-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk) RETURN c.hash AS hash",
        params={"sid": "s-2"},
    )
    assert [r["hash"] for r in rows] == ["hash-for-s2"]


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_sessions_with_no_content_never_touch_lightrag(graph, actions_graph):
    """A session with no reconcilable text must complete without being
    counted in the shared batch -- an empty text would just waste a slot."""
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-empty"))
    actions_graph.create_session(Session(session_id="s-has-content"))
    actions_graph.record_message(session_id="s-has-content", role=MessageRole.USER, content="real content")
    lightrag_wrapper = _fake_lightrag_wrapper()

    with (
        patch(
            "unstructured2graph.enqueue_texts", new=AsyncMock(return_value=[[Chunk(text="x", hash="h1")]])
        ) as mock_enqueue,
        patch(
            "unstructured2graph.process_enqueued_and_finalize",
            new=AsyncMock(return_value=_all_processed([[Chunk(text="x", hash="h1")]])),
        ),
    ):
        summaries = await graph.reconcile_sessions_batch(
            ["s-empty", "s-has-content"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    assert mock_enqueue.call_args.args[0] == ["user: real content"]
    by_id = {s.session_id: s for s in summaries}
    assert by_id["s-empty"].status == "completed"
    assert by_id["s-empty"].texts_deduped == 0
    assert by_id["s-has-content"].status == "completed"


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_shared_pass_failure_marks_every_session_failed(graph, actions_graph):
    """Documented, coarser-than-reconcile_session behaviour: a failure in the
    shared processing pass fails every session in that call, since there is
    no per-document result to attribute it to (yet)."""
    from actions_graph import Session

    for sid in ("s-1", "s-2"):
        actions_graph.create_session(Session(session_id=sid))
        actions_graph.record_message(session_id=sid, role=MessageRole.USER, content=f"content {sid}")
    lightrag_wrapper = _fake_lightrag_wrapper()

    with patch("unstructured2graph.enqueue_texts", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        summaries = await graph.reconcile_sessions_batch(
            ["s-1", "s-2"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    assert all(s.status == "failed" for s in summaries)
    assert all("LLM down" in s.error for s in summaries)


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_one_finalize_failure_does_not_fail_the_rest(graph, actions_graph):
    """Unlike the shared pass, finalize has no cross-session state -- one
    session's episode-summary call failing must not cost its siblings
    their result."""
    from actions_graph import Session

    for sid in ("s-1", "s-2"):
        actions_graph.create_session(Session(session_id=sid))
        actions_graph.record_message(session_id=sid, role=MessageRole.USER, content=f"content {sid}")

    lightrag_wrapper = MagicMock()
    lightrag_wrapper.get_lightrag.return_value.llm_model_func = AsyncMock(
        side_effect=[RuntimeError("summary failed for s-1"), "ok summary for s-2"]
    )
    fake_chunks = [[Chunk(text="c", hash="h1")], [Chunk(text="c2", hash="h2")]]

    with (
        patch("unstructured2graph.enqueue_texts", new=AsyncMock(return_value=fake_chunks)),
        patch(
            "unstructured2graph.process_enqueued_and_finalize", new=AsyncMock(return_value=_all_processed(fake_chunks))
        ),
    ):
        summaries = await graph.reconcile_sessions_batch(
            ["s-1", "s-2"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    by_id = {s.session_id: s for s in summaries}
    assert by_id["s-1"].status == "failed"
    assert by_id["s-2"].status == "completed"


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_reports_a_real_per_document_failure(graph, actions_graph):
    """The exact defect the review caught: process_enqueued_and_finalize
    returning without raising is NOT proof every document succeeded --
    LightRAG swallows a per-document extraction failure and records it in
    doc_status instead. A session whose own chunk shows anything other than
    "processed" must be reported failed, not blindly marked completed
    because the shared call didn't raise."""
    from actions_graph import Session

    for sid in ("s-1", "s-2"):
        actions_graph.create_session(Session(session_id=sid))
        actions_graph.record_message(session_id=sid, role=MessageRole.USER, content=f"content {sid}")
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunks = [[Chunk(text="c1", hash="h1")], [Chunk(text="c2", hash="h2")]]
    doc_statuses = {
        "h1": {"status": "processed"},
        "h2": {"status": "failed", "error_msg": "simulated extraction failure for h2"},
    }

    with (
        patch("unstructured2graph.enqueue_texts", new=AsyncMock(return_value=fake_chunks)),
        patch("unstructured2graph.process_enqueued_and_finalize", new=AsyncMock(return_value=doc_statuses)),
    ):
        summaries = await graph.reconcile_sessions_batch(
            ["s-1", "s-2"], lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph
        )

    by_id = {s.session_id: s for s in summaries}
    assert by_id["s-1"].status == "completed"
    assert by_id["s-2"].status == "failed"
    assert "simulated extraction failure for h2" in by_id["s-2"].error

    rows = graph._db.query("MATCH (s:Session {session_id: 's-2'}) RETURN s.reconciliation_status AS status")
    assert rows[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_reconcile_sessions_batch_rejects_non_positive_summary_concurrency():
    """Semaphore(0) would deadlock the finalize step forever -- and only
    after the batch's extraction has already been billed. Must be caught
    up front, before any paid work (before even resolving actions_graph),
    not discovered as a hang."""
    g = _graph()

    with pytest.raises(ValueError, match="summary_concurrency"):
        await g.reconcile_sessions_batch(["s-1"], lightrag_wrapper=MagicMock(), summary_concurrency=0)


def test_get_pending_reconciliation_sessions_maps_rows():
    db = _stub_db()
    db.query.return_value = [{"session_id": "s-1"}, {"session_id": "s-2"}]
    g = _graph(db)

    assert g.get_pending_reconciliation_sessions() == ["s-1", "s-2"]


def test_get_memories_for_session_maps_rows():
    db = _stub_db()
    db.query.return_value = [
        {
            "memory_id": "m-1",
            "user_id": "alice",
            "content": "Prefers Python",
            "created_at": "2026-01-01T00:00:00+00:00",
            "session_id": "s-1",
        }
    ]
    g = _graph(db)

    result = g.get_memories_for_session("s-1")
    assert len(result) == 1
    assert result[0].content == "Prefers Python"


class TestSpeakerAttribution:
    """Who said something is part of what was said (#328).

    Measured on 39 reconciled sessions before this: 0% of 400 sampled chunks
    carried any role marker, and the extractor produced 3,185 entities from
    assistant turns against 229 from user turns -- 13.9:1, with no overlap. The
    sample was dominated by products the assistant had recommended (Todoist,
    Trello, Asana), not by anything the user asserted.

    An extractor that cannot tell an assertion from a suggestion cannot weight
    them differently, and a memory tier built from that is mostly the model's
    own output reflected back.
    """

    def test_a_user_message_is_attributed_to_the_user(self):
        action = Message(session_id="s1", role=MessageRole.USER, content="I adopted a beagle named Max")

        assert extract_reconcilable_text(action) == "user: I adopted a beagle named Max"

    def test_an_assistant_message_is_attributed_to_the_assistant(self):
        action = Message(session_id="s1", role=MessageRole.ASSISTANT, content="Congratulations on the new dog!")

        assert extract_reconcilable_text(action) == "assistant: Congratulations on the new dog!"

    def test_two_speakers_saying_the_same_words_stay_distinct(self):
        """Reconciliation dedupes by content hash. Without the speaker in the
        text, a user's assertion and an assistant's echo of it collapse into one
        source, and whichever is written first silently wins."""
        said = "The deployment target is staging"
        user = extract_reconcilable_text(Message(session_id="s1", role=MessageRole.USER, content=said))
        assistant = extract_reconcilable_text(Message(session_id="s1", role=MessageRole.ASSISTANT, content=said))

        assert user != assistant

    def test_tool_results_are_not_given_a_speaker(self):
        """Only conversation turns have a speaker. A tool result is output, and
        labelling it as one would assert something untrue."""
        action = ToolResult(session_id="s1", content="exit code 0")

        assert extract_reconcilable_text(action) == "exit code 0"


@pytest.mark.asyncio
async def test_reconcile_session_keeps_a_turn_whole(graph, actions_graph):
    """A turn is the unit worth extracting from, and the default ~500-char cap
    cut it into roughly 3.6 fragments -- measured, 468 reconcilable actions
    became 1,677 LightRAG documents at two LLM calls each (#327).

    Splitting mid-utterance also hands the extractor a fragment with no
    surrounding context, which is the judgement relationship typing needs
    (#127)."""
    from actions_graph import Session

    actions_graph.create_session(Session(session_id="s-1"))
    actions_graph.record_message(session_id="s-1", role=MessageRole.USER, content="A long turn. " * 80)
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="whole", hash=content_hash("whole"))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        await graph.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    chunk_kwargs = mock_from_texts.call_args.kwargs["chunk_kwargs"]
    assert chunk_kwargs["max_characters"] >= MAX_RECONCILABLE_CHARS
