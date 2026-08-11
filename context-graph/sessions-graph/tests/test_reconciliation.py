"""Unit tests for Sessions Graph reconciliation (extract_reconcilable_text,
build_reconciliation_sources, SessionsGraph.reconcile_session).

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
        assert extract_reconcilable_text(action) == "Hello there"

    def test_message_with_content_blocks_joins_text(self):
        action = Message(
            session_id="s-1",
            role=MessageRole.ASSISTANT,
            content=[{"type": "text", "text": "Part one"}, {"type": "text", "text": "Part two"}],
        )
        assert extract_reconcilable_text(action) == "Part one\nPart two"

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
        assert sources[0] == ReconciliationSource(kind="action", node_id=actions[0].action_id, text="Question")
        assert sources[1] == ReconciliationSource(kind="action", node_id=actions[1].action_id, text="Answer")
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
# SessionsGraph.reconcile_session (stubbed Memgraph + mocked ActionsGraph/LightRAG)
# ---------------------------------------------------------------------------


def _stub_db():
    db = MagicMock()
    db.query.return_value = []
    return db


def _graph(db=None):
    from sessions_graph.core import SessionsGraph

    g = SessionsGraph.__new__(SessionsGraph)
    g._db = db or _stub_db()
    return g


def _fake_actions_graph(actions):
    ag = MagicMock()
    ag.get_session_actions.return_value = actions
    return ag


def _fake_lightrag_wrapper(summary_text: str = "A narrative summary of the session."):
    wrapper = MagicMock()
    wrapper.get_lightrag.return_value.llm_model_func = AsyncMock(return_value=summary_text)
    return wrapper


@pytest.mark.asyncio
async def test_reconcile_session_success_marks_completed_and_links_chunks():
    db = _stub_db()
    g = _graph(db)
    actions = [Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine.")]
    actions_graph = _fake_actions_graph(actions)
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        summary = await g.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "completed"
    assert summary.texts_considered == 1
    assert summary.texts_deduped == 1
    mock_from_texts.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_session_writes_episode_from_dedicated_llm_call():
    db = _stub_db()
    g = _graph(db)
    actions = [Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine.")]
    actions_graph = _fake_actions_graph(actions)
    lightrag_wrapper = _fake_lightrag_wrapper("Alice was discussed working on the graph engine.")

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])):
        summary = await g.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.summary_written is True
    lightrag_wrapper.get_lightrag.return_value.llm_model_func.assert_awaited_once()
    episode_calls = [call for call in db.query.call_args_list if "HAS_EPISODE" in call.args[0]]
    assert len(episode_calls) == 1
    assert episode_calls[0].kwargs["params"]["summary"] == "Alice was discussed working on the graph engine."
    assert episode_calls[0].kwargs["params"]["session_id"] == "s-1"


@pytest.mark.asyncio
async def test_reconcile_session_passes_promotion_and_ontology_kwargs_through_to_from_texts():
    db = _stub_db()
    g = _graph(db)
    actions = [Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Alice works on the graph engine.")]
    actions_graph = _fake_actions_graph(actions)
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="Alice works on the graph engine.", hash=content_hash("Alice works on the graph engine."))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        await g.reconcile_session(
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
async def test_reconcile_session_dedupes_identical_text_before_calling_lightrag():
    db = _stub_db()
    g = _graph(db)
    actions = [
        Message(session_id="s-1", role=MessageRole.USER, content="Same question"),
        Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Same question"),
    ]
    actions_graph = _fake_actions_graph(actions)
    lightrag_wrapper = _fake_lightrag_wrapper()

    fake_chunk = Chunk(text="Same question", hash=content_hash("Same question"))
    with patch("unstructured2graph.from_texts", new=AsyncMock(return_value=[[fake_chunk]])) as mock_from_texts:
        summary = await g.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.texts_considered == 2
    assert summary.texts_deduped == 1
    mock_from_texts.assert_awaited_once()
    called_texts = mock_from_texts.call_args.args[0]
    assert called_texts == ["Same question"]


@pytest.mark.asyncio
async def test_reconcile_session_no_reconcilable_content_skips_lightrag_but_still_completes():
    db = _stub_db()
    g = _graph(db)
    actions_graph = _fake_actions_graph([])
    lightrag_wrapper = MagicMock()

    with patch("unstructured2graph.from_texts", new=AsyncMock()) as mock_from_texts:
        summary = await g.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "completed"
    assert summary.texts_considered == 0
    assert summary.summary_written is False
    mock_from_texts.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_session_failure_marks_failed_and_returns_error():
    db = _stub_db()
    g = _graph(db)
    actions = [Message(session_id="s-1", role=MessageRole.ASSISTANT, content="Some content")]
    actions_graph = _fake_actions_graph(actions)
    lightrag_wrapper = MagicMock()

    with patch("unstructured2graph.from_texts", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        summary = await g.reconcile_session("s-1", lightrag_wrapper=lightrag_wrapper, actions_graph=actions_graph)

    assert summary.status == "failed"
    assert "LLM down" in summary.error


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
