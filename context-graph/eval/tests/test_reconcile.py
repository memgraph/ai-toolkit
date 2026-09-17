"""Tests for reconcile.py's environment-tuning helpers and reconcile_batch's
own validation.

Not e2e: these don't touch Memgraph or an LLM.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from context_graph_eval.reconcile import _resolve_reconciliation_tuning, reconcile_batch


def test_sets_a_looser_merge_threshold_when_unset(monkeypatch):
    """LightRAG's own default (8) re-triggers a paid merge-summary call on every
    later mention of an entity that has crossed 8 raw mentions -- confirmed by
    reading operate.py's merge path. Eval batches share one workspace across
    thousands of sessions, so a recurring entity keeps re-paying that cost for
    the rest of the run. This raises the eval-only default well past what most
    entities in a single batch will realistically reach."""
    monkeypatch.delenv("FORCE_LLM_SUMMARY_ON_MERGE", raising=False)

    _resolve_reconciliation_tuning()

    assert os.environ["FORCE_LLM_SUMMARY_ON_MERGE"] == "30"


def test_never_overrides_an_operators_own_value(monkeypatch):
    """setdefault semantics: production sessions-graph reconciliation never sets
    this at all (and keeps LightRAG's default of 8), and an operator who has
    exported their own value is deliberately tuning it -- either way, this must
    not clobber what's already there."""
    monkeypatch.setenv("FORCE_LLM_SUMMARY_ON_MERGE", "8")

    _resolve_reconciliation_tuning()

    assert os.environ["FORCE_LLM_SUMMARY_ON_MERGE"] == "8"


def test_lowers_embedding_concurrency_despite_raising_llm_concurrency(monkeypatch):
    """Measured live: raising MAX_PARALLEL_INSERT/MAX_ASYNC_LLM for the
    OpenAI-backed extraction LLM also puts more documents' embedding calls in
    flight at once -- all against the SAME local, CPU-bound bge-m3 model
    (#331), which doesn't parallelize like a rate-limited remote API does.
    9 of 10 sessions timed out in one real batch before this was tuned down
    and the timeout raised to compensate."""
    monkeypatch.delenv("EMBEDDING_FUNC_MAX_ASYNC", raising=False)
    monkeypatch.delenv("EMBEDDING_TIMEOUT", raising=False)

    _resolve_reconciliation_tuning()

    assert os.environ["EMBEDDING_FUNC_MAX_ASYNC"] == "2"
    assert os.environ["EMBEDDING_TIMEOUT"] == "120"


def test_embedding_tuning_also_never_overrides_an_operators_own_value(monkeypatch):
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "8")
    monkeypatch.setenv("EMBEDDING_TIMEOUT", "30")

    _resolve_reconciliation_tuning()

    assert os.environ["EMBEDDING_FUNC_MAX_ASYNC"] == "8"
    assert os.environ["EMBEDDING_TIMEOUT"] == "30"


@pytest.mark.asyncio
async def test_reconcile_batch_rejects_non_positive_sessions_per_call():
    """range(0, N, sessions_per_call) with a negative step is silently
    empty (reports "nothing to do" while sessions sit pending, untouched)
    and with 0 raises ValueError from deep inside range() -- neither is an
    obviously-relevant error. Must be validated up front, before even
    querying for pending sessions."""
    db = MagicMock()

    with pytest.raises(ValueError, match="sessions_per_call"):
        await reconcile_batch(db, sessions_per_call=0)

    with pytest.raises(ValueError, match="sessions_per_call"):
        await reconcile_batch(db, sessions_per_call=-5)

    db.query.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_batch_rejects_an_unknown_extraction_backend():
    db = MagicMock()

    with pytest.raises(ValueError, match="extraction_backend"):
        await reconcile_batch(db, extraction_backend="anthropic")

    db.query.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_batch_gliner2_mode_reconciles_one_session_at_a_time():
    """No batch/queue pipeline exists for a backend with no shared busy-lock
    to fan out over -- gliner2 mode must go through reconcile_session, not
    reconcile_sessions_batch (map #322's pipeline is LightRAG-specific)."""
    db = MagicMock()
    db.query.return_value = [{"session_id": "s-1"}, {"session_id": "s-2"}]
    lightrag_wrapper = MagicMock()
    fake_backend = MagicMock()

    fake_graph = MagicMock()
    fake_graph.reconcile_session = AsyncMock(
        side_effect=lambda session_id, **_: SimpleNamespace(session_id=session_id, status="completed", error=None)
    )
    fake_graph.reconcile_sessions_batch = AsyncMock()

    with (
        patch("sessions_graph.SessionsGraph", return_value=fake_graph),
        patch("unstructured2graph.gliner2_backend.GLiNER2Backend", return_value=fake_backend),
    ):
        outcome = await reconcile_batch(
            db,
            memgraph_url="bolt://fake:7687",
            lightrag_wrapper=lightrag_wrapper,
            extraction_backend="gliner2",
            progress=False,
        )

    assert outcome.reconciled == 2
    assert fake_graph.reconcile_session.await_count == 2
    fake_graph.reconcile_session.assert_any_await(
        "s-1", lightrag_wrapper=lightrag_wrapper, extraction_backend=fake_backend, enforce_ontology=True
    )
    fake_graph.reconcile_sessions_batch.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_batch_lightrag_mode_still_uses_the_batch_pipeline():
    """Regression guard alongside the gliner2 test above: the default path
    must be untouched by extraction_backend's introduction."""
    db = MagicMock()
    db.query.return_value = [{"session_id": "s-1"}]
    lightrag_wrapper = MagicMock()

    fake_graph = MagicMock()
    fake_graph.reconcile_sessions_batch = AsyncMock(
        return_value=[SimpleNamespace(session_id="s-1", status="completed", error=None)]
    )
    fake_graph.reconcile_session = AsyncMock()

    with patch("sessions_graph.SessionsGraph", return_value=fake_graph):
        outcome = await reconcile_batch(
            db, memgraph_url="bolt://fake:7687", lightrag_wrapper=lightrag_wrapper, progress=False
        )

    assert outcome.reconciled == 1
    fake_graph.reconcile_sessions_batch.assert_awaited_once()
    fake_graph.reconcile_session.assert_not_called()
