"""Tests for reconcile.py's environment-tuning helpers, reconcile_batch's own
validation, and (real Memgraph, stubbed/faked LLM boundary, no OPENAI_API_KEY
needed) which reconciliation strategy each extraction_backend drives.
"""

import os
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from conftest import EVAL_MEMGRAPH_URL
from context_graph_eval.reconcile import ExtractionBackendName, _resolve_reconciliation_tuning, reconcile_batch


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
    """The Literal type catches this at the call sites we control; this
    guards the runtime path for a value that reached here anyway (a stale
    saved value, an env var, anything outside static analysis) -- hence the
    explicit cast, not a type error, to construct that value here."""
    db = MagicMock()

    with pytest.raises(ValueError, match="extraction_backend"):
        await reconcile_batch(db, extraction_backend=cast("ExtractionBackendName", "anthropic"))

    db.query.assert_not_called()


# --- Real SessionsGraph/ActionsGraph/Memgraph below (testing policy: prefer
# a real instance of another package's class over a hand-rolled fake -- a
# fake SessionsGraph proved nothing about whether reconcile_batch actually
# drives the real batch pipeline, or the real per-session one, correctly).
# Only the LLM boundary is stubbed (free, deterministic, unstructured2graph's
# own established pattern) and GLiNER2's underlying model is faked via the
# class's own documented no-gliner2-install escape hatch -- neither needs
# OPENAI_API_KEY or a real `gliner2` install.


async def _stub_no_entities(prompt, system_prompt=None, history_messages=None, **kwargs):
    """A minimal, valid LightRAG extraction response meaning "nothing found"."""
    return "<|COMPLETE|>"


class _FakeGLiNER2Schema:
    """Mirrors gliner2's real chainable schema builder -- see
    unstructured2graph/tests/test_gliner2_backend.py's own _FakeSchema, which
    established this as GLiNER2Backend's supported way to run without the
    real `gliner2` package."""

    def entities(self, schema):
        return self

    def relations(self, schema):
        return self


class _FakeGLiNER2Model:
    def create_schema(self):
        return _FakeGLiNER2Schema()

    def extract_long(self, text, schema, **kwargs):
        return {"entities": {}}


def _one_session_fixture(session_id: str):
    from context_graph_eval.convert.longmemeval import SessionFixture, Turn

    return SessionFixture(
        session_id=session_id,
        date="2023/05/20 (Sat) 14:03",
        turns=[Turn(role="user", content=f"I adopted a beagle named Max, in {session_id}")],
        holds_evidence=True,
    )


def _session_row(eval_graph, session_id: str) -> dict:
    rows = eval_graph._db.query(
        "MATCH (s:Session {session_id: $id}) RETURN s.reconciliation_status AS status, "
        "s.extraction_backend AS extraction_backend",
        {"id": session_id},
    )
    return rows[0]


@pytest.mark.asyncio
async def test_reconcile_batch_gliner2_mode_reconciles_one_session_at_a_time(eval_graph, monkeypatch):
    """No batch/queue pipeline exists for a backend with no shared busy-lock
    to fan out over -- gliner2 mode must go through reconcile_session, not
    reconcile_sessions_batch (map #322's pipeline is LightRAG-specific)."""
    from context_graph_eval.inject import inject_batch

    from lightrag_memgraph import MemgraphLightRAGWrapper

    inject_batch([_one_session_fixture("s-1"), _one_session_fixture("s-2")], graph=eval_graph)

    # LightRAG's Memgraph storage backend reads this at construction time,
    # before reconcile_batch itself sets it -- needed regardless of
    # extraction_backend, since the summarization LLM call still goes
    # through this same wrapper either way.
    monkeypatch.setenv("MEMGRAPH_URL", EVAL_MEMGRAPH_URL)
    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(working_dir="./lightrag_storage.test_reconcile_gliner2", llm_model_func=_stub_no_entities)

    # Imported before patching: a lookup done *while* the patch is active
    # would resolve to the mock itself, calling it recursively.
    from unstructured2graph.gliner2_backend import GLiNER2Backend as RealGLiNER2Backend

    def _fake_gliner2_backend():
        return RealGLiNER2Backend(model=_FakeGLiNER2Model())

    try:
        with patch("unstructured2graph.gliner2_backend.GLiNER2Backend", side_effect=_fake_gliner2_backend):
            outcome = await reconcile_batch(
                eval_graph._db,
                memgraph_url=EVAL_MEMGRAPH_URL,
                lightrag_wrapper=wrapper,
                extraction_backend="gliner2",
                progress=False,
            )
    finally:
        await wrapper.afinalize()

    assert outcome.reconciled == 2
    assert outcome.failed == 0
    for session_id in ("s-1", "s-2"):
        row = _session_row(eval_graph, session_id)
        assert row["status"] == "completed"
        assert row["extraction_backend"] == "GLiNER2Backend"


@pytest.mark.asyncio
async def test_reconcile_batch_lightrag_mode_still_uses_the_batch_pipeline(eval_graph, monkeypatch):
    """Regression guard alongside the gliner2 test above: the default path
    must be untouched by extraction_backend's introduction."""
    from context_graph_eval.inject import inject_batch

    from lightrag_memgraph import MemgraphLightRAGWrapper

    inject_batch([_one_session_fixture("s-1")], graph=eval_graph)

    monkeypatch.setenv("MEMGRAPH_URL", EVAL_MEMGRAPH_URL)
    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(working_dir="./lightrag_storage.test_reconcile_lightrag", llm_model_func=_stub_no_entities)

    try:
        outcome = await reconcile_batch(
            eval_graph._db, memgraph_url=EVAL_MEMGRAPH_URL, lightrag_wrapper=wrapper, progress=False
        )
    finally:
        await wrapper.afinalize()

    assert outcome.reconciled == 1
    row = _session_row(eval_graph, "s-1")
    assert row["status"] == "completed"
    assert row["extraction_backend"] == "LightRAGBackend"
