"""Tests for reconcile.py's environment-tuning helpers and reconcile_batch's
own validation.

Not e2e: these don't touch Memgraph or an LLM.
"""

import os
from unittest.mock import MagicMock

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
