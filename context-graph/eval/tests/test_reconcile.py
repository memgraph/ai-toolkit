"""Tests for reconcile.py's environment-tuning helpers.

Not e2e: these don't touch Memgraph or an LLM, only the env-var contract
LightRAG reads its cost knobs from.
"""

import os

from context_graph_eval.reconcile import _resolve_reconciliation_tuning


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
