"""Tests for reconcile's input validation; reconciling itself is tested against Memgraph in test_e2e_reconcile."""

from unittest.mock import MagicMock

import pytest
from context_graph_eval.pipeline import BuildPlan
from context_graph_eval.reconcile import reconcile_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [0, -5])
async def test_a_concurrency_below_one_is_refused_before_querying(concurrency):
    db = MagicMock()

    with pytest.raises(ValueError, match="concurrency"):
        await reconcile_pending(db, concurrency=concurrency)

    db.query.assert_not_called()


def test_a_build_extracts_under_the_default_or_a_learned_ontology_only():
    """The hand-written LongMemEval vocabulary ('fixed') is gone: every build is the product's own."""
    assert BuildPlan(ontology="learned").ontology == "learned"
    with pytest.raises(ValueError, match="ontology"):
        BuildPlan(ontology="fixed")
