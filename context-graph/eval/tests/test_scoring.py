"""Tests for scoring a retrieval result against its Golden.

The judged half (ContextualRecall, the GEval coverage rubric) is LLM-backed and
lives behind a gate. What is tested here without a key is everything that
decides how a score is *composed*: the deterministic efficiency count, the
coverage gate, and how tiers are kept apart.
"""

from context_graph_eval.retrieval import Retrieved
from context_graph_eval.scoring import (
    DEFAULT_TOKENIZER,
    Scored,
    aggregate,
    build_metrics,
    efficiency_tokens,
    gate_and_rank,
    tokenizer_in_use,
)
from deepeval.models import DeepEvalBaseLLM


def _scored(name, *, tier=1, covered=True, tokens=100, abstention=False):
    return Scored(
        name=name,
        tier=tier,
        coverage=1.0 if covered else 0.0,
        covered=covered,
        efficiency_tokens=tokens,
        abstention=abstention,
    )


class _StubJudge(DeepEvalBaseLLM):
    """A judge that is never called.

    These tests are about which metrics get *built*, not about running them.
    Passing ``judge=None`` makes deepeval fall back to an OpenAI model and
    demand a key at construction time, which would make a pure composition test
    require credentials.
    """

    def __init__(self):
        super().__init__(model_name="stub")

    def load_model(self):
        return self

    def get_model_name(self):
        return "stub"

    def generate(self, *args, **kwargs):
        raise AssertionError("the stub judge should never be called")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("the stub judge should never be called")


def test_an_ordinary_question_is_scored_on_retrieval_and_answer():
    names = [type(m).__name__ for m in build_metrics(_StubJudge(), abstention=False)]

    assert "ContextualRecallMetric" in names
    assert "GEval" in names


def test_an_abstention_question_is_not_scored_on_contextual_recall():
    """ContextualRecall asks whether the retrieved context supports the expected
    output. For an abstention question the correct retrieved context is EMPTY,
    so the metric scores ~0 by construction -- and because coverage takes the
    weakest metric, it made every abstention question unpassable no matter how
    well the agent behaved.

    Measured before this fix: abstention scored 0/8, while the agent had
    correctly answered "not in memory" on at least four of them.
    """
    names = [type(m).__name__ for m in build_metrics(_StubJudge(), abstention=True)]

    assert "ContextualRecallMetric" not in names
    assert "GEval" in names


def test_the_recorded_tokenizer_is_the_one_actually_used():
    """A run records what it measured with, so two runs counted in different
    units cannot compare cleanly. Reporting the configured name while silently
    word-splitting would defeat compare()'s tokenizer check, which reads that
    name and would see a match."""
    assert tokenizer_in_use() == DEFAULT_TOKENIZER


def test_efficiency_counts_the_payload_handed_back():
    """#309: efficiency is how many tokens were returned to answer the
    question. Fewer for the same answer is better."""
    retrieved = Retrieved(answer="A beagle.", retrieval_context=["one two three", "four five"])

    assert efficiency_tokens(retrieved) > 0


def test_efficiency_grows_with_a_larger_payload():
    small = Retrieved(answer="x", retrieval_context=["a short row"])
    large = Retrieved(answer="x", retrieval_context=["a short row"] * 20)

    assert efficiency_tokens(large) > efficiency_tokens(small)


def test_returning_nothing_costs_nothing():
    """The degenerate case the coverage gate exists to catch: an empty payload
    is maximally 'efficient' and useless."""
    assert efficiency_tokens(Retrieved(answer="", retrieval_context=[])) == 0


def test_only_questions_that_cleared_coverage_are_ranked():
    """Coverage is a hard gate, not a weighted term (#309) -- otherwise a
    retrieval change could trade real coverage for token savings and still
    show a flat or improved headline."""
    ranked = gate_and_rank(
        [
            _scored("passed-cheap", covered=True, tokens=50),
            _scored("failed-cheapest", covered=False, tokens=1),
            _scored("passed-costly", covered=True, tokens=500),
        ]
    )

    assert [s.name for s in ranked] == ["passed-cheap", "passed-costly"]


def test_a_failing_question_cannot_win_on_efficiency():
    ranked = gate_and_rank([_scored("failed", covered=False, tokens=0)])

    assert ranked == []


def test_tiers_are_aggregated_separately():
    """#303: a blended number would let an organizational-recall regression
    hide behind a personal-memory gain."""
    report = aggregate(
        [
            _scored("t1-a", tier=1, covered=True),
            _scored("t1-b", tier=1, covered=True),
            _scored("t2-a", tier=2, covered=False),
        ]
    )

    assert report.by_tier[1].coverage_rate == 1.0
    assert report.by_tier[2].coverage_rate == 0.0


def test_aggregate_refuses_to_produce_a_single_blended_number():
    report = aggregate([_scored("t1", tier=1), _scored("t2", tier=2, covered=False)])

    assert not hasattr(report, "overall_coverage_rate")


def test_median_efficiency_uses_only_questions_that_passed():
    report = aggregate(
        [
            _scored("cheap", covered=True, tokens=10),
            _scored("costly", covered=True, tokens=30),
            _scored("failed", covered=False, tokens=1),
        ]
    )

    assert report.by_tier[1].median_efficiency_tokens == 20


def test_abstention_questions_are_reported_apart():
    """For these the correct answer is 'not in memory', so a confident answer
    is the failure -- averaging them in with ordinary recall hides that."""
    report = aggregate(
        [
            _scored("ordinary", covered=True),
            _scored("abstain", covered=False, abstention=True),
        ]
    )

    assert report.by_tier[1].abstention_total == 1
    assert report.by_tier[1].abstention_correct == 0


def test_an_empty_run_reports_nothing_rather_than_dividing_by_zero():
    report = aggregate([])

    assert report.by_tier == {}


def test_a_tier_with_no_passing_question_has_no_median():
    report = aggregate([_scored("failed", covered=False)])

    assert report.by_tier[1].median_efficiency_tokens is None
