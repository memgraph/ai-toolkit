"""Tests for scoring a retrieval result against its Golden.

The judged half (ContextualRecall, the GEval coverage rubric) is LLM-backed and
lives behind a gate. What is tested here without a key is everything that
decides how a score is *composed*: the deterministic efficiency count, the
coverage gate, and how tiers are kept apart.
"""

import pytest
from context_graph_eval.retrieval import Retrieved
from context_graph_eval.scoring import (
    DEFAULT_TOKENIZER,
    PREFERENCE_STEPS,
    Scored,
    aggregate,
    bleu_score,
    build_metrics,
    efficiency_tokens,
    enforce_retrieval_floor,
    evidence_recall,
    gate_and_rank,
    gate_score,
    rubric_for,
    token_f1_score,
    tokenizer_in_use,
)
from deepeval.models import DeepEvalBaseLLM


def _scored(
    name, *, tier=1, covered=True, tokens=100, abstention=False, metric_scores=None, bleu=0.0, f1=0.0, latency=0.0
):
    return Scored(
        name=name,
        tier=tier,
        coverage=1.0 if covered else 0.0,
        covered=covered,
        efficiency_tokens=tokens,
        abstention=abstention,
        # Non-empty by default: a Scored with no metric scores means the judge
        # could not score it, which is a different thing from scoring zero.
        metric_scores=metric_scores if metric_scores is not None else {"Coverage": 1.0 if covered else 0.0},
        bleu=bleu,
        f1=f1,
        latency_seconds=latency,
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
    names = [type(m).__name__ for m in build_metrics(_StubJudge())]

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
    names = [type(m).__name__ for m in build_metrics(_StubJudge(), rubric="abstention")]

    assert "ContextualRecallMetric" not in names
    assert "GEval" in names


def test_the_recorded_tokenizer_is_the_one_actually_used():
    """A run records what it measured with, so two runs counted in different
    units cannot compare cleanly. Reporting the configured name while silently
    word-splitting would defeat compare()'s tokenizer check, which reads that
    name and would see a match."""
    assert tokenizer_in_use() == DEFAULT_TOKENIZER


def test_a_question_the_judge_could_not_score_is_not_counted_as_a_failure():
    """Observed live: the judge's provider ran out of credit, every metric
    errored, and the run reported "coverage 0/2 (0%)" -- an outage presented as
    a measurement. "Could not be scored" and "scored zero" must not collapse
    into the same number."""
    report = aggregate(
        [
            _scored("judged", covered=True, metric_scores={"Coverage": 1.0}),
            Scored(name="unjudged", tier=1, coverage=0.0, covered=False, efficiency_tokens=10),
        ]
    )

    assert report.by_tier[1].unscored == 1
    # One question was judged and it passed. The unscoreable one is excluded
    # rather than dragging the rate to 50%.
    assert report.by_tier[1].questions == 1
    assert report.by_tier[1].coverage_rate == 1.0


def test_a_score_with_no_reasons_recorded_defaults_to_empty_not_missing():
    """A Scored built without metric_reasons (every caller before this field
    existed, and every judge-free question) must not raise on .get() -- an
    absent reason is a normal case, not an error."""
    scored = Scored(name="q1", tier=1, coverage=1.0, covered=True, efficiency_tokens=10)

    assert scored.metric_reasons == {}


def test_a_fully_unscoreable_tier_reports_no_rate():
    """Better to say nothing than to report 0%."""
    report = aggregate([Scored(name="q", tier=1, coverage=0.0, covered=False, efficiency_tokens=0)])

    assert report.by_tier[1].unscored == 1
    assert report.by_tier[1].questions == 0
    assert report.by_tier[1].coverage_rate is None


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


# --- BLEU/F1: deterministic, judge-free cross-checks against the answer
# key, computed the same way regardless of whether a judge ran. ---


def test_bleu_scores_an_exact_match_as_perfect():
    assert bleu_score("Admon was assigned the day shift.", "Admon was assigned the day shift.") == 1.0


def test_bleu_scores_unrelated_text_near_zero():
    assert bleu_score("Admon was assigned the day shift.", "Completely unrelated sentence about kayaking.") < 0.2


def test_bleu_is_zero_for_an_empty_answer():
    """An empty answer -- a failed or abstaining retrieval -- has zero token
    overlap with any real expected output. Not an error: a real outcome."""
    assert bleu_score("Admon was assigned the day shift.", "") == 0.0


def test_bleu_is_zero_for_an_empty_expected_output():
    assert bleu_score("", "Admon was assigned the day shift.") == 0.0


def test_f1_scores_an_exact_match_as_perfect():
    assert token_f1_score("Admon was assigned the day shift.", "Admon was assigned the day shift.") == 1.0


def test_f1_rewards_partial_token_overlap():
    """Half the expected tokens present, none extra: precision 1.0, recall
    0.5, F1 the harmonic mean of the two -- not their average."""
    f1 = token_f1_score("the day shift starts at eight am", "the day shift")

    assert 0.0 < f1 < 1.0


def test_f1_is_zero_for_no_overlap_at_all():
    assert token_f1_score("Admon was assigned the day shift.", "Completely different words entirely.") == 0.0


def test_f1_counts_token_multiplicity_not_just_membership():
    """'the the the' against a single 'the' must not score a perfect match on
    either side -- Counter intersection, not set intersection."""
    f1 = token_f1_score("the cat sat", "the the the")

    assert f1 < 1.0


def test_f1_is_zero_for_an_empty_answer():
    assert token_f1_score("Admon was assigned the day shift.", "") == 0.0


def test_f1_ignores_punctuation_the_answer_key_happens_to_carry():
    """Whitespace-splitting glued the expected output's trailing period onto
    its last token, so "A beagle." against "A beagle" scored only 0.5 despite
    being the same answer -- punctuation is not a fact the judge cares
    about."""
    assert token_f1_score("A beagle.", "A beagle") == 1.0


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


def test_mean_bleu_f1_latency_include_every_scored_question_not_just_covered():
    """Unlike efficiency's gate (#309, specific to that axis), BLEU/F1/latency
    are their own independent, judge-free signal -- a failing question's
    answer still has a real BLEU/F1/latency, and dropping it would hide
    exactly the questions most worth looking at."""
    report = aggregate(
        [
            _scored("passed", covered=True, bleu=1.0, f1=1.0, latency=2.0),
            _scored("failed", covered=False, bleu=0.0, f1=0.0, latency=4.0),
        ]
    )

    summary = report.by_tier[1]
    assert summary.mean_bleu == 0.5
    assert summary.mean_f1 == 0.5
    assert summary.mean_latency_seconds == 3.0


def test_mean_bleu_f1_latency_survive_a_judge_free_run():
    """The bug this guards against (#342): with no judge (or one that
    errored on every question) metric_scores is empty for every row, so
    aggregating BLEU/F1/latency over the judge-scored subset -- rather than
    over all_rows -- silently produced three None aggregates instead of the
    judge-free run's only headline numbers."""
    report = aggregate(
        [
            _scored("q1", metric_scores={}, bleu=0.4, f1=0.6, latency=1.0),
            _scored("q2", metric_scores={}, bleu=0.8, f1=0.2, latency=3.0),
        ]
    )

    summary = report.by_tier[1]
    # Unscored by the judge -- coverage and the gated efficiency median stay
    # unavailable, since neither can mean anything without a verdict.
    assert summary.coverage_rate is None
    assert summary.unscored == 2
    # But BLEU/F1/latency need no verdict, so they must not be None too.
    assert summary.mean_bleu == pytest.approx(0.6)
    assert summary.mean_f1 == pytest.approx(0.4)
    assert summary.mean_latency_seconds == pytest.approx(2.0)


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


# --- Empty retrieval must not pass (#309's degenerate case, seen live) ---


def test_a_question_answered_from_nothing_cannot_pass():
    """Observed in calibration run 2: gpt4_59149c77, a temporal-reasoning
    question whose expected answer is "7 days", retrieved ZERO tokens, answered
    "not in memory", and scored Contextual Recall 1.0 and Coverage [GEval] 1.0
    -- a perfect pass.

    ContextualRecall over an empty context is vacuously satisfied, and the
    rubric let a refusal stand in for a fact. Coverage measured on a question
    the graph was never consulted for is not a measurement of recall, and it
    inflates the headline in the one direction nobody checks.
    """
    scored = enforce_retrieval_floor(
        [_scored("q1", covered=True, metric_scores={"Contextual Recall": 1.0, "Coverage [GEval]": 1.0})],
        retrieved_tokens={"q1": 0},
    )

    assert scored[0].covered is False
    assert scored[0].coverage == 0.0


def test_an_abstention_question_may_legitimately_retrieve_nothing():
    """For these the correct answer IS "not in memory", so an empty payload is
    the right behaviour rather than the degenerate case -- applying the floor
    here would make them unpassable, which is the bug this loop already fixed
    once."""
    scored = enforce_retrieval_floor(
        [_scored("q1", covered=True, abstention=True, metric_scores={"Coverage": 1.0})],
        retrieved_tokens={"q1": 0},
    )

    assert scored[0].covered is True


def test_a_question_that_retrieved_something_is_left_alone():
    scored = enforce_retrieval_floor(
        [_scored("q1", covered=True, metric_scores={"Coverage": 1.0})],
        retrieved_tokens={"q1": 250},
    )

    assert scored[0].covered is True


def test_abstention_is_judged_on_refusing_not_on_reciting_the_near_miss():
    """Measured across three runs: the agent declined on 5/8, 3/8 and 5/8
    abstention questions, and scored 0/8 every time.

    The cause is upstream's expected outputs, which pair the refusal with a
    contrastive fact -- "You mentioned your cat Luna but not your hamster",
    "You mentioned trying Korean restaurants but not Italian restaurants". The
    shared Coverage rubric asks whether the answer contains every fact in the
    expected output, so a correct "not in memory" was marked down for omitting
    the near-miss detail: partial credit of 0.3-0.6, under the 0.7 gate, every
    time.

    That also contradicted the rubric's own stated criterion, which said a
    refusal was what these questions required.

    Abstention exists to measure not fabricating an answer, so that is what it
    scores.
    """
    (metric,) = build_metrics(_StubJudge(), rubric="abstention")

    assert metric.name == "Abstention"
    criteria = metric.criteria.lower()
    assert "decline" in criteria or "refus" in criteria
    # The failure mode being fixed: demanding the expected output's facts back.
    assert "every fact" not in criteria


def test_rubrics_carry_fixed_evaluation_steps_so_the_judge_never_writes_them():
    """Without them deepeval generates steps per question: an extra judge call each, and drifting rubrics."""
    from context_graph_eval.scoring import ABSTENTION_STEPS, COVERAGE_STEPS, build_metrics

    judge = _StubJudge()
    coverage = next(m for m in build_metrics(judge) if getattr(m, "name", "") == "Coverage")
    abstention = build_metrics(judge, rubric="abstention")[0]
    assert coverage.evaluation_steps == COVERAGE_STEPS
    assert abstention.evaluation_steps == ABSTENTION_STEPS


def test_the_answer_rubric_gates_and_contextual_recall_does_not():
    """A computed answer ("17 days") has no single retrieved row stating it, so
    Contextual Recall scored correct answers 0 and failed them (#397)."""
    assert gate_score({"Contextual Recall": 0.0, "Coverage [GEval]": 0.9}) == 0.9
    assert gate_score({"Contextual Recall": 1.0, "Coverage [GEval]": 0.2}) == 0.2
    assert gate_score({"Abstention [GEval]": 0.8}) == 0.8


def test_a_judge_that_returned_only_the_retrieval_signal_gates_nothing_through():
    assert gate_score({"Contextual Recall": 1.0}) == 0.0
    assert gate_score({}) == 0.0


def test_retrieval_is_reported_per_tier_without_gating():
    report = aggregate(
        [
            _scored("q1", covered=True, metric_scores={"Contextual Recall": 0.0, "Coverage [GEval]": 1.0}),
            _scored("q2", covered=True, metric_scores={"Contextual Recall": 1.0, "Coverage [GEval]": 1.0}),
            _scored("q3", covered=True, abstention=True, metric_scores={"Abstention [GEval]": 1.0}),
        ]
    ).by_tier[1]

    assert report.covered == 3
    assert (report.retrieval_supported, report.retrieval_judged) == (1, 2)


def test_each_question_is_routed_to_its_answer_rubric():
    assert rubric_for({"question_type": "multi-session"}) == "answer"
    assert rubric_for({"question_type": "single-session-preference"}) == "preference"
    assert rubric_for({"question_type": "single-session-preference", "abstention": True}) == "abstention"
    assert rubric_for({}) == "answer"


def test_preference_questions_are_judged_on_using_the_users_preferences_not_on_every_fact():
    """Their expected output is a rubric for a personalized answer, so Coverage's
    every-fact check failed recommendations that did what the rubric asked:
    0/6 at 0.2-0.6 (#404). Retrieval is still reported beside it."""
    metrics = build_metrics(_StubJudge(), rubric="preference")

    assert [type(m).__name__ for m in metrics] == ["ContextualRecallMetric", "GEval"]
    preference = metrics[1]
    assert preference.name == "Preference"
    assert preference.evaluation_steps == PREFERENCE_STEPS
    assert "need not reflect every point" in preference.criteria


def test_evidence_recall_is_the_share_of_evidence_turns_retrieved():
    evidence = ["user: I took a week-long break in January.", "user: Then a 10-day break in February."]

    assert evidence_recall(evidence, ["TURN [s1, 2023-01-20, user]: I took a week-long break in January."]) == 0.5
    assert evidence_recall(evidence, ["FACT: unrelated"]) == 0.0
    assert evidence_recall(evidence, [f"TURN [...]: {turn.split(': ', 1)[1]}" for turn in evidence]) == 1.0


def test_a_question_without_marked_evidence_has_no_evidence_recall():
    """Abstention questions mark no evidence: there is nothing to retrieve, not a miss."""
    assert evidence_recall([], ["anything"]) is None
    assert evidence_recall(None, ["anything"]) is None


def test_a_long_evidence_turn_counts_when_its_opening_was_retrieved():
    """Rows truncate long turns (hybrid's turn_chars), so the whole turn can't be required;
    whitespace differences from rendering don't count either."""
    turn = "user: " + "I visited the museum. " * 200
    row = "TURN [...]: " + " ".join(turn.split(": ", 1)[1].split())[:1500]

    assert evidence_recall([turn], [row.replace(" ", "  ", 3)]) == 1.0


def test_evidence_recall_is_reported_per_tier():
    report = aggregate(
        [
            Scored(name="q1", tier=1, coverage=1.0, covered=True, efficiency_tokens=1, evidence_recall=1.0),
            Scored(name="q2", tier=1, coverage=0.0, covered=False, efficiency_tokens=1, evidence_recall=0.5),
            Scored(name="q3", tier=1, coverage=1.0, covered=True, efficiency_tokens=1, abstention=True),
        ]
    ).by_tier[1]

    assert (report.evidence_complete, report.evidence_judged) == (1, 2)
    assert report.mean_evidence_recall == 0.75
