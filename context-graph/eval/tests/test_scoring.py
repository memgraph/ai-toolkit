"""Tests for scoring a retrieval result against its Golden.

The judged half (ContextualRecall, the GEval coverage rubric) is LLM-backed and
lives behind a gate. What is tested here without a key is everything that
decides how a score is *composed*: the deterministic efficiency count, BLEU
and F1, evidence recall, and which rubric judges which question.
"""

from context_graph_eval.answering import Retrieved
from context_graph_eval.scoring import (
    PREFERENCE_STEPS,
    bleu_score,
    build_metrics,
    efficiency_tokens,
    evidence_recall,
    gate_score,
    rubric_for,
    token_f1_score,
)
from deepeval.models import DeepEvalBaseLLM


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


# --- Empty retrieval must not pass (#309's degenerate case, seen live) ---


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
