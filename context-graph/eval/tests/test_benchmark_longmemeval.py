"""Tests for how the LongMemEval adapter turns judged answers into rows.

The judges themselves are LLM calls, stubbed here at that boundary; what is
tested is who decides a question's score and what overrides it.
"""

import pytest
from context_graph_eval import official_judge
from context_graph_eval.answering import Retrieved
from context_graph_eval.benchmarks.longmemeval import LongMemEval, _Judged, accuracy
from context_graph_eval.goldslice import GOLD_SLICE_USER, gold_slice_goldens
from deepeval.dataset import Golden


def _question(name, *, question_type="multi-session", abstention=False) -> Golden:
    return Golden(
        input=f"question {name}?",
        expected_output="Paris",
        context=["user: I moved to Paris."],
        name=name,
        additional_metadata={
            "tier": 1,
            "question_type": question_type,
            "question_date": "2023/06/01 (Thu) 10:00",
            "abstention": abstention,
        },
    )


FOUND = Retrieved(answer="Paris", retrieval_context=["TURN [session s, 2023-05-30, user]: I moved to Paris."])
NOTHING = Retrieved(answer="Paris")


@pytest.fixture
def verdicts(monkeypatch):
    """The official judge's verdict per question name, as the test sets them."""
    decided: dict[str, bool | None] = {}

    async def judge_all(prompts, **_):
        return {name: decided.get(name) for name in prompts}

    monkeypatch.setattr(official_judge, "judge_all", judge_all)
    return decided


async def test_the_official_judge_decides_the_score(verdicts):
    verdicts.update(q1=True, q2=False)

    rows = await LongMemEval().judge([_question("q1"), _question("q2")], [FOUND, FOUND])

    assert [row.score for row in rows] == [1.0, 0.0]
    assert rows[0].metrics["judged_by"] == "official"


async def test_a_failed_official_call_is_unjudged_never_a_fallback_to_our_rubrics(verdicts, monkeypatch):
    adapter = LongMemEval(rubric_judge=object())
    monkeypatch.setattr(adapter, "_rubric_scores", lambda *_: {"q1": _Judged(scores={"Coverage": 1.0})})

    rows = await adapter.judge([_question("q1")], [FOUND])

    assert rows[0].score is None
    assert rows[0].metrics["rubric_coverage"] == 1.0


async def test_an_answer_from_an_empty_retrieval_cannot_pass(verdicts):
    """An empty payload is not evidence of recall, whatever the judge made of the answer."""
    verdicts.update(q1=True)

    rows = await LongMemEval().judge([_question("q1")], [NOTHING])

    assert rows[0].score == 0.0


async def test_an_abstention_may_rightly_retrieve_nothing(verdicts):
    verdicts.update(q1_abs=True)

    rows = await LongMemEval().judge([_question("q1_abs", abstention=True)], [NOTHING])

    assert rows[0].score == 1.0
    assert rows[0].evidence_recall is None


async def test_the_rubrics_decide_the_gold_slice_which_no_official_template_covers(verdicts, monkeypatch):
    question = gold_slice_goldens()[0]
    judged = _Judged(scores={"Coverage": 0.9, "Contextual Recall": 0.2}, reasons={"Coverage": "ok"})
    adapter = LongMemEval(rubric_judge=object())
    monkeypatch.setattr(adapter, "_rubric_scores", lambda *_: {question.name: judged})

    rows = await adapter.judge([question], [FOUND])

    assert rows[0].score == 1.0
    assert rows[0].metrics["judged_by"] == "rubric"
    assert rows[0].metrics["contextual_recall"] == 0.2
    assert rows[0].detail["judge_reasons"] == {"Coverage": "ok"}


async def test_without_any_judge_every_question_is_unjudged(verdicts):
    rows = await LongMemEval(official_judge_model=None).judge([_question("q1")], [FOUND])

    assert rows[0].score is None


def test_each_question_is_its_own_users_history_and_the_gold_slice_the_planting_sessions():
    adapter = LongMemEval()

    assert adapter.user_id(_question("q1")) == "q1"
    assert adapter.user_id(gold_slice_goldens()[0]) == GOLD_SLICE_USER
    assert adapter.today(_question("q1")) == "2023/06/01 (Thu) 10:00"


async def test_accuracy_is_over_judged_questions_and_notes_report_abstention(verdicts):
    verdicts.update(q1=True, q2=False, q3_abs=True)
    questions = [_question("q1"), _question("q2"), _question("q3_abs", abstention=True), _question("q4")]

    rows = await LongMemEval().judge(questions, [FOUND, FOUND, NOTHING, FOUND])

    assert accuracy(rows) == pytest.approx(2 / 3)
    assert "abstention    1/1 correct" in LongMemEval().notes(rows)
