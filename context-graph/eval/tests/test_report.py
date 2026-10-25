"""Tests for rendering a run and comparing two runs of the same workload.

A comparison informs a promotion decision a person makes (#299), so its job is
to say what changed and -- critically -- whether the change is real rather than
judge noise (#304).
"""

from dataclasses import replace
from typing import Any

import pytest
from context_graph_eval.report import Verdict, compare, render, render_comparison
from context_graph_eval.results import Row, Run, RunMeta


def _meta(**overrides) -> RunMeta:
    # Annotated: a kwargs bag would otherwise infer dict[str, str | int].
    fields: dict[str, Any] = {
        "label": "baseline",
        "workload": "lme-s-dev",
        "benchmark": "longmemeval-s",
        "revision": "98d7416c24c7",
        "questions": 4,
        "judge_model": "gpt-4o-2024-08-06",
        "agent_model": "openai:gpt-4.1",
    }
    fields.update(overrides)
    return RunMeta(**fields)


def _row(name, score=1.0, *, tokens=1000, question_type="multi-session", **fields) -> Row:
    return Row(name=name, user_id=name, question_type=question_type, score=score, efficiency_tokens=tokens, **fields)


def _run(rows, **meta) -> Run:
    return Run(meta=_meta(questions=len(rows), **meta), rows=rows)


BASE = [_row("q1"), _row("q2"), _row("q3", 0.0), _row("q4", 0.0, question_type="temporal-reasoning")]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workload", "lme-s"),
        ("revision", "other"),
        ("judge_model", "gpt-4o-mini"),
        ("agent_model", "openai:gpt-5"),
        ("tokenizer", "o200k_base"),
    ],
)
def test_runs_measured_differently_are_refused(field, value):
    """A pin change would otherwise be reported as though it were the change under test."""
    with pytest.raises(ValueError, match=field):
        compare(_run(BASE), _run(BASE, **{field: value}))


def test_runs_that_asked_different_questions_are_refused():
    with pytest.raises(ValueError, match="different questions"):
        compare(_run(BASE), _run([*BASE[:3], _row("q5")]))


def test_a_gain_inside_the_noise_floor_is_not_called_real():
    candidate = _run([BASE[0], BASE[1], _row("q3"), BASE[3]], label="candidate")

    comparison = compare(_run(BASE), candidate, noise_floor_pp=30)

    assert comparison.delta_pp == pytest.approx(25.0)
    assert (comparison.is_real, comparison.verdict) == (False, Verdict.INCONCLUSIVE)


def test_a_gain_beyond_the_noise_floor_is_called_real():
    candidate = _run([BASE[0], BASE[1], _row("q3"), BASE[3]])

    assert compare(_run(BASE), candidate, noise_floor_pp=10).verdict == Verdict.IMPROVED


def test_a_real_loss_is_a_regression_and_names_the_questions_that_stopped_passing():
    candidate = _run([_row("q1", 0.0), BASE[1], BASE[2], BASE[3]])

    comparison = compare(_run(BASE), candidate, noise_floor_pp=10)

    assert comparison.verdict == Verdict.REGRESSED
    assert comparison.regressions == ["q1"]


def test_without_calibration_no_delta_is_claimed_real():
    comparison = compare(_run(BASE), _run([_row("q1"), _row("q2"), _row("q3"), _row("q4")]))

    assert comparison.is_real is None
    assert comparison.verdict == Verdict.INCONCLUSIVE
    assert "NOT CALIBRATED" in render_comparison(comparison)


def test_tokens_are_compared_only_over_questions_both_runs_passed():
    """Returning less is only cheaper if it still answers (#309): a question that
    started failing on an empty payload must not make the candidate look cheap."""
    candidate = _run([_row("q1", tokens=500), _row("q2", 0.0, tokens=0), BASE[2], BASE[3]])

    comparison = compare(_run(BASE), candidate)

    assert (comparison.baseline_tokens, comparison.candidate_tokens) == (1000, 500)


def test_beam_headlines_weight_abilities_not_questions():
    rows = [
        _row("a1", 1.0, question_type="abstention"),
        _row("a2", 1.0, question_type="abstention"),
        _row("s1", 0.0, question_type="summarization"),
    ]
    beam = {"workload": "beam-100k-dev", "benchmark": "beam-100k"}

    comparison = compare(_run(rows, **beam), _run(rows, **beam))

    assert comparison.baseline_headline == 0.5


def test_the_comparison_says_what_differs_about_the_system():
    base = _run(BASE, build={"inputs": {"ontology": "default", "construction_code": "a"}}, hybrid_lanes="turns,text")
    cand = _run(BASE, build={"inputs": {"ontology": "learned", "construction_code": "a"}}, hybrid_lanes="turns")

    text = render_comparison(compare(base, cand))

    assert "build ontology: default -> learned" in text
    assert "recall lanes: turns,text -> turns" in text
    assert "recall packing" not in text
    assert "construction_code" not in text


def test_a_change_of_recall_packing_is_named():
    base, cand = _run(BASE), _run(BASE, packing="exchanges")

    assert "recall packing: turns -> exchanges" in render_comparison(compare(base, cand))


def test_a_run_renders_its_headline_per_type_breakdown_and_unjudged_questions():
    run = _run([*BASE, _row("q5", None)])

    text = render(run, notes=["abstention    1/1 correct"])

    assert "HEADLINE" in text and "50.0%" in text
    assert "temporal-reasoning" in text
    assert "UNJUDGED      1 question(s)" in text
    assert "abstention    1/1 correct" in text


def test_a_run_asked_of_a_stale_graph_says_so():
    run = replace(_run(BASE), meta=_meta(build={"fingerprint": "abc", "current": False, "commit": "1234567"}))

    assert "STALE" in render(run)
