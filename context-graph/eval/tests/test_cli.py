"""Tests for what a run prints.

The report is the whole product of an eval run -- a number nobody reads is a
number nobody acts on -- so how it renders is behaviour, not formatting. These
test the printer at its seam by capturing stdout, rather than reaching into the
branch structure.
"""

from context_graph_eval.cli import _print_report
from context_graph_eval.runner import BatchReport
from context_graph_eval.scoring import Scored, aggregate


def _report(scored: list[Scored]) -> BatchReport:
    return BatchReport(by_tier=aggregate(scored).by_tier, scored=scored)


def _scored(name, *, covered=True, tokens=100, judged=True):
    return Scored(
        name=name,
        tier=1,
        coverage=1.0 if covered else 0.0,
        covered=covered,
        efficiency_tokens=tokens,
        metric_scores={"Coverage": 1.0 if covered else 0.0} if judged else {},
    )


def test_a_judged_run_reports_both_coverage_and_efficiency(capsys):
    """The two halves of the rubric (#309): coverage gates, efficiency ranks
    within it. Printing one without the other loses the number that decides
    whether a retrieval change was worth its payload."""
    _print_report(_report([_scored("q1", covered=True, tokens=120)]), judged=True)

    out = capsys.readouterr().out
    assert "coverage      1/1 (100%)" in out
    assert "efficiency    median 120 tokens" in out


def test_a_judge_outage_is_reported_as_unscored_not_as_zero(capsys):
    """Observed live: the judge's provider ran out of credit, every metric
    errored, and the run printed "coverage 0/2 (0%)" -- an outage rendered as a
    measurement, which reads as a real regression."""
    _print_report(_report([_scored("q1", judged=False), _scored("q2", judged=False)]), judged=True)

    out = capsys.readouterr().out
    assert "UNSCORED      2 question(s)" in out
    assert "(0%)" not in out


def test_a_run_without_a_judge_does_not_cry_outage(capsys):
    """No judge configured is an ordinary efficiency-only run, not a failure.
    Sharing the unscored warning with the outage case would fire it every
    time."""
    _print_report(_report([_scored("q1", judged=False)]), judged=False)

    out = capsys.readouterr().out
    assert "UNSCORED" not in out
    assert "not judged" in out
