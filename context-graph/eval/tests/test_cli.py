"""Tests for what a run prints.

The report is the whole product of an eval run -- a number nobody reads is a
number nobody acts on -- so how it renders is behaviour, not formatting. These
test the printer at its seam by capturing stdout, rather than reaching into the
branch structure.
"""

from context_graph_eval.cli import (
    DEFAULT_JUDGE_MODEL,
    _build_model,
    _parse_model_spec,
    _print_report,
    _resolved_spec,
    select_goldens,
)
from context_graph_eval.runner import BatchReport
from context_graph_eval.scoring import Scored, aggregate


def _report(scored: list[Scored], *, indexed_turns: int = 0, reconciled: int = 0) -> BatchReport:
    return BatchReport(
        by_tier=aggregate(scored).by_tier, scored=scored, indexed_turns=indexed_turns, reconciled=reconciled
    )


def _scored(name, *, covered=True, tokens=100, judged=True, metric_reasons=None):
    return Scored(
        name=name,
        tier=1,
        coverage=1.0 if covered else 0.0,
        covered=covered,
        efficiency_tokens=tokens,
        metric_scores={"Coverage": 1.0 if covered else 0.0} if judged else {},
        metric_reasons=metric_reasons if metric_reasons is not None else {},
    )


def test_a_limited_run_asks_only_that_many_questions():
    """--limit was ignored for a while after `run` switched to reading the
    committed corpus, so a run asked for 2 questions quietly did all 20. Each
    question costs sessions of reconciliation at ~2 LLM calls apiece, so the
    overrun was expensive and invisible."""
    assert select_goldens(list("abcdefgh"), limit=2, gold_slice=False) == ["a", "b"]


def test_an_unlimited_run_asks_the_whole_corpus():
    assert select_goldens(list("abc"), limit=None, gold_slice=False) == ["a", "b", "c"]


def test_the_gold_slice_is_not_trimmed_by_a_tier_1_limit():
    """The gold slice is Tier 2 (#303), scored apart, and the only question that
    exercises the capture layer -- so a Tier 1 limit must not drop it."""
    selected = select_goldens(list("abcdefgh"), limit=1, gold_slice=True)

    assert selected[:1] == ["a"]
    assert len(selected) > 1


def test_a_judged_run_reports_both_coverage_and_efficiency(capsys):
    """The two halves of the rubric (#309): coverage gates, efficiency ranks
    within it. Printing one without the other loses the number that decides
    whether a retrieval change was worth its payload."""
    _print_report(_report([_scored("q1", covered=True, tokens=120)]), judged=True)

    out = capsys.readouterr().out
    assert "coverage      1/1 (100%)" in out
    assert "efficiency    median 120 tokens" in out


def test_a_text_search_run_reports_the_index_not_reconciliation(capsys):
    """The text-search baseline never reconciles anything -- printing
    "reconciled 0 sessions" would read as an outage rather than as the
    strategy's whole point."""
    _print_report(_report([_scored("q1", covered=True)], indexed_turns=7, reconciled=0), judged=True)

    out = capsys.readouterr().out
    assert "text-search index: 7 turns indexed" in out
    assert "reconciled" not in out


def test_a_judge_outage_is_reported_as_unscored_not_as_zero(capsys):
    """Observed live: the judge's provider ran out of credit, every metric
    errored, and the run printed "coverage 0/2 (0%)" -- an outage rendered as a
    measurement, which reads as a real regression."""
    _print_report(_report([_scored("q1", judged=False), _scored("q2", judged=False)]), judged=True)

    out = capsys.readouterr().out
    assert "UNSCORED      2 question(s)" in out
    assert "(0%)" not in out


def test_a_failure_shows_the_judges_own_reason_not_just_a_count(capsys):
    """A count says a metric failed N times; the reason says why -- retrieval
    missed the fact, or the answer dropped it -- without rerunning the
    question by hand to find out."""
    _print_report(
        _report(
            [
                _scored(
                    "q1",
                    covered=False,
                    metric_reasons={"Coverage": "the answer named the dog but omitted the breed"},
                )
            ]
        ),
        judged=True,
    )

    out = capsys.readouterr().out
    assert "failed on     Coverage: 1" in out
    assert "e.g. q1: the answer named the dog but omitted the breed" in out


def test_a_failure_with_no_reason_recorded_prints_no_example_line(capsys):
    """A Scored built before this field existed, or one whose judge simply
    didn't return a reason, must not print a blank or crash -- absence is
    silent, not an empty "e.g." line."""
    _print_report(_report([_scored("q1", covered=False)]), judged=True)

    out = capsys.readouterr().out
    assert "failed on     Coverage: 1" in out
    assert "e.g." not in out


def test_a_run_without_a_judge_does_not_cry_outage(capsys):
    """No judge configured is an ordinary efficiency-only run, not a failure.
    Sharing the unscored warning with the outage case would fire it every
    time."""
    _print_report(_report([_scored("q1", judged=False)]), judged=False)

    out = capsys.readouterr().out
    assert "UNSCORED" not in out
    assert "not judged" in out


# --- Provider-qualified model specs (#329) ---


def test_a_provider_qualified_spec_overrides_the_default_provider():
    assert _parse_model_spec("openai:gpt-4o", default_provider="anthropic") == ("openai", "gpt-4o")


def test_a_bare_model_id_keeps_the_default_provider():
    """Pre-#329 --judge-model/--agent-model values had no provider prefix --
    they must keep meaning what they meant."""
    assert _parse_model_spec("claude-opus-4-1", default_provider="anthropic") == ("anthropic", "claude-opus-4-1")


def test_omitting_the_flag_keeps_the_default_provider_with_no_model_id():
    assert _parse_model_spec(None, default_provider="openai") == ("openai", None)


def test_resolved_spec_reports_the_actual_fallback_model():
    """RunMeta must reflect what ran, not what was typed -- omitting a model
    id still resolves to a concrete model per-provider."""
    assert _resolved_spec("anthropic", None) == f"anthropic:{DEFAULT_JUDGE_MODEL}"
    assert _resolved_spec("openai", None) == "openai:default"
    assert _resolved_spec("openai", "gpt-4o") == "openai:gpt-4o"


def test_an_unknown_provider_is_refused_not_guessed(capsys):
    assert _build_model("azure", "gpt-4o") is None
    assert "unknown model provider" in capsys.readouterr().err


def test_no_model_is_built_without_a_matching_api_key(monkeypatch):
    """A judge/agent silently disappearing when its key is absent is existing,
    relied-on behaviour (an unjudged run is a valid efficiency-only run) --
    this just must still hold now that the provider is resolved from a table
    instead of a hardcoded branch."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert _build_model("anthropic", None) is None
    assert _build_model("openai", None) is None
