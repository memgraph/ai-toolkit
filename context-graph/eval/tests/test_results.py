"""Tests for the run format every benchmark shares: summaries, files, publishing and merging shards."""

from typing import Any

import pytest
from context_graph_eval.results import Row, Run, RunMeta, load, merge, public, save, summarize


def _meta(**overrides) -> RunMeta:
    # Annotated: a kwargs bag would otherwise infer dict[str, str | int].
    fields: dict[str, Any] = {
        "label": "run",
        "workload": "lme-s",
        "benchmark": "longmemeval-s",
        "revision": "r",
        "questions": 2,
        "judge_model": "j",
        "agent_model": "a",
    }
    fields.update(overrides)
    return RunMeta(**fields)


def _row(name, score=1.0, **fields) -> Row:
    fields.setdefault("question_type", "multi-session")
    return Row(name=name, user_id=name, score=score, **fields)


def test_an_unjudged_question_is_left_out_of_every_score_not_counted_as_zero():
    """A judge outage is not a measurement."""
    summary = summarize([_row("q1", 1.0), _row("q2", None)], headline=1.0)

    assert (summary.questions, summary.judged) == (2, 1)
    assert summary.by_type[0].mean_score == 1.0


def test_median_tokens_are_over_questions_that_passed():
    summary = summarize(
        [_row("q1", 1.0, efficiency_tokens=100), _row("q2", 0.0, efficiency_tokens=0)],
        headline=0.5,
    )

    assert summary.median_tokens_passed == 100
    assert summary.mean_tokens == 50


def test_types_are_listed_in_the_benchmarks_own_order_first():
    rows = [_row("q1", question_type="b"), _row("q2", question_type="z"), _row("q3", question_type="a")]

    summary = summarize(rows, headline=None, type_order=("z", "a"))

    assert [t.question_type for t in summary.by_type] == ["z", "a", "b"]


def test_recall_latency_skips_rows_whose_retrieval_raised():
    rows = [
        _row("q1", latency_seconds=4.0, recall_seconds=1.0),
        _row("q2", latency_seconds=2.0, recall_seconds=3.0),
        _row("q3", latency_seconds=0.1, recall_seconds=None),
    ]

    times = summarize(rows, headline=None).latency

    assert times["mean_e2e_seconds"] == pytest.approx(6.1 / 3)
    assert times["mean_recall_seconds"] == 2.0
    assert times["p95_recall_seconds"] == 3.0


def test_an_empty_run_summarises_to_nothing_rather_than_dividing_by_zero():
    summary = summarize([], headline=None)

    assert (summary.questions, summary.mean_tokens, summary.mean_evidence_recall) == (0, None, None)


def test_a_run_round_trips_through_disk(tmp_path):
    run = Run(meta=_meta(build={"fingerprint": "f"}), rows=[_row("q1", metrics={"bleu": 0.5}), _row("q2", None)])

    restored = load(save(run, tmp_path / "run.json"))

    assert restored == run


def test_a_published_run_holds_no_benchmark_text(tmp_path):
    """BEAM is CC BY-SA and LongMemEval's haystack is never committed (#302)."""
    run = Run(meta=_meta(), rows=[_row("q1", detail={"question": "Where?", "answer": "Paris"}, metrics={"f1": 1.0})])

    published = load(save(public(run), tmp_path / "baseline.json"))

    assert published.rows[0].detail == {}
    assert published.rows[0].metrics == {"f1": 1.0}
    assert "Paris" not in (tmp_path / "baseline.json").read_text()


def test_shards_merge_into_one_run_with_their_build_time_summed():
    first = Run(meta=_meta(build={"shard": 0, "build_seconds": 10.0}, commit="abc"), rows=[_row("q1")])
    second = Run(meta=_meta(build={"shard": 1, "build_seconds": 5.0}, commit="abc"), rows=[_row("q2")])

    merged = merge([first, second], label="lme-s-full")

    assert [row.name for row in merged.rows] == ["q1", "q2"]
    assert (merged.meta.label, merged.meta.questions, merged.meta.commit) == ("lme-s-full", 2, "abc")
    assert merged.meta.build == {"shards": 2, "build_seconds": 15.0}


def test_shards_measured_differently_do_not_merge():
    with pytest.raises(ValueError, match="judge_model"):
        merge(
            [Run(meta=_meta(), rows=[_row("q1")]), Run(meta=_meta(judge_model="other"), rows=[_row("q2")])],
            label="x",
        )


def test_a_question_in_two_shards_does_not_merge():
    with pytest.raises(ValueError, match="more than one shard"):
        merge([Run(meta=_meta(), rows=[_row("q1")]), Run(meta=_meta(), rows=[_row("q1")])], label="x")
