"""Tests for pinning and sampling upstream LongMemEval data.

The network fetch itself is not unit-tested -- mocking HTTP would only test the
mock. What is tested is everything that decides *what* gets fetched and *which*
records survive into a committed corpus.
"""

import collections

import pytest
from context_graph_eval.convert.longmemeval import (
    DEFAULT_REVISION,
    build_corpus,
    download_url,
)


def test_sampling_roughly_preserves_upstream_proportions():
    """Mirrors the real dataset's shape: six question types of very unequal size,
    each with a small abstention run at its end (30 of 500 records, ~6%).

    Round-robin across strata over-samples the rare ones badly -- it produced 24
    abstention questions in a 60-question sample, 40% against an upstream 6%,
    which would let abstention behaviour dominate the aggregate score.
    """
    types = {
        "multi-session": 128,
        "temporal-reasoning": 128,
        "knowledge-update": 73,
        "single-session-user": 65,
        "single-session-assistant": 51,
        "single-session-preference": 25,
    }
    records = []
    for question_type, count in types.items():
        records += [_record(f"{question_type}-{i}", question_type) for i in range(count)]
        records += [_record(f"{question_type}-{i}_abs", question_type) for i in range(5)]

    goldens = build_corpus(records, limit=60)
    abstention = sum(1 for g in goldens if g.additional_metadata["abstention"])

    # Upstream is 30/500 = 6%, so ~4 of 60. The floor lifts that a little; 40%
    # is not "a little".
    assert abstention <= 15


def test_equally_sized_strata_get_equal_shares():
    """The per-stratum floor forces rounding drift that has to come out of the
    large strata. Taking it all from whichever stratum sorts first would leave
    two equally-sized categories with very different sample counts -- observed
    against the real dataset as multi-session 17 vs temporal-reasoning 11,
    despite both holding 133 records upstream."""
    records = []
    for question_type in ("alpha-type", "beta-type"):
        records += [_record(f"{question_type}-{i}", question_type) for i in range(128)]
        records += [_record(f"{question_type}-{i}_abs", question_type) for i in range(5)]

    goldens = build_corpus(records, limit=60)
    counts = collections.Counter(g.additional_metadata["question_type"] for g in goldens)

    assert abs(counts["alpha-type"] - counts["beta-type"]) <= 1


def test_a_rare_stratum_still_survives_a_small_sample():
    """The floor: strictly proportional sampling would round a rare category to
    zero at small limits, silently dropping it."""
    records = [_record(f"common{i}", "multi-session") for i in range(495)]
    records += [_record(f"rare{i}", "single-session-preference") for i in range(5)]

    goldens = build_corpus(records, limit=20)
    rare = [g for g in goldens if g.additional_metadata["question_type"] == "single-session-preference"]

    assert rare


def test_sampling_reaches_abstention_questions():
    """Abstention is orthogonal to question_type, and upstream clusters those
    records in contiguous runs at the end of each type block (verified: the 30
    abstention records in the real dataset sit at positions 64-69, 126-129, and
    so on). Stratifying on question_type alone therefore samples zero of them,
    silently dropping the category where a confident answer is the failure."""
    records = [_record(f"plain{i}") for i in range(20)]
    records += [_record(f"abs{i}_abs") for i in range(20)]

    goldens = build_corpus(records, limit=10)

    assert any(g.additional_metadata["abstention"] for g in goldens)
    assert any(not g.additional_metadata["abstention"] for g in goldens)


def _record(question_id: str, question_type: str = "single-session-user"):
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": f"question {question_id}?",
        "answer": f"answer {question_id}",
        "question_date": "2023/06/15 (Thu) 09:12",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/05/20 (Sat) 14:03"],
        "haystack_sessions": [[{"role": "user", "content": "a fact", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }


def test_download_url_pins_an_explicit_revision():
    url = download_url("s", revision="abc123")

    assert url == (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/abc123/longmemeval_s_cleaned.json"
    )


def test_download_url_defaults_to_the_pinned_revision_not_main():
    """A moving ref would silently change the corpus between runs, breaking the
    cross-version comparison the corpus exists to support."""
    url = download_url("s")

    assert DEFAULT_REVISION in url
    assert "/main/" not in url


def test_the_oracle_variant_is_refused():
    """oracle ships evidence sessions only. With no distractors, retrieval
    precision -- and the payload-size efficiency metric -- would score well by
    construction."""
    with pytest.raises(ValueError, match="oracle"):
        download_url("oracle")


def test_build_corpus_converts_every_record_when_unlimited():
    goldens = build_corpus([_record("a"), _record("b")])

    assert [g.name for g in goldens] == ["a", "b"]


def test_build_corpus_honours_a_limit():
    goldens = build_corpus([_record(str(i)) for i in range(10)], limit=3)

    assert len(goldens) == 3


def test_sampling_is_deterministic():
    """The corpus is committed, so the same upstream data must always yield the
    same sample -- otherwise a regenerated corpus shows a spurious diff and two
    runs are no longer comparable."""
    records = [_record(str(i)) for i in range(10)]

    first = build_corpus(records, limit=4)
    second = build_corpus(records, limit=4)

    assert [g.name for g in first] == [g.name for g in second]


def test_sampling_spreads_across_question_types():
    """Taking the first N would bias the sample if upstream groups records by
    type -- a corpus of nothing but temporal-reasoning questions would report a
    misleading score."""
    records = [_record(f"a{i}", "single-session-user") for i in range(10)]
    records += [_record(f"b{i}", "temporal-reasoning") for i in range(10)]

    goldens = build_corpus(records, limit=10)
    types = {g.additional_metadata["question_type"] for g in goldens}

    assert types == {"single-session-user", "temporal-reasoning"}
