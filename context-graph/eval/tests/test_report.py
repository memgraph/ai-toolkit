"""Tests for the human-gated comparison report.

The report exists to inform a promotion decision a person makes (#299), so its
job is to say what changed and -- critically -- whether the change is real
rather than judge noise. #304 provides the noise floor via repeat-and-compare
calibration.
"""

from typing import Any

import pytest
from context_graph_eval.report import (
    RunMeta,
    SavedRun,
    Verdict,
    compare,
    load_run,
    render,
    save_run,
)
from context_graph_eval.scoring import Scored


def _meta(**overrides) -> RunMeta:
    # Annotated because this is a kwargs bag: without it the literal infers
    # dict[str, str | int] and every str-typed parameter rejects the int.
    fields: dict[str, Any] = {
        "label": "baseline",
        "corpus_revision": "98d7416c24c7",
        "corpus_variant": "s",
        "judge_model": "claude-sonnet-4-5",
        "tokenizer": "cl100k_base",
        "questions": 20,
    }
    fields.update(overrides)
    return RunMeta(**fields)


def _run(meta: RunMeta, scored: list[Scored]) -> SavedRun:
    return SavedRun(meta=meta, scored=scored)


def _scored(name, *, tier=1, covered=True, tokens=1000, abstention=False):
    return Scored(
        name=name,
        tier=tier,
        coverage=1.0 if covered else 0.0,
        covered=covered,
        efficiency_tokens=tokens,
        abstention=abstention,
    )


def test_a_run_round_trips_through_disk(tmp_path):
    """Comparison needs a previous run to compare against, so a run has to
    outlive the process that produced it."""
    path = tmp_path / "baseline.json"
    save_run(_run(_meta(), [_scored("q1"), _scored("q2", covered=False)]), path)

    restored = load_run(path)

    assert restored.meta.corpus_revision == "98d7416c24c7"
    assert [s.name for s in restored.scored] == ["q1", "q2"]


def test_runs_pinned_to_different_corpora_are_refused():
    """#302 pinned the corpus revision precisely so two runs are comparable.
    Comparing across pins measures the corpus change as if it were a schema
    change -- a wrong answer presented confidently."""
    baseline = _run(_meta(corpus_revision="aaaaaaa"), [_scored("q1")])
    candidate = _run(_meta(corpus_revision="bbbbbbb"), [_scored("q1")])

    with pytest.raises(ValueError, match="corpus"):
        compare(baseline, candidate)


def test_runs_judged_by_different_models_are_refused():
    """#304 pinned the judge for the same reason, and treats a judge bump as
    invalidating prior baselines."""
    baseline = _run(_meta(judge_model="claude-sonnet-4-5"), [_scored("q1")])
    candidate = _run(_meta(judge_model="claude-opus-4-1"), [_scored("q1")])

    with pytest.raises(ValueError, match="judge"):
        compare(baseline, candidate)


def test_runs_using_different_agent_models_are_refused():
    """Answers depend on the agent model too (#329) -- previously untracked,
    so this comparison used to run silently across two agent models with
    nothing to say so."""
    baseline = _run(_meta(agent_model="openai:gpt-4o"), [_scored("q1")])
    candidate = _run(_meta(agent_model="openai:gpt-4o-mini"), [_scored("q1")])

    with pytest.raises(ValueError, match="agent model"):
        compare(baseline, candidate)


def test_runs_using_different_extraction_backends_are_refused():
    """A LightRAG-built and a GLiNER2-built graph are different systems under
    test -- comparing across them would measure the backend swap as though it
    were the retrieval/schema change under test."""
    baseline = _run(_meta(extraction_backend="lightrag"), [_scored("q1")])
    candidate = _run(_meta(extraction_backend="gliner2"), [_scored("q1")])

    with pytest.raises(ValueError, match="extraction backend"):
        compare(baseline, candidate)


def test_a_text_search_run_is_comparable_to_a_graph_agent_run():
    """retrieval_strategy is the one axis compare() must NOT pin: a
    text-search baseline vs the graph-agent memory pipeline is exactly the
    comparison this exists to enable. extraction_backend="none" (what a
    text-search run actually records, since it never reconciles) must not
    trip the extraction-backend pin either -- that is not a backend
    disagreement, it is one side having none."""
    baseline = _run(
        _meta(retrieval_strategy="text-search", extraction_backend="none"),
        [_scored("q1")],
    )
    candidate = _run(
        _meta(label="cand", retrieval_strategy="graph-agent", extraction_backend="lightrag"),
        [_scored("q1")],
    )

    comparison = compare(baseline, candidate)

    assert comparison.baseline.retrieval_strategy == "text-search"
    assert comparison.candidate.retrieval_strategy == "graph-agent"


def test_extraction_backend_still_matters_between_two_graph_agent_runs():
    """The "none" exemption must not become a blanket bypass: two real,
    reconciled runs built with different backends are still a mismatch, and
    still refused."""
    baseline = _run(_meta(extraction_backend="lightrag"), [_scored("q1")])
    candidate = _run(_meta(extraction_backend="gliner2"), [_scored("q1")])

    with pytest.raises(ValueError, match="extraction backend"):
        compare(baseline, candidate)


def test_runs_over_different_numbers_of_questions_are_refused():
    """Coverage is reported as a rate, so a 20-question baseline and a
    60-question candidate yield comparable-looking percentages over different
    corpora -- the sampling change measured as though it were the change under
    test."""
    baseline = _run(_meta(questions=20), [_scored("q1")])
    candidate = _run(_meta(questions=60), [_scored("q1")])

    with pytest.raises(ValueError, match="corpus size"):
        compare(baseline, candidate)


def test_runs_measured_with_different_tokenizers_are_refused():
    """A tokenizer change silently shifts every efficiency number."""
    baseline = _run(_meta(tokenizer="cl100k_base"), [_scored("q1")])
    candidate = _run(_meta(tokenizer="o200k_base"), [_scored("q1")])

    with pytest.raises(ValueError, match="tokenizer"):
        compare(baseline, candidate)


def test_a_coverage_gain_inside_the_noise_floor_is_not_called_real():
    baseline = _run(_meta(), [_scored(f"q{i}", covered=i < 12) for i in range(20)])
    candidate = _run(_meta(label="cand"), [_scored(f"q{i}", covered=i < 13) for i in range(20)])

    result = compare(baseline, candidate, noise_floor_pp=10.0)

    assert result.tiers[1].coverage_is_real is False


def test_a_coverage_gain_beyond_the_noise_floor_is_called_real():
    baseline = _run(_meta(), [_scored(f"q{i}", covered=i < 5) for i in range(20)])
    candidate = _run(_meta(label="cand"), [_scored(f"q{i}", covered=i < 18) for i in range(20)])

    result = compare(baseline, candidate, noise_floor_pp=4.0)

    assert result.tiers[1].coverage_is_real is True


def test_without_calibration_no_delta_is_claimed_real():
    """#304's noise floor comes from a calibration run. Absent one, the honest
    answer is that we cannot tell -- not that the delta counts."""
    baseline = _run(_meta(), [_scored(f"q{i}", covered=i < 5) for i in range(20)])
    candidate = _run(_meta(label="cand"), [_scored(f"q{i}", covered=i < 18) for i in range(20)])

    result = compare(baseline, candidate, noise_floor_pp=None)

    assert result.tiers[1].coverage_is_real is None
    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_real_regression_outranks_a_real_improvement():
    """Coverage regressing is the thing the eval exists to catch, so it decides
    the verdict even when efficiency improved."""
    baseline = _run(_meta(), [_scored(f"q{i}", covered=i < 18, tokens=2000) for i in range(20)])
    candidate = _run(_meta(label="cand"), [_scored(f"q{i}", covered=i < 5, tokens=100) for i in range(20)])

    result = compare(baseline, candidate, noise_floor_pp=4.0)

    assert result.verdict is Verdict.REGRESSED


def test_efficiency_alone_cannot_declare_an_improvement():
    """Efficiency ranks within the coverage gate (#309); a payload that shrank
    while coverage held flat inside the noise floor is not a demonstrated win."""
    baseline = _run(_meta(), [_scored(f"q{i}", covered=i < 12, tokens=2000) for i in range(20)])
    candidate = _run(_meta(label="cand"), [_scored(f"q{i}", covered=i < 12, tokens=200) for i in range(20)])

    result = compare(baseline, candidate, noise_floor_pp=4.0)

    assert result.verdict is Verdict.INCONCLUSIVE


def test_questions_that_stopped_being_covered_are_named():
    """A rate alone is not actionable -- the human needs to know which questions
    to go and look at."""
    baseline = _run(_meta(), [_scored("kept"), _scored("lost")])
    candidate = _run(_meta(label="cand"), [_scored("kept"), _scored("lost", covered=False)])

    result = compare(baseline, candidate, noise_floor_pp=4.0)

    assert result.tiers[1].regressions == ["lost"]


def test_tiers_are_compared_separately():
    baseline = _run(_meta(), [_scored("t1", tier=1), _scored("t2", tier=2, covered=False)])
    candidate = _run(_meta(label="cand"), [_scored("t1", tier=1, covered=False), _scored("t2", tier=2)])

    result = compare(baseline, candidate, noise_floor_pp=4.0)

    assert set(result.tiers) == {1, 2}


def test_the_rendered_report_states_what_it_was_measured_against():
    """Provenance is what makes a number mean anything later."""
    baseline = _run(_meta(), [_scored("q1")])
    candidate = _run(_meta(label="cand"), [_scored("q1")])

    text = render(compare(baseline, candidate, noise_floor_pp=4.0))

    assert "98d7416c24c7" in text
    assert "claude-sonnet-4-5" in text


def test_the_rendered_report_says_when_the_noise_floor_is_unknown():
    baseline = _run(_meta(), [_scored("q1")])
    candidate = _run(_meta(label="cand"), [_scored("q1", covered=False)])

    text = render(compare(baseline, candidate, noise_floor_pp=None))

    assert "not calibrated" in text.lower()


def test_the_rendered_report_names_the_agent_model_too():
    """Provenance covers the agent as well as the judge (#329) -- a reader
    deciding whether a delta is real needs to know what produced the answers,
    not only what scored them."""
    baseline = _run(_meta(agent_model="openai:gpt-4o"), [_scored("q1")])
    candidate = _run(_meta(label="cand", agent_model="openai:gpt-4o"), [_scored("q1")])

    text = render(compare(baseline, candidate, noise_floor_pp=4.0))

    assert "openai:gpt-4o" in text


def test_the_rendered_report_names_both_retrieval_strategies_when_they_differ():
    """Not pinned by compare() (deliberately -- it is usually the thing being
    measured), so a reader must not mistake "not pinned" for "the same": a
    text-search-vs-graph-agent comparison must be unmissable in the report."""
    baseline = _run(_meta(retrieval_strategy="text-search", extraction_backend="none"), [_scored("q1")])
    candidate = _run(_meta(label="cand", retrieval_strategy="graph-agent"), [_scored("q1")])

    text = render(compare(baseline, candidate, noise_floor_pp=4.0))

    assert "text-search" in text
    assert "graph-agent" in text


def test_the_rendered_report_omits_the_retrieval_line_when_strategies_match():
    baseline = _run(_meta(), [_scored("q1")])
    candidate = _run(_meta(label="cand"), [_scored("q1")])

    text = render(compare(baseline, candidate, noise_floor_pp=4.0))

    assert "retrieval " not in text


def test_a_same_provider_run_is_flagged_in_the_report():
    """#304's cross-provider independence is a property of the run's own
    configuration, not something compare() can see from a coverage delta --
    so it must be recorded and surfaced explicitly, not left silent (#329)."""
    baseline = _run(_meta(same_provider=True), [_scored("q1")])
    candidate = _run(_meta(label="cand", same_provider=True), [_scored("q1")])

    text = render(compare(baseline, candidate, noise_floor_pp=4.0))

    assert "share a provider" in text.lower()


def test_a_cross_provider_run_is_not_flagged():
    baseline = _run(_meta(same_provider=False), [_scored("q1")])
    candidate = _run(_meta(label="cand", same_provider=False), [_scored("q1")])

    text = render(compare(baseline, candidate, noise_floor_pp=4.0))

    assert "share a provider" not in text.lower()


# --- Efficiency has to be measured on the same questions (#309) ---


def test_efficiency_is_compared_only_on_questions_both_runs_passed():
    """Efficiency's median is taken over the questions that cleared coverage,
    and that set changes between runs. Measured across three identical
    calibration runs, the passing sets were {0a995998, 6a1eabeb, gpt4_f49edff3},
    {8a2466db, gpt4_59149c77} and {0a995998, 7161e7e2, gpt4_f49edff3} -- the
    first two entirely disjoint -- and the median swung 1,113 -> 10,256 -> 164
    tokens.

    That looked like a 62x change in retrieval cost. It was a median over two
    or three different questions each time. Comparing those numbers compares
    nothing.
    """
    baseline = _run(_meta(), [_scored("q1", covered=True, tokens=100), _scored("q2", covered=False, tokens=50)])
    candidate = _run(
        _meta(label="candidate"), [_scored("q1", covered=False, tokens=100), _scored("q2", covered=True, tokens=9000)]
    )

    result = compare(baseline, candidate)
    tier = result.tiers[1]

    # Disjoint passing sets: there is no question both runs measured, so there
    # is no efficiency comparison to make.
    assert tier.baseline_efficiency is None
    assert tier.candidate_efficiency is None


def test_efficiency_uses_the_shared_passing_set_when_there_is_one():
    baseline = _run(_meta(), [_scored("q1", covered=True, tokens=100), _scored("q2", covered=True, tokens=8000)])
    candidate = _run(
        _meta(label="candidate"), [_scored("q1", covered=True, tokens=250), _scored("q2", covered=False, tokens=10)]
    )

    tier = compare(baseline, candidate).tiers[1]

    # Only q1 cleared coverage in both, so only q1's payload is comparable.
    assert tier.baseline_efficiency == 100
    assert tier.candidate_efficiency == 250


def test_a_report_says_when_efficiency_could_not_be_compared(capsys):
    baseline = _run(_meta(questions=1), [_scored("q1", covered=True, tokens=100)])
    candidate = _run(_meta(label="candidate", questions=1), [_scored("q1", covered=False, tokens=100)])

    print(render(compare(baseline, candidate)))

    assert "no question cleared coverage in both runs" in capsys.readouterr().out
