"""Tests for deriving a noise floor from repeated runs.

#304 chose repeat-and-compare over trusting a single run, because temperature=0
is not actually deterministic on hosted APIs. Until a floor exists, the
comparison report refuses to call any delta real -- so this is what turns
`--noise-floor` from a hand-typed number into a measured one.
"""

import pytest
from context_graph_eval.calibrate import describe_stability, noise_floor_pp


def test_identical_runs_have_no_noise():
    assert noise_floor_pp([0.6, 0.6, 0.6]) == 0.0


def test_the_floor_spans_the_observed_spread():
    """The floor is what a delta must exceed to be believed, so it has to cover
    the movement seen when nothing changed at all."""
    assert noise_floor_pp([0.50, 0.60, 0.55]) == pytest.approx(10.0)


def test_a_single_run_cannot_establish_a_floor():
    """One measurement has no spread, and reporting 0.0 would say 'any delta is
    real' -- the opposite of the truth. #304 calls for repeats precisely
    because a single run cannot speak to its own variance."""
    with pytest.raises(ValueError, match="at least"):
        noise_floor_pp([0.6])


def test_an_empty_calibration_is_refused():
    with pytest.raises(ValueError, match="at least"):
        noise_floor_pp([])


# --- Efficiency needs its own stability report (#309) ---


def test_calibration_reports_which_questions_pass_consistently():
    """The coverage floor can look tight while the underlying set churns
    completely. Measured across three identical runs: rates 15%, 10%, 15% --
    a +/-5pp floor -- yet ZERO questions passed in all three, and the first two
    runs' passing sets were entirely disjoint.

    A floor on the aggregate is technically correct and materially misleading
    if nobody is told the set beneath it is unstable."""
    summary = describe_stability(
        [
            {"a", "b", "c"},
            {"d", "e"},
            {"a", "f", "c"},
        ]
    )

    assert "0 of 6" in summary
    assert "never passes twice" in summary or "unstable" in summary.lower()


def test_a_stable_pass_set_is_reported_as_such():
    summary = describe_stability([{"a", "b"}, {"a", "b"}, {"a", "b"}])

    assert "2 of 2" in summary
