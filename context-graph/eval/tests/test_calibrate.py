"""Tests for deriving a noise floor from repeated runs.

#304 chose repeat-and-compare over trusting a single run, because temperature=0
is not actually deterministic on hosted APIs. Until a floor exists, the
comparison report refuses to call any delta real -- so this is what turns
`--noise-floor` from a hand-typed number into a measured one.
"""

import pytest
from context_graph_eval.calibrate import noise_floor_pp


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
