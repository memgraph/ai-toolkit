"""Establish the noise floor a comparison needs before it can claim anything.

#304 decided the judge is checked by **repeat-and-compare**: run the same
questions against the same graph several times and see how far the score moves
when nothing has changed. That spread is the floor a real delta must clear.

Without it, ``report.compare`` reports "cannot tell" rather than guessing --
deliberately, since ``temperature=0`` is not actually deterministic on hosted
APIs and a single run cannot speak to its own variance.

This covers the *noise* half of #304's calibration. The *bias* half -- roughly
25 human-graded items, to catch a judge that is stable and consistently wrong --
is not something code can supply, and remains outstanding.
"""

from statistics import mean

#: Repeats needed before a spread means anything. Two is the arithmetic
#: minimum; #304 specified three.
MIN_RUNS = 2


def noise_floor_pp(coverage_rates: list[float]) -> float:
    """Percentage-point spread across repeated runs of the same questions.

    The full range rather than a standard deviation: the floor's job is to stop
    a human believing a delta that the judge would have produced anyway, so it
    should cover the movement actually observed, not a summary of it.
    """
    if len(coverage_rates) < MIN_RUNS:
        raise ValueError(
            f"a noise floor needs at least {MIN_RUNS} runs of the same questions; "
            f"got {len(coverage_rates)}. One run has no spread, and reporting 0.0 "
            "would assert that any delta is real."
        )
    return (max(coverage_rates) - min(coverage_rates)) * 100


def describe(coverage_rates: list[float]) -> str:
    """One-line summary of a calibration run."""
    floor = noise_floor_pp(coverage_rates)
    rates = ", ".join(f"{rate:.0%}" for rate in coverage_rates)
    return (
        f"{len(coverage_rates)} runs: {rates} (mean {mean(coverage_rates):.0%})\n"
        f"noise floor: +/-{floor:.0f}pp -- pass this to `compare --noise-floor {floor:.0f}`"
    )
