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


def describe_stability(passing_sets: list[set[str]]) -> str:
    """How stable the *set* of passing questions is across repeated runs.

    The coverage floor alone can flatter a run badly. Measured across three
    identical repeats: rates of 15%, 10% and 15% gave a +/-5pp floor, while
    **zero** questions passed in all three and the first two runs' passing sets
    were entirely disjoint. The aggregate looked steady because the same *count*
    kept passing, not the same questions.

    That distinction decides what a delta can mean. If no question passes
    reliably, a coverage change between two runs is resampling, and an
    efficiency median -- taken over whichever questions happened to pass -- is
    not measuring a fixed quantity at all.
    """
    if len(passing_sets) < MIN_RUNS:
        raise ValueError(f"stability needs at least {MIN_RUNS} runs; got {len(passing_sets)}")

    always = set.intersection(*passing_sets)
    ever = set.union(*passing_sets)
    if not ever:
        return "no question passed in any run -- nothing to compare."

    line = f"stable passes: {len(always)} of {len(ever)} questions that passed at least once"
    if not always:
        return (
            f"{line}\n"
            "  UNSTABLE: no question passes in every run, so the passing set is resampled each "
            "time. Coverage deltas below the floor are noise, and the efficiency median is taken "
            "over a different sample per run -- do not compare it."
        )
    if len(always) < len(ever) / 2:
        return f"{line}\n  unstable: most passes never repeat; treat efficiency comparisons with care."
    return line
