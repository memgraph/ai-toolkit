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


def noise_floor_pp(headlines: list[float]) -> float:
    """Percentage-point spread across repeated runs of the same questions.

    The full range rather than a standard deviation: the floor's job is to stop
    a human believing a delta that the judge would have produced anyway, so it
    should cover the movement actually observed, not a summary of it.
    """
    if len(headlines) < MIN_RUNS:
        raise ValueError(
            f"a noise floor needs at least {MIN_RUNS} runs of the same questions; "
            f"got {len(headlines)}. One run has no spread, and reporting 0.0 "
            "would assert that any delta is real."
        )
    return (max(headlines) - min(headlines)) * 100


def describe(headlines: list[float]) -> str:
    """One-line summary of a calibration run."""
    floor = noise_floor_pp(headlines)
    rates = ", ".join(f"{rate:.1%}" for rate in headlines)
    return (
        f"{len(headlines)} runs: {rates} (mean {mean(headlines):.1%})\n"
        f"noise floor: +/-{floor:.1f}pp -- pass this to `compare --noise-floor {floor:.1f}`"
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


def per_question_pass_rate(passing_sets: list[set[str]], all_names: set[str]) -> dict[str, float]:
    """Each question in ``all_names``'s pass rate across repeats; one that never passed gets 0.0.

    Names the flaky questions, which ``describe_stability``'s aggregate cannot.
    """
    if len(passing_sets) < MIN_RUNS:
        raise ValueError(f"per-question pass rates need at least {MIN_RUNS} runs; got {len(passing_sets)}")
    n = len(passing_sets)
    return {name: sum(1 for s in passing_sets if name in s) / n for name in all_names}


def describe_per_question_rates(rates: dict[str, float]) -> str:
    """Render per-question pass rates, listing the flaky ones (neither always nor never pass) by name."""
    if not rates:
        return "no questions scored."

    always = sorted(name for name, rate in rates.items() if rate == 1.0)
    never = sorted(name for name, rate in rates.items() if rate == 0.0)
    flaky = sorted(((rate, name) for name, rate in rates.items() if 0 < rate < 1), reverse=True)

    lines = [f"per-question pass rate ({len(rates)} questions, {len(flaky)} flaky):"]
    if flaky:
        lines.append("  flaky (neither always nor never passes):")
        lines += [f"    {name}: {rate:.0%}" for rate, name in flaky]
    lines.append(f"  always passes: {len(always)}  |  never passes: {len(never)}")
    return "\n".join(lines)
