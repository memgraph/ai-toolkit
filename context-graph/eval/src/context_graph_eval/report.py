"""Compare two eval runs and render the report a human decides on.

#299 chose human-gated promotion: eval produces a comparison a person reads and
acts on, rather than a threshold that promotes automatically. So this report's
job is not to decide -- it is to make the decision *makeable*, which means
saying plainly what changed and whether the change is real.

"Whether it's real" is the part that would otherwise go wrong. Judged scores
vary run to run, so a human shown a bare 12/20 -> 13/20 will read a win into
noise. #304's repeat-and-compare calibration exists to give that variance a
number; this report refuses to call anything real without it.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from statistics import median

from .scoring import Scored


class Verdict(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class RunMeta:
    """What a run was measured against.

    Every field here is something that, if it differed between two runs, would
    make their numbers incomparable -- which is why :func:`compare` refuses
    rather than quietly producing a delta.
    """

    label: str
    corpus_revision: str
    corpus_variant: str
    judge_model: str
    tokenizer: str
    questions: int
    changed: str = ""


@dataclass(frozen=True)
class SavedRun:
    meta: RunMeta
    scored: list[Scored]


@dataclass(frozen=True)
class TierComparison:
    tier: int
    baseline_covered: int
    candidate_covered: int
    questions: int
    coverage_delta_pp: float
    #: True/False when calibrated, None when no noise floor is known -- the
    #: distinction between "not real" and "cannot tell" matters to a decision.
    coverage_is_real: bool | None
    baseline_efficiency: int | None
    candidate_efficiency: int | None
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Comparison:
    baseline: RunMeta
    candidate: RunMeta
    verdict: Verdict
    noise_floor_pp: float | None
    tiers: dict[int, TierComparison] = field(default_factory=dict)


def save_run(run: SavedRun, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": asdict(run.meta), "scored": [asdict(s) for s in run.scored]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_run(path: Path) -> SavedRun:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SavedRun(
        meta=RunMeta(**payload["meta"]),
        scored=[Scored(**row) for row in payload["scored"]],
    )


def compare(baseline: SavedRun, candidate: SavedRun, noise_floor_pp: float | None = None) -> Comparison:
    """Compare two runs, refusing when they were not measured the same way.

    The refusals are the point. #302 pinned the corpus revision and #304 pinned
    the judge model and tokenizer *so that* two runs could be compared; a
    comparison across different pins measures the pin change as though it were
    the schema change under test, and reports it confidently.
    """
    _require_same(baseline.meta, candidate.meta, "corpus_revision", "corpus")
    _require_same(baseline.meta, candidate.meta, "corpus_variant", "corpus")
    _require_same(baseline.meta, candidate.meta, "judge_model", "judge")
    _require_same(baseline.meta, candidate.meta, "tokenizer", "tokenizer")
    # Question count too: coverage is reported as a rate, so a 20-question
    # baseline and a 60-question candidate produce comparable-looking
    # percentages over different corpora. That is the same "measures the
    # sampling change as though it were the change under test" failure the
    # other three refusals exist to prevent.
    _require_same(baseline.meta, candidate.meta, "questions", "corpus size")

    tiers: dict[int, TierComparison] = {}
    for tier in sorted({s.tier for s in baseline.scored} | {s.tier for s in candidate.scored}):
        tiers[tier] = _compare_tier(tier, baseline.scored, candidate.scored, noise_floor_pp)

    return Comparison(
        baseline=baseline.meta,
        candidate=candidate.meta,
        verdict=_verdict(tiers),
        noise_floor_pp=noise_floor_pp,
        tiers=tiers,
    )


def _require_same(baseline: RunMeta, candidate: RunMeta, attr: str, noun: str) -> None:
    before, after = getattr(baseline, attr), getattr(candidate, attr)
    if before != after:
        raise ValueError(
            f"runs are not comparable: {noun} differs ({attr} {before!r} vs {after!r}). "
            f"A {noun} change invalidates prior baselines -- re-run the baseline before comparing."
        )


def _compare_tier(
    tier: int,
    baseline: list[Scored],
    candidate: list[Scored],
    noise_floor_pp: float | None,
) -> TierComparison:
    before = {s.name: s for s in baseline if s.tier == tier}
    after = {s.name: s for s in candidate if s.tier == tier}
    shared = before.keys() & after.keys()

    baseline_covered = sum(1 for s in before.values() if s.covered)
    candidate_covered = sum(1 for s in after.values() if s.covered)
    questions = max(len(before), len(after)) or 1
    delta_pp = (candidate_covered - baseline_covered) / questions * 100

    is_real: bool | None = None if noise_floor_pp is None else abs(delta_pp) > noise_floor_pp

    return TierComparison(
        tier=tier,
        baseline_covered=baseline_covered,
        candidate_covered=candidate_covered,
        questions=questions,
        coverage_delta_pp=delta_pp,
        coverage_is_real=is_real,
        baseline_efficiency=_median_efficiency(before.values()),
        candidate_efficiency=_median_efficiency(after.values()),
        # Named, not just counted: a rate tells a human something moved, but
        # only the names tell them where to look.
        regressions=sorted(n for n in shared if before[n].covered and not after[n].covered),
        improvements=sorted(n for n in shared if not before[n].covered and after[n].covered),
    )


def _median_efficiency(scored) -> int | None:
    """Median payload over questions that cleared coverage (#309)."""
    passing = [s.efficiency_tokens for s in scored if s.covered]
    return int(median(passing)) if passing else None


def _verdict(tiers: dict[int, TierComparison]) -> Verdict:
    """A verdict, deliberately conservative.

    A real coverage regression decides the outcome even when efficiency
    improved: coverage is the gate (#309), and a cheaper answer that is missing
    facts is not a better one. Efficiency alone never declares an improvement
    for the same reason -- returning less is only a win if coverage held, and
    "held" cannot be established inside the noise floor.
    """
    if any(t.coverage_is_real and t.coverage_delta_pp < 0 for t in tiers.values()):
        return Verdict.REGRESSED
    if any(t.coverage_is_real and t.coverage_delta_pp > 0 for t in tiers.values()):
        return Verdict.IMPROVED
    return Verdict.INCONCLUSIVE


def render(comparison: Comparison) -> str:
    """Render the report as plain text."""
    meta = comparison.candidate
    lines = [
        f"context-graph eval - {meta.label} vs {comparison.baseline.label}",
        f"corpus longmemeval-{meta.corpus_variant}@{meta.corpus_revision} ({meta.questions}q)",
        f"judge {meta.judge_model} - tok {meta.tokenizer}",
        "",
        f"VERDICT  {comparison.verdict.value}",
    ]

    if comparison.noise_floor_pp is None:
        lines.append("  noise floor NOT CALIBRATED - no delta can be called real.")
        lines.append("  Run the repeat-and-compare check (#304) to establish one.")
    else:
        lines.append(f"  noise floor +/-{comparison.noise_floor_pp:.0f}pp")

    for tier, t in sorted(comparison.tiers.items()):
        lines += ["", f"Tier {tier}{'':14}base    cand   delta"]
        verdict_note = _delta_note(t)
        lines.append(
            f"  coverage         {t.baseline_covered:>5}/{t.questions:<2}"
            f"{t.candidate_covered:>4}/{t.questions:<2}"
            f"{t.coverage_delta_pp:>+7.0f}pp  {verdict_note}"
        )
        lines.append(
            f"  efficiency med   {_fmt(t.baseline_efficiency):>7} {_fmt(t.candidate_efficiency):>7}"
            f"{_efficiency_delta(t):>10}"
        )
        if t.regressions:
            lines.append(f"  regressions      {', '.join(t.regressions)}")
        if t.improvements:
            lines.append(f"  improvements     {', '.join(t.improvements)}")

    if meta.changed:
        lines += ["", f"changed  {meta.changed}"]

    # Stated outright: the report informs a decision, it does not make one.
    lines += ["", "This report is input to a human promotion decision (#299), not a gate."]
    return "\n".join(lines)


def _delta_note(t: TierComparison) -> str:
    if t.coverage_is_real is None:
        return "(uncalibrated)"
    return "REAL" if t.coverage_is_real else "(noise)"


def _efficiency_delta(t: TierComparison) -> str:
    if not t.baseline_efficiency or t.candidate_efficiency is None:
        return ""
    change = (t.candidate_efficiency - t.baseline_efficiency) / t.baseline_efficiency * 100
    return f"{change:+.0f}%"


def _fmt(value: int | None) -> str:
    return f"{value:,}" if value is not None else "n/a"
