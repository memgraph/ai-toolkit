"""Render a run, and compare two runs of the same workload.

#299 chose human-gated promotion: eval produces a comparison a person reads
and acts on, rather than a threshold that promotes automatically. So a
comparison's job is not to decide -- it is to make the decision makeable,
which means saying plainly what changed and whether the change is real.

Judged scores vary run to run, so a human shown a bare 0.62 -> 0.64 will read
a win into noise. #304's repeat-and-compare calibration gives that variance a
number; a comparison refuses to call anything real without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median

from .benchmarks import headline, type_order
from .results import Row, Run, Summary, summarize


class Verdict(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"


#: RunMeta fields that, differing, make two runs incomparable: a change to any
#: of them is measured as though it were the change under test.
PINNED = ("workload", "revision", "judge_model", "agent_model", "tokenizer")


def summary_of(run: Run) -> Summary:
    """``run``'s summary, around its benchmark's headline."""
    return summarize(
        run.rows, headline=headline(run.meta.benchmark, run.rows), type_order=type_order(run.meta.benchmark)
    )


def _score(value: float | None, benchmark: str) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if benchmark != "beam-100k" else f"{value:.3f}"


def _seconds(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "-"


def render(run: Run, *, notes: list[str] | None = None) -> str:
    """``run`` as the text a person reads after it finishes."""
    meta = run.meta
    summary = summary_of(run)
    lines = [
        f"{meta.workload} -- {meta.label} ({meta.benchmark} @ {meta.revision[:12]})",
        f"answering {meta.agent_model} | judge {meta.judge_model} | packing {meta.packing}"
        + (f" | rubrics {meta.rubric_judge_model}" if meta.rubric_judge_model != "none" else ""),
        "",
        f"{'question type':<28} {'n':>4} {'judged':>6} {'score':>7} {'evidence':>8}",
    ]
    for t in summary.by_type:
        evidence = f"{t.mean_evidence_recall:.2f}" if t.mean_evidence_recall is not None else "-"
        lines.append(
            f"{t.question_type:<28} {t.questions:>4} {t.judged:>6} {_score(t.mean_score, meta.benchmark):>7} {evidence:>8}"
        )
    lines.append(
        f"{'HEADLINE':<28} {summary.questions:>4} {summary.judged:>6} {_score(summary.headline, meta.benchmark):>7}"
    )
    lines.append("")
    if summary.judged < summary.questions:
        # With a judge configured, unjudged means it errored -- an outage, not a 0.
        lines.append(f"UNJUDGED      {summary.questions - summary.judged} question(s): left out of every score")
    if summary.mean_evidence_recall is not None:
        lines.append(f"evidence      mean recall {summary.mean_evidence_recall:.2f}")
    if summary.mean_tokens is not None:
        passed = f", median {summary.median_tokens_passed:,.0f} over passed" if summary.median_tokens_passed else ""
        lines.append(f"tokens        mean {summary.mean_tokens:,.0f} retrieved{passed}")
    times = summary.latency
    lines.append(
        f"recall only   mean {_seconds(times['mean_recall_seconds'])}, median {_seconds(times['median_recall_seconds'])}, "
        f"p95 {_seconds(times['p95_recall_seconds'])}"
    )
    lines.append(f"end to end    mean {_seconds(times['mean_e2e_seconds'])}, p95 {_seconds(times['p95_e2e_seconds'])}")
    if summary.errored:
        lines.append(f"errors        {summary.errored} question(s) whose retrieval reported an error")
    lines.extend(notes or [])
    build = meta.build
    if build.get("fingerprint"):
        stale = "" if build.get("current", True) else " (STALE: asked with --reuse after its inputs changed)"
        seconds = build.get("build_seconds")
        took = f", {seconds / 60:.0f} min" if seconds else ""
        lines.append(f"graph         build {build['fingerprint']} at {build.get('commit') or '?'}{took}{stale}")
    return "\n".join(lines)


@dataclass(frozen=True)
class TypeDelta:
    question_type: str
    baseline: float | None
    candidate: float | None


@dataclass(frozen=True)
class Comparison:
    baseline: Run
    candidate: Run
    baseline_headline: float | None
    candidate_headline: float | None
    #: Headline change in percentage points.
    delta_pp: float | None
    noise_floor_pp: float | None
    #: True/False when calibrated, None when no noise floor is known -- "not
    #: real" and "cannot tell" are different answers to a decision.
    is_real: bool | None
    verdict: Verdict
    by_type: list[TypeDelta] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    #: Median retrieval tokens over questions both runs passed (#309).
    baseline_tokens: float | None = None
    candidate_tokens: float | None = None
    baseline_evidence: float | None = None
    candidate_evidence: float | None = None
    baseline_recall_seconds: float | None = None
    candidate_recall_seconds: float | None = None
    #: What differs about the system under test: build inputs, lanes, commit.
    changes: list[str] = field(default_factory=list)


def compare(baseline: Run, candidate: Run, *, noise_floor_pp: float | None = None) -> Comparison:
    """Compare two runs of the same questions, refusing when they were not measured the same way.

    Raises:
        ValueError: a pinned field (``PINNED``) differs, or the runs asked different questions.
    """
    for attr in PINNED:
        before, after = getattr(baseline.meta, attr), getattr(candidate.meta, attr)
        if before != after:
            raise ValueError(
                f"runs are not comparable: {attr} differs ({before!r} vs {after!r}). "
                "Re-run the baseline the same way before comparing."
            )
    if {row.name for row in baseline.rows} != {row.name for row in candidate.rows}:
        raise ValueError("runs are not comparable: they asked different questions")

    benchmark = baseline.meta.benchmark
    before = {row.name: row for row in baseline.rows}
    after = {row.name: row for row in candidate.rows}
    base_headline = headline(benchmark, baseline.rows)
    cand_headline = headline(benchmark, candidate.rows)
    delta = (cand_headline - base_headline) * 100 if base_headline is not None and cand_headline is not None else None
    is_real = None if noise_floor_pp is None or delta is None else abs(delta) > noise_floor_pp
    both_passed = [name for name in before if before[name].passed and after[name].passed]

    return Comparison(
        baseline=baseline,
        candidate=candidate,
        baseline_headline=base_headline,
        candidate_headline=cand_headline,
        delta_pp=delta,
        noise_floor_pp=noise_floor_pp,
        is_real=is_real,
        verdict=_verdict(delta, is_real),
        by_type=_by_type(benchmark, baseline.rows, candidate.rows),
        regressions=sorted(name for name in before if before[name].passed and not after[name].passed),
        improvements=sorted(name for name in before if not before[name].passed and after[name].passed),
        baseline_tokens=_median([before[name].efficiency_tokens for name in both_passed]),
        candidate_tokens=_median([after[name].efficiency_tokens for name in both_passed]),
        baseline_evidence=_mean([row.evidence_recall for row in baseline.rows]),
        candidate_evidence=_mean([row.evidence_recall for row in candidate.rows]),
        baseline_recall_seconds=_mean([row.recall_seconds for row in baseline.rows]),
        candidate_recall_seconds=_mean([row.recall_seconds for row in candidate.rows]),
        changes=_changes(baseline, candidate),
    )


def _verdict(delta: float | None, is_real: bool | None) -> Verdict:
    """Deliberately conservative: nothing is called better or worse inside the noise floor, or without one."""
    if not is_real or delta is None:
        return Verdict.INCONCLUSIVE
    return Verdict.IMPROVED if delta > 0 else Verdict.REGRESSED


def _by_type(benchmark: str, baseline: list[Row], candidate: list[Row]) -> list[TypeDelta]:
    def means(rows: list[Row]) -> dict[str, float]:
        by: dict[str, list[float]] = {}
        for row in rows:
            if row.score is not None:
                by.setdefault(row.question_type, []).append(row.score)
        return {key: mean(values) for key, values in by.items()}

    before, after = means(baseline), means(candidate)
    present = before.keys() | after.keys()
    ordered = [t for t in type_order(benchmark) if t in present] + sorted(present - set(type_order(benchmark)))
    return [TypeDelta(t, before.get(t), after.get(t)) for t in ordered]


def _changes(baseline: Run, candidate: Run) -> list[str]:
    changes = []
    old, new = baseline.meta.build.get("inputs") or {}, candidate.meta.build.get("inputs") or {}
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) != new.get(key) and key not in ("shard",):
            changes.append(f"build {key}: {old.get(key)} -> {new.get(key)}")
    if baseline.meta.packing != candidate.meta.packing:
        changes.append(f"recall packing: {baseline.meta.packing} -> {candidate.meta.packing}")
    if baseline.meta.hybrid_lanes != candidate.meta.hybrid_lanes:
        changes.append(f"recall lanes: {baseline.meta.hybrid_lanes} -> {candidate.meta.hybrid_lanes}")
    if baseline.meta.commit != candidate.meta.commit:
        changes.append(f"commit: {baseline.meta.commit} -> {candidate.meta.commit}")
    return changes


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _median(values: list[int]) -> float | None:
    return median(values) if values else None


def render_comparison(comparison: Comparison) -> str:
    """The comparison as plain text."""
    base, cand = comparison.baseline.meta, comparison.candidate.meta
    benchmark = base.benchmark
    lines = [
        f"{cand.workload} -- {cand.label} vs {base.label}",
        f"answering {cand.agent_model} | judge {cand.judge_model}",
        "",
        f"VERDICT  {comparison.verdict.value}",
    ]
    if comparison.noise_floor_pp is None:
        lines.append("  noise floor NOT CALIBRATED - no delta can be called real (see `calibrate`).")
    else:
        lines.append(f"  noise floor +/-{comparison.noise_floor_pp:.1f}pp")
    lines += ["", f"{'':<28} {'base':>8} {'cand':>8} {'delta':>8}"]
    delta = f"{comparison.delta_pp:+.1f}pp" if comparison.delta_pp is not None else "-"
    note = {None: "(uncalibrated)", True: "REAL", False: "(noise)"}[comparison.is_real]
    lines.append(
        f"{'HEADLINE':<28} {_score(comparison.baseline_headline, benchmark):>8} "
        f"{_score(comparison.candidate_headline, benchmark):>8} {delta:>8}  {note}"
    )
    for t in comparison.by_type:
        change = (
            f"{(t.candidate - t.baseline) * 100:+.1f}pp" if t.baseline is not None and t.candidate is not None else ""
        )
        lines.append(
            f"  {t.question_type:<26} {_score(t.baseline, benchmark):>8} {_score(t.candidate, benchmark):>8} {change:>8}"
        )
    lines.append("")
    if comparison.baseline_evidence is not None and comparison.candidate_evidence is not None:
        lines.append(f"evidence      {comparison.baseline_evidence:.2f} -> {comparison.candidate_evidence:.2f}")
    if comparison.baseline_tokens is not None and comparison.candidate_tokens is not None:
        lines.append(
            f"tokens        median {comparison.baseline_tokens:,.0f} -> {comparison.candidate_tokens:,.0f} "
            "(questions both runs passed)"
        )
    else:
        lines.append("tokens        no question passed in both runs -- nothing comparable")
    lines.append(
        f"recall only   mean {_seconds(comparison.baseline_recall_seconds)} -> "
        f"{_seconds(comparison.candidate_recall_seconds)}"
    )
    if comparison.regressions:
        lines.append(f"stopped passing ({len(comparison.regressions)}): {', '.join(comparison.regressions)}")
    if comparison.improvements:
        lines.append(f"started passing ({len(comparison.improvements)}): {', '.join(comparison.improvements)}")
    if comparison.changes:
        lines += ["", "what differs:", *(f"  {change}" for change in comparison.changes)]
    if cand.changed:
        lines += ["", f"changed  {cand.changed}"]
    lines += ["", "This report is input to a human promotion decision (#299), not a gate."]
    return "\n".join(lines)
