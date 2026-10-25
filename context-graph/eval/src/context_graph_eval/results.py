"""A run's results in one shape for every benchmark: one row per question, plus what it was measured against.

Every benchmark's judge reduces a question to a ``score`` between 0 and 1
(LongMemEval: right or wrong; BEAM: partial credit per rubric item), so the
summary, the comparison and the noise floor are written once. How question
scores become the headline is the benchmark's (``Benchmark.headline``):
LongMemEval averages questions, BEAM averages abilities.

A run file holds benchmark text -- questions, answers, what recall
returned -- under each row's ``detail``. A published baseline must not
(BEAM is CC BY-SA, LongMemEval's haystack is never committed), so
:func:`public` strips it before a run goes into ``baselines/``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

#: A question scoring at least this counts as passed: covered, on
#: LongMemEval; at least half the rubric, on BEAM. Used for pass stability and
#: for which questions efficiency is compared over (#309).
PASS_SCORE = 0.5


@dataclass(frozen=True)
class Row:
    """One question's outcome."""

    name: str
    user_id: str
    question_type: str
    #: The benchmark judge's score, 0-1; None when it could not judge (an outage, never a 0).
    score: float | None
    abstention: bool = False
    #: Share of the question's evidence turns found in what recall returned;
    #: None when the question cites none.
    evidence_recall: float | None = None
    #: Tokens of retrieval payload (#309), cl100k_base.
    efficiency_tokens: int = 0
    #: Recall plus the answering call.
    latency_seconds: float = 0.0
    #: ``sessions_graph.recall`` alone; None when recall raised.
    recall_seconds: float | None = None
    errored: bool = False
    #: Benchmark-specific numbers (official verdict, BLEU, F1, rubric scores): public.
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Benchmark text -- question, answer, retrieval, judge reasons, errors: never published.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score is not None and self.score >= PASS_SCORE


@dataclass(frozen=True)
class RunMeta:
    """What a run was measured against.

    The fields :func:`report.compare` pins are the ones that, differing,
    would make two runs' numbers incomparable: the workload, the upstream
    revision, the judge, the answering model and the tokenizer. The rest
    describe the system under test, and are what a comparison measures.
    """

    label: str
    workload: str
    benchmark: str
    revision: str
    questions: int
    judge_model: str
    agent_model: str
    tokenizer: str = "cl100k_base"
    changed: str = ""
    hybrid_lanes: str = ""
    #: Recall's packing: "turns" (separate turn and fact rows, as every run before
    #: exchanges became the default was measured) or "exchanges".
    packing: str = "turns"
    #: The build the questions were asked of: its fingerprint, inputs and timing (see instance.py).
    build: dict[str, Any] = field(default_factory=dict)
    commit: str | None = None
    memgraph_version: str | None = None
    #: The deepeval rubric judge reported beside LongMemEval's own, or "none".
    rubric_judge_model: str = "none"
    #: Hand-written facts about a published run: hardware, dates, caveats.
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    meta: RunMeta
    rows: list[Row]


@dataclass(frozen=True)
class TypeSummary:
    """One question type: how many were asked, judged, and how they scored."""

    question_type: str
    questions: int
    judged: int
    mean_score: float | None
    mean_evidence_recall: float | None


@dataclass(frozen=True)
class Summary:
    """A run's numbers, computed from its rows the same way for every benchmark."""

    headline: float | None
    questions: int
    judged: int
    by_type: list[TypeSummary]
    mean_evidence_recall: float | None
    mean_tokens: float | None
    #: Over questions that passed: returning less is only cheaper if it still answers (#309).
    median_tokens_passed: float | None
    latency: dict[str, float | None]
    errored: int


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (non-empty), as latency tails are reported."""
    import math

    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def summarize(rows: list[Row], *, headline: float | None, type_order: tuple[str, ...] = ()) -> Summary:
    """Summarize ``rows`` around the benchmark's ``headline``.

    Unjudged questions are left out of every score mean, never counted as 0:
    an outage is not a measurement.
    """
    by_type: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_type[row.question_type].append(row)
    ordered = [t for t in type_order if t in by_type] + sorted(t for t in by_type if t not in type_order)
    judged = [row for row in rows if row.score is not None]
    evidence = [row.evidence_recall for row in rows if row.evidence_recall is not None]
    passed_tokens = [row.efficiency_tokens for row in rows if row.passed]
    return Summary(
        headline=headline,
        questions=len(rows),
        judged=len(judged),
        by_type=[_type_summary(question_type, by_type[question_type]) for question_type in ordered],
        mean_evidence_recall=mean(evidence) if evidence else None,
        mean_tokens=mean(row.efficiency_tokens for row in rows) if rows else None,
        median_tokens_passed=median(passed_tokens) if passed_tokens else None,
        latency=latency(rows),
        errored=sum(1 for row in rows if row.errored),
    )


def _type_summary(question_type: str, rows: list[Row]) -> TypeSummary:
    scores = [row.score for row in rows if row.score is not None]
    evidence = [row.evidence_recall for row in rows if row.evidence_recall is not None]
    return TypeSummary(
        question_type=question_type,
        questions=len(rows),
        judged=len(scores),
        mean_score=mean(scores) if scores else None,
        mean_evidence_recall=mean(evidence) if evidence else None,
    )


def latency(rows: list[Row]) -> dict[str, float | None]:
    """End-to-end and recall-only latency: mean, median and p95, in seconds.

    Recall-only is over the rows that report it, so a question whose
    retrieval raised doesn't count as an instant recall.
    """
    summary: dict[str, float | None] = {}
    for name, values in (
        ("e2e", [row.latency_seconds for row in rows]),
        ("recall", [row.recall_seconds for row in rows if row.recall_seconds is not None]),
    ):
        summary[f"mean_{name}_seconds"] = mean(values) if values else None
        summary[f"median_{name}_seconds"] = percentile(values, 50) if values else None
        summary[f"p95_{name}_seconds"] = percentile(values, 95) if values else None
    return summary


def save(run: Run, path: Path, *, summary: Summary | None = None) -> Path:
    """Write ``run`` to ``path`` as JSON, with its summary beside the rows for a reader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": asdict(run.meta),
        "summary": asdict(summary) if summary is not None else None,
        "rows": [asdict(row) for row in run.rows],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load(path: Path) -> Run:
    """Read a run written by :func:`save`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(RunMeta)}
    meta = RunMeta(**{key: value for key, value in payload["meta"].items() if key in known})
    return Run(meta=meta, rows=[Row(**row) for row in payload["rows"]])


def public(run: Run) -> Run:
    """``run`` without benchmark text: every row's ``detail`` dropped. What ``baselines/`` holds."""
    return replace(run, rows=[replace(row, detail={}) for row in run.rows])


def merge(runs: list[Run], *, label: str) -> Run:
    """One run from shards of the same workload, measured the same way.

    Raises:
        ValueError: shards of different workloads, revisions, judges or
            answering models, or a question asked in two shards.
    """
    if not runs:
        raise ValueError("nothing to merge")
    first = runs[0].meta
    for run in runs[1:]:
        for attr in ("workload", "revision", "judge_model", "agent_model", "tokenizer"):
            if getattr(run.meta, attr) != getattr(first, attr):
                raise ValueError(f"shards differ in {attr}: {getattr(first, attr)!r} vs {getattr(run.meta, attr)!r}")
    rows = [row for run in runs for row in run.rows]
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("a question appears in more than one shard")
    build_seconds = [run.meta.build.get("build_seconds") for run in runs]
    build = {**first.build, "shards": len(runs)}
    build.pop("shard", None)
    if all(seconds is not None for seconds in build_seconds):
        build["build_seconds"] = sum(float(seconds) for seconds in build_seconds if seconds is not None)
    commits = sorted({run.meta.commit for run in runs if run.meta.commit})
    return Run(
        meta=replace(
            first,
            label=label,
            questions=len(rows),
            build=build,
            commit=commits[0] if len(commits) == 1 else ", ".join(commits) or None,
        ),
        rows=rows,
    )
