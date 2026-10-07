"""Run BEAM against the product's read path: inject, reconcile, recall, judge.

The pipeline is the hybrid one the LongMemEval runs measure -- the same
injection, reconciliation, ``sessions_graph.recall`` and ``answer_prompt`` --
so a BEAM score and a LongMemEval score are two readings of one system.
Only the questions and the judge differ: BEAM's own rubric judge
(``beam_judge``) decides every score.

Separate from ``runner`` rather than a mode of it: ``runner``'s scoring is
built around a yes/no verdict and LongMemEval's question types, while BEAM
scores each question on a 0-1 scale per rubric and reports by ability.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from .convert.beam import ABILITIES, BeamChat, to_goldens, to_session_fixtures
from .hybrid import RecallConfig, ensure_recall_ready, retrieve_hybrid
from .inject import inject_batch
from .reconcile import ExtractionBackendName, reconcile_batch
from .retrieval import ReadOnlyGraph, Retrieved
from .runner import _require_reconciled, check_offline
from .scoring import efficiency_tokens, evidence_recall

if TYPE_CHECKING:
    from deepeval.dataset import Golden

    from actions_graph import ActionsGraph

    from .beam_judge import BeamJudge
    from .retrieval import LLM


@dataclass(frozen=True)
class BeamRow:
    """One question: what was asked, answered, retrieved, and how BEAM's judge scored it."""

    name: str
    user_id: str
    ability: str
    question: str
    expected: str
    answer: str
    rubric: list[str]
    #: The number upstream's report averages; None when unjudged.
    score: float | None = None
    llm_judge_score: float | None = None
    item_scores: list[float | None] = field(default_factory=list)
    item_reasons: list[str] = field(default_factory=list)
    ordering: dict[str, float] | None = None
    #: Share of the messages the question cites found in retrieval; None when it cites none.
    evidence_recall: float | None = None
    efficiency_tokens: int = 0
    latency_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    retrieval_context: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BeamRun:
    """A run's rows, plus how much of the graph reconciliation actually built."""

    rows: list[BeamRow]
    reconciled: int = 0
    reconcile_failures: int = 0


def select_questions(goldens: list[Golden], limit: int | None) -> list[Golden]:
    """A deterministic subset of ``limit`` questions, balanced across abilities and spread across chats.

    Each ability gets an equal share (earlier abilities absorb the remainder),
    because BEAM's score weights abilities equally: an ability with one
    question would swing the headline on a single verdict. Within an ability,
    each chat's first question is preferred, and each ability starts at a
    different chat, so the chats share the questions evenly too.
    """
    if limit is None or limit >= len(goldens):
        return list(goldens)
    by_ability: dict[str, list[Golden]] = defaultdict(list)
    for golden in goldens:
        by_ability[(golden.additional_metadata or {})["question_type"]].append(golden)
    abilities = [ability for ability in ABILITIES if by_ability.get(ability)]
    share, extra = divmod(limit, len(abilities))

    selected: list[Golden] = []
    for position, ability in enumerate(abilities):
        quota = share + (1 if position < extra else 0)
        users = list(dict.fromkeys((g.additional_metadata or {})["user_id"] for g in by_ability[ability]))
        offset = (position * share) % len(users)
        rotation = users[offset:] + users[:offset]
        ranked = sorted(
            by_ability[ability],
            key=lambda g: (
                int((g.name or "").rsplit("-", 1)[1]),
                rotation.index((g.additional_metadata or {})["user_id"]),
            ),
        )
        selected.extend(ranked[:quota])
    names = {golden.name for golden in selected}
    return [golden for golden in goldens if golden.name in names]


async def run_beam(
    chats: list[BeamChat],
    *,
    graph: ActionsGraph,
    llm: LLM,
    judge: BeamJudge | None,
    memgraph_url: str,
    reuse_graph: bool = False,
    extraction_backend: ExtractionBackendName = "lightrag",
    hybrid: RecallConfig | None = None,
    max_concurrent: int = 4,
    limit: int | None = None,
) -> BeamRun:
    """Run every probing question of ``chats`` end to end.

    ``reuse_graph`` skips the wipe, injection and reconciliation, and refuses
    a graph that doesn't hold these chats' sessions fully reconciled -- the
    same check, for the same reason, as ``runner``'s ``--skip-reconcile``.
    ``limit`` asks only a subset (``select_questions``); every chat is still
    injected and reconciled, so the graph is the one a full run would use.
    """
    check_offline()
    fixtures = [fixture for chat in chats for fixture in to_session_fixtures(chat)]
    goldens = select_questions([golden for chat in chats for golden in to_goldens(chat)], limit)

    reconciled = failures = 0
    if reuse_graph:
        _require_reconciled(fixtures, graph=graph, extraction_backend=extraction_backend)
    else:
        inject_batch(fixtures, graph=graph)
        outcome = await reconcile_batch(graph.db, memgraph_url=memgraph_url, extraction_backend=extraction_backend)
        reconciled, failures = outcome.reconciled, outcome.failed

    ensure_recall_ready(graph.db)
    retrieved = await _retrieve_all(goldens, ReadOnlyGraph(graph.db), llm, hybrid or RecallConfig(), max_concurrent)
    rows = await _judge_all(goldens, retrieved, judge)
    return BeamRun(rows=rows, reconciled=reconciled, reconcile_failures=failures)


async def _retrieve_all(
    goldens: list[Golden], graph: ReadOnlyGraph, llm: LLM, config: RecallConfig, max_concurrent: int
) -> list[Retrieved]:
    """Recall and answer each question from its chat's user's history.

    No question date is given: upstream asks its questions with none.
    A question whose retrieval raises is an empty answer, kept and judged, so
    a failure is a reported miss rather than a silently shorter run.
    """
    limiter = asyncio.Semaphore(max_concurrent)

    async def one(golden: Golden) -> Retrieved:
        async with limiter:
            started = time.monotonic()
            user_id = (golden.additional_metadata or {})["user_id"]
            try:
                return await retrieve_hybrid(golden.input, graph=graph, llm=llm, user_id=user_id, config=config)
            except Exception as exc:
                return Retrieved(answer="", errors=[str(exc)], latency_seconds=time.monotonic() - started)

    return list(await asyncio.gather(*(one(golden) for golden in goldens)))


async def _judge_all(goldens: list[Golden], retrieved: list[Retrieved], judge: BeamJudge | None) -> list[BeamRow]:
    async def one(golden: Golden, result: Retrieved) -> BeamRow:
        metadata = golden.additional_metadata or {}
        ability = metadata["question_type"]
        rubric = list(metadata.get("rubric", []))
        verdict = await judge.judge(ability, rubric, result.answer) if judge is not None else None
        return BeamRow(
            name=golden.name or golden.input,
            user_id=metadata["user_id"],
            ability=ability,
            question=golden.input,
            expected=golden.expected_output or "",
            answer=result.answer,
            rubric=rubric,
            score=verdict.score if verdict else None,
            llm_judge_score=verdict.llm_judge_score if verdict else None,
            item_scores=verdict.item_scores if verdict else [],
            item_reasons=verdict.item_reasons if verdict else [],
            ordering=verdict.ordering if verdict else None,
            evidence_recall=evidence_recall(golden.context, result.retrieval_context),
            efficiency_tokens=efficiency_tokens(result),
            latency_seconds=result.latency_seconds,
            errors=list(result.errors),
            retrieval_context=list(result.retrieval_context),
        )

    return list(await asyncio.gather(*(one(g, r) for g, r in zip(goldens, retrieved, strict=True))))


@dataclass(frozen=True)
class AbilityScore:
    """One ability's mean over its judged questions."""

    ability: str
    questions: int
    judged: int
    mean_score: float | None
    mean_evidence_recall: float | None


def summarize(rows: list[BeamRow]) -> tuple[list[AbilityScore], float | None]:
    """Per-ability means and their mean -- upstream's report, which weights every ability equally.

    Unjudged questions are left out of a mean rather than counted as zero, and
    ``judged`` says how many that left.
    """
    by_ability: dict[str, list[BeamRow]] = defaultdict(list)
    for row in rows:
        by_ability[row.ability].append(row)
    scores = []
    for ability in ABILITIES:
        group = by_ability.get(ability, [])
        if not group:
            continue
        judged = [row.score for row in group if row.score is not None]
        evidence = [row.evidence_recall for row in group if row.evidence_recall is not None]
        scores.append(
            AbilityScore(
                ability=ability,
                questions=len(group),
                judged=len(judged),
                mean_score=mean(judged) if judged else None,
                mean_evidence_recall=mean(evidence) if evidence else None,
            )
        )
    means = [score.mean_score for score in scores if score.mean_score is not None]
    return scores, (mean(means) if means else None)


def render(run: BeamRun) -> str:
    """The run's report as text."""
    scores, overall = summarize(run.rows)
    lines = [f"{'ability':<26} {'n':>3} {'judged':>6} {'score':>6} {'evidence':>8}"]
    for score in scores:
        mean_score = f"{score.mean_score:.3f}" if score.mean_score is not None else "-"
        evidence = f"{score.mean_evidence_recall:.2f}" if score.mean_evidence_recall is not None else "-"
        lines.append(f"{score.ability:<26} {score.questions:>3} {score.judged:>6} {mean_score:>6} {evidence:>8}")
    lines.append(f"{'BEAM score (mean of abilities)':<38} {overall:.3f}" if overall is not None else "BEAM score: -")
    errors = sum(1 for row in run.rows if row.errors)
    lines.append(f"reconciled {run.reconciled} sessions, {run.reconcile_failures} failed; {errors} retrieval errors")
    return "\n".join(lines)


def save(run: BeamRun, path: Path, *, meta: dict[str, Any]) -> Path:
    """Write the run, its settings and every row to ``path`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores, overall = summarize(run.rows)
    payload = {
        "meta": meta,
        "overall": overall,
        "abilities": [asdict(score) for score in scores],
        "reconciled": run.reconciled,
        "reconcile_failures": run.reconcile_failures,
        "rows": [asdict(row) for row in run.rows],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_rows(path: Path) -> list[BeamRow]:
    """The rows of a saved run."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BeamRow(**row) for row in payload["rows"]]
