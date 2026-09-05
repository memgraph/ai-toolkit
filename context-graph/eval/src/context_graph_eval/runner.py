"""Drive a whole eval batch: inject, reconcile, retrieve, score.

This is the *pipeline* loop. deepeval owns the scoring loop underneath, but it
knows nothing about injection, reconciliation, or retrieval -- all of which must
happen before an ``actual_output`` exists for it to score. The division is:

    runner  ->  inject -> reconcile -> retrieve  ->  deepeval  ->  metrics

Ordering is the runner's real responsibility. Retrieving before injection would
query an empty graph and score every question a miss; scoring before
reconciliation would score raw turns rather than emerged memory, which is the
thing actually under test.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .convert.longmemeval import to_session_fixtures
from .inject import PENDING, inject_batch
from .reconcile import reconcile_batch
from .retrieval import ReadOnlyGraph, Retrieved, retrieve
from .scoring import (
    DEFAULT_COVERAGE_THRESHOLD,
    Scored,
    aggregate,
    efficiency_tokens,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from deepeval.dataset import Golden

    from actions_graph import ActionsGraph

    from .retrieval import LLM


@dataclass(frozen=True)
class RunPlan:
    """What a run does, and how much of it.

    ``reconcile`` is separable because it dominates cost: an LLM-backed pass
    over a batch's sessions is far more expensive than the retrieval it
    enables, so iterating on retrieval or scoring against an already-reconciled
    graph should not pay for it again.
    """

    reconcile: bool = True
    #: Reuse whatever is already in the graph instead of wiping and re-injecting.
    #: Without this, skipping reconciliation does not save anything: injection
    #: wipes first, so the run would re-inject raw sessions with no memory tier
    #: and silently measure retrieval against the collection tier alone (#322).
    reuse_graph: bool = False
    judge: Any | None = None
    reconcile_limit: int | None = None
    max_concurrent: int = 4
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    #: Trim each question's haystack. Reconciliation cost scales with sessions
    #: while coverage needs questions, and upstream couples them ~47:1. Any
    #: score measured with this set is an UPPER BOUND: fewer distractors make
    #: retrieval easier, so it is not comparable to a full-haystack run.
    max_sessions_per_question: int | None = None
    #: Which Memgraph reconciliation writes to. Not optional in practice:
    #: LightRAG's storage backends resolve their connection from the
    #: environment rather than the client passed in, so without this
    #: reconciliation either refuses to start or writes to whatever
    #: MEMGRAPH_URL happens to name -- a different graph than the one being
    #: evaluated, silently.
    memgraph_url: str | None = None


@dataclass(frozen=True)
class BatchReport:
    """Per-tier aggregates plus the per-question rows behind them."""

    by_tier: dict[int, Any] = field(default_factory=dict)
    scored: list[Scored] = field(default_factory=list)
    reconciled: int = 0
    reconcile_failures: int = 0


def _require_reconciled(fixtures: list, *, graph: "ActionsGraph") -> None:
    """Refuse to reuse a graph that cannot answer the questions about to be run.

    Reuse exists to skip the dominant cost (#322), but it hands the run a graph
    nobody just built, so the two ways it can be wrong are both silent. Either
    would score every affected question as a recall miss and report it as an
    ordinary result -- the same manufactured-zero shape as abstention questions
    judged on ContextualRecall, or a judge outage rendered as 0%.

    Missing sessions: the graph holds a different batch, or none.

    Present but unreconciled: injection ran without distillation, so there is no
    Chunk, Episode or entity to retrieve -- only the raw collection tier, which
    is a different system from the one under test.
    """
    wanted = {fixture.session_id for fixture in fixtures}
    if not wanted:
        return

    rows = graph._db.query(
        "MATCH (s:Session) WHERE s.session_id IN $ids "
        "RETURN s.session_id AS session_id, s.reconciliation_status AS status",
        {"ids": sorted(wanted)},
    )
    found = {row["session_id"]: row["status"] for row in rows}

    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(
            f"cannot reuse the graph: {len(missing)} of {len(wanted)} sessions this run needs are "
            f"not in it (e.g. {missing[:3]}). Run once without --skip-reconcile first."
        )

    pending = sorted(sid for sid, status in found.items() if status == PENDING)
    if pending:
        raise ValueError(
            f"cannot reuse the graph: {len(pending)} of {len(wanted)} sessions are still pending "
            f"reconciliation (e.g. {pending[:3]}), so there is no distilled memory to retrieve "
            "from -- only the raw collection tier. Run once without --skip-reconcile first."
        )


def check_offline() -> None:
    """Refuse to run if results would be exported to a third party.

    #302 kept the corpus out of a vendor cloud on this project's own owned-IP
    grounds -- the accumulated graph is the thing you own, unlike rented
    intelligence. deepeval uploads a test run whenever a Confident AI key is
    present, so a stray environment variable would quietly send eval results
    there. Fail loudly instead.
    """
    if os.environ.get("CONFIDENT_API_KEY"):
        raise RuntimeError(
            "CONFIDENT_API_KEY is set: deepeval would upload this run to Confident AI. "
            "Eval results stay local (#302). Unset it to continue."
        )


async def run_batch(
    goldens: list["Golden"],
    *,
    records: list[dict],
    graph: "ActionsGraph",
    llm: "LLM",
    plan: RunPlan | None = None,
) -> BatchReport:
    """Run one eval batch end to end and return its report.

    ``records`` are the upstream records the goldens came from -- the haystack
    lives there, not on the Golden, which carries only the answer key.
    """
    plan = plan or RunPlan()
    check_offline()

    # Every score, report row, and comparison in a run is keyed by question
    # name, so a nameless golden is unattributable: its scores would collide
    # with every other nameless one under a single key, and the questions it
    # displaced would report as unscored. Checked here rather than at scoring
    # time because by then the run has already paid for injection, distillation
    # and retrieval.
    unnamed = [i for i, g in enumerate(goldens) if not g.name]
    if unnamed:
        raise ValueError(f"every golden must carry a name to be scored; goldens at {unnamed} have none")

    fixtures = [
        fixture
        for record in records
        for fixture in to_session_fixtures(record, max_sessions=plan.max_sessions_per_question)
    ]
    if plan.reuse_graph:
        _require_reconciled(fixtures, graph=graph)
    else:
        inject_batch(fixtures, graph=graph)

    reconciled = failures = 0
    if plan.reconcile:
        outcome = await reconcile_batch(graph._db, limit=plan.reconcile_limit, memgraph_url=plan.memgraph_url)
        reconciled, failures = outcome.reconciled, outcome.failed

    read_only = ReadOnlyGraph(graph._db)
    retrieved = await _retrieve_all(goldens, read_only, llm, plan.max_concurrent)

    scored = _score(goldens, retrieved, plan)
    report = aggregate(scored)
    return BatchReport(
        by_tier=report.by_tier,
        scored=scored,
        reconciled=reconciled,
        reconcile_failures=failures,
    )


async def _retrieve_all(
    goldens: list["Golden"],
    graph: ReadOnlyGraph,
    llm: "LLM",
    max_concurrent: int,
) -> list[Retrieved]:
    """Retrieve for every question, bounded so a batch cannot stampede the model.

    A question whose retrieval raises becomes an empty result rather than
    propagating: a coverage rate computed over a silently shortened corpus is
    wrong, not merely noisy, so a failure has to be reported as a miss.
    """
    limiter = asyncio.Semaphore(max_concurrent)

    async def one(golden: "Golden") -> Retrieved:
        async with limiter:
            try:
                return await retrieve(golden.input, graph=graph, llm=llm)
            except Exception as exc:
                return Retrieved(answer="", errors=[str(exc)])

    return list(await asyncio.gather(*(one(golden) for golden in goldens)))


def _score(goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan) -> list[Scored]:
    """Turn retrieval results into per-question scores.

    Efficiency is computed regardless of whether a judge ran -- it is
    deterministic (#304), so there is no reason to make it wait on an LLM.
    """
    judged = _judge(goldens, retrieved, plan) if plan.judge is not None else {}

    scored: list[Scored] = []
    for golden, result in zip(goldens, retrieved, strict=True):
        metadata = golden.additional_metadata or {}
        metric_scores = judged.get(golden.name, {})
        # The weakest metric gates: passing one check while failing another is
        # not a pass. The individual scores are kept alongside so a failure can
        # still be attributed to retrieval or to the answer.
        coverage = min(metric_scores.values()) if metric_scores else 0.0
        scored.append(
            Scored(
                name=golden.name or golden.input,
                tier=metadata.get("tier", 1),
                coverage=coverage,
                covered=coverage >= plan.coverage_threshold,
                efficiency_tokens=efficiency_tokens(result),
                abstention=bool(metadata.get("abstention")),
                answer=result.answer,
                metric_scores=metric_scores,
            )
        )
    return scored


def _judge(goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan) -> dict[str, dict[str, float]]:
    """Score answer quality with deepeval, returning per-metric scores per question.

    Abstention and ordinary questions are judged in **separate passes**, because
    they need different metrics: ContextualRecall is structurally inapplicable
    to a question whose correct retrieved context is empty (see
    ``scoring.build_metrics``). Scoring them together made every abstention
    question unpassable.

    deepeval's ``evaluate`` is synchronous and drives its own event loop, so it
    runs after all pipeline work rather than inside it.
    """
    paired = list(zip(goldens, retrieved, strict=True))
    judged: dict[str, dict[str, float]] = {}
    for abstention in (False, True):
        group = [(g, r) for g, r in paired if bool((g.additional_metadata or {}).get("abstention")) is abstention]
        if group:
            judged.update(_judge_group(group, plan, abstention=abstention))
    return judged


def _judge_group(
    group: list[tuple["Golden", Retrieved]], plan: RunPlan, *, abstention: bool
) -> dict[str, dict[str, float]]:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

    from .scoring import build_metrics, to_test_case

    goldens = [g for g, _ in group]
    cases = [to_test_case(g, r) for g, r in group]
    result = evaluate(
        test_cases=cases,
        metrics=build_metrics(plan.judge, abstention=abstention),
        async_config=AsyncConfig(max_concurrent=plan.max_concurrent),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        # One question the judge cannot score should not abandon the batch --
        # the same reasoning reconciliation uses for an undistillable session.
        error_config=ErrorConfig(ignore_errors=True),
    )

    judged: dict[str, dict[str, float]] = {}
    for golden, test_result in zip(goldens, result.test_results, strict=False):
        # Kept per metric, not collapsed. The weakest still decides the gate --
        # passing one check while failing another is not a pass -- but which
        # one failed is what tells you whether retrieval or the answer was at
        # fault, and #304 pointed out that attribution is free here.
        # run_batch has already rejected nameless goldens; asserted rather than
        # re-checked so the type narrows and the invariant stays stated once.
        assert golden.name is not None
        judged[golden.name] = {m.name: m.score for m in (test_result.metrics_data or []) if m.score is not None}
    return judged
