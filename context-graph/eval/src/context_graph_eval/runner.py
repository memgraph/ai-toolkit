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
from .inject import inject_batch
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
    judge: Any | None = None
    reconcile_limit: int | None = None
    max_concurrent: int = 4
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD


@dataclass(frozen=True)
class BatchReport:
    """Per-tier aggregates plus the per-question rows behind them."""

    by_tier: dict[int, Any] = field(default_factory=dict)
    scored: list[Scored] = field(default_factory=list)
    reconciled: int = 0
    reconcile_failures: int = 0


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

    fixtures = [fixture for record in records for fixture in to_session_fixtures(record)]
    inject_batch(fixtures, graph=graph)

    reconciled = failures = 0
    if plan.reconcile:
        outcome = await reconcile_batch(graph._db, limit=plan.reconcile_limit)
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
        coverage = judged.get(golden.name, 0.0)
        scored.append(
            Scored(
                name=golden.name or golden.input,
                tier=metadata.get("tier", 1),
                coverage=coverage,
                covered=coverage >= plan.coverage_threshold,
                efficiency_tokens=efficiency_tokens(result),
                abstention=bool(metadata.get("abstention")),
                answer=result.answer,
            )
        )
    return scored


def _judge(goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan) -> dict[str, float]:
    """Score answer quality with deepeval, returning coverage per question.

    deepeval's ``evaluate`` is synchronous and drives its own event loop, so it
    runs after all pipeline work rather than inside it.
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

    from .scoring import build_metrics, to_test_case

    cases = [to_test_case(g, r) for g, r in zip(goldens, retrieved, strict=True)]
    result = evaluate(
        test_cases=cases,
        metrics=build_metrics(plan.judge),
        async_config=AsyncConfig(max_concurrent=plan.max_concurrent),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        # One question the judge cannot score should not abandon the batch --
        # the same reasoning reconciliation uses for an undistillable session.
        error_config=ErrorConfig(ignore_errors=True),
    )

    coverage: dict[str, float] = {}
    for golden, test_result in zip(goldens, result.test_results, strict=False):
        scores = [m.score for m in (test_result.metrics_data or []) if m.score is not None]
        # The weakest metric decides: coverage is a gate, so passing one check
        # while failing another is not a pass.
        coverage[golden.name] = min(scores) if scores else 0.0
    return coverage
