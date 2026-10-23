"""Drive a whole eval batch: inject, reconcile, retrieve, score.

The runner owns the pipeline loop; deepeval owns the scoring loop underneath:

    runner  ->  inject -> reconcile -> retrieve  ->  deepeval  ->  metrics

Ordering is the runner's real responsibility: retrieving before injection
scores every question a miss, and scoring before reconciliation scores raw
turns rather than emerged memory.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from . import official_judge
from .convert.longmemeval import to_session_fixtures
from .hybrid import RecallConfig, ensure_recall_ready, retrieve_hybrid
from .inject import PENDING, inject_batch
from .reconcile import BACKEND_CLASS_NAMES, ExtractionBackendName, reconcile_batch
from .retrieval import ReadOnlyGraph, Retrieved, retrieve
from .scoring import (
    DEFAULT_COVERAGE_THRESHOLD,
    RUBRICS,
    Scored,
    aggregate,
    bleu_score,
    efficiency_tokens,
    enforce_retrieval_floor,
    evidence_recall,
    gate_score,
    rubric_for,
    token_f1_score,
)
from .text_search import DEFAULT_LIMIT as DEFAULT_TEXT_SEARCH_LIMIT
from .text_search import ensure_turn_text_index, retrieve_by_text_search

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from deepeval.dataset import Golden

    from actions_graph import ActionsGraph

    from .retrieval import LLM

T = TypeVar("T")

#: retrieval.py, text_search.py and hybrid.py respectively. The one axis
#: report.compare() does not pin: it is usually what a comparison measures.
RetrievalStrategyName = Literal["graph-agent", "text-search", "hybrid"]
RETRIEVAL_STRATEGIES: tuple[RetrievalStrategyName, ...] = ("graph-agent", "text-search", "hybrid")


@dataclass(frozen=True)
class RunPlan:
    """What a run does, and how much of it."""

    #: Separable because it dominates run cost.
    reconcile: bool = True
    #: Reuse the graph as-is instead of wiping and re-injecting; checked by _require_reconciled.
    reuse_graph: bool = False
    judge: Any | None = None
    #: "fixed" extracts against the LongMemEval vocabulary; "default" under hygm's
    #: default model; "learned" the same, then derived once per user (learned.py).
    ontology: str = "fixed"
    derive_seeds: int = 1
    derive_workers: int = 4
    max_concurrent: int = 4
    #: LongMemEval's own judge, which decides ``covered`` when set; None leaves the deepeval answer rubrics deciding.
    official_judge_model: str | None = None
    #: Trim each question's distractors. A score measured this way is an upper bound.
    max_sessions_per_question: int | None = None
    #: The Memgraph reconciliation writes to; see reconcile_batch for why it is needed.
    memgraph_url: str | None = None
    #: Ignored by "text-search", which never reconciles.
    extraction_backend: ExtractionBackendName = "gliner2"
    #: "text-search" forces reconciliation off: it reads only raw turns.
    retrieval_strategy: RetrievalStrategyName = "graph-agent"
    #: "text-search" only.
    text_search_limit: int = DEFAULT_TEXT_SEARCH_LIMIT
    #: "hybrid" only: which recall lanes run and how wide (sessions_graph.RecallConfig).
    hybrid: RecallConfig = field(default_factory=RecallConfig)


@dataclass(frozen=True)
class BatchReport:
    """Per-tier aggregates plus the per-question rows behind them."""

    by_tier: dict[int, Any] = field(default_factory=dict)
    scored: list[Scored] = field(default_factory=list)
    reconciled: int = 0
    reconcile_failures: int = 0
    #: Turns text_search.ensure_turn_text_index indexed; 0 for other strategies.
    indexed_turns: int = 0


def _require_reconciled(fixtures: list, *, graph: "ActionsGraph", extraction_backend: ExtractionBackendName) -> None:
    """Refuse to reuse a graph that cannot answer the questions about to be run.

    A reused graph was built by nobody this run, so every way it can be wrong
    is silent and would score as ordinary recall misses: sessions missing,
    sessions still pending reconciliation, or sessions reconciled by a
    different backend than ``extraction_backend`` -- which ``RunMeta`` would
    otherwise record unverified.

    Raises:
        ValueError: on any of the three.
    """
    wanted = {fixture.session_id for fixture in fixtures}
    if not wanted:
        return

    rows = graph.db.query(
        "MATCH (s:Session) WHERE s.session_id IN $ids "
        "RETURN s.session_id AS session_id, s.reconciliation_status AS status, "
        "s.extraction_backend AS extraction_backend",
        {"ids": sorted(wanted)},
    )
    found = {row["session_id"]: row for row in rows}

    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(
            f"cannot reuse the graph: {len(missing)} of {len(wanted)} sessions this run needs are "
            f"not in it (e.g. {missing[:3]}). Run once without --skip-reconcile first."
        )

    pending = sorted(sid for sid, row in found.items() if row["status"] == PENDING)
    if pending:
        raise ValueError(
            f"cannot reuse the graph: {len(pending)} of {len(wanted)} sessions are still pending "
            f"reconciliation (e.g. {pending[:3]}), so there is no distilled memory to retrieve "
            "from -- only the raw collection tier. Run once without --skip-reconcile first."
        )

    # None (no reconcilable content) is not attributable to any backend, so it is not a mismatch.
    expected_label = BACKEND_CLASS_NAMES[extraction_backend]
    mismatched = sorted(
        sid
        for sid, row in found.items()
        if row["extraction_backend"] is not None and row["extraction_backend"] != expected_label
    )
    if mismatched:
        raise ValueError(
            f"cannot reuse the graph: {len(mismatched)} of {len(wanted)} sessions were reconciled with a "
            f"different extraction backend than --extraction-backend={extraction_backend!r} claims "
            f"(e.g. {mismatched[:3]}). Run once without --skip-reconcile to rebuild the graph with this "
            "backend, or pass the backend that actually built it."
        )


def check_offline() -> None:
    """Refuse to run if deepeval would upload results to Confident AI (#302 keeps eval results local).

    Raises:
        RuntimeError: ``CONFIDENT_API_KEY`` is set.
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

    # Every score is keyed by question name; checked before paying for the pipeline.
    unnamed = [i for i, g in enumerate(goldens) if not g.name]
    if unnamed:
        raise ValueError(f"every golden must carry a name to be scored; goldens at {unnamed} have none")

    fixtures = [
        fixture
        for record in records
        for fixture in to_session_fixtures(record, max_sessions=plan.max_sessions_per_question)
    ]
    # text-search never reconciles, so it has nothing to reuse and always re-injects.
    if plan.reuse_graph and plan.retrieval_strategy != "text-search":
        _require_reconciled(fixtures, graph=graph, extraction_backend=plan.extraction_backend)
    else:
        inject_batch(fixtures, graph=graph)

    reconciled = failures = indexed_turns = 0
    if plan.retrieval_strategy == "text-search":
        indexed_turns = ensure_turn_text_index(graph).turns
    elif plan.reconcile:
        outcome = await reconcile_batch(
            graph.db,
            memgraph_url=plan.memgraph_url,
            extraction_backend=plan.extraction_backend,
            ontology=plan.ontology,
        )
        reconciled, failures = outcome.reconciled, outcome.failed
        if plan.ontology == "learned":
            from .learned import derive_users

            assert plan.memgraph_url is not None, "a learned-ontology run derives through --memgraph-url"
            derived = await derive_users(
                graph.db, memgraph_url=plan.memgraph_url, seeds=plan.derive_seeds, workers=plan.derive_workers
            )
            print(f"derivation: {derived.counts()}", flush=True)

    read_only = ReadOnlyGraph(graph.db)
    if plan.retrieval_strategy == "hybrid":
        ensure_recall_ready(graph.db)
    retrieved = await _retrieve_all(goldens, read_only, llm, plan)

    official = await _official_labels(goldens, retrieved, plan)
    scored = _score(goldens, retrieved, plan, official)
    report = aggregate(scored)
    return BatchReport(
        by_tier=report.by_tier,
        scored=scored,
        reconciled=reconciled,
        reconcile_failures=failures,
        indexed_turns=indexed_turns,
    )


async def retrieve_concurrently(
    items: Sequence[T], retrieve_one: Callable[[T], Awaitable[Retrieved]], *, max_concurrent: int
) -> list[Retrieved]:
    """Run ``retrieve_one`` over ``items``, at most ``max_concurrent`` at once, in input order.

    An item whose retrieval raises becomes an empty, timed result -- a
    reported miss rather than a silently shorter run.
    """
    limiter = asyncio.Semaphore(max_concurrent)

    async def one(item: T) -> Retrieved:
        async with limiter:
            started = time.monotonic()
            try:
                return await retrieve_one(item)
            except Exception as exc:
                return Retrieved(answer="", errors=[str(exc)], latency_seconds=time.monotonic() - started)

    return list(await asyncio.gather(*(one(item) for item in items)))


async def _retrieve_all(
    goldens: list["Golden"],
    graph: ReadOnlyGraph,
    llm: "LLM",
    plan: RunPlan,
) -> list[Retrieved]:
    """Retrieve for every question with ``plan``'s strategy, at most ``plan.max_concurrent`` at once."""

    async def one(golden: "Golden") -> Retrieved:
        # The corpus's question_date, the "now" relative dates count from.
        today = (golden.additional_metadata or {}).get("question_date")
        if plan.retrieval_strategy == "hybrid":
            # Each question is its own user's history (to_session_fixtures).
            if golden.name is None:
                raise ValueError("hybrid recall needs the question's user, carried as the golden's name")
            return await retrieve_hybrid(
                golden.input,
                graph=graph,
                llm=llm,
                config=plan.hybrid,
                today=today,
                user_id=golden.name,
            )
        if plan.retrieval_strategy == "text-search":
            return await retrieve_by_text_search(
                golden.input, graph=graph, llm=llm, limit=plan.text_search_limit, today=today
            )
        return await retrieve(golden.input, graph=graph, llm=llm, today=today)

    return await retrieve_concurrently(goldens, one, max_concurrent=plan.max_concurrent)


@dataclass(frozen=True)
class _Judged:
    """One question's judge output: per-metric scores and the judge's reasons, kept as one unit."""

    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


async def _official_labels(
    goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan
) -> dict[str, bool | None]:
    """LongMemEval's judge's verdict per question it has a template for; {} when it isn't configured."""
    if plan.official_judge_model is None:
        return {}
    prompts = {}
    for golden, result in zip(goldens, retrieved, strict=True):
        metadata = golden.additional_metadata or {}
        abstention = bool(metadata.get("abstention"))
        if golden.name and official_judge.judges(metadata.get("question_type"), abstention=abstention):
            prompts[golden.name] = official_judge.anscheck_prompt(
                metadata.get("question_type", ""),
                golden.input,
                golden.expected_output or "",
                result.answer,
                abstention=abstention,
            )
    return await official_judge.judge_all(prompts, model=plan.official_judge_model, max_concurrent=plan.max_concurrent)


def _score(
    goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan, official: dict[str, bool | None] | None = None
) -> list[Scored]:
    """Turn retrieval results into per-question scores.

    ``covered`` comes from LongMemEval's judge where it judged the question,
    else from the deepeval answer rubric. Efficiency, BLEU, F1, latency and
    evidence recall need no judge and are always computed.
    """
    judged = _judge(goldens, retrieved, plan) if plan.judge is not None else {}

    scored: list[Scored] = []
    for golden, result in zip(goldens, retrieved, strict=True):
        metadata = golden.additional_metadata or {}
        outcome = judged.get(golden.name, _Judged())
        coverage = gate_score(outcome.scores)
        # A failed official-judge call is unjudged, never a fallback to our own rubrics.
        judged_by = "official" if official and golden.name in official else "answer"
        verdict = official.get(golden.name) if official and golden.name else None
        scored.append(
            Scored(
                name=golden.name or golden.input,
                tier=metadata.get("tier", 1),
                coverage=coverage,
                covered=bool(verdict) if judged_by == "official" else coverage >= DEFAULT_COVERAGE_THRESHOLD,
                judged_by=judged_by,
                official_correct=verdict,
                efficiency_tokens=efficiency_tokens(result),
                abstention=bool(metadata.get("abstention")),
                answer=result.answer,
                metric_scores=outcome.scores,
                metric_reasons=outcome.reasons,
                bleu=bleu_score(golden.expected_output, result.answer),
                f1=token_f1_score(golden.expected_output, result.answer),
                latency_seconds=result.latency_seconds,
                recall_seconds=result.recall_seconds,
                evidence_recall=(
                    None if metadata.get("abstention") else evidence_recall(golden.context, result.retrieval_context)
                ),
            )
        )
    # After judging, so metric_scores keep what the judge said while the floor overrides ``covered``.
    return enforce_retrieval_floor(
        scored,
        retrieved_tokens={s.name: s.efficiency_tokens for s in scored},
    )


def _judge(goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan) -> dict[str, _Judged]:
    """Score answer quality with deepeval, one pass per answer rubric (``scoring.rubric_for``).

    The rubrics take different metrics (see ``scoring.build_metrics``).
    deepeval's ``evaluate`` drives its own event loop, so it runs after all
    pipeline work rather than inside it.
    """
    paired = list(zip(goldens, retrieved, strict=True))
    judged: dict[str, _Judged] = {}
    for rubric in RUBRICS:
        group = [(g, r) for g, r in paired if rubric_for(g.additional_metadata or {}) == rubric]
        if group:
            judged.update(_judge_group(group, plan, rubric=rubric))
    return judged


def _judge_group(group: list[tuple["Golden", Retrieved]], plan: RunPlan, *, rubric: str) -> dict[str, _Judged]:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

    from .scoring import build_metrics, to_test_case

    goldens = [g for g, _ in group]
    cases = [to_test_case(g, r) for g, r in group]
    result = evaluate(
        test_cases=cases,
        metrics=build_metrics(plan.judge, rubric=rubric),
        async_config=AsyncConfig(max_concurrent=plan.max_concurrent),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        # One unscoreable question must not abandon the batch.
        error_config=ErrorConfig(ignore_errors=True),
    )

    judged: dict[str, _Judged] = {}
    # Matched by name, never position: deepeval returns results in completion order.
    by_name = {test_result.name: test_result for test_result in result.test_results}
    for golden in goldens:
        test_result = by_name.get(golden.name)
        if test_result is None:
            continue
        # run_batch has already rejected nameless goldens.
        assert golden.name is not None
        metrics = test_result.metrics_data or []
        judged[golden.name] = _Judged(
            scores={m.name: m.score for m in metrics if m.score is not None},
            reasons={m.name: m.reason for m in metrics if m.reason},
        )
    return judged
