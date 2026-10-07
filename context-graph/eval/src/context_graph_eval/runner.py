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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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

#: "graph-agent" (default, #300's existing baseline: an agent writes its own
#: Cypher against the reconciled memory) or "text-search" (this eval's cheaper
#: comparison point: Memgraph's own full-text index over raw, unreconciled
#: turns -- see text_search.py). Deliberately the one axis report.compare()
#: does NOT pin: it is usually the thing a run using this field is measuring.
RetrievalStrategyName = Literal["graph-agent", "text-search", "hybrid"]
RETRIEVAL_STRATEGIES: tuple[RetrievalStrategyName, ...] = ("graph-agent", "text-search", "hybrid")


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
    #: False skips each session's summary LLM call; recall never reads Episodes.
    summaries: bool = True
    #: "fixed" extracts against the LongMemEval vocabulary; "learned" under each
    #: user's adopted version, derived once per user after reconciling (learned.py).
    ontology: str = "fixed"
    derive_seeds: int = 1
    derive_workers: int = 4
    max_concurrent: int = 4
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    #: LongMemEval's own judge (official_judge.py), which decides ``covered``
    #: when set; None leaves the deepeval answer rubrics deciding it.
    official_judge_model: str | None = None
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
    #: "gliner2" (default) or "lightrag" -- see reconcile.EXTRACTION_BACKENDS
    #: and reconcile_batch's docstring for what each implies. Meaningless when
    #: retrieval_strategy="text-search" (nothing reconciles), and ignored then.
    extraction_backend: ExtractionBackendName = "gliner2"
    #: "graph-agent" (default) or "text-search" -- see RETRIEVAL_STRATEGIES
    #: above. "text-search" forces reconciliation off regardless of
    #: ``reconcile`` above: the whole point of that baseline is to skip the
    #: dominant cost, and running reconciliation anyway would just discard its
    #: output unused.
    retrieval_strategy: RetrievalStrategyName = "graph-agent"
    #: How many text-search hits to hand the answering LLM. Ignored for
    #: "graph-agent". See text_search.DEFAULT_LIMIT for why this is not tuned.
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
    #: Turns text_search.ensure_turn_text_index indexed.
    #: Always 0 for retrieval_strategy="graph-agent", which never calls it.
    indexed_turns: int = 0


def _require_reconciled(fixtures: list, *, graph: "ActionsGraph", extraction_backend: ExtractionBackendName) -> None:
    """Refuse to reuse a graph that cannot answer the questions about to be run.

    Reuse exists to skip the dominant cost (#322), but it hands the run a graph
    nobody just built, so the ways it can be wrong are all silent. Each would
    score every affected question as a recall miss and report it as an
    ordinary result -- the same manufactured-zero shape as abstention questions
    judged on ContextualRecall, or a judge outage rendered as 0%.

    Missing sessions: the graph holds a different batch, or none.

    Present but unreconciled: injection ran without distillation, so there is no
    Chunk, Episode or entity to retrieve -- only the raw collection tier, which
    is a different system from the one under test.

    Reconciled by a different backend: ``extraction_backend`` names what this
    run is *claiming* built the graph (recorded on ``RunMeta`` so ``compare()``
    can refuse across a mismatch), but ``--skip-reconcile`` never actually
    builds anything -- without this check the claim was trusted, not verified.
    A LightRAG-built graph reused with ``--extraction-backend gliner2`` would
    silently save "gliner2" in ``RunMeta`` despite every entity in the graph
    coming from LightRAG, letting two runs that used the same real backend
    compare as a mismatch, or two that didn't compare as identical.
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

    # None (no reconcilable content, or a graph built before this property
    # existed) is not attributable to any backend -- excluded rather than
    # treated as a mismatch, the same reasoning enforce_retrieval_floor's
    # abstention exemption and aggregate()'s unscored rows already use
    # elsewhere in this package for "not applicable" vs. "wrong".
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
    # reuse_graph exists to skip reconciliation's dominant LLM cost (#322) --
    # text-search has no such cost to skip (indexing is deterministic and
    # cheap), and _require_reconciled's pending-status check is meaningless
    # for a strategy that never reconciles anything, so text-search always
    # re-injects rather than reusing.
    if plan.reuse_graph and plan.retrieval_strategy != "text-search":
        _require_reconciled(fixtures, graph=graph, extraction_backend=plan.extraction_backend)
    else:
        inject_batch(fixtures, graph=graph)

    reconciled = failures = indexed_turns = 0
    if plan.retrieval_strategy == "text-search":
        # No reconciliation regardless of plan.reconcile: this baseline's
        # whole point is to skip the dominant cost, and reconciling anyway
        # would just build memory this strategy never reads.
        indexed_turns = ensure_turn_text_index(graph).turns
    elif plan.reconcile:
        outcome = await reconcile_batch(
            graph.db,
            limit=plan.reconcile_limit,
            memgraph_url=plan.memgraph_url,
            extraction_backend=plan.extraction_backend,
            summaries=plan.summaries,
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


async def _retrieve_all(
    goldens: list["Golden"],
    graph: ReadOnlyGraph,
    llm: "LLM",
    plan: RunPlan,
) -> list[Retrieved]:
    """Retrieve for every question, bounded so a batch cannot stampede the model.

    A question whose retrieval raises becomes an empty result rather than
    propagating: a coverage rate computed over a silently shortened corpus is
    wrong, not merely noisy, so a failure has to be reported as a miss.
    """
    limiter = asyncio.Semaphore(plan.max_concurrent)

    async def one(golden: "Golden") -> Retrieved:
        async with limiter:
            started = time.monotonic()
            # When the question is asked -- the corpus's question_date, as a
            # real session's "now" -- without which "how many days ago" and
            # "this year" have nothing to count from (#367).
            today = (golden.additional_metadata or {}).get("question_date")
            try:
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
            except Exception as exc:
                # retrieve() times itself, but that timing rides out on the
                # Retrieved it returns -- a raise never produces one, so the
                # attempt is timed here too. A failure can have spent real
                # time (a slow model call that then errored) before raising;
                # reporting a hard-coded 0.0 there would make it look instant.
                return Retrieved(answer="", errors=[str(exc)], latency_seconds=time.monotonic() - started)

    return list(await asyncio.gather(*(one(golden) for golden in goldens)))


@dataclass(frozen=True)
class _Judged:
    """One question's judge output: the gating scores, and why.

    Kept as one unit rather than two parallel dicts so ``_judge``'s two-pass
    ``dict.update`` (ordinary questions, then abstention) cannot update scores
    for a question without also updating its reasons.
    """

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

    Efficiency, BLEU, F1 and latency are all computed regardless of whether a
    judge ran, so none of them wait on an LLM. Efficiency, BLEU and F1 are
    deterministic (#304); latency is not -- it is wall-clock time, which
    varies run to run -- but it needs no judge either, so it belongs in this
    same judge-free group despite not sharing that reason.
    """
    judged = _judge(goldens, retrieved, plan) if plan.judge is not None else {}

    scored: list[Scored] = []
    for golden, result in zip(goldens, retrieved, strict=True):
        metadata = golden.additional_metadata or {}
        outcome = judged.get(golden.name, _Judged())
        # The answer rubric's score -- the gate when LongMemEval's judge didn't
        # run; Contextual Recall rides along in metric_scores as the retrieval
        # signal (scoring.RETRIEVAL_SIGNAL).
        coverage = gate_score(outcome.scores)
        # LongMemEval's judge decides when it judged this question (#409); its
        # failed call is unjudged, never a fallback to our own rubrics.
        judged_by = "official" if official and golden.name in official else "answer"
        verdict = official.get(golden.name) if official and golden.name else None
        scored.append(
            Scored(
                name=golden.name or golden.input,
                tier=metadata.get("tier", 1),
                coverage=coverage,
                covered=bool(verdict) if judged_by == "official" else coverage >= plan.coverage_threshold,
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
                evidence_recall=(
                    None if metadata.get("abstention") else evidence_recall(golden.context, result.retrieval_context)
                ),
            )
        )
    # Applied after judging, not before: the per-metric scores are kept as the
    # judge reported them so attribution still shows what it thought, while the
    # gate stops an empty payload from counting as recall.
    return enforce_retrieval_floor(
        scored,
        retrieved_tokens={s.name: s.efficiency_tokens for s in scored},
    )


def _judge(goldens: list["Golden"], retrieved: list[Retrieved], plan: RunPlan) -> dict[str, _Judged]:
    """Score answer quality with deepeval, returning per-metric scores per question.

    Each answer rubric is judged in its **own pass** (``scoring.rubric_for``),
    because the metrics differ: ContextualRecall is structurally inapplicable
    to an abstention question, whose correct retrieved context is empty, and a
    preference question's expected output is a rubric rather than facts (see
    ``scoring.build_metrics``). Scoring them together made every abstention
    question unpassable.

    deepeval's ``evaluate`` is synchronous and drives its own event loop, so it
    runs after all pipeline work rather than inside it.
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
        # One question the judge cannot score should not abandon the batch --
        # the same reasoning reconciliation uses for an undistillable session.
        error_config=ErrorConfig(ignore_errors=True),
    )

    judged: dict[str, _Judged] = {}
    # Matched by name, never by position: deepeval returns test results in
    # completion order under async concurrency, so zipping them with the input
    # order handed every question another question's scores -- verified with a
    # fake metric of random latency, and visible in saved runs as Contextual
    # Recall reasons quoting a different question's expected answer.
    by_name = {test_result.name: test_result for test_result in result.test_results}
    for golden in goldens:
        test_result = by_name.get(golden.name)
        if test_result is None:
            continue
        # Kept per metric, not collapsed: the answer rubric decides the gate,
        # and the retrieval signal beside it tells you whether retrieval or the
        # answer was at fault -- #304 pointed out that attribution is free here.
        # run_batch has already rejected nameless goldens; asserted rather than
        # re-checked so the type narrows and the invariant stays stated once.
        assert golden.name is not None
        metrics = test_result.metrics_data or []
        # deepeval already generates `reason` alongside every score -- for
        # ContextualRecallMetric it is itself a synthesis of that metric's
        # per-sentence supported/unsupported verdicts (see contextual_recall.py
        # in deepeval) -- so keeping it costs no extra judge call. Previously
        # discarded here, which meant a failed question said only "0.4" with no
        # way to tell whether retrieval or the answer was at fault without
        # rerunning by hand.
        judged[golden.name] = _Judged(
            scores={m.name: m.score for m in metrics if m.score is not None},
            reasons={m.name: m.reason for m in metrics if m.reason},
        )
    return judged
