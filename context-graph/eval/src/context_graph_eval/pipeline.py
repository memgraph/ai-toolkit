"""The one pipeline every benchmark runs through: build the graph, then ask it.

    build:  inject -> reconcile -> (derive) -> embed      once per construction
    ask:    recall -> answer -> benchmark's judge         every run

Building is where a run spends its hours, so a built graph is kept on the
workload's instance with a manifest (an ``(:EvalBuild)`` node) recording
what built it: a fingerprint over the workload's questions, the build
settings, and the source of every module that shapes the graph. The next run
reuses the graph when the fingerprint matches, resumes it when a build of
the same fingerprint was interrupted, and rebuilds from empty otherwise --
so a recall change re-asks in minutes and a construction change rebuilds
without anyone deciding which it was.

Ordering is the pipeline's real responsibility: asking before injection
scores every question a miss, and asking before reconciliation scores raw
turns rather than emerged memory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .answering import (
    LLM,
    ReadOnlyGraph,
    RecallConfig,
    Retrieved,
    answer_all,
    ensure_recall_ready,
    retrieve,
    setup_with_retry,
)
from .inject import PENDING, inject, wipe
from .reconcile import EXTRACTION_BACKEND, ONTOLOGIES, reconcile_pending
from .results import Run, RunMeta, merge
from .workloads import Workload, shards

if TYPE_CHECKING:
    from deepeval.dataset import Golden

    from memgraph_toolbox.api.memgraph import Memgraph

    from .benchmarks import Benchmark
    from .convert.longmemeval import SessionFixture

MANIFEST_LABEL = "EvalBuild"

#: Modules whose source decides what a build writes. A change to any of them
#: rebuilds; a change anywhere else (recall, answering, judging) doesn't.
#: Packages are hashed whole; for sessions-graph only its construction
#: modules, since recall lives there too.
_CONSTRUCTION_PACKAGES = ("actions_graph", "unstructured2graph", "hygm")
_CONSTRUCTION_MODULES = (
    "sessions_graph.core",
    "sessions_graph.reconciler",
    "sessions_graph.reconciliation",
    "sessions_graph.extraction",
    "sessions_graph.embeddings",
    "sessions_graph.passages",
    "sessions_graph.ontology",
    "sessions_graph.derivation",
    "sessions_graph.derive_llm",
    "sessions_graph.models",
    "context_graph_eval.inject",
    "context_graph_eval.reconcile",
    "context_graph_eval.learned",
    "context_graph_eval.convert.longmemeval",
    "context_graph_eval.convert.beam",
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


@dataclass(frozen=True)
class BuildPlan:
    """How a graph is built, beyond the workload itself."""

    #: "default": hygm's default model; "learned": then each user's own, derived and adopted (map #431).
    ontology: str = "default"
    derive_seeds: int = 1
    derive_workers: int = 4
    #: Sessions extracted at once.
    concurrency: int = 4

    def __post_init__(self) -> None:
        if self.ontology not in ONTOLOGIES:
            raise ValueError(f"ontology must be one of {ONTOLOGIES}, got {self.ontology!r}")


def construction_code() -> str:
    """A hash of the source of every module that shapes the graph (see ``_CONSTRUCTION_MODULES``)."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for package in _CONSTRUCTION_PACKAGES:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            files.extend(sorted(Path(location).rglob("*.py")))
            files.extend(sorted(Path(location).rglob("*.yaml")))
    for module in _CONSTRUCTION_MODULES:
        spec = importlib.util.find_spec(module)
        if spec is not None and spec.origin:
            files.append(Path(spec.origin))
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_inputs(workload: Workload, shard: int | None, questions: list[Golden], plan: BuildPlan) -> dict[str, Any]:
    """Everything a build's graph depends on; two builds with equal inputs write the same graph."""
    from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL
    from sessions_graph.passages import PASSAGE_SCHEME

    from unstructured2graph import gliner2_backend

    names = hashlib.sha256("\n".join(sorted(q.name or q.input for q in questions)).encode()).hexdigest()[:16]
    return {
        "workload": workload.name,
        "shard": shard,
        "questions": names,
        "revision": workload.revision,
        "max_sessions_per_question": workload.max_sessions_per_question,
        "ontology": plan.ontology,
        "derive_seeds": plan.derive_seeds if plan.ontology == "learned" else None,
        "gliner2_checkpoint": "fastino/gliner2.5-base-v1",
        "gliner2_decoder": gliner2_backend.DECODER,
        "gliner2_candidate_cap": gliner2_backend.DEFAULT_CANDIDATE_CAP,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "passage_scheme": PASSAGE_SCHEME,
        "construction_code": construction_code(),
    }


def fingerprint(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Manifest:
    """What built the graph an instance holds."""

    fingerprint: str
    inputs: dict[str, Any]
    #: "building" until every stage finished; a crash leaves it there.
    status: str
    commit: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    #: Summed over every attempt that built this fingerprint, resumed ones included.
    build_seconds: float = 0.0


def read_manifest(db: Memgraph) -> Manifest | None:
    rows = db.query(f"MATCH (b:{MANIFEST_LABEL}) RETURN properties(b) AS b LIMIT 1")
    if not rows:
        return None
    properties = rows[0]["b"]
    return Manifest(
        fingerprint=properties["fingerprint"],
        inputs=json.loads(properties.get("inputs") or "{}"),
        status=properties.get("status", "building"),
        commit=properties.get("commit"),
        started_at=properties.get("started_at"),
        completed_at=properties.get("completed_at"),
        build_seconds=float(properties.get("build_seconds") or 0.0),
    )


def _write_manifest(db: Memgraph, manifest: Manifest) -> None:
    # One manifest per instance: an instance holds one build.
    db.query(
        f"MERGE (b:{MANIFEST_LABEL}) SET b.fingerprint = $fingerprint, b.inputs = $inputs, b.status = $status, "
        "b.commit = $commit, b.started_at = $started_at, b.completed_at = $completed_at, "
        "b.build_seconds = $build_seconds",
        {**asdict(manifest), "inputs": json.dumps(manifest.inputs)},
    )


@dataclass(frozen=True)
class Decision:
    """What a run does with the graph it finds, and why."""

    action: Literal["reuse", "resume", "build"]
    reason: str


def decide(
    manifest: Manifest | None, inputs: dict[str, Any], *, rebuild: bool = False, reuse: bool = False
) -> Decision:
    """Reuse, resume or rebuild the instance's graph for a build with ``inputs``.

    ``reuse`` keeps a finished graph whose construction inputs changed (the
    run records it as stale); it never reuses a graph of other questions.
    """
    if rebuild:
        return Decision("build", "--rebuild")
    if manifest is None:
        return Decision("build", "no build on this instance yet")
    if manifest.fingerprint == fingerprint(inputs):
        if manifest.status == "complete":
            return Decision(
                "reuse", f"built at {manifest.commit or 'an unknown commit'}, nothing that shapes it changed"
            )
        return Decision("resume", "an interrupted build of the same inputs")
    changed = sorted(
        key for key in inputs.keys() | manifest.inputs.keys() if inputs.get(key) != manifest.inputs.get(key)
    )
    same_questions = all(inputs.get(key) == manifest.inputs.get(key) for key in ("workload", "shard", "questions"))
    if reuse and same_questions and manifest.status == "complete":
        return Decision("reuse", f"--reuse, though {', '.join(changed)} changed since it was built")
    return Decision("build", f"{', '.join(changed)} changed since the last build")


def require_built(db: Memgraph, fixtures: list[SessionFixture]) -> None:
    """Refuse to ask a graph that cannot answer: sessions missing, still pending, or built by another backend.

    Every way a kept graph can be wrong is otherwise silent, scoring as
    ordinary recall misses.

    Raises:
        ValueError: on any of the three.
    """
    wanted = {fixture.session_id for fixture in fixtures}
    if not wanted:
        return
    rows = db.query(
        "MATCH (s:Session) WHERE s.session_id IN $ids "
        "RETURN s.session_id AS session_id, s.reconciliation_status AS status, "
        "s.extraction_backend AS extraction_backend",
        {"ids": sorted(wanted)},
    )
    found = {row["session_id"]: row for row in rows}
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"the graph lacks {len(missing)} of {len(wanted)} sessions (e.g. {missing[:3]}); rebuild it")
    pending = sorted(sid for sid, row in found.items() if row["status"] == PENDING)
    if pending:
        raise ValueError(
            f"{len(pending)} of {len(wanted)} sessions are still pending reconciliation (e.g. {pending[:3]})"
        )
    # None means no reconcilable content, which no backend is to blame for.
    foreign = sorted(sid for sid, row in found.items() if row["extraction_backend"] not in (None, EXTRACTION_BACKEND))
    if foreign:
        raise ValueError(f"{len(foreign)} sessions were reconciled by another extraction backend (e.g. {foreign[:3]})")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def build(
    db: Memgraph,
    *,
    fixtures: list[SessionFixture],
    inputs: dict[str, Any],
    plan: BuildPlan,
    decision: Decision,
    memgraph_url: str,
    commit: str | None = None,
) -> Manifest:
    """Build (or finish building) the graph ``fixtures`` make, and record it in the manifest.

    ``memgraph_url`` must name the instance ``db`` is connected to: a learned
    build derives each user in a process of its own, which connects by URL.
    """
    from actions_graph import ActionsGraph

    graph = ActionsGraph(memgraph=db)
    previous = read_manifest(db) if decision.action == "resume" else None
    if decision.action == "build":
        wipe(graph)
    manifest = Manifest(
        fingerprint=fingerprint(inputs),
        inputs=inputs,
        status="building",
        commit=commit,
        started_at=previous.started_at if previous else _now(),
        build_seconds=previous.build_seconds if previous else 0.0,
    )
    _write_manifest(db, manifest)

    started = time.monotonic()
    setup_with_retry(graph)
    written = inject(fixtures, graph=graph)
    print(f"injected {written.sessions} sessions ({written.turns} turns), {written.kept} already there", flush=True)
    outcome = await reconcile_pending(db, concurrency=plan.concurrency)
    print(f"reconciled {outcome.reconciled} sessions, {outcome.failed} failed", flush=True)
    for error in outcome.errors[:5]:
        print(f"  {error}", flush=True)
    if plan.ontology == "learned":
        from .learned import derive_users

        derived = await derive_users(
            db, memgraph_url=memgraph_url, seeds=plan.derive_seeds, workers=plan.derive_workers
        )
        print(f"derivation: {derived.counts()}", flush=True)
    embedded = ensure_recall_ready(db)
    print(f"embedded {embedded} sessions", flush=True)

    manifest = replace(
        manifest,
        status="complete",
        completed_at=_now(),
        build_seconds=manifest.build_seconds + time.monotonic() - started,
    )
    _write_manifest(db, manifest)
    return manifest


async def ask(
    db: Memgraph,
    benchmark: Benchmark,
    questions: list[Golden],
    *,
    llm: LLM,
    recall: RecallConfig,
    max_concurrent: int = 4,
) -> list[Retrieved]:
    """Recall and answer every question from its own user's history, at most ``max_concurrent`` at once."""
    ensure_recall_ready(db)
    graph = ReadOnlyGraph(db)

    async def one(question: Golden) -> Retrieved:
        return await retrieve(
            question.input,
            graph=graph,
            llm=llm,
            user_id=benchmark.user_id(question),
            config=recall,
            today=benchmark.today(question),
        )

    return await answer_all(questions, one, max_concurrent=max_concurrent)


@dataclass(frozen=True)
class RunSettings:
    """What a run asks with and records, beyond the workload."""

    label: str
    agent_model: str
    recall: RecallConfig = field(default_factory=RecallConfig)
    lanes: str = ""
    changed: str = ""
    plan: BuildPlan = field(default_factory=BuildPlan)
    max_concurrent: int = 4
    rebuild: bool = False
    reuse: bool = False
    #: Build the graph and stop: no questions asked, nothing judged.
    build_only: bool = False
    rubric_judge_model: str = "none"
    commit: str | None = None


@dataclass(frozen=True)
class Connection:
    """How a run reaches its instance; the CLI starts the workload's own, tests pass one in."""

    db: Memgraph
    url: str


async def run(
    workload: Workload,
    benchmark: Benchmark,
    *,
    connection: Connection,
    llm: LLM | None,
    settings: RunSettings,
    shard_dir: Path | None = None,
) -> Run | None:
    """Run ``workload`` end to end: one graph, or one shard at a time when it doesn't fit in memory.

    A shard whose result is already in ``shard_dir`` is skipped, so an
    interrupted official run resumes where it stopped. Returns None when
    ``settings.build_only``.

    Raises:
        ValueError: a sharded workload asked to build only, or a planted
            workload whose fact isn't in the graph.
    """
    check_offline()
    questions = benchmark.questions(workload)
    parts = shards(workload, questions)
    if settings.build_only and len(parts) > 1:
        raise ValueError(f"{workload.name} is built one shard at a time, as `run` asks it; it can't be built alone")

    results: list[Run] = []
    for index, part in enumerate(parts):
        shard = index if len(parts) > 1 else None
        saved = shard_dir / f"shard-{index:02d}.json" if shard_dir is not None and shard is not None else None
        if saved is not None and saved.exists():
            from .results import load

            print(f"shard {index + 1}/{len(parts)}: already done ({saved})", flush=True)
            results.append(load(saved))
            continue
        if shard is not None:
            print(f"shard {index + 1}/{len(parts)}: {len(part)} questions", flush=True)
        result = await _run_part(workload, benchmark, part, shard, connection=connection, llm=llm, settings=settings)
        if result is None:
            return None
        if saved is not None:
            from .results import save

            save(result, saved)
        results.append(result)
    return merge(results, label=settings.label) if len(results) > 1 else results[0]


async def _run_part(
    workload: Workload,
    benchmark: Benchmark,
    questions: list[Golden],
    shard: int | None,
    *,
    connection: Connection,
    llm: LLM | None,
    settings: RunSettings,
) -> Run | None:
    db = connection.db
    build_meta: dict[str, Any] = {"shard": shard}
    if workload.planted:
        _require_planted(db)
    else:
        fixtures = benchmark.fixtures(workload, questions)
        inputs = build_inputs(workload, shard, questions, settings.plan)
        decision = decide(read_manifest(db), inputs, rebuild=settings.rebuild, reuse=settings.reuse)
        print(f"{workload.instance}: {decision.action} -- {decision.reason}", flush=True)
        if decision.action == "reuse":
            manifest = read_manifest(db)
            assert manifest is not None
        else:
            manifest = await build(
                db,
                fixtures=fixtures,
                inputs=inputs,
                plan=settings.plan,
                decision=decision,
                memgraph_url=connection.url,
                commit=settings.commit,
            )
            print(f"built in {manifest.build_seconds / 60:.1f} min", flush=True)
        require_built(db, fixtures)
        build_meta.update(
            fingerprint=manifest.fingerprint,
            current=manifest.fingerprint == fingerprint(inputs),
            inputs=manifest.inputs,
            commit=manifest.commit,
            completed_at=manifest.completed_at,
            build_seconds=manifest.build_seconds,
        )
    if settings.build_only:
        return None
    if llm is None:
        raise ValueError("asking needs an answering model")

    retrieved = await ask(
        db, benchmark, questions, llm=llm, recall=settings.recall, max_concurrent=settings.max_concurrent
    )
    rows = await benchmark.judge(questions, retrieved)
    return Run(
        meta=RunMeta(
            label=settings.label,
            workload=workload.name,
            benchmark=workload.benchmark,
            revision=workload.revision,
            questions=len(rows),
            judge_model=benchmark.judge_model,
            agent_model=settings.agent_model,
            changed=settings.changed,
            hybrid_lanes=settings.lanes,
            packing=settings.recall.packing,
            build=build_meta,
            commit=settings.commit,
            memgraph_version=_memgraph_version(db),
            rubric_judge_model=settings.rubric_judge_model,
        ),
        rows=rows,
    )


def _require_planted(db: Memgraph) -> None:
    from actions_graph import ActionsGraph

    from .goldslice import evidence_is_planted

    if not evidence_is_planted(ActionsGraph(memgraph=db)):
        raise ValueError(
            "the gold slice's fact is not in the graph, so asking would score a guaranteed miss. "
            "Plant it first with `context-graph-eval gold-slice`."
        )


def _memgraph_version(db: Memgraph) -> str | None:
    try:
        return db.query("SHOW VERSION")[0]["version"]
    except Exception:
        return None
