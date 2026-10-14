"""Context Graph eval: build the corpus, run and compare batches, calibrate the judge, run BEAM."""

import argparse
import os
import re
import sys
from pathlib import Path

from .beam_judge import BEAM_JUDGE_MODEL
from .convert.beam import CHAT_COUNTS as BEAM_CHAT_COUNTS
from .convert.beam import DEFAULT_REVISION as BEAM_REVISION
from .convert.longmemeval import DEFAULT_REVISION, build_corpus, fetch, haystack_path, load_raw
from .corpus import write_corpus
from .hybrid import LANES as HYBRID_LANES
from .hybrid import RecallConfig
from .official_judge import OFFICIAL_JUDGE_MODEL
from .reconcile import EXTRACTION_BACKENDS
from .runner import RETRIEVAL_STRATEGIES
from .text_search import DEFAULT_LIMIT as DEFAULT_TEXT_SEARCH_LIMIT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-graph-eval", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser(
        "build-corpus",
        help="fetch a pinned LongMemEval release and write the Tier 1 corpus",
    )
    build.add_argument(
        "--variant",
        default="s",
        help="LongMemEval variant: 's' (~40 sessions) or 'm' (~500). 'oracle' is refused: "
        "it has no distractor sessions, so retrieval would score well by construction.",
    )
    build.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="pinned upstream revision; changing it invalidates prior baselines",
    )
    build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample down to this many questions, stratified by question type",
    )
    build.add_argument(
        "--out",
        type=Path,
        default=Path("corpus/tier1-longmemeval.jsonl"),
        help="where to write the corpus JSONL",
    )

    run = subcommands.add_parser("run", help="run an eval batch end to end and print the report")
    run.add_argument("--variant", default="s", help="LongMemEval variant the corpus was built from")
    run.add_argument("--revision", default=DEFAULT_REVISION, help="pinned upstream revision")
    run.add_argument("--limit", type=int, default=20, help="how many questions to run")
    run.add_argument(
        "--corpus",
        type=Path,
        default=Path("context-graph/eval/corpus/tier1-longmemeval.jsonl"),
        help="the committed corpus to run, so compared runs provably ask the same questions (#302)",
    )
    run.add_argument(
        "--gold-slice",
        action="store_true",
        help="also run Tier 2's gold-slice questions, scored separately from Tier 1",
    )
    run.add_argument(
        "--memgraph-url",
        default="bolt://localhost:7689",
        help="the DEDICATED eval instance. It is wiped before each batch -- never point this at a "
        "shared or development database.",
    )
    run.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="reuse the already-reconciled graph: no wipe, injection or distillation. Refuses unless "
        "the graph holds this run's sessions, reconciled by --extraction-backend.",
    )
    run.add_argument(
        "--official-judge-model",
        default=OFFICIAL_JUDGE_MODEL,
        help=f"OpenAI model for LongMemEval's own judge, which decides coverage (default {OFFICIAL_JUDGE_MODEL}, "
        "upstream's). 'none' leaves the deepeval answer rubrics deciding it.",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        help="'provider:model_id' for the deepeval judge, e.g. 'anthropic:claude-sonnet-4-5-20250929'. "
        "A bare model id keeps the default provider (anthropic); sharing --agent-model's provider is "
        "warned about (#304). Omit to skip judging.",
    )
    run.add_argument(
        "--agent-model",
        default=None,
        help="'provider:model_id' for the retrieval agent, e.g. 'openai:gpt-4o'. A bare model id "
        "keeps the default provider (openai).",
    )
    run.add_argument(
        "--extraction-backend",
        choices=EXTRACTION_BACKENDS,
        default="gliner2",
        help="entity extraction for reconciliation: 'gliner2' (default; local, LLM-free) or 'lightrag' (needs an LLM key).",
    )
    _learned_args(run)
    run.add_argument(
        "--retrieval-strategy",
        choices=RETRIEVAL_STRATEGIES,
        default="graph-agent",
        help="'hybrid' (sessions-graph's recall: turns by vector and text search plus the typed graph), "
        "'graph-agent' (an agent writes Cypher against the reconciled graph), or 'text-search' "
        "(full-text over raw turns; never reconciles).",
    )
    run.add_argument(
        "--hybrid-lanes",
        default=",".join(HYBRID_LANES),
        help=f"comma-separated lanes for --retrieval-strategy hybrid, from {', '.join(HYBRID_LANES)}. "
        "Dropping 'entities' and 'facts' measures what the graph adds over turns alone.",
    )
    run.add_argument(
        "--text-search-limit",
        type=int,
        default=DEFAULT_TEXT_SEARCH_LIMIT,
        help="how many text-search hits to hand the answering LLM (text-search only)",
    )
    run.add_argument(
        "--max-sessions-per-question",
        type=int,
        default=None,
        help="trim each question's distractor sessions (evidence is always kept). The score is an "
        "upper bound, not comparable to a full-haystack run.",
    )
    run.add_argument("--save", type=Path, default=None, help="persist this run so it can be compared later")
    run.add_argument("--label", default="run", help="name for this run in a later comparison")
    run.add_argument("--changed", default="", help="what this run changed, shown in the comparison report")

    gold = subcommands.add_parser(
        "gold-slice",
        help="plant the gold-slice fixture by driving a REAL, billed Claude Code session",
    )
    gold.add_argument("--memgraph-url", default="bolt://localhost:7689", help="the dedicated eval instance")
    gold.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="working directory for the session; hook commands resolve `uv run` against it",
    )
    gold.add_argument(
        "--keep",
        action="store_true",
        help="do not wipe first, so the fact is planted among already-loaded Tier 1 distractors",
    )

    cal = subcommands.add_parser(
        "calibrate",
        help="derive a noise floor from repeated runs of the same questions (#304)",
    )
    cal.add_argument("runs", type=Path, nargs="+", help="two or more saved runs of the SAME questions")

    cmp_ = subcommands.add_parser("compare", help="compare two saved runs and print the report")
    cmp_.add_argument("baseline", type=Path)
    cmp_.add_argument("candidate", type=Path)
    cmp_.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="coverage noise floor in percentage points, from `calibrate`. Without it no delta is called real.",
    )

    beam = subcommands.add_parser(
        "beam",
        help="run BEAM's probing questions through the hybrid read path, judged by BEAM's own judge",
    )
    beam.add_argument("--size", default="100K", choices=sorted(BEAM_CHAT_COUNTS), help="BEAM chat size")
    beam.add_argument(
        "--chats",
        default="1",
        help="which chats of that size, e.g. '1', '1,3' or '1-20'; each is one user's history",
    )
    beam.add_argument("--revision", default=BEAM_REVISION, help="pinned upstream BEAM commit")
    beam.add_argument(
        "--memgraph-url",
        default="bolt://localhost:7689",
        help="the DEDICATED eval instance. It is wiped before each batch -- never point this at a "
        "shared or development database.",
    )
    beam.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="reuse the already-reconciled graph: no wipe, no injection, no distillation. Refuses "
        "unless the graph holds these chats' sessions, reconciled.",
    )
    beam.add_argument("--extraction-backend", choices=EXTRACTION_BACKENDS, default="lightrag")
    _learned_args(beam)
    beam.add_argument(
        "--hybrid-lanes",
        default=",".join(HYBRID_LANES),
        help=f"comma-separated recall lanes, from {', '.join(HYBRID_LANES)}",
    )
    beam.add_argument(
        "--agent-model",
        default=None,
        help="'provider:model_id' for the answering model; the same default as `run`'s.",
    )
    beam.add_argument(
        "--judge-model",
        default=BEAM_JUDGE_MODEL,
        help=f"OpenAI model for BEAM's judge (default {BEAM_JUDGE_MODEL}, upstream's). 'none' skips judging.",
    )
    beam.add_argument(
        "--limit",
        type=int,
        default=None,
        help="ask only this many questions, balanced across abilities and chats; every chat is still loaded",
    )
    beam.add_argument("--max-concurrent", type=int, default=4)
    beam.add_argument("--save", type=Path, default=None, help="write every row and the report to this JSON file")
    beam.add_argument("--label", default="beam", help="name recorded in the saved run")

    kg = subcommands.add_parser(
        "kg",
        help="measure the default knowledge-graph construction on BEAM, against the committed baseline",
    )
    kg.add_argument("--size", default="100K", choices=sorted(BEAM_CHAT_COUNTS), help="BEAM chat size")
    kg.add_argument("--chats", default="1-20", help="which chats, e.g. '1-20' (the baseline's)")
    kg.add_argument("--revision", default=BEAM_REVISION, help="pinned upstream BEAM commit")
    kg.add_argument("--windows", type=int, default=200, help="windows sampled, evenly spread (default 200)")
    kg.add_argument("--observe", action="store_true", help="also time the open-endpoint pass derivation runs")
    kg.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="compare against this saved run (the committed one: context-graph/eval/baselines/kg-beam-100k.json)",
    )
    kg.add_argument("--save", type=Path, default=None, help="write the run (metrics and fingerprints) here")
    kg.add_argument("--label", default="kg", help="name recorded in the saved run")

    args = parser.parse_args(argv)

    if args.command == "kg":
        return _kg(args)
    if args.command == "build-corpus":
        return _build_corpus(args)
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "gold-slice":
        return _gold_slice(args)
    if args.command == "beam":
        return _beam(args)
    return 1


def parse_chat_ids(spec: str) -> list[int]:
    """Chat ids from '1', '1,3' or '1-20' (inclusive), in order, without repeats.

    Raises:
        ValueError: anything else.
    """
    ids: list[int] = []
    for part in spec.split(","):
        start, _, end = part.strip().partition("-")
        ids.extend(range(int(start), int(end or start) + 1))
    return list(dict.fromkeys(ids))


def _beam(args) -> int:
    import asyncio

    from actions_graph import ActionsGraph
    from memgraph_toolbox.api.memgraph import Memgraph

    from .beam import render, run_beam, save
    from .beam_judge import BeamJudge
    from .convert.beam import fetch_chat
    from .reconcile import _resolve_llm_credentials, _resolve_reconciliation_tuning
    from .retrieval import DeepEvalLLM
    from .runner import check_offline

    check_offline()
    _resolve_llm_credentials()
    _resolve_reconciliation_tuning()

    chat_ids = parse_chat_ids(args.chats)
    chats = [fetch_chat(args.size, chat_id, revision=args.revision) for chat_id in chat_ids]
    total = sum(len(group) for chat in chats for group in chat.probing_questions.values())
    questions = min(total, args.limit) if args.limit is not None else total
    print(f"running {questions} of {total} questions over {len(chats)} BEAM {args.size} chats @ {args.revision[:12]}")

    agent_provider, agent_model_id = _parse_model_spec(args.agent_model, default_provider=DEFAULT_AGENT_PROVIDER)
    agent = _build_model(agent_provider, agent_model_id)
    if agent is None:
        print("no agent model configured: set --agent-model or an OPENAI_API_KEY", file=sys.stderr)
        return 1
    judge = None
    if args.judge_model != "none":
        if not os.environ.get("OPENAI_API_KEY"):
            print("BEAM's judge needs an OPENAI_API_KEY; pass --judge-model none to skip judging", file=sys.stderr)
            return 1
        judge = BeamJudge(model=args.judge_model, max_concurrent=args.max_concurrent * 2)

    db = Memgraph(url=args.memgraph_url, username="", password="")
    run = asyncio.run(
        run_beam(
            chats,
            graph=ActionsGraph(memgraph=db),
            llm=DeepEvalLLM(agent),
            judge=judge,
            memgraph_url=args.memgraph_url,
            reuse_graph=args.skip_reconcile,
            extraction_backend=args.extraction_backend,
            hybrid=RecallConfig.from_mapping({"lanes": args.hybrid_lanes}),
            max_concurrent=args.max_concurrent,
            limit=args.limit,
            ontology=args.ontology,
            derive_seeds=args.derive_seeds,
            derive_workers=args.derive_workers,
        )
    )
    print(render(run))

    if args.save:
        saved = save(
            run,
            args.save,
            meta={
                "label": args.label,
                "benchmark": f"beam-{args.size}",
                "revision": args.revision,
                "chats": chat_ids,
                "limit": args.limit,
                "judge_model": args.judge_model,
                "agent_model": _resolved_spec(agent_provider, agent_model_id),
                "extraction_backend": args.extraction_backend,
                "hybrid_lanes": args.hybrid_lanes,
                "ontology": args.ontology,
                "derive_seeds": args.derive_seeds if args.ontology == "learned" else None,
            },
        )
        print(f"\nsaved run to {saved}")
    return 0


def _kg(args) -> int:
    from .convert.beam import fetch_chat
    from .kg import compare as compare_kg
    from .kg import load, render, run_kg, save

    chat_ids = parse_chat_ids(args.chats)
    chats = [fetch_chat(args.size, chat_id, revision=args.revision) for chat_id in chat_ids]
    print(f"kg: {len(chats)} BEAM {args.size} chats @ {args.revision[:12]}, {args.windows} windows", flush=True)
    run = run_kg(
        chats,
        windows=args.windows,
        observe=args.observe,
        meta={
            "label": args.label,
            "benchmark": f"beam-{args.size}",
            "revision": args.revision,
            "chats": chat_ids,
            "windows_requested": args.windows,
        },
    )
    print(render(run))
    if args.baseline:
        print("\n" + "\n".join(compare_kg(run, load(args.baseline))))
    if args.save:
        print(f"\nsaved run to {save(run, args.save)}")
    return 0


def _learned_args(parser) -> None:
    """The reconcile options `run` and `beam` share: a learned ontology (map #431)."""
    parser.add_argument(
        "--ontology",
        choices=("fixed", "default", "learned"),
        default="fixed",
        help="'fixed': extract against the LongMemEval vocabulary. 'default': extract under hygm's default "
        "model, nothing derived. 'learned': the same, then derive and adopt each user's own model "
        "(sessions-graph derive) before answering. The last two need GLiNER2.",
    )
    parser.add_argument("--derive-seeds", type=int, default=1, help="candidates per user derivation (learned)")
    parser.add_argument("--derive-workers", type=int, default=4, help="user derivations at once (learned)")


def _gold_slice(args) -> int:
    """Plant the gold-slice fixture with a real Claude Code session."""
    from actions_graph import ActionsGraph
    from memgraph_toolbox.api.memgraph import Memgraph

    from .goldslice import (
        GOLD_SLICE_FACT,
        GOLD_SLICE_PROMPT,
        evidence_is_nested,
        evidence_is_planted,
        evidence_is_top_level,
    )
    from .live import LiveSessionError, drive_session, hooks_pointed_at

    db = Memgraph(url=args.memgraph_url, username="", password="")
    graph = ActionsGraph(memgraph=db)

    if not args.keep:
        # Wipe BEFORE the session, never after: the whole point is that the
        # planted fact survives into the batch that scores it.
        db.query("MATCH (n) DETACH DELETE n")
        print("wiped the eval instance")

    print("driving a real Claude Code session (billed)...")
    try:
        with hooks_pointed_at(args.memgraph_url) as env:
            session_id, transcript = drive_session(GOLD_SLICE_PROMPT, repo_root=args.repo_root, env=env)
    except LiveSessionError as exc:
        print(f"gold slice: {exc}", file=sys.stderr)
        return 1

    print(f"session {session_id} complete")
    # Always shown: a zero exit does not mean the model delegated or hooks
    # recorded anything, and re-running to find out costs another billed session.
    print("--- session transcript ---")
    print(transcript.strip()[:2000])
    print("--- end transcript ---")

    # Not recall results: each decides whether a recall result would mean anything.
    if not evidence_is_planted(graph):
        print(
            "gold slice: the fact never reached the graph. Hooks may not be installed, or the "
            "session did not read the file. Nothing here is scoreable.",
            file=sys.stderr,
        )
        return 1

    if not evidence_is_nested(graph, session_id, GOLD_SLICE_FACT):
        print(
            "gold slice: the fact is in the graph but NOT inside a subagent. The model declined "
            "to delegate, so this run cannot test nested recall -- scoring it would pass or fail "
            "for the wrong reason. Re-run; delegation is model-decided and not deterministic.",
            file=sys.stderr,
        )
        return 1

    if evidence_is_top_level(graph, session_id, GOLD_SLICE_FACT):
        print(
            "gold slice: the fact is nested BUT also present at top level -- the subagent quoted "
            "it in its report, which is recorded as the parent's Task ToolResult. Retrieval could "
            "then answer without traversing HAS_AGENT, so this run tests nothing. Re-run; whether "
            "the subagent obeys 'do not quote the value' is model-decided.",
            file=sys.stderr,
        )
        return 1

    print("fixture planted, nested, and absent from top level. Score it with:")
    print(f"  context-graph-eval run --gold-slice --skip-reconcile --memgraph-url {args.memgraph_url}")
    return 0


def _calibrate(args) -> int:
    from .calibrate import describe, describe_per_question_rates, describe_stability, per_question_pass_rate
    from .report import load_run

    runs = [load_run(path) for path in args.runs]

    # Same questions, or the spread measures the corpus rather than the judge.
    question_sets = {tuple(sorted(s.name for s in run.scored)) for run in runs}
    if len(question_sets) > 1:
        print(
            "refusing to calibrate: these runs cover different questions, so their "
            "spread would measure the corpus change rather than judge variance.",
            file=sys.stderr,
        )
        return 1

    rates = [sum(1 for s in run.scored if s.covered) / len(run.scored) for run in runs if run.scored]
    passing_sets = [{s.name for s in run.scored if s.covered} for run in runs]
    all_names = {s.name for run in runs for s in run.scored}
    try:
        print(describe(rates))
        # Stability beside the floor, never instead: a tight floor over an
        # unstable passing set is the misleading case.
        print()
        print(describe_stability(passing_sets))
        print()
        print(describe_per_question_rates(per_question_pass_rate(passing_sets, all_names)))
    except ValueError as exc:
        print(f"refusing to calibrate: {exc}", file=sys.stderr)
        return 1
    return 0


def _compare(args) -> int:
    from .report import compare, load_run, render

    try:
        comparison = compare(
            load_run(args.baseline),
            load_run(args.candidate),
            noise_floor_pp=args.noise_floor,
        )
    except ValueError as exc:
        print(f"refusing to compare: {exc}", file=sys.stderr)
        return 1

    print(render(comparison))
    return 0


def _build_corpus(args) -> int:
    raw = fetch(args.variant, args.revision, dest=haystack_path(args.variant, args.revision))
    records = load_raw(raw)
    print(f"fetched {len(records)} records from longmemeval-{args.variant} @ {args.revision[:12]}")

    goldens = build_corpus(records, limit=args.limit)
    written = write_corpus(goldens, args.out)
    print(f"wrote {written} goldens to {args.out}")
    return 0


def select_goldens(corpus: list, *, limit: int | None, gold_slice: bool) -> list:
    """The first ``limit`` corpus questions, plus the gold slice when asked.

    The gold slice is appended rather than counted against ``limit``: it is
    Tier 2, scored apart, and a Tier 1 limit must not trim it away.
    """
    selected = corpus[:limit] if limit is not None else list(corpus)
    if gold_slice:
        from .goldslice import gold_slice_goldens

        selected += gold_slice_goldens()
    return selected


def _run(args) -> int:
    import asyncio

    from actions_graph import ActionsGraph
    from memgraph_toolbox.api.memgraph import Memgraph

    from .corpus import read_corpus
    from .goldslice import evidence_is_planted
    from .reconcile import _resolve_llm_credentials, _resolve_reconciliation_tuning
    from .retrieval import DeepEvalLLM
    from .runner import RunPlan, check_offline, run_batch

    check_offline()
    _resolve_llm_credentials()
    # Before lightrag is first imported: it reads its tuning env vars once, as dataclass defaults.
    _resolve_reconciliation_tuning()

    # The committed corpus, not a re-conversion, so compared runs provably ask the same questions (#302).
    if not args.corpus.exists():
        print(
            f"no corpus at {args.corpus}. Build one first:\n"
            f"  context-graph-eval build-corpus --limit {args.limit} --out {args.corpus}",
            file=sys.stderr,
        )
        return 1
    goldens = select_goldens(read_corpus(args.corpus), limit=args.limit, gold_slice=args.gold_slice)

    # The haystack is not committed (#302); the pinned revision makes it immutable instead.
    records = load_raw(fetch(args.variant, args.revision, dest=haystack_path(args.variant, args.revision)))

    wanted = {g.name for g in goldens}
    used_records = [r for r in records if r["question_id"] in wanted]
    print(
        f"running {len(goldens)} questions from {args.corpus} "
        f"(fixtures: longmemeval-{args.variant} @ {args.revision[:12]})"
    )

    db = Memgraph(url=args.memgraph_url, username="", password="")
    graph = ActionsGraph(memgraph=db)

    judge_provider, judge_model_id = _parse_model_spec(args.judge_model, default_provider=DEFAULT_JUDGE_PROVIDER)
    agent_provider, agent_model_id = _parse_model_spec(args.agent_model, default_provider=DEFAULT_AGENT_PROVIDER)
    judge = _build_model(judge_provider, judge_model_id, minimal_effort=True)
    agent = _build_model(agent_provider, agent_model_id)
    if agent is None:
        print("no agent model configured: set --agent-model or an OPENAI_API_KEY", file=sys.stderr)
        return 1

    official_model = _official_judge_model(args.official_judge_model, judging=judge is not None)
    same_provider = judge is not None and judge_provider == agent_provider
    if same_provider:
        # A legitimate experiment, but never a silent one: #304's judge independence does not hold.
        print(
            f"WARNING: judge ({judge_provider}) and agent ({agent_provider}) share a provider -- "
            "#304's cross-provider independence does not hold for this run.",
            file=sys.stderr,
        )

    if args.gold_slice and not evidence_is_planted(graph):
        print(
            "--gold-slice: the fixture is not in the graph, so scoring it would report a guaranteed "
            "zero as a recall failure. Plant it first with `context-graph-eval gold-slice`.",
            file=sys.stderr,
        )
        return 1

    report = asyncio.run(
        run_batch(
            goldens,
            records=used_records,
            graph=graph,
            llm=DeepEvalLLM(agent),
            plan=RunPlan(
                reconcile=not args.skip_reconcile,
                reuse_graph=args.skip_reconcile,
                judge=judge,
                max_sessions_per_question=args.max_sessions_per_question,
                memgraph_url=args.memgraph_url,
                extraction_backend=args.extraction_backend,
                retrieval_strategy=args.retrieval_strategy,
                text_search_limit=args.text_search_limit,
                hybrid=RecallConfig.from_mapping({"lanes": args.hybrid_lanes}),
                official_judge_model=official_model,
                ontology=args.ontology,
                derive_seeds=args.derive_seeds,
                derive_workers=args.derive_workers,
            ),
        )
    )
    _print_report(report, judged=judge is not None)

    if args.save:
        from .report import RunMeta, SavedRun, save_run
        from .scoring import COVERAGE_GATE, RUBRIC_GATE, tokenizer_in_use

        saved = save_run(
            SavedRun(
                meta=RunMeta(
                    label=args.label,
                    corpus_revision=args.revision,
                    corpus_variant=args.variant,
                    # What actually ran, after provider fallbacks -- not the raw CLI args.
                    judge_model=_resolved_spec(judge_provider, judge_model_id) if judge is not None else "none",
                    agent_model=_resolved_spec(agent_provider, agent_model_id),
                    same_provider=same_provider,
                    tokenizer=tokenizer_in_use(),
                    questions=len(goldens),
                    changed=args.changed,
                    # "none" for text-search, which never reconciles; compare() exempts it.
                    extraction_backend=args.extraction_backend if args.retrieval_strategy == "graph-agent" else "none",
                    retrieval_strategy=args.retrieval_strategy,
                    ontology=args.ontology,
                    coverage_gate=COVERAGE_GATE if official_model else RUBRIC_GATE,
                    official_judge_model=official_model or "none",
                ),
                scored=report.scored,
            ),
            args.save,
        )
        print(f"\nsaved run to {saved}")
    return 0


#: Dated rather than a moving alias (#304): a judge that changes underneath
#: you silently invalidates every prior baseline.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

#: Each role's provider for a bare model id. #304 asks only that the judge's
#: provider differ from the pipeline's; these defaults satisfy that today.
DEFAULT_JUDGE_PROVIDER = "anthropic"
DEFAULT_AGENT_PROVIDER = "openai"

#: provider -> (API key env var, fallback model id used when none is given).
_PROVIDER_KEYS: dict[str, tuple[str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", DEFAULT_JUDGE_MODEL),
    # GPTModel has its own built-in default when given no model id.
    "openai": ("OPENAI_API_KEY", None),
}


def _parse_model_spec(spec: str | None, *, default_provider: str) -> tuple[str, str | None]:
    """Split a 'provider:model_id' CLI value into (provider, model_id).

    A bare model id, or None, keeps ``default_provider``; None also leaves the
    model id None, meaning that provider's own default.
    """
    if spec and ":" in spec:
        provider, _, model_id = spec.partition(":")
        return provider, (model_id or None)
    return default_provider, spec


def _resolved_spec(provider: str, model_id: str | None) -> str:
    """The 'provider:model_id' actually used, after that provider's fallback, for recording on RunMeta."""
    _, fallback = _PROVIDER_KEYS.get(provider, (None, None))
    return f"{provider}:{model_id or fallback or 'default'}"


def _clear_deepeval_anthropic_secret() -> None:
    """Make deepeval's Anthropic client usable at all.

    ``AnthropicModel._build_client`` resolves
    ``settings.ANTHROPIC_API_KEY or self._anthropic_api_key``. The settings
    value is a pydantic ``SecretStr``, which is truthy and therefore always
    wins that ``or`` -- and httpx then refuses it outright ("Header value must
    be str or bytes, not SecretStr") before a single request is made. Passing
    the key explicitly cannot help while settings still holds one.

    Settings are cached at first access, so the cached object is corrected
    directly. The environment variable is deliberately left alone: clearing it
    would make a later ``_build_model`` return None, silently dropping the judge.
    """
    try:
        from deepeval.config.settings import get_settings

        get_settings().ANTHROPIC_API_KEY = None
    except Exception:
        pass


#: Anthropic model families that accept output_config.effort. Older ones
#: (Sonnet 4.5, Haiku 4.5) reject it -- and run without extended thinking
#: unless it is requested, which is already their minimum.
_ANTHROPIC_EFFORT_MODELS = re.compile(r"^claude-(opus|sonnet|fable|mythos)-(4-[6-9]|[5-9])")
#: OpenAI reasoning models, and the lowest reasoning_effort each accepts.
_OPENAI_REASONING_MODELS = ((re.compile(r"^gpt-5"), "minimal"), (re.compile(r"^o[1-9]"), "low"))


def minimal_effort_kwargs(provider: str, model_id: str) -> dict:
    """Generation kwargs that run ``model_id`` at its lowest reasoning effort, or {} when it has no such knob.

    Every model here is built at its minimum: a judge grades a short answer
    against a key, and the eval's cost is dominated by judge calls. Effort is
    set only where the model accepts it, since passing it elsewhere is a 400.
    """
    if provider == "anthropic" and _ANTHROPIC_EFFORT_MODELS.match(model_id):
        return {"output_config": {"effort": "low"}}
    if provider == "openai":
        for pattern, effort in _OPENAI_REASONING_MODELS:
            if pattern.match(model_id):
                return {"reasoning_effort": effort}
    return {}


def _build_model(provider: str, model_id: str | None, *, minimal_effort: bool = False):
    """A deepeval model for ``provider``, or None when it is unknown, has no API key, or fails to build."""
    if provider not in _PROVIDER_KEYS:
        print(f"unknown model provider {provider!r}; supported: {', '.join(_PROVIDER_KEYS)}", file=sys.stderr)
        return None
    env_var, _ = _PROVIDER_KEYS[provider]
    key = os.environ.get(env_var)
    if not key:
        return None
    try:
        if provider == "anthropic":
            from deepeval.models import AnthropicModel

            _clear_deepeval_anthropic_secret()
            model = model_id or DEFAULT_JUDGE_MODEL
            kwargs = minimal_effort_kwargs(provider, model) if minimal_effort else {}
            return AnthropicModel(model=model, _anthropic_api_key=key, generation_kwargs=kwargs)
        from deepeval.models import GPTModel

        kwargs = minimal_effort_kwargs(provider, model_id or "") if minimal_effort else {}
        return GPTModel(model=model_id, generation_kwargs=kwargs) if model_id else GPTModel(generation_kwargs=kwargs)
    except Exception as exc:
        print(f"could not build {provider} model {model_id!r}: {exc}", file=sys.stderr)
        return None


def _official_judge_model(requested: str, *, judging: bool) -> str | None:
    """The model LongMemEval's judge runs on, or None when judging is off, it was turned off, or no OpenAI key exists.

    Without a key the run says so and falls back to the answer rubrics; the
    saved run records which gate decided, so compare() refuses to mix them.
    """
    if not judging or requested == "none":
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "WARNING: no OPENAI_API_KEY for LongMemEval's judge -- coverage falls back to the deepeval answer rubrics.",
            file=sys.stderr,
        )
        return None
    return requested


def _print_attribution(failures) -> None:
    """Print which answer rubric the failures failed, whether retrieval found their evidence, and one judge reason.

    The split tells a retrieval problem from an answering one without rerunning by hand.
    """
    from .scoring import DEFAULT_COVERAGE_THRESHOLD, RETRIEVAL_SIGNAL

    blamed: dict[str, list] = {}
    for row in failures:
        answer = {name: score for name, score in row.metric_scores.items() if name != RETRIEVAL_SIGNAL}
        if answer:
            blamed.setdefault(min(answer, key=lambda name: answer[name]), []).append(row)
    for metric, rows in sorted(blamed.items(), key=lambda kv: -len(kv[1])):
        print(f"  failed on     {metric}: {len(rows)}")
        recall = [row.metric_scores[RETRIEVAL_SIGNAL] for row in rows if RETRIEVAL_SIGNAL in row.metric_scores]
        if recall:
            missed = sum(1 for score in recall if score < DEFAULT_COVERAGE_THRESHOLD)
            print(f"                contextual recall low on {missed}, passed on {len(recall) - missed}")
        evidence = [row.evidence_recall for row in rows if row.evidence_recall is not None]
        if evidence:
            complete = sum(1 for share in evidence if share == 1.0)
            print(
                f"                evidence: all retrieved on {complete}, some on "
                f"{sum(1 for share in evidence if 0 < share < 1)}, none on {sum(1 for share in evidence if share == 0)}"
            )
        example = min(rows, key=lambda row: row.metric_scores[metric])
        reason = example.metric_reasons.get(metric)
        if reason:
            print(f"                e.g. {example.name}: {reason}")


def _print_report(report, *, judged: bool) -> None:
    from .scoring import gate_and_rank

    if report.indexed_turns:
        print(f"text-search index: {report.indexed_turns} turns indexed (no reconciliation)")
    elif report.reconciled or report.reconcile_failures:
        print(f"reconciled {report.reconciled} sessions ({report.reconcile_failures} failed)")

    if not report.by_tier:
        print("no questions scored")
        return

    # Per tier, never blended (#303).
    for tier, summary in sorted(report.by_tier.items()):
        print(f"\nTier {tier}: {summary.questions} questions scored")
        if summary.unscored and judged:
            # With a judge configured, unscored means it errored -- an outage, not a 0%.
            print(f"  UNSCORED      {summary.unscored} question(s) -- the judge errored on them.")
            print("                Not counted as failures. Check judge credentials/credit.")
        if judged:
            if summary.coverage_rate is not None:
                judge_name = " -- LongMemEval judge" if summary.rubric_covered is not None else ""
                print(
                    f"  coverage      {summary.covered}/{summary.questions} ({summary.coverage_rate:.0%}){judge_name}"
                )
                if summary.rubric_covered is not None:
                    print(f"  rubrics       {summary.rubric_covered}/{summary.questions} (deepeval answer rubrics)")
            else:
                print("  coverage      n/a -- nothing in this tier could be scored")
            median = summary.median_efficiency_tokens
            print(
                f"  efficiency    median {median} tokens returned (over questions that cleared coverage)"
                if median is not None
                else "  efficiency    n/a -- no question cleared the coverage gate"
            )
        else:
            print("  coverage      not judged (no judge model configured)")
            # Labelled ungated: #309 defines the comparable number only within the gate.
            payloads = sorted(s.efficiency_tokens for s in report.scored if s.tier == tier)
            if payloads:
                print(f"  payload       median {payloads[len(payloads) // 2]} tokens returned (UNGATED)")
        if summary.abstention_total:
            print(f"  abstention    {summary.abstention_correct}/{summary.abstention_total} correct")
        if summary.retrieval_judged:
            print(
                f"  retrieval     contextual recall passed on {summary.retrieval_supported}/"
                f"{summary.retrieval_judged} (reported, not gated)"
            )

        # Judge-free, so printed whether or not a judge ran.
        if summary.mean_bleu is not None:
            print(f"  bleu          mean {summary.mean_bleu:.2f}")
        if summary.mean_f1 is not None:
            print(f"  f1            mean {summary.mean_f1:.2f}")
        if summary.mean_evidence_recall is not None:
            print(
                f"  evidence      every evidence turn retrieved on {summary.evidence_complete}/"
                f"{summary.evidence_judged} (mean recall {summary.mean_evidence_recall:.2f})"
            )
        if summary.mean_latency_seconds is not None:
            print(f"  latency       mean {summary.mean_latency_seconds:.1f}s per question")

        # Gate-then-rank (#309): only questions that cleared coverage, cheapest first.
        ranked = gate_and_rank([s for s in report.scored if s.tier == tier])
        if len(ranked) > 1:
            print(f"  cheapest      {ranked[0].name} ({ranked[0].efficiency_tokens:,} tokens)")
            print(f"  costliest     {ranked[-1].name} ({ranked[-1].efficiency_tokens:,} tokens)")

        _print_attribution([s for s in report.scored if s.tier == tier and not s.covered])


if __name__ == "__main__":
    raise SystemExit(main())
