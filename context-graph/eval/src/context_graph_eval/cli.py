"""Context Graph eval: run a workload, compare runs, calibrate the noise floor, build corpora."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .answering import LANES
from .beam_judge import BEAM_JUDGE_MODEL
from .convert.longmemeval import DEFAULT_REVISION
from .models import DEFAULT_AGENT_MODEL
from .official_judge import OFFICIAL_JUDGE_MODEL
from .reconcile import ONTOLOGIES
from .workloads import EVAL_DIR, WORKLOADS

RUNS_DIR = EVAL_DIR / "runs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-graph-eval", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("workloads", help="list the workloads, their eval instances and the graphs they keep")

    run = subcommands.add_parser(
        "run",
        help="run a workload: build its graph if what shapes it changed, then ask, judge and save",
    )
    _workload_arg(run)
    _build_args(run)
    run.add_argument(
        "--reuse",
        action="store_true",
        help="ask the kept graph even though what shapes it changed since it was built; the run is marked stale",
    )
    run.add_argument(
        "--lanes",
        default=",".join(LANES),
        help=f"comma-separated recall lanes, from {', '.join(LANES)}. 'turns,text' is the graph ablation.",
    )
    run.add_argument(
        "--packing",
        choices=("exchanges", "turns"),
        default=None,
        help="recall's packing (default: recall's own, exchanges). 'turns' is how the version points were measured.",
    )
    run.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help=f"'provider:model_id' for the answering model (default {DEFAULT_AGENT_MODEL})",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        help=f"the benchmark's own judge (default: upstream's -- {OFFICIAL_JUDGE_MODEL} for LongMemEval, "
        f"{BEAM_JUDGE_MODEL} for BEAM). 'none' skips judging.",
    )
    run.add_argument(
        "--rubric-judge",
        default=None,
        help="LongMemEval only: 'provider:model_id' for the deepeval rubrics reported beside the official "
        "judge, e.g. 'anthropic:claude-sonnet-4-5-20250929'. The gold slice needs it.",
    )
    run.add_argument("--max-concurrent", type=int, default=4, help="questions answered at once")
    run.add_argument("--label", default=None, help="name for this run (default: the commit)")
    run.add_argument("--changed", default="", help="what this run changed, shown when it is compared")
    run.add_argument(
        "--save",
        type=Path,
        default=None,
        help="where to write the run (default: runs/<workload>/<label>.json in the eval package)",
    )

    build = subcommands.add_parser("build", help="build a workload's graph and stop, e.g. overnight")
    _workload_arg(build)
    _build_args(build)

    drop = subcommands.add_parser("drop", help="delete a workload's eval instance and the graph it keeps")
    _workload_arg(drop)

    cmp_ = subcommands.add_parser("compare", help="compare two runs of the same workload")
    cmp_.add_argument("baseline", type=Path)
    cmp_.add_argument("candidate", type=Path)
    cmp_.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="headline noise floor in percentage points, from `calibrate`. Without it no delta is called real.",
    )

    cal = subcommands.add_parser("calibrate", help="derive a noise floor from repeated runs of the same questions")
    cal.add_argument("runs", type=Path, nargs="+", help="two or more runs of the SAME workload")

    publish = subcommands.add_parser(
        "publish",
        help="copy a run without benchmark text (questions, answers, retrieval) for baselines/",
    )
    publish.add_argument("run", type=Path)
    publish.add_argument("out", type=Path)

    corpus = subcommands.add_parser("build-corpus", help="fetch the pinned LongMemEval-S release and write a corpus")
    corpus.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="pinned upstream revision; changing it invalidates prior baselines",
    )
    corpus.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample down to this many questions, stratified by question type (the dev corpus is 50)",
    )
    corpus.add_argument("--out", type=Path, required=True, help="where to write the corpus JSONL")

    gold = subcommands.add_parser(
        "gold-slice",
        help="plant the gold slice in its instance by driving a REAL, billed Claude Code session",
    )
    gold.add_argument("--memgraph-url", default=None, help="an instance managed elsewhere, instead of its own")
    gold.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="working directory for the session; hook commands resolve `uv run` against it",
    )
    gold.add_argument("--keep", action="store_true", help="do not wipe first")

    kg = subcommands.add_parser(
        "kg",
        help="measure the default knowledge-graph construction on BEAM, against the committed baseline",
    )
    kg.add_argument("--chats", default="1-20", help="which BEAM 100K chats, e.g. '1-20' (the baseline's)")
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
    commands = {
        "workloads": _workloads,
        "run": _run,
        "build": _run,
        "drop": _drop,
        "compare": _compare,
        "calibrate": _calibrate,
        "publish": _publish,
        "build-corpus": _build_corpus,
        "gold-slice": _gold_slice,
        "kg": _kg,
    }
    try:
        return commands[args.command](args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"context-graph-eval {args.command}: {exc}", file=sys.stderr)
        return 1


def _workload_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workload", choices=sorted(WORKLOADS), help="see `context-graph-eval workloads`")


def _build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memgraph-url",
        default=None,
        help="an instance managed elsewhere (e.g. CI) instead of the workload's own; a rebuild wipes it",
    )
    parser.add_argument("--rebuild", action="store_true", help="wipe and rebuild the graph even if it is current")
    parser.add_argument(
        "--ontology",
        choices=ONTOLOGIES,
        default="default",
        help="'default': extract under hygm's default model. 'learned': then derive and adopt each user's "
        "own model (sessions-graph derive) before asking.",
    )
    parser.add_argument("--derive-seeds", type=int, default=1, help="candidates per user derivation (learned)")
    parser.add_argument("--derive-workers", type=int, default=4, help="user derivations at once (learned)")


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


def default_label() -> str:
    """The commit a run measures, marked when the working tree has uncommitted changes."""
    from .kg import _commit

    commit = _commit() or "uncommitted"
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        dirty = ""
    return f"{commit}-dirty" if dirty.strip() else commit


def _connect(workload, url: str | None):
    """The workload's own instance, started if needed, or the one ``url`` names."""
    from memgraph_toolbox.api.memgraph import Memgraph

    from .instance import ensure_running, wait_until_ready
    from .pipeline import Connection

    if url is None:
        url = ensure_running(workload).url
    else:
        wait_until_ready(url, timeout=30)
    return Connection(db=Memgraph(url=url, username="", password=""), url=url)


def _benchmark(workload, args):
    """The workload's benchmark adapter, with the judges ``args`` ask for."""
    if workload.benchmark == "beam-100k":
        from .benchmarks.beam import Beam

        judge = args.judge_model or BEAM_JUDGE_MODEL
        if judge != "none" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("BEAM's judge needs an OPENAI_API_KEY; pass --judge-model none to skip judging")
        return Beam(judge_model=None if judge == "none" else judge, max_concurrent=args.max_concurrent), "none"

    from .benchmarks.longmemeval import LongMemEval
    from .models import DEFAULT_JUDGE_PROVIDER, build_model, parse_model_spec, resolved_spec

    official = args.judge_model or OFFICIAL_JUDGE_MODEL
    if official != "none" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("LongMemEval's judge needs an OPENAI_API_KEY; pass --judge-model none to skip judging")
    rubric_judge, rubric_name = None, "none"
    if args.rubric_judge:
        provider, model_id = parse_model_spec(args.rubric_judge, default_provider=DEFAULT_JUDGE_PROVIDER)
        rubric_judge = build_model(provider, model_id, minimal_effort=True)
        if rubric_judge is None:
            raise RuntimeError(f"could not build the rubric judge {args.rubric_judge!r}: is its API key set?")
        rubric_name = resolved_spec(provider, model_id)
    adapter = LongMemEval(
        official_judge_model=None if official == "none" else official,
        rubric_judge=rubric_judge,
        max_concurrent=args.max_concurrent,
    )
    return adapter, rubric_name


def _run(args) -> int:
    import asyncio

    from .answering import DeepEvalLLM
    from .models import DEFAULT_AGENT_PROVIDER, build_model, parse_model_spec, resolve_llm_credentials, resolved_spec
    from .pipeline import BuildPlan, RunSettings, run
    from .report import render, summary_of
    from .results import save
    from .workloads import workload as named

    resolve_llm_credentials()
    workload = named(args.workload)
    build_only = args.command == "build"
    commit = default_label()
    label = getattr(args, "label", None) or commit

    llm, agent_spec, benchmark, rubric_name = None, "none", None, "none"
    if build_only:
        from .benchmarks.beam import Beam
        from .benchmarks.longmemeval import LongMemEval

        benchmark = (
            Beam(judge_model=None) if workload.benchmark == "beam-100k" else LongMemEval(official_judge_model=None)
        )
    else:
        benchmark, rubric_name = _benchmark(workload, args)
        provider, model_id = parse_model_spec(args.agent_model, default_provider=DEFAULT_AGENT_PROVIDER)
        agent = build_model(provider, model_id)
        if agent is None:
            raise RuntimeError(f"could not build the answering model {args.agent_model!r}: is its API key set?")
        llm, agent_spec = DeepEvalLLM(agent), resolved_spec(provider, model_id)

    save_path = getattr(args, "save", None) or RUNS_DIR / workload.name / f"{label}.json"
    print(f"{workload.name}: {workload.description}")
    connection = _connect(workload, args.memgraph_url)
    try:
        result = asyncio.run(
            run(
                workload,
                benchmark,
                connection=connection,
                llm=llm,
                settings=RunSettings(
                    label=label,
                    agent_model=agent_spec,
                    recall=_recall_config(args),
                    lanes=getattr(args, "lanes", ",".join(LANES)),
                    changed=getattr(args, "changed", ""),
                    plan=BuildPlan(
                        ontology=args.ontology, derive_seeds=args.derive_seeds, derive_workers=args.derive_workers
                    ),
                    max_concurrent=getattr(args, "max_concurrent", 4),
                    rebuild=args.rebuild,
                    reuse=getattr(args, "reuse", False),
                    build_only=build_only,
                    rubric_judge_model=rubric_name,
                    commit=commit,
                ),
                shard_dir=save_path.with_suffix(".shards"),
            )
        )
    finally:
        connection.db.close()
    if result is None:
        print(f"{workload.instance}: built")
        return 0
    print()
    print(render(result, notes=benchmark.notes(result.rows)))
    print(f"\nsaved run to {save(result, save_path, summary=summary_of(result))}")
    return 0


def _recall_config(args):
    """Recall's settings for this run: its defaults, with the lanes and packing ``args`` name."""
    from .answering import RecallConfig

    overrides: dict[str, object] = {"lanes": getattr(args, "lanes", ",".join(LANES))}
    if getattr(args, "packing", None):
        overrides["packing"] = args.packing
    return RecallConfig.from_mapping(overrides)


def _workloads(args) -> int:
    from .instance import state
    from .pipeline import read_manifest

    for workload in WORKLOADS.values():
        current = state(workload)
        kept = ""
        if current == "running":
            from memgraph_toolbox.api.memgraph import Memgraph

            db = Memgraph(url=f"bolt://localhost:{workload.port}", username="", password="")
            try:
                manifest = read_manifest(db)
            finally:
                db.close()
            if manifest is not None:
                kept = f"  graph {manifest.fingerprint} ({manifest.status}, {manifest.commit or '?'})"
        print(f"{workload.name:<15} {workload.kind:<9} {workload.description}")
        print(f"{'':<15} {workload.instance} :{workload.port} {current}{kept}")
    return 0


def _drop(args) -> int:
    from .instance import remove
    from .workloads import workload as named

    workload = named(args.workload)
    remove(workload)
    print(f"removed {workload.instance} and its graph")
    return 0


def _compare(args) -> int:
    from .report import compare, render_comparison
    from .results import load

    print(render_comparison(compare(load(args.baseline), load(args.candidate), noise_floor_pp=args.noise_floor)))
    return 0


def _calibrate(args) -> int:
    from .benchmarks import headline
    from .calibrate import describe, describe_per_question_rates, describe_stability, per_question_pass_rate
    from .results import load

    runs = [load(path) for path in args.runs]
    # Same questions, or the spread measures the corpus rather than the judge.
    if len({run.meta.workload for run in runs}) > 1 or len({frozenset(r.name for r in run.rows) for run in runs}) > 1:
        raise ValueError("these runs asked different questions, so their spread would measure that, not the judge")
    headlines = [headline(run.meta.benchmark, run.rows) for run in runs]
    if any(value is None for value in headlines):
        raise ValueError("a run has no judged questions")
    passing_sets = [{row.name for row in run.rows if row.passed} for run in runs]
    all_names = {row.name for run in runs for row in run.rows}
    # Stability beside the floor, never instead: a tight floor over an unstable passing set is the misleading case.
    print(describe([value for value in headlines if value is not None]))
    print()
    print(describe_stability(passing_sets))
    print()
    print(describe_per_question_rates(per_question_pass_rate(passing_sets, all_names)))
    return 0


def _publish(args) -> int:
    from .report import summary_of
    from .results import load, public, save

    run = public(load(args.run))
    print(f"wrote {save(run, args.out, summary=summary_of(run))} without benchmark text")
    return 0


def _build_corpus(args) -> int:
    from .convert.longmemeval import build_corpus, fetch, haystack_path, load_raw
    from .corpus import write_corpus

    raw = fetch("s", args.revision, dest=haystack_path("s", args.revision))
    records = load_raw(raw)
    print(f"fetched {len(records)} records from longmemeval-s @ {args.revision[:12]}")
    written = write_corpus(build_corpus(records, limit=args.limit), args.out)
    print(f"wrote {written} questions to {args.out}")
    return 0


def _kg(args) -> int:
    from .convert.beam import DEFAULT_REVISION as BEAM_REVISION
    from .convert.beam import fetch_chat
    from .kg import compare as compare_kg
    from .kg import load, render, run_kg, save

    chat_ids = parse_chat_ids(args.chats)
    chats = [fetch_chat("100K", chat_id, revision=BEAM_REVISION) for chat_id in chat_ids]
    print(f"kg: {len(chats)} BEAM 100K chats @ {BEAM_REVISION[:12]}, {args.windows} windows", flush=True)
    run = run_kg(
        chats,
        windows=args.windows,
        observe=args.observe,
        meta={
            "label": args.label,
            "benchmark": "beam-100K",
            "revision": BEAM_REVISION,
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


def _gold_slice(args) -> int:
    """Plant the gold-slice fixture with a real Claude Code session."""
    from actions_graph import ActionsGraph

    from .goldslice import (
        GOLD_SLICE_FACT,
        GOLD_SLICE_PROMPT,
        GOLD_SLICE_USER,
        evidence_is_nested,
        evidence_is_planted,
        evidence_is_top_level,
    )
    from .inject import wipe
    from .live import LiveSessionError, drive_session, hooks_pointed_at
    from .workloads import workload as named

    connection = _connect(named("gold-slice"), args.memgraph_url)
    graph = ActionsGraph(memgraph=connection.db)
    if not args.keep:
        # Wipe BEFORE the session, never after: the planted fact must survive into the run that scores it.
        wipe(graph)
        print("wiped the gold-slice instance")

    print("driving a real Claude Code session (billed)...")
    try:
        with hooks_pointed_at(connection.url, user_id=GOLD_SLICE_USER) as env:
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
    print("  context-graph-eval run gold-slice --rubric-judge anthropic:claude-sonnet-4-5-20250929")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
