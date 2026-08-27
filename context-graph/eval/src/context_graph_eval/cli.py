"""Build a committed eval corpus from a pinned upstream benchmark.

Fetches a pinned LongMemEval release, converts it to deepeval Goldens, and
writes the JSONL that gets committed. The downloaded file is a build artifact
and is not committed -- only the converted corpus is (see #302).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .convert.longmemeval import DEFAULT_REVISION, build_corpus, fetch, load_raw
from .corpus import write_corpus


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
        help="pinned upstream revision. Changing this invalidates prior baselines.",
    )
    build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample down to this many questions, spread across question types. "
        "Tier 1 starts small and scales only when the signal is too noisy to decide on.",
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
        "--memgraph-url",
        default="bolt://localhost:7689",
        help="the DEDICATED eval instance. It is wiped before each batch -- never point this at a "
        "shared or development database.",
    )
    run.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="reuse an already-reconciled graph. Reconciliation dominates run cost, so iterating "
        "on retrieval or scoring should not pay for it again.",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        help="Anthropic model id for the judge (#304 keeps the judge on a different provider from "
        "the OpenAI-backed pipeline so their blind spots do not correlate). Omit to skip judging "
        "and report efficiency only.",
    )
    run.add_argument("--agent-model", default=None, help="model id for the retrieval agent")
    run.add_argument("--save", type=Path, default=None, help="persist this run so it can be compared later")
    run.add_argument("--label", default="run", help="name for this run in a later comparison")
    run.add_argument("--changed", default="", help="what this run changed, shown in the comparison report")

    cmp_ = subcommands.add_parser("compare", help="compare two saved runs and print the report")
    cmp_.add_argument("baseline", type=Path)
    cmp_.add_argument("candidate", type=Path)
    cmp_.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="coverage noise floor in percentage points, from #304's repeat-and-compare "
        "calibration. Without it no delta is claimed real -- 'cannot tell' is reported "
        "rather than a guess.",
    )

    args = parser.parse_args(argv)

    if args.command == "build-corpus":
        return _build_corpus(args)
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    return 1


def _compare(args) -> int:
    from .report import compare, load_run, render

    try:
        comparison = compare(
            load_run(args.baseline),
            load_run(args.candidate),
            noise_floor_pp=args.noise_floor,
        )
    except ValueError as exc:
        # Not comparable is a refusal, not a warning: a delta computed across
        # different pins measures the pin change as if it were the change under
        # test, and says so confidently.
        print(f"refusing to compare: {exc}", file=sys.stderr)
        return 1

    print(render(comparison))
    return 0


def _build_corpus(args) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        raw = fetch(args.variant, args.revision, dest=Path(workdir) / "upstream.json")
        records = load_raw(raw)
        print(f"fetched {len(records)} records from longmemeval-{args.variant} @ {args.revision[:12]}")

        goldens = build_corpus(records, limit=args.limit)
        written = write_corpus(goldens, args.out)
        print(f"wrote {written} goldens to {args.out}")
    return 0


def _run(args) -> int:
    import asyncio

    from actions_graph import ActionsGraph
    from memgraph_toolbox.api.memgraph import Memgraph

    from .convert.longmemeval import build_corpus as build_goldens
    from .reconcile import _resolve_llm_credentials
    from .retrieval import DeepEvalLLM
    from .runner import RunPlan, check_offline, run_batch

    check_offline()
    _resolve_llm_credentials()

    with tempfile.TemporaryDirectory() as workdir:
        raw = fetch(args.variant, args.revision, dest=Path(workdir) / "upstream.json")
        records = load_raw(raw)

    goldens = build_goldens(records, limit=args.limit)
    chosen = {g.name for g in goldens}
    # Only the sampled questions' haystacks get injected: injecting the whole
    # upstream file would reconcile thousands of sessions nobody asks about.
    used_records = [r for r in records if r["question_id"] in chosen]
    print(f"running {len(goldens)} questions from longmemeval-{args.variant} @ {args.revision[:12]}")

    db = Memgraph(url=args.memgraph_url, username="", password="")
    graph = ActionsGraph(memgraph=db)

    judge = _build_model(args.judge_model, anthropic=True)
    agent = _build_model(args.agent_model, anthropic=False)
    if agent is None:
        print("no agent model configured: set --agent-model or an OPENAI_API_KEY", file=sys.stderr)
        return 1

    report = asyncio.run(
        run_batch(
            goldens,
            records=used_records,
            graph=graph,
            llm=DeepEvalLLM(agent),
            plan=RunPlan(reconcile=not args.skip_reconcile, judge=judge),
        )
    )
    _print_report(report, judged=judge is not None)

    if args.save:
        from .report import RunMeta, SavedRun, save_run
        from .scoring import DEFAULT_TOKENIZER

        saved = save_run(
            SavedRun(
                meta=RunMeta(
                    label=args.label,
                    corpus_revision=args.revision,
                    corpus_variant=args.variant,
                    # Recorded even when absent: a comparison must refuse to
                    # measure an unjudged run against a judged one.
                    judge_model=args.judge_model or "none",
                    tokenizer=DEFAULT_TOKENIZER,
                    questions=len(goldens),
                    changed=args.changed,
                ),
                scored=report.scored,
            ),
            args.save,
        )
        print(f"\nsaved run to {saved}")
    return 0


#: Default judge. Dated rather than a moving alias, per #304: a judge that
#: changes underneath you silently invalidates every prior baseline, which is
#: the same reason the corpus revision is pinned.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"


def _clear_deepeval_anthropic_secret() -> None:
    """Make deepeval's Anthropic client usable at all.

    ``AnthropicModel._build_client`` resolves
    ``settings.ANTHROPIC_API_KEY or self._anthropic_api_key``. The settings
    value is a pydantic ``SecretStr``, which is truthy and therefore always
    wins that ``or`` -- and httpx then refuses it outright ("Header value must
    be str or bytes, not SecretStr") before a single request is made. Passing
    the key explicitly cannot help while settings still holds one.

    Clearing the environment variable is not sufficient: settings are cached at
    first access, and merely importing another deepeval model earlier in the
    process is enough to populate them. The cached object has to be corrected
    directly.

    This is load-bearing rather than cosmetic: #304 chose Anthropic as the
    judge precisely so it would not share the OpenAI-backed pipeline's blind
    spots, and `eval-run.yaml` supplies ANTHROPIC_API_KEY as an environment
    variable -- so without this the CI eval job fails before scoring anything.
    """
    try:
        from deepeval.config.settings import get_settings

        get_settings().ANTHROPIC_API_KEY = None
    except Exception:
        pass
    os.environ.pop("ANTHROPIC_API_KEY", None)


def _build_model(model_id: str | None, *, anthropic: bool):
    """Instantiate a deepeval model, or None when nothing is configured."""
    try:
        if anthropic:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                return None
            from deepeval.models import AnthropicModel

            _clear_deepeval_anthropic_secret()
            return AnthropicModel(model=model_id or DEFAULT_JUDGE_MODEL, _anthropic_api_key=key)
        if not (model_id or os.environ.get("OPENAI_API_KEY")):
            return None
        from deepeval.models import GPTModel

        return GPTModel(model=model_id) if model_id else GPTModel()
    except Exception as exc:
        print(f"could not build model {model_id!r}: {exc}", file=sys.stderr)
        return None


def _print_report(report, *, judged: bool) -> None:
    if report.reconciled or report.reconcile_failures:
        print(f"reconciled {report.reconciled} sessions ({report.reconcile_failures} failed)")

    if not report.by_tier:
        print("no questions scored")
        return

    # Printed per tier, never blended: a single cross-tier number would let an
    # organizational-recall regression hide behind a personal-memory gain (#303).
    for tier, summary in sorted(report.by_tier.items()):
        print(f"\nTier {tier}: {summary.questions} questions")
        if judged:
            print(f"  coverage      {summary.covered}/{summary.questions} ({summary.coverage_rate:.0%})")
            median = summary.median_efficiency_tokens
            print(
                f"  efficiency    median {median} tokens returned (over questions that cleared coverage)"
                if median is not None
                else "  efficiency    n/a -- no question cleared the coverage gate"
            )
        else:
            print("  coverage      not judged (no judge model configured)")
            # The gated median is empty without a judge, since nothing clears an
            # unscored gate. The raw payload size is still deterministic and
            # worth seeing -- labelled ungated so it is never mistaken for the
            # comparable number, which #309 defines only within the gate.
            payloads = sorted(s.efficiency_tokens for s in report.scored if s.tier == tier)
            if payloads:
                print(f"  payload       median {payloads[len(payloads) // 2]} tokens returned (UNGATED)")
        if summary.abstention_total:
            print(f"  abstention    {summary.abstention_correct}/{summary.abstention_total} correct")


if __name__ == "__main__":
    raise SystemExit(main())
