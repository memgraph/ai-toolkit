"""Build a committed eval corpus from a pinned upstream benchmark.

Fetches a pinned LongMemEval release, converts it to deepeval Goldens, and
writes the JSONL that gets committed. The downloaded file is a build artifact
and is not committed -- only the converted corpus is (see #302).
"""

import argparse
import os
import sys
from pathlib import Path

from .convert.longmemeval import DEFAULT_REVISION, build_corpus, fetch, haystack_path, load_raw
from .corpus import write_corpus
from .reconcile import EXTRACTION_BACKENDS


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
        "--corpus",
        type=Path,
        default=Path("context-graph/eval/corpus/tier1-longmemeval.jsonl"),
        help="the COMMITTED corpus to run. Read rather than re-derived, so two runs being "
        "compared provably answer the same questions (#302).",
    )
    run.add_argument(
        "--gold-slice",
        action="store_true",
        help="also run Tier 2's gold-slice questions, scored separately from Tier 1 (#303).",
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
        help="reuse the already-reconciled graph as-is: no wipe, no re-injection, no distillation. "
        "Reconciliation dominates run cost, so iterating on retrieval or scoring should not pay "
        "for it again. Refuses if the graph does not already hold this run's sessions, distilled "
        "-- reusing the wrong graph would score every question as a miss and report it as a "
        "result (#322).",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        help="'provider:model_id' for the judge, e.g. 'anthropic:claude-sonnet-4-5-20250929' or "
        "'openai:gpt-4o'. A bare model id keeps the default provider (anthropic). #304 keeps the "
        "judge on a different provider from the OpenAI-backed pipeline so their blind spots do not "
        "correlate -- picking the same provider as --agent-model is a legitimate experiment (#329), "
        "not a default, and is flagged loudly when it happens. Omit entirely to skip judging and "
        "report efficiency only.",
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
        default="lightrag",
        help="what reconciliation uses to extract entities: 'lightrag' (default, LLM-based) or "
        "'gliner2' (local, LLM-free -- requires 'pip install gliner2[local]>=2.0.0' manually, see "
        "unstructured2graph.gliner2_backend's module docstring). Narrative summarization always "
        "runs via a LightRAG wrapper's own LLM regardless of this choice -- GLiNER2 has no "
        "generative capability -- so an LLM key is still needed either way.",
    )
    run.add_argument(
        "--max-sessions-per-question",
        type=int,
        default=None,
        help="trim each question's DISTRACTOR sessions. Reconciliation cost scales with "
        "sessions while coverage needs questions, and upstream couples them ~47:1. Evidence "
        "sessions are always kept, so a question with more evidence than this cap exceeds it "
        "rather than becoming unanswerable. A score measured this way is an UPPER BOUND -- "
        "fewer distractors make retrieval easier -- and is NOT comparable to a full-haystack run.",
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
        help="do not wipe first. Use when Tier 1 fixtures are already loaded and the gold-slice "
        "fact should be planted on top of them, among their distractors.",
    )

    cal = subcommands.add_parser(
        "calibrate",
        help="derive a noise floor by running the same questions repeatedly (#304)",
    )
    cal.add_argument("runs", type=Path, nargs="+", help="two or more saved runs of the SAME questions")

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
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "gold-slice":
        return _gold_slice(args)
    return 1


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
    # Always shown. A zero exit means the CLI did not crash, not that the model
    # delegated or that hooks recorded anything -- and re-running to find out
    # costs another billed session.
    print("--- session transcript ---")
    print(transcript.strip()[:2000])
    print("--- end transcript ---")

    # Two checks, because they fail for different reasons and mean different
    # things. Neither is a recall result -- they decide whether a recall result
    # would mean anything at all.
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
    from .calibrate import describe, describe_stability
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
    try:
        print(describe(rates))
        # Printed with the floor, never instead of it: a tight floor over an
        # unstable passing set is the misleading case, and only this line makes
        # it visible.
        print()
        print(describe_stability([{s.name for s in run.scored if s.covered} for run in runs]))
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
        # Not comparable is a refusal, not a warning: a delta computed across
        # different pins measures the pin change as if it were the change under
        # test, and says so confidently.
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
    """Which questions this run will actually ask.

    Its own function because a run's question set is the one thing every number
    downstream is relative to, and getting it silently wrong is expensive:
    ``--limit`` was ignored for a while after ``run`` switched to reading the
    committed corpus, so a run asked for 2 questions quietly did all 20 -- a
    "minimal" check that turned out to be 40 sessions of reconciliation, killed
    twice before anyone knew why.

    The gold slice is appended rather than counted against ``--limit``: it is
    Tier 2 (#303), scored apart, so letting a Tier 1 limit trim it would drop
    the only questions that exercise the capture layer.
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
    # Must run before reconciliation's lazy unstructured2graph/lightrag import:
    # LightRAG reads FORCE_LLM_SUMMARY_ON_MERGE once, as a dataclass default
    # baked in when lightrag.lightrag is first imported.
    _resolve_reconciliation_tuning()

    # Read the COMMITTED corpus rather than re-deriving it. This is the whole
    # point of #302 putting it in git: two runs being compared must provably be
    # answering the same questions, and re-converting upstream each time proves
    # nothing -- a change in sampling, conversion, or upstream would silently
    # alter the question set between a baseline and its candidate.
    if not args.corpus.exists():
        print(
            f"no corpus at {args.corpus}. Build one first:\n"
            f"  context-graph-eval build-corpus --limit {args.limit} --out {args.corpus}",
            file=sys.stderr,
        )
        return 1
    goldens = select_goldens(read_corpus(args.corpus), limit=args.limit, gold_slice=args.gold_slice)

    # The haystack is NOT committed alongside the corpus: at 20 questions it is
    # ~9.5MB of reshaped upstream text, which is what #302 rejected vendoring.
    # Its immutability comes from the pinned revision instead of from git.
    # Cached across runs rather than fetched into a temp dir. The haystack is
    # ~277MB, so re-downloading it per run made iterating on retrieval or
    # scoring painfully slow and put a large flaky transfer in front of every
    # run -- which is what killed the first end-to-end attempt.
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
    judge = _build_model(judge_provider, judge_model_id)
    agent = _build_model(agent_provider, agent_model_id)
    if agent is None:
        print("no agent model configured: set --agent-model or an OPENAI_API_KEY", file=sys.stderr)
        return 1

    same_provider = judge is not None and judge_provider == agent_provider
    if same_provider:
        # #304's independence property (judge decorrelated from the pipeline's
        # own blind spots) does not hold here. A legitimate experiment (#329,
        # #324's model-vs-rubric diagnosis needs exactly this), just never the
        # unannounced default -- so this is a warning, not a refusal.
        print(
            f"WARNING: judge ({judge_provider}) and agent ({agent_provider}) share a provider -- "
            "#304's cross-provider independence does not hold for this run.",
            file=sys.stderr,
        )

    if args.gold_slice and not evidence_is_planted(graph):
        # Refuse rather than report a guaranteed zero. The gold slice's fixture
        # is planted by a real harness session (#307), and that driver is not
        # built yet -- so scoring the question here measures a fact the graph
        # never contained and reports it as a recall failure.
        print(
            "--gold-slice: the fixture is not in the graph. It is planted by a real "
            "Claude Code session, and that driver is not built yet (#307), so scoring "
            "this question now would report a guaranteed zero as a recall failure.",
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
            ),
        )
    )
    _print_report(report, judged=judge is not None)

    if args.save:
        from .report import RunMeta, SavedRun, save_run
        from .scoring import tokenizer_in_use

        saved = save_run(
            SavedRun(
                meta=RunMeta(
                    label=args.label,
                    corpus_revision=args.revision,
                    corpus_variant=args.variant,
                    # Provider-qualified and reflecting the resolved fallback
                    # model, not the raw CLI arg (#329) -- e.g. omitting
                    # --judge-model with ANTHROPIC_API_KEY set still judges
                    # with DEFAULT_JUDGE_MODEL, and recording "none" for that
                    # would let an unjudged and a judged run compare cleanly.
                    judge_model=_resolved_spec(judge_provider, judge_model_id) if judge is not None else "none",
                    # Previously untracked entirely: a run's answers depend on
                    # this model too, and compare() below now pins it for the
                    # same reason it already pins the judge and tokenizer.
                    agent_model=_resolved_spec(agent_provider, agent_model_id),
                    same_provider=same_provider,
                    # What was actually used, not what was configured -- a run
                    # counted in fallback units must not compare cleanly
                    # against one counted in real tokens.
                    tokenizer=tokenizer_in_use(),
                    questions=len(goldens),
                    changed=args.changed,
                    # A LightRAG-built and a GLiNER2-built graph are different
                    # systems under test (different entities, no cross-chunk
                    # coreference for GLiNER2, a different workspace label) --
                    # compare() below refuses across them for the same reason
                    # it refuses across judges or tokenizers.
                    extraction_backend=args.extraction_backend,
                ),
                scored=report.scored,
            ),
            args.save,
        )
        print(f"\nsaved run to {saved}")
    return 0


#: Default judge model. Dated rather than a moving alias, per #304: a judge
#: that changes underneath you silently invalidates every prior baseline,
#: which is the same reason the corpus revision is pinned.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

#: Default provider for each role when a bare (unprefixed) model id -- or
#: nothing at all -- is given. #304's decision was "a different provider than
#: the pipeline", not "Anthropic" specifically (#329); these are only the
#: defaults that satisfy it given the pipeline happens to be OpenAI-backed
#: today, not a hardcoded pairing. --judge-model/--agent-model can override
#: either independently via a 'provider:model_id' spec.
DEFAULT_JUDGE_PROVIDER = "anthropic"
DEFAULT_AGENT_PROVIDER = "openai"

#: provider -> (API key env var, fallback model id used when none is given).
#: Adding a provider is the whole change needed to make it reachable from
#: either --judge-model or --agent-model -- nothing else in _build_model is
#: provider-specific beyond the deepeval class each one maps to.
_PROVIDER_KEYS: dict[str, tuple[str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", DEFAULT_JUDGE_MODEL),
    # GPTModel has its own built-in default when given no model id.
    "openai": ("OPENAI_API_KEY", None),
}


def _parse_model_spec(spec: str | None, *, default_provider: str) -> tuple[str, str | None]:
    """Split a 'provider:model_id' CLI value into (provider, model_id).

    A bare model id (no colon) keeps ``default_provider`` -- so existing
    --judge-model/--agent-model values written before #329 keep meaning what
    they meant. ``spec=None`` (the flag omitted) also keeps the default
    provider, with no model id, i.e. "use that provider's own default."
    """
    if spec and ":" in spec:
        provider, _, model_id = spec.partition(":")
        return provider, (model_id or None)
    return default_provider, spec


def _resolved_spec(provider: str, model_id: str | None) -> str:
    """The 'provider:model_id' actually used, after applying that provider's
    fallback default -- for recording on RunMeta, not for building a model.

    Comment at the call site explains why this matters more than it looks:
    RunMeta.judge_model must reflect what ran, not what was typed on the
    command line, or two runs that used different fallbacks could compare as
    identical.
    """
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

    Settings are cached at first access, and merely constructing another
    deepeval model earlier in the process populates them, so the cached object
    has to be corrected directly.

    The environment variable is deliberately left alone. Clearing it as well
    looked like harmless belt-and-braces and was actively wrong: the key would
    then be gone for every later caller, so a second ``_build_model`` returned
    None and the judge silently disappeared mid-process -- a run reporting "not
    judged" instead of failing.

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


def _build_model(provider: str, model_id: str | None):
    """Instantiate a deepeval model for ``provider``, or None when nothing is
    configured (no matching API key -- via ADR 0002's config-file resolution
    or the environment directly -- present for it).

    Keyed by the resolved provider rather than a two-vendor ``anthropic: bool``
    (#329): the special-casing below is what deepeval itself requires per
    provider (a settings-clearing workaround for Anthropic, a plain optional
    model id for OpenAI), not a judge/agent distinction -- either role can
    resolve to either provider.
    """
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
            return AnthropicModel(model=model_id or DEFAULT_JUDGE_MODEL, _anthropic_api_key=key)
        from deepeval.models import GPTModel

        return GPTModel(model=model_id) if model_id else GPTModel()
    except Exception as exc:
        print(f"could not build {provider} model {model_id!r}: {exc}", file=sys.stderr)
        return None


def _print_attribution(failures) -> None:
    """Say which metric was the weakest link across the failures."""
    blamed: dict[str, int] = {}
    for row in failures:
        if row.metric_scores:
            worst = min(row.metric_scores, key=lambda name: row.metric_scores[name])
            blamed[worst] = blamed.get(worst, 0) + 1
    for metric, count in sorted(blamed.items(), key=lambda kv: -kv[1]):
        print(f"  failed on     {metric}: {count}")


def _print_report(report, *, judged: bool) -> None:
    from .scoring import gate_and_rank

    if report.reconciled or report.reconcile_failures:
        print(f"reconciled {report.reconciled} sessions ({report.reconcile_failures} failed)")

    if not report.by_tier:
        print("no questions scored")
        return

    # Printed per tier, never blended: a single cross-tier number would let an
    # organizational-recall regression hide behind a personal-memory gain (#303).
    for tier, summary in sorted(report.by_tier.items()):
        print(f"\nTier {tier}: {summary.questions} questions scored")
        if summary.unscored and judged:
            # Only alarming when a judge WAS configured: then unscored means it
            # errored, and folding that into the rate would report an outage as
            # a real 0%. Without a judge, unscored is simply expected.
            print(f"  UNSCORED      {summary.unscored} question(s) -- the judge errored on them.")
            print("                Not counted as failures. Check judge credentials/credit.")
        if judged:
            if summary.coverage_rate is not None:
                print(f"  coverage      {summary.covered}/{summary.questions} ({summary.coverage_rate:.0%})")
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
            # The gated median is empty without a judge, since nothing clears an
            # unscored gate. The raw payload size is still deterministic and
            # worth seeing -- labelled ungated so it is never mistaken for the
            # comparable number, which #309 defines only within the gate.
            payloads = sorted(s.efficiency_tokens for s in report.scored if s.tier == tier)
            if payloads:
                print(f"  payload       median {payloads[len(payloads) // 2]} tokens returned (UNGATED)")
        if summary.abstention_total:
            print(f"  abstention    {summary.abstention_correct}/{summary.abstention_total} correct")

        # Gate-then-rank made visible (#309): only questions that cleared
        # coverage are ranked, cheapest payload first. Showing the extremes is
        # what makes an efficiency regression actionable -- a median tells you
        # something moved, these tell you where to look.
        ranked = gate_and_rank([s for s in report.scored if s.tier == tier])
        if len(ranked) > 1:
            print(f"  cheapest      {ranked[0].name} ({ranked[0].efficiency_tokens:,} tokens)")
            print(f"  costliest     {ranked[-1].name} ({ranked[-1].efficiency_tokens:,} tokens)")

        # Which stage failed, not just how many. ContextualRecall scores
        # retrieval and the rubric scores the answer, so a run failing mostly on
        # the former is a retrieval problem and on the latter an answering one --
        # different work. #304 noted this attribution is free; collapsing to a
        # single coverage number was discarding it.
        _print_attribution([s for s in report.scored if s.tier == tier and not s.covered])


if __name__ == "__main__":
    raise SystemExit(main())
