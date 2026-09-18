"""Score retrieval results against their Goldens, and aggregate a run.

The rubric splits by mechanism (#304): the LLM judges *quality* -- did retrieval
surface the facts the answer needed -- while plain code counts *cost*. Asking a
model to grade a number you can count adds variance to the headline for no
information, and makes an efficiency regression arguable rather than factual.

Coverage is a hard gate and efficiency ranks within it (#309). Efficiency alone
is trivially gamed by returning nothing, and a weighted composite would let a
retrieval change trade real coverage for token savings while the headline stayed
flat -- exactly the regression this exists to catch.

Tiers are aggregated separately and never blended (#303): Tier 1 is adopted from
upstream and asks whether recall works mechanically; Tier 2 is authored and asks
whether it works for what is actually being built. One averaged number would let
an organizational-recall regression hide behind a personal-memory gain.
"""

from dataclasses import dataclass, field, replace
from statistics import mean, median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from deepeval.dataset import Golden

    from .retrieval import Retrieved

#: Coverage at or above this counts as cleared. deepeval metrics express the
#: same idea through their own ``threshold``; kept explicit here so the gate is
#: visible rather than buried in metric configuration.
DEFAULT_COVERAGE_THRESHOLD = 0.7

#: Tokenizer for the efficiency count. Pinned for the same reason #304 pins the
#: judge model: a tokenizer change silently shifts every efficiency number, and
#: two runs measured differently are not comparable.
DEFAULT_TOKENIZER = "cl100k_base"


@dataclass(frozen=True)
class Scored:
    """One question's outcome."""

    name: str
    tier: int
    coverage: float
    covered: bool
    efficiency_tokens: int
    abstention: bool = False
    answer: str = ""
    #: Per-metric scores behind ``coverage``. Kept because ``coverage`` is
    #: min() of them, which gates correctly but discards which stage failed --
    #: ContextualRecall scores retrieval, the GEval rubric scores the answer.
    #: #304 noted that attribution "falls out for nothing"; collapsing to one
    #: number was throwing it away. Absent for abstention questions, which are
    #: judged on the rubric alone.
    metric_scores: dict[str, float] = field(default_factory=dict)
    #: Deterministic, judge-free cross-checks against the answer key -- not
    #: gated by coverage, unlike efficiency_tokens (#309's gate is specific
    #: to that axis). Computed regardless of whether a judge ran, same
    #: reasoning as efficiency_tokens: no LLM call needed for either.
    bleu: float = 0.0
    f1: float = 0.0
    #: Wall-clock seconds retrieve() took for this question -- unlike bleu/f1
    #: above, not deterministic (it's timing, not counting), but likewise
    #: independent of whether a judge ran. Reflects real elapsed time even for
    #: a question whose retrieval raised outright: runner._retrieve_all times
    #: the attempt itself in that case, since retrieve() has no Retrieved to
    #: report the elapsed time through when it never returns.
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class TierReport:
    """Aggregate for one tier. Deliberately per-tier -- there is no overall."""

    #: Questions the judge actually scored. Excludes unscoreable ones, so the
    #: rate below is over what was measured rather than what was attempted.
    questions: int
    covered: int
    #: None when nothing in the tier could be scored. Deliberately not 0.0:
    #: observed live, the judge's provider ran out of credit, every metric
    #: errored, and the run reported "coverage 0/2 (0%)" -- an outage presented
    #: as a measurement.
    coverage_rate: float | None
    median_efficiency_tokens: int | None
    abstention_total: int = 0
    abstention_correct: int = 0
    #: Questions with no metric scores at all. A judge failure, not a low score.
    unscored: int = 0
    #: Mean, not median: unlike efficiency_tokens (#309's gate, deliberately
    #: robust to one pathological payload), these are the blog-comparison
    #: metrics themselves -- a single outlier answer should show up in the
    #: mean, not be shrugged off by a robust statistic. Aggregated over every
    #: question in the tier, not just the judge-scored subset coverage and
    #: efficiency use above: BLEU, F1 and latency are their own judge-free
    #: signal, so a judge-free run -- or a judge that failed on some
    #: questions -- must still report all three rather than three Nones.
    mean_bleu: float | None = None
    mean_f1: float | None = None
    mean_latency_seconds: float | None = None


@dataclass(frozen=True)
class RunReport:
    """A whole run, kept split by tier.

    There is intentionally no blended headline field: a single number across
    tiers is the thing #303 ruled out.
    """

    by_tier: dict[int, TierReport] = field(default_factory=dict)


def efficiency_tokens(retrieved: "Retrieved", tokenizer: str = DEFAULT_TOKENIZER) -> int:
    """Tokens handed back to answer the question (#309).

    Counts the retrieval payload, not the agent's own consumption: for the same
    answer quality, returning less is better.

    Raises rather than falling back if the tokenizer is unavailable. There used
    to be a whitespace-splitting fallback, which was worse than useless: it
    produced numbers roughly a third smaller while the run still recorded the
    pinned tokenizer's name, so two runs counted in different units compared
    cleanly and ``compare()`` -- which checks that recorded name -- saw a match.
    An efficiency figure that quietly changes units is more dangerous than one
    that fails, so ``tiktoken`` is a declared dependency and its absence is an
    error.
    """
    payload = "\n".join(retrieved.retrieval_context)
    if not payload:
        return 0
    return len(_encoding(tokenizer).encode(payload))


#: word_tokenize (which Scorer.sentence_bleu_score calls) needs this data
#: package on disk; nltk does not bundle it. Downloaded lazily, once per
#: process, rather than assumed present -- a fresh CI runner or a
#: contributor's first run has no reason to already have it, and deepeval's
#: own import of nltk is guarded by a try/except that only prints on
#: failure, so an absent download would otherwise surface as a LookupError
#: raised deep inside nltk's tokenizer, not at an obvious call site.
_nltk_tokenizer_data_ready = False


def _ensure_nltk_tokenizer_data() -> None:
    global _nltk_tokenizer_data_ready
    if _nltk_tokenizer_data_ready:
        return
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _nltk_tokenizer_data_ready = True


def bleu_score(expected_output: str | None, answer: str) -> float:
    """BLEU-1 similarity between the retrieved answer and the expected
    output -- a standard NLP metric other memory-benchmark suites (LoCoMo,
    LongMemEval, BEAM) also report, added so a run here is comparable on the
    same axis.

    Not reimplemented: this calls deepeval's own ``Scorer.sentence_bleu_score``
    (nltk's n-gram BLEU under the hood, already a workspace dependency via
    deepeval itself) rather than hand-rolling n-gram/brevity-penalty math
    deepeval already ships correctly.

    BLEU-1 (unigram overlap), not BLEU-4: an answer here is a sentence or
    two, not a paragraph, so there usually is not enough text for 4-grams to
    match at all -- BLEU-4 would floor near zero regardless of answer
    quality and measure sentence length more than correctness.

    Returns 0.0 for an empty answer or expected_output rather than raising:
    an abstention question's correct answer can legitimately be a short
    refusal, and a failed retrieval's answer can legitimately be empty --
    both are real "no similarity" outcomes, not scoring errors.
    """
    if not answer or not expected_output:
        return 0.0
    import warnings

    from deepeval.scorer import Scorer

    _ensure_nltk_tokenizer_data()
    with warnings.catch_warnings():
        # nltk warns whenever the candidate has zero 2/3/4-gram matches, even
        # though bleu1's weights=(1,0,0,0) never uses those orders in the
        # returned score -- expected for a short answer (most of them here),
        # not a real problem, and noisy enough over a whole run to bury
        # anything that IS worth seeing in the output.
        warnings.simplefilter("ignore", UserWarning)
        return Scorer.sentence_bleu_score(references=expected_output, prediction=answer, bleu_type="bleu1")


def token_f1_score(expected_output: str | None, answer: str) -> float:
    """Token-level F1 between the retrieved answer and the expected output --
    precision and recall over shared tokens, the same formula LongMemEval's
    own evaluation script and the blog above report as "F1 Score".

    deepeval's ``Scorer`` ships ROUGE (summarization-oriented, weights
    matches differently) and BERTScore (needs its own model download) but
    not this specific, simpler formula, so it is computed directly here --
    not a reimplementation of a metric deepeval already provides, since
    neither of deepeval's options is the same metric.

    Case-insensitive, whitespace-tokenized, and counts token *multiplicity*
    (``Counter`` intersection, not set intersection): "the the the" against
    "the" should not score a perfect match on either precision or recall.
    """
    if not answer or not expected_output:
        return 0.0
    from collections import Counter

    expected_tokens = expected_output.lower().split()
    answer_tokens = answer.lower().split()
    if not expected_tokens or not answer_tokens:
        return 0.0
    overlap = sum((Counter(expected_tokens) & Counter(answer_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def tokenizer_in_use(tokenizer: str = DEFAULT_TOKENIZER) -> str:
    """The tokenizer name to record on a run, verified to actually load."""
    _encoding(tokenizer)
    return tokenizer


def _encoding(tokenizer: str = DEFAULT_TOKENIZER):
    import tiktoken

    return tiktoken.get_encoding(tokenizer)


def gate_and_rank(scored: list[Scored]) -> list[Scored]:
    """Questions that cleared coverage, cheapest payload first.

    Anything that failed coverage is dropped rather than ranked: it has no
    meaningful efficiency, and letting a zero-token failure top the ranking is
    precisely the gaming this guards against.
    """
    passed = [s for s in scored if s.covered]
    return sorted(passed, key=lambda s: (s.efficiency_tokens, s.name))


def aggregate(scored: list[Scored]) -> RunReport:
    """Summarise a run, per tier."""
    by_tier: dict[int, TierReport] = {}
    for tier in sorted({s.tier for s in scored}):
        all_rows = [s for s in scored if s.tier == tier]
        # A row with no metric scores was never judged -- the judge errored, or
        # none ran. Counting it as a failure turns an outage into a reported
        # score, so it is excluded from the rate and surfaced separately.
        # Coverage and (gated) efficiency need a judge's verdict, so they are
        # aggregated over this judge-scored subset only, not over all_rows.
        judge_scored_rows = [s for s in all_rows if s.metric_scores]
        unscored = len(all_rows) - len(judge_scored_rows)
        covered = [s for s in judge_scored_rows if s.covered]
        abstentions = [s for s in judge_scored_rows if s.abstention]
        by_tier[tier] = TierReport(
            unscored=unscored,
            questions=len(judge_scored_rows),
            covered=len(covered),
            coverage_rate=(len(covered) / len(judge_scored_rows) if judge_scored_rows else None),
            # Median, not mean: one pathological payload should not drag the
            # number that gets compared across schema versions.
            median_efficiency_tokens=(int(median([s.efficiency_tokens for s in covered])) if covered else None),
            abstention_total=len(abstentions),
            abstention_correct=sum(1 for s in abstentions if s.covered),
            # Over all_rows, not judge_scored_rows: BLEU/F1/latency need no
            # judge, so a judge-free run (or one where the judge errored on
            # some questions) must still aggregate them over everything that
            # was actually scored, rather than over an empty or shrunken
            # judge-scored subset (#342).
            mean_bleu=(mean(s.bleu for s in all_rows) if all_rows else None),
            mean_f1=(mean(s.f1 for s in all_rows) if all_rows else None),
            mean_latency_seconds=(mean(s.latency_seconds for s in all_rows) if all_rows else None),
        )
    return RunReport(by_tier=by_tier)


def enforce_retrieval_floor(scored: list[Scored], *, retrieved_tokens: dict[str, int]) -> list[Scored]:
    """Fail any question that was answered without retrieving anything.

    A question whose retrieval payload was empty told the graph nothing and
    learned nothing from it, so whatever the judge made of the answer, it is not
    evidence of recall.

    The judge cannot catch this on its own. ``ContextualRecallMetric`` asks
    whether the retrieved context supports the expected output, and an empty
    context satisfies that vacuously -- observed live in a calibration run:
    ``gpt4_59149c77``, a temporal-reasoning question whose expected answer is
    "7 days", retrieved zero tokens, replied "not in memory", and scored 1.0 on
    both metrics. A perfect pass for consulting nothing.

    That inflates coverage, which is the direction nobody audits: a zero gets
    investigated, a pass gets believed.

    Abstention questions are exempt, and must be. For those the correct answer
    really is "not in memory", so an empty payload is right rather than
    degenerate -- applying the floor to them would make them unpassable by
    construction, which is a bug this rubric has already had once.
    """
    floored: list[Scored] = []
    for row in scored:
        if row.abstention or retrieved_tokens.get(row.name, 0) > 0:
            floored.append(row)
            continue
        floored.append(replace(row, coverage=0.0, covered=False))
    return floored


def build_metrics(judge: Any | None = None, *, abstention: bool = False) -> list[Any]:
    """The judged half of the rubric: a deliberately minimal pair (#304).

    ``ContextualRecallMetric`` scores retrieval-side coverage -- its required
    params are exactly the Golden fields #302 locked -- and one ``GEval`` rubric
    scores the answer itself, since no built-in asks whether ``actual_output``
    contains every fact in ``expected_output``, which is the real question when
    an answer key exists.

    ``Faithfulness`` and ``AnswerRelevancy`` are deliberately omitted: both
    exist mainly for the no-ground-truth case, and every extra metric is another
    judge call per question, multiplied again by re-running per schema
    candidate.

    **Abstention questions drop ContextualRecall entirely.** That metric asks
    whether the retrieved context supports the expected output -- but for a
    question whose correct answer is "that isn't in memory", the correct
    retrieved context is *empty*. It therefore scores near zero by
    construction, and since coverage takes the weakest metric, it made every
    abstention question unpassable however well the agent behaved. Measured
    before this fix: abstention scored 0/8 while the agent had correctly
    declined on at least four. Only the rubric, which knows to require a
    refusal, applies to these.
    """
    from deepeval.metrics import ContextualRecallMetric, GEval
    from deepeval.test_case import LLMTestCaseParams

    metrics: list[Any] = []
    if abstention:
        # Its own rubric, because these questions measure a different thing.
        # Upstream pairs the refusal with a contrastive fact -- "You mentioned
        # your cat Luna but not your hamster" -- so the Coverage rubric below,
        # which demands every fact in the expected output, marked a correct
        # "not in memory" down for omitting the near-miss detail. Measured: the
        # agent declined on 5/8, 3/8 and 5/8 across three runs and scored 0/8
        # every time, landing at 0.3-0.6 against a 0.7 gate.
        #
        # What abstention is for is not fabricating an answer, so that is what
        # is scored. Naming the near-miss is a finer-grained skill and would be
        # its own metric, not a silent precondition of this one.
        return [
            GEval(
                name="Abstention",
                criteria=(
                    "The expected output states that the information is not in memory. "
                    "Did the actual output decline to answer, rather than inventing one? "
                    "A refusal, a statement that the information is absent, or a correct "
                    "zero count all pass. A confident specific answer is a failure. "
                    "Naming what the user did mention instead is a bonus, not a requirement."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=DEFAULT_COVERAGE_THRESHOLD,
                model=judge,
            )
        ]

    metrics.append(ContextualRecallMetric(threshold=DEFAULT_COVERAGE_THRESHOLD, model=judge))
    metrics.append(
        GEval(
            name="Coverage",
            criteria=(
                "Does the actual output contain every fact present in the expected output? "
                "Extra detail is acceptable. A missing fact is a failure."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            model=judge,
        )
    )
    return metrics


def to_test_case(golden: "Golden", retrieved: "Retrieved") -> Any:
    """Pair a Golden with what retrieval produced, for the judge."""
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=golden.input,
        actual_output=retrieved.answer,
        expected_output=golden.expected_output,
        retrieval_context=retrieved.retrieval_context or None,
        context=golden.context,
    )
