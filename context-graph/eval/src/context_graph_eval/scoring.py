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

from dataclasses import dataclass, field
from statistics import median
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
    reason: str = ""


@dataclass(frozen=True)
class TierReport:
    """Aggregate for one tier. Deliberately per-tier -- there is no overall."""

    questions: int
    covered: int
    coverage_rate: float
    median_efficiency_tokens: int | None
    abstention_total: int = 0
    abstention_correct: int = 0


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
        rows = [s for s in scored if s.tier == tier]
        covered = [s for s in rows if s.covered]
        abstentions = [s for s in rows if s.abstention]
        by_tier[tier] = TierReport(
            questions=len(rows),
            covered=len(covered),
            coverage_rate=len(covered) / len(rows) if rows else 0.0,
            # Median, not mean: one pathological payload should not drag the
            # number that gets compared across schema versions.
            median_efficiency_tokens=(int(median([s.efficiency_tokens for s in covered])) if covered else None),
            abstention_total=len(abstentions),
            abstention_correct=sum(1 for s in abstentions if s.covered),
        )
    return RunReport(by_tier=by_tier)


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
    if not abstention:
        metrics.append(ContextualRecallMetric(threshold=DEFAULT_COVERAGE_THRESHOLD, model=judge))
    metrics.append(
        GEval(
            name="Coverage",
            criteria=(
                "Does the actual output contain every fact present in the expected output? "
                "Extra detail is acceptable. A missing fact is a failure. "
                "If the expected output says the information is not in memory, then the actual "
                "output must decline to answer -- a confident answer is a failure."
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
