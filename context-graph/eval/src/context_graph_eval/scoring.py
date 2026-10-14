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

#: deepeval's name for ContextualRecallMetric: the retrieval signal. Reported,
#: never gating: a correct computed answer ("17 days") has no single row
#: supporting it, so it scores 0 (#397).
RETRIEVAL_SIGNAL = "Contextual Recall"

#: Which judge decides ``covered``, recorded on every saved run so compare()
#: refuses across a gate change: LongMemEval's own judge (#409), or the
#: deepeval answer rubric when that isn't configured.
COVERAGE_GATE = "official"
RUBRIC_GATE = "answer"

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
    #: Every judge score, keyed by metric name. ``coverage`` is the answer
    #: rubric's (see gate_score); RETRIEVAL_SIGNAL is kept beside it so a
    #: failure can still be attributed to retrieval or to the answer.
    #: Abstention questions carry no retrieval signal.
    metric_scores: dict[str, float] = field(default_factory=dict)
    #: The judge's own explanation behind each metric_scores entry, same keys.
    metric_reasons: dict[str, str] = field(default_factory=dict)
    #: Judge-free cross-checks against the answer key, not gated by coverage.
    bleu: float = 0.0
    f1: float = 0.0
    #: Wall-clock seconds for retrieval and answering, including a failed attempt.
    latency_seconds: float = 0.0
    #: Share of the question's evidence turns found in what retrieval returned;
    #: None when the question marks no evidence, as abstention questions don't.
    evidence_recall: float | None = None
    #: Who decided ``covered``: "official" (LongMemEval's judge) or "answer"
    #: (the deepeval answer rubric, whose score is ``coverage`` either way).
    judged_by: str = "answer"
    #: LongMemEval's judge's verdict; None when it didn't run or its call failed.
    official_correct: bool | None = None

    @property
    def judged(self) -> bool:
        """Whether the judge that decides ``covered`` produced a verdict -- not an outage."""
        return self.official_correct is not None if self.judged_by == "official" else bool(self.metric_scores)


@dataclass(frozen=True)
class TierReport:
    """Aggregate for one tier. Deliberately per-tier -- there is no overall."""

    #: Questions the judge actually scored. Excludes unscoreable ones, so the
    #: rate below is over what was measured rather than what was attempted.
    questions: int
    covered: int
    #: None, not 0.0, when nothing in the tier could be scored: an outage is not a measurement.
    coverage_rate: float | None
    median_efficiency_tokens: int | None
    abstention_total: int = 0
    abstention_correct: int = 0
    #: Questions with no metric scores at all. A judge failure, not a low score.
    unscored: int = 0
    #: Of the judge-scored questions carrying a retrieval signal, how many had
    #: Contextual Recall at the threshold. Reported, not gated.
    retrieval_supported: int = 0
    retrieval_judged: int = 0
    #: Means over every question in the tier, judged or not: these need no judge.
    mean_bleu: float | None = None
    mean_f1: float | None = None
    mean_latency_seconds: float | None = None
    #: The deepeval answer rubrics' own coverage count, shown beside the
    #: official judge's when that decides ``covered``; None when it doesn't.
    rubric_covered: int | None = None
    #: Questions with marked evidence, and how many had every evidence turn retrieved.
    evidence_judged: int = 0
    evidence_complete: int = 0
    mean_evidence_recall: float | None = None


@dataclass(frozen=True)
class RunReport:
    """A whole run, kept split by tier.

    There is intentionally no blended headline field: a single number across
    tiers is the thing #303 ruled out.
    """

    by_tier: dict[int, TierReport] = field(default_factory=dict)


def efficiency_tokens(retrieved: "Retrieved", tokenizer: str = DEFAULT_TOKENIZER) -> int:
    """Tokens of retrieval payload handed back to answer the question (#309), not the agent's own consumption.

    Raises rather than falling back when the tokenizer is unavailable: a
    fallback count would silently change units while the run still records
    the pinned tokenizer's name.
    """
    payload = "\n".join(retrieved.retrieval_context)
    if not payload:
        return 0
    return len(_encoding(tokenizer).encode(payload))


#: Scorer.sentence_bleu_score needs nltk's punkt_tab data, which nltk does not
#: bundle; downloaded lazily, once per process.
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
    """BLEU-1 between the answer and the expected output, via deepeval's ``Scorer``; 0.0 when either is empty.

    Unigram rather than BLEU-4: answers are a sentence or two, too short for
    4-grams to match regardless of quality.
    """
    if not answer or not expected_output:
        return 0.0
    import warnings

    from deepeval.scorer import Scorer

    _ensure_nltk_tokenizer_data()
    with warnings.catch_warnings():
        # nltk warns on zero 2/3/4-gram matches, which bleu1's weights never use.
        warnings.simplefilter("ignore", UserWarning)
        return Scorer.sentence_bleu_score(references=expected_output, prediction=answer, bleu_type="bleu1")


#: How much of an evidence passage must appear in a retrieved row for its turn
#: to count as found: its opening, whitespace-collapsed.
EVIDENCE_MATCH_CHARS = 200


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def evidence_recall(evidence: list[str] | None, retrieved: list[str]) -> float | None:
    """The share of ``evidence`` turns of which some passage appears in a ``retrieved`` row, or None without evidence.

    Evidence turns are the corpus's ``"role: content"`` strings (see
    ``convert.longmemeval._evidence_turns``). Unlike Contextual Recall this
    checks the turns an answer is built from, not the answer itself, so a
    computed answer ("17 days") whose parts were all retrieved scores 1.0.

    Recall shows a long turn as its matched passages, so any one of the
    turn's passages counts. Neither benchmark marks where in a turn its fact
    is, so for a long turn this is an upper bound: the passage shown may not
    be the one holding the fact.
    """
    from sessions_graph.passages import split_passages

    if not evidence:
        return None
    haystack = _collapsed("\n".join(retrieved))
    found = 0
    for turn in evidence:
        _, _, content = turn.partition(": ")
        openings = (_collapsed(passage)[:EVIDENCE_MATCH_CHARS] for passage in split_passages(content))
        if any(opening and opening in haystack for opening in openings):
            found += 1
    return found / len(evidence)


def token_f1_score(expected_output: str | None, answer: str) -> float:
    """Token-level F1 between the answer and the expected output, as LongMemEval's script reports it.

    Tokens are case-insensitive ``\\w+`` runs, so punctuation never splits a
    match, counted with multiplicity. 0.0 when either side is empty.
    """
    if not answer or not expected_output:
        return 0.0
    import re
    from collections import Counter

    expected_tokens = re.findall(r"\w+", expected_output.lower())
    answer_tokens = re.findall(r"\w+", answer.lower())
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


def gate_score(scores: dict[str, float]) -> float:
    """A question's coverage: its answer rubric's score, or 0.0 when the judge returned none.

    Every score but RETRIEVAL_SIGNAL is an answer rubric (Coverage, or
    Abstention), and the weakest of them gates.
    """
    answer = [score for name, score in scores.items() if name != RETRIEVAL_SIGNAL]
    return min(answer) if answer else 0.0


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
        # An unjudged row is an outage, not a failure: excluded from coverage
        # and gated efficiency, and counted as unscored instead.
        judge_scored_rows = [s for s in all_rows if s.judged]
        official = any(s.judged_by == "official" for s in all_rows)
        unscored = len(all_rows) - len(judge_scored_rows)
        covered = [s for s in judge_scored_rows if s.covered]
        abstentions = [s for s in judge_scored_rows if s.abstention]
        recall = [s.metric_scores[RETRIEVAL_SIGNAL] for s in judge_scored_rows if RETRIEVAL_SIGNAL in s.metric_scores]
        evidence = [s.evidence_recall for s in all_rows if s.evidence_recall is not None]
        by_tier[tier] = TierReport(
            unscored=unscored,
            questions=len(judge_scored_rows),
            covered=len(covered),
            coverage_rate=(len(covered) / len(judge_scored_rows) if judge_scored_rows else None),
            # Median, not mean: robust to one pathological payload.
            median_efficiency_tokens=(int(median([s.efficiency_tokens for s in covered])) if covered else None),
            abstention_total=len(abstentions),
            abstention_correct=sum(1 for s in abstentions if s.covered),
            retrieval_supported=sum(1 for score in recall if score >= DEFAULT_COVERAGE_THRESHOLD),
            retrieval_judged=len(recall),
            # Judge-free metrics, so over all_rows.
            mean_bleu=(mean(s.bleu for s in all_rows) if all_rows else None),
            mean_f1=(mean(s.f1 for s in all_rows) if all_rows else None),
            mean_latency_seconds=(mean(s.latency_seconds for s in all_rows) if all_rows else None),
            rubric_covered=(
                sum(1 for s in judge_scored_rows if s.metric_scores and s.coverage >= DEFAULT_COVERAGE_THRESHOLD)
                if official
                else None
            ),
            evidence_judged=len(evidence),
            evidence_complete=sum(1 for share in evidence if share == 1.0),
            mean_evidence_recall=(mean(evidence) if evidence else None),
        )
    return RunReport(by_tier=by_tier)


def enforce_retrieval_floor(scored: list[Scored], *, retrieved_tokens: dict[str, int]) -> list[Scored]:
    """Fail any question answered without retrieving anything.

    An empty payload is not evidence of recall, whatever the judge made of the
    answer -- ``ContextualRecallMetric`` is vacuously satisfied by an empty
    context. Abstention questions are exempt: for them an empty payload is
    correct.
    """
    floored: list[Scored] = []
    for row in scored:
        if row.abstention or retrieved_tokens.get(row.name, 0) > 0:
            floored.append(row)
            continue
        floored.append(replace(row, coverage=0.0, covered=False))
    return floored


#: The rubrics' evaluation steps, fixed rather than generated. deepeval copies
#: each metric per test case, so a GEval without explicit steps asks the judge
#: to write them afresh for every question: one extra judge call per question
#: per rubric, and every question graded against slightly different steps.
COVERAGE_STEPS = [
    "List every distinct fact stated in the expected output.",
    "For each fact, check whether the actual output states it, allowing paraphrase and equivalent values.",
    "Extra detail in the actual output does not reduce the score; a missing or contradicted fact does.",
    "Score high only if every fact is present, and low if any is missing.",
]
#: LongMemEval's own preference judge accepts a response that "recalls and
#: utilizes the user's personal information correctly" without reflecting every
#: point of the rubric; these steps follow it so scores stay comparable (#404).
PREFERENCE_STEPS = [
    "The expected output is a rubric describing a desired personalized response, not a list of facts.",
    "Check whether the actual output recalls the user's personal information the rubric points to "
    "and uses it to shape its answer.",
    "The actual output need not reflect every point of the rubric.",
    "A generic answer, one that ignores or contradicts the user's stated preferences, "
    'or "not in memory" is a failure.',
]
ABSTENTION_STEPS = [
    "The expected output says the information is not in memory.",
    "Check whether the actual output declines to answer, says the information is absent, or gives a correct zero count.",
    "A confident specific answer that is not a zero count is a failure.",
    "Mentioning what the user did say instead is a bonus, never a requirement.",
]


#: LongMemEval's question type whose expected output is a rubric for a personalized answer.
PREFERENCE_QUESTION_TYPE = "single-session-preference"
#: Which answer rubric judges a question; see build_metrics.
RUBRICS = ("answer", "abstention", "preference")


def rubric_for(metadata: dict[str, Any]) -> str:
    """The answer rubric a golden is judged by, from its ``additional_metadata``."""
    if metadata.get("abstention"):
        return "abstention"
    if metadata.get("question_type") == PREFERENCE_QUESTION_TYPE:
        return "preference"
    return "answer"


def build_metrics(judge: Any | None = None, *, rubric: str = "answer") -> list[Any]:
    """The deepeval metrics for ``rubric``: a deliberately minimal set (#304).

    "answer": a ``GEval`` Coverage rubric (does the answer state every
    expected fact; it gates) plus ``ContextualRecallMetric`` (reported).
    "preference": Coverage swapped for a Preference rubric, since the
    expected output describes a good personalized answer, as LongMemEval's
    own judge treats it. "abstention": an Abstention rubric alone --
    Contextual Recall scores an empty, correct context near zero.
    """
    from deepeval.metrics import ContextualRecallMetric, GEval
    from deepeval.test_case import LLMTestCaseParams

    metrics: list[Any] = []
    if rubric == "abstention":
        # Scores not fabricating. Upstream pairs the refusal with a contrastive
        # fact ("your cat Luna but not your hamster"); naming it is not required.
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
                evaluation_steps=ABSTENTION_STEPS,
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
    if rubric == "preference":
        metrics.append(
            GEval(
                name="Preference",
                criteria=(
                    "The expected output is a rubric for a desired personalized response. Does the actual "
                    "output satisfy it? It need not reflect every point of the rubric; it is correct as long "
                    "as it recalls and uses the user's personal information correctly."
                ),
                evaluation_steps=PREFERENCE_STEPS,
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
    metrics.append(
        GEval(
            name="Coverage",
            criteria=(
                "Does the actual output contain every fact present in the expected output? "
                "Extra detail is acceptable. A missing fact is a failure."
            ),
            evaluation_steps=COVERAGE_STEPS,
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
        # Named so the judge's results can be matched back to their question:
        # deepeval returns them in completion order, not input order.
        name=golden.name,
        input=golden.input,
        actual_output=retrieved.answer,
        expected_output=golden.expected_output,
        retrieval_context=retrieved.retrieval_context or None,
        context=golden.context,
    )
