"""Per-question measurements shared by the benchmarks: cost, evidence, and the deepeval rubrics.

The split is by mechanism (#304): an LLM judges *quality* -- did the answer
state what the key asks -- while plain code counts *cost* and *evidence*.
Asking a model to grade a number you can count adds variance for no
information, and makes an efficiency regression arguable rather than factual.

The deepeval rubrics here are diagnostics: each benchmark's own judge decides
its score (``benchmarks/``), and the rubrics decide only questions no
upstream judge covers -- the gold slice.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from deepeval.dataset import Golden

    from .answering import Retrieved

#: Coverage at or above this counts as cleared. deepeval metrics express the
#: same idea through their own ``threshold``; kept explicit here so the gate is
#: visible rather than buried in metric configuration.
DEFAULT_COVERAGE_THRESHOLD = 0.7

#: deepeval's name for ContextualRecallMetric: the retrieval signal. Reported,
#: never gating: a correct computed answer ("17 days") has no single row
#: supporting it, so it scores 0 (#397).
RETRIEVAL_SIGNAL = "Contextual Recall"

#: Tokenizer for the efficiency count. Pinned for the same reason #304 pins the
#: judge model: a tokenizer change silently shifts every efficiency number, and
#: two runs measured differently are not comparable.
DEFAULT_TOKENIZER = "cl100k_base"


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
