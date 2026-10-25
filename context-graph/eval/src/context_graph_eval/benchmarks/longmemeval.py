"""LongMemEval-S, and the gold slice judged the same way.

A question's score is LongMemEval's own judge's verdict (``official_judge``,
vendored verbatim, gpt-4o-2024-08-06): 1 when correct, 0 when not. The
deepeval rubrics (``scoring.build_metrics``) are optional diagnostics beside
it, and decide only questions the official judge has no template for -- the
gold slice. A question answered from an empty retrieval scores 0, except an
abstention, for which retrieving nothing is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any

from .. import official_judge
from ..official_judge import OFFICIAL_JUDGE_MODEL
from ..results import Row
from ..scoring import (
    DEFAULT_COVERAGE_THRESHOLD,
    RETRIEVAL_SIGNAL,
    RUBRICS,
    bleu_score,
    efficiency_tokens,
    evidence_recall,
    gate_score,
    rubric_for,
    token_f1_score,
)

if TYPE_CHECKING:
    from deepeval.dataset import Golden

    from ..answering import Retrieved
    from ..convert.longmemeval import SessionFixture
    from ..workloads import Workload

#: Upstream's question types, in its paper's order.
QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
)


def accuracy(rows: list[Row]) -> float | None:
    """LongMemEval's headline: the share of judged questions answered correctly."""
    scores = [row.score for row in rows if row.score is not None]
    return mean(scores) if scores else None


@dataclass(frozen=True)
class _Judged:
    """One question's deepeval output: per-metric scores and the judge's reasons, kept as one unit."""

    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


class LongMemEval:
    """LongMemEval-S questions over their own haystacks; also the gold slice's judge."""

    def __init__(
        self,
        *,
        official_judge_model: str | None = OFFICIAL_JUDGE_MODEL,
        rubric_judge: Any | None = None,
        max_concurrent: int = 4,
    ):
        """
        Args:
            official_judge_model: LongMemEval's judge; None leaves every
                question to the rubric judge, or unjudged without one.
            rubric_judge: A deepeval model for the rubrics, or None to skip them.
            max_concurrent: Judge calls in flight at once.
        """
        self._official_judge_model = official_judge_model
        self._rubric_judge = rubric_judge
        self._max_concurrent = max_concurrent

    @property
    def judge_model(self) -> str:
        return self._official_judge_model or "none"

    def questions(self, workload: Workload) -> list[Golden]:
        from ..corpus import read_corpus

        if workload.benchmark == "gold-slice":
            from ..goldslice import gold_slice_goldens

            return gold_slice_goldens()
        if workload.corpus_path is None or not workload.corpus_path.exists():
            raise FileNotFoundError(
                f"{workload.name}: no corpus at {workload.corpus_path}. Build it with `context-graph-eval build-corpus`."
            )
        return read_corpus(workload.corpus_path)

    def fixtures(self, workload: Workload, questions: list[Golden]) -> list[SessionFixture]:
        from ..convert.longmemeval import fetch, haystack_path, load_raw, to_session_fixtures

        if workload.planted:
            return []
        # The haystack is not committed (#302); the pinned revision makes it immutable instead.
        records = load_raw(fetch("s", workload.revision, dest=haystack_path("s", workload.revision)))
        wanted = {question.name for question in questions}
        return [
            fixture
            for record in records
            if record["question_id"] in wanted
            for fixture in to_session_fixtures(record, max_sessions=workload.max_sessions_per_question)
        ]

    def user_id(self, question: Golden) -> str:
        # Each LongMemEval question is its own user's history (to_session_fixtures).
        user = (question.additional_metadata or {}).get("user_id") or question.name
        if not user:
            raise ValueError(f"question {question.input!r} has no name to scope recall to")
        return user

    def today(self, question: Golden) -> str | None:
        # The corpus's question_date, the "now" relative dates count from.
        return (question.additional_metadata or {}).get("question_date")

    async def judge(self, questions: list[Golden], retrieved: list[Retrieved]) -> list[Row]:
        official = await self._official_labels(questions, retrieved)
        rubrics = self._rubric_scores(questions, retrieved) if self._rubric_judge is not None else {}
        return [
            self._row(question, result, official, rubrics.get(question.name or ""))
            for question, result in zip(questions, retrieved, strict=True)
        ]

    def headline(self, rows: list[Row]) -> float | None:
        return accuracy(rows)

    def notes(self, rows: list[Row]) -> list[str]:
        lines: list[str] = []
        abstentions = [row for row in rows if row.abstention and row.score is not None]
        if abstentions:
            lines.append(f"abstention    {sum(1 for row in abstentions if row.passed)}/{len(abstentions)} correct")
        rubric = [row.metrics["rubric_coverage"] for row in rows if row.metrics.get("rubric_coverage") is not None]
        if rubric:
            passed = sum(1 for score in rubric if score >= DEFAULT_COVERAGE_THRESHOLD)
            lines.append(f"rubrics       {passed}/{len(rubric)} (deepeval answer rubrics, reported beside)")
        recall = [row.metrics["contextual_recall"] for row in rows if row.metrics.get("contextual_recall") is not None]
        if recall:
            passed = sum(1 for score in recall if score >= DEFAULT_COVERAGE_THRESHOLD)
            lines.append(f"retrieval     contextual recall passed on {passed}/{len(recall)} (reported, not gated)")
        bleu = [row.metrics["bleu"] for row in rows if "bleu" in row.metrics]
        f1 = [row.metrics["f1"] for row in rows if "f1" in row.metrics]
        if bleu and f1:
            lines.append(f"bleu / f1     {mean(bleu):.3f} / {mean(f1):.3f}")
        missed = [row for row in rows if row.score is not None and not row.passed and row.evidence_recall is not None]
        if missed:
            complete = sum(1 for row in missed if row.evidence_recall == 1.0)
            none = sum(1 for row in missed if row.evidence_recall == 0)
            lines.append(
                f"misses        {len(missed)}: evidence all retrieved on {complete}, "
                f"some on {len(missed) - complete - none}, none on {none}"
            )
        return lines

    def _row(
        self,
        question: Golden,
        result: Retrieved,
        official: dict[str, bool | None],
        rubric: _Judged | None,
    ) -> Row:
        metadata = question.additional_metadata or {}
        abstention = bool(metadata.get("abstention"))
        name = question.name or question.input
        tokens = efficiency_tokens(result)
        rubric_coverage = gate_score(rubric.scores) if rubric and rubric.scores else None
        if name in official:
            # A failed official-judge call is unjudged, never a fallback to our own rubrics.
            verdict = official[name]
            score = None if verdict is None else float(verdict)
        elif rubric_coverage is not None:
            score = float(rubric_coverage >= DEFAULT_COVERAGE_THRESHOLD)
        else:
            score = None
        # An empty payload is not evidence of recall, whatever the judge made of the answer.
        if score is not None and not abstention and tokens == 0:
            score = 0.0
        return Row(
            name=name,
            user_id=self.user_id(question),
            question_type=metadata.get("question_type", ""),
            score=score,
            abstention=abstention,
            evidence_recall=None if abstention else evidence_recall(question.context, result.retrieval_context),
            efficiency_tokens=tokens,
            latency_seconds=result.latency_seconds,
            recall_seconds=result.recall_seconds,
            errored=bool(result.errors),
            metrics={
                "judged_by": "official" if name in official else ("rubric" if rubric_coverage is not None else None),
                "official_correct": official.get(name),
                "rubric_coverage": rubric_coverage,
                "contextual_recall": rubric.scores.get(RETRIEVAL_SIGNAL) if rubric else None,
                "bleu": bleu_score(question.expected_output, result.answer),
                "f1": token_f1_score(question.expected_output, result.answer),
            },
            detail={
                "question": question.input,
                "expected": question.expected_output or "",
                "answer": result.answer,
                "judge_reasons": rubric.reasons if rubric else {},
                "errors": list(result.errors),
                "retrieval_context": list(result.retrieval_context),
            },
        )

    async def _official_labels(self, questions: list[Golden], retrieved: list[Retrieved]) -> dict[str, bool | None]:
        """LongMemEval's verdict per question it has a template for; {} when it is off."""
        if self._official_judge_model is None:
            return {}
        prompts = {}
        for question, result in zip(questions, retrieved, strict=True):
            metadata = question.additional_metadata or {}
            abstention = bool(metadata.get("abstention"))
            if question.name and official_judge.judges(metadata.get("question_type"), abstention=abstention):
                prompts[question.name] = official_judge.anscheck_prompt(
                    metadata.get("question_type", ""),
                    question.input,
                    question.expected_output or "",
                    result.answer,
                    abstention=abstention,
                )
        return await official_judge.judge_all(
            prompts, model=self._official_judge_model, max_concurrent=self._max_concurrent
        )

    def _rubric_scores(self, questions: list[Golden], retrieved: list[Retrieved]) -> dict[str, _Judged]:
        """Score answers with the deepeval rubrics, one pass per rubric (``scoring.rubric_for``).

        deepeval's ``evaluate`` drives its own event loop, so it runs after
        every answer is in rather than inside the answering loop.
        """
        paired = list(zip(questions, retrieved, strict=True))
        judged: dict[str, _Judged] = {}
        for rubric in RUBRICS:
            group = [(q, r) for q, r in paired if rubric_for(q.additional_metadata or {}) == rubric]
            if group:
                judged.update(self._judge_group(group, rubric=rubric))
        return judged

    def _judge_group(self, group: list[tuple[Golden, Retrieved]], *, rubric: str) -> dict[str, _Judged]:
        from deepeval import evaluate
        from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

        from ..scoring import build_metrics, to_test_case

        result = evaluate(
            test_cases=[to_test_case(q, r) for q, r in group],
            metrics=build_metrics(self._rubric_judge, rubric=rubric),
            async_config=AsyncConfig(max_concurrent=self._max_concurrent),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
            # One unscoreable question must not abandon the run.
            error_config=ErrorConfig(ignore_errors=True),
        )
        # Matched by name, never position: deepeval returns results in completion order.
        by_name = {test_result.name: test_result for test_result in result.test_results}
        judged: dict[str, _Judged] = {}
        for question, _ in group:
            test_result = by_name.get(question.name)
            if test_result is None or question.name is None:
                continue
            metrics = test_result.metrics_data or []
            judged[question.name] = _Judged(
                scores={m.name: m.score for m in metrics if m.score is not None},
                reasons={m.name: m.reason for m in metrics if m.reason},
            )
        return judged
