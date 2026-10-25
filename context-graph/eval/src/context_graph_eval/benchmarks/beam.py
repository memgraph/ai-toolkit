"""BEAM: long multi-domain chats, one user each, scored by BEAM's own rubric judge.

A question's score is ``beam_judge``'s (vendored verbatim, gpt-4.1-mini):
the mean of its rubric items at 1, 0.5 or 0, or for event ordering the
normalised Kendall tau. The headline is upstream's: the mean over abilities,
which weights every ability equally. Questions are asked with no date, as
upstream asks them.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from functools import cache
from statistics import mean
from typing import TYPE_CHECKING

from ..beam_judge import BEAM_JUDGE_MODEL, BeamJudge
from ..convert.beam import BeamChat, fetch_chat, to_goldens, to_session_fixtures
from ..results import Row
from ..scoring import efficiency_tokens, evidence_recall

if TYPE_CHECKING:
    from deepeval.dataset import Golden

    from ..answering import Retrieved
    from ..convert.longmemeval import SessionFixture
    from ..workloads import Workload

#: Every BEAM workload is the 100K size; the larger ones would not fit one graph.
SIZE = "100K"


@cache
def _chat(chat_id: int, revision: str) -> BeamChat:
    return fetch_chat(SIZE, chat_id, revision=revision)


def beam_headline(rows: list[Row]) -> float | None:
    """Upstream's report: the mean of each ability's mean, unjudged questions left out."""
    by_ability: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.score is not None:
            by_ability[row.question_type].append(row.score)
    means = [mean(scores) for scores in by_ability.values()]
    return mean(means) if means else None


class Beam:
    """BEAM 100K chats and their probing questions."""

    def __init__(self, *, judge_model: str | None = BEAM_JUDGE_MODEL, max_concurrent: int = 4, client=None):
        """
        Args:
            judge_model: BEAM's judge; None leaves every question unjudged.
            max_concurrent: Questions judged at once; each makes one call per rubric item.
            client: An OpenAI-compatible client, for tests.
        """
        self._judge = (
            BeamJudge(model=judge_model, max_concurrent=max_concurrent * 2, client=client) if judge_model else None
        )
        self._judge_model = judge_model

    @property
    def judge_model(self) -> str:
        return self._judge_model or "none"

    def questions(self, workload: Workload) -> list[Golden]:
        return [golden for chat_id in workload.chats for golden in to_goldens(_chat(chat_id, workload.revision))]

    def fixtures(self, workload: Workload, questions: list[Golden]) -> list[SessionFixture]:
        # Every question is asked of its chat's whole history, so every chat is loaded whole.
        return [
            fixture for chat_id in workload.chats for fixture in to_session_fixtures(_chat(chat_id, workload.revision))
        ]

    def user_id(self, question: Golden) -> str:
        return (question.additional_metadata or {})["user_id"]

    def today(self, question: Golden) -> str | None:
        return None

    async def judge(self, questions: list[Golden], retrieved: list[Retrieved]) -> list[Row]:
        async def one(question: Golden, result: Retrieved) -> Row:
            metadata = question.additional_metadata or {}
            ability = metadata["question_type"]
            rubric = list(metadata.get("rubric", []))
            verdict = await self._judge.judge(ability, rubric, result.answer) if self._judge is not None else None
            return Row(
                name=question.name or question.input,
                user_id=metadata["user_id"],
                question_type=ability,
                score=verdict.score if verdict else None,
                abstention=ability == "abstention",
                evidence_recall=evidence_recall(question.context, result.retrieval_context),
                efficiency_tokens=efficiency_tokens(result),
                latency_seconds=result.latency_seconds,
                recall_seconds=result.recall_seconds,
                errored=bool(result.errors),
                metrics={
                    "llm_judge_score": verdict.llm_judge_score if verdict else None,
                    "item_scores": list(verdict.item_scores) if verdict else [],
                    "ordering": verdict.ordering if verdict else None,
                },
                detail={
                    "question": question.input,
                    "expected": question.expected_output or "",
                    "answer": result.answer,
                    "rubric": rubric,
                    "judge_reasons": list(verdict.item_reasons) if verdict else [],
                    "errors": list(result.errors),
                    "retrieval_context": list(result.retrieval_context),
                },
            )

        return list(await asyncio.gather(*(one(q, r) for q, r in zip(questions, retrieved, strict=True))))

    def headline(self, rows: list[Row]) -> float | None:
        return beam_headline(rows)

    def notes(self, rows: list[Row]) -> list[str]:
        return []
