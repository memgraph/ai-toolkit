"""Benchmark adapters: what differs between benchmarks, behind one interface the pipeline drives.

The pipeline (``pipeline.py``) is the same for every benchmark: inject the
haystack, reconcile it, embed it, recall and answer each question. A
benchmark brings only its questions and haystack, its own judge, and how its
question scores add up to a headline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from deepeval.dataset import Golden

    from ..answering import Retrieved
    from ..convert.longmemeval import SessionFixture
    from ..results import Row
    from ..workloads import Workload


class Benchmark(Protocol):
    """One upstream benchmark, adapted to the pipeline."""

    @property
    def judge_model(self) -> str:
        """The judge that decides ``Row.score``, recorded on every run."""
        ...

    def questions(self, workload: Workload) -> list[Golden]:
        """The workload's questions, in a fixed order."""
        ...

    def fixtures(self, workload: Workload, questions: list[Golden]) -> list[SessionFixture]:
        """The haystack sessions ``questions`` are asked of."""
        ...

    def user_id(self, question: Golden) -> str:
        """Whose history the question is asked of; recall is scoped to it."""
        ...

    def today(self, question: Golden) -> str | None:
        """The date the question is asked on, or None when the benchmark gives none."""
        ...

    async def judge(self, questions: list[Golden], retrieved: list[Retrieved]) -> list[Row]:
        """Score every answer with the benchmark's own judge."""
        ...

    def headline(self, rows: list[Row]) -> float | None:
        """The one number the benchmark reports, from its question scores."""
        ...

    def notes(self, rows: list[Row]) -> list[str]:
        """Benchmark-specific report lines beyond the shared summary."""
        ...


def headline(benchmark: str, rows: list[Row]) -> float | None:
    """``benchmark``'s headline over ``rows``, without building its judge."""
    from .beam import beam_headline
    from .longmemeval import accuracy

    return beam_headline(rows) if benchmark == "beam-100k" else accuracy(rows)


def type_order(benchmark: str) -> tuple[str, ...]:
    """The order ``benchmark`` reports its question types in."""
    from ..convert.beam import ABILITIES
    from .longmemeval import QUESTION_TYPES

    return ABILITIES if benchmark == "beam-100k" else QUESTION_TYPES
