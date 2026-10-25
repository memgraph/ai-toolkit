"""The fixed workloads a run asks: official ones we publish, and dev ones we iterate on.

An **official** workload is a whole benchmark, uncapped: its result is the
number we communicate (the README's "Official numbers", backed by a file in
``baselines/``). A **dev** workload is a small, fixed slice of the same
benchmark, sized so a construction change rebuilds in hours and a recall
change re-asks in minutes on the kept graph. Dev scores are for comparing
two versions of the system with each other, never for quoting: a capped
haystack makes LongMemEval easier, and four BEAM chats are not BEAM.

Each workload owns a named eval instance (``instance.py``), so a built graph
stays where it is between runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .convert.beam import DEFAULT_REVISION as BEAM_REVISION
from .convert.longmemeval import DEFAULT_REVISION as LONGMEMEVAL_REVISION

if TYPE_CHECKING:
    from deepeval.dataset import Golden

EVAL_DIR = Path(__file__).resolve().parents[2]

#: The order shards are drawn in: the official LongMemEval-S run shuffled its
#: corpus with this seed and asked it 20 at a time.
SHARD_SEED = 0


@dataclass(frozen=True)
class Workload:
    """A named, fixed set of questions and the haystack they are asked of."""

    name: str
    kind: Literal["official", "dev"]
    benchmark: Literal["longmemeval-s", "beam-100k", "gold-slice"]
    description: str
    #: The host port of the workload's own eval instance.
    port: int
    revision: str
    #: LongMemEval: the committed corpus file, relative to the eval package.
    corpus: str | None = None
    #: BEAM: which 100K chats, each one user's history.
    chats: tuple[int, ...] = ()
    #: LongMemEval: distractor sessions kept per question; evidence is always kept.
    max_sessions_per_question: int | None = None
    #: Questions per graph when the whole workload doesn't fit in memory at once.
    shard_size: int | None = None
    #: Staged by a live Claude Code session (``context-graph-eval gold-slice``), not injected.
    planted: bool = False

    @property
    def instance(self) -> str:
        """The name of the workload's Docker container."""
        return f"context-graph-eval-{self.name}"

    @property
    def corpus_path(self) -> Path | None:
        return EVAL_DIR / self.corpus if self.corpus else None


WORKLOADS: dict[str, Workload] = {
    workload.name: workload
    for workload in (
        Workload(
            name="lme-s",
            kind="official",
            benchmark="longmemeval-s",
            description="LongMemEval-S, all 500 questions, full haystacks; 25 shards of 20",
            port=7791,
            revision=LONGMEMEVAL_REVISION,
            corpus="corpus/longmemeval-s-500.jsonl",
            # One graph for 500 questions (~24k sessions) does not fit in 16 GB.
            shard_size=20,
        ),
        Workload(
            name="beam-100k",
            kind="official",
            benchmark="beam-100k",
            description="BEAM 100K, all 20 chats and 400 questions",
            port=7793,
            revision=BEAM_REVISION,
            chats=tuple(range(1, 21)),
        ),
        Workload(
            name="lme-s-dev",
            kind="dev",
            benchmark="longmemeval-s",
            description="LongMemEval-S, 50 questions stratified by type, haystacks capped at 10 sessions",
            port=7792,
            revision=LONGMEMEVAL_REVISION,
            corpus="corpus/longmemeval-s-dev.jsonl",
            max_sessions_per_question=10,
        ),
        Workload(
            name="beam-100k-dev",
            kind="dev",
            benchmark="beam-100k",
            description="BEAM 100K, chats 1, 6, 11 and 16 with all 80 of their questions",
            port=7794,
            revision=BEAM_REVISION,
            chats=(1, 6, 11, 16),
        ),
        Workload(
            name="gold-slice",
            kind="dev",
            benchmark="gold-slice",
            description="Tier 2: one fact that exists only inside a subagent, planted by a live session",
            port=7795,
            revision=LONGMEMEVAL_REVISION,
            planted=True,
        ),
    )
}


def workload(name: str) -> Workload:
    """The workload called ``name``.

    Raises:
        ValueError: no such workload.
    """
    try:
        return WORKLOADS[name]
    except KeyError:
        raise ValueError(f"unknown workload {name!r}; expected one of {', '.join(WORKLOADS)}") from None


def shards(workload: Workload, questions: list[Golden]) -> list[list[Golden]]:
    """``questions`` split into the graphs they are built in: one, or ``shard_size`` at a time.

    A shard is independent of the others -- each question is its own user --
    so splitting changes cost, not what is measured.
    """
    if workload.shard_size is None:
        return [list(questions)]
    order = list(questions)
    random.Random(SHARD_SEED).shuffle(order)
    return [order[start : start + workload.shard_size] for start in range(0, len(order), workload.shard_size)]
