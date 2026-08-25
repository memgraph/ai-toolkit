"""Load a batch of session fixtures into the eval database.

Injection stages content only. It does not distil anything -- reconciliation is
a separate, LLM-backed pass the runner triggers afterwards, exactly as it would
run over a real harness session.

Isolation is per **batch**, not per question: a batch-wide graph is what gives
retrieval distractors to get wrong, and without distractors both precision and
the payload-size efficiency metric would score well by construction.

The eval instance is cleared before each batch so every run starts from known,
fixed state -- otherwise a question could be answered from a previous run's
sessions rather than this batch's fixtures, and two runs would not be
comparable. Clearing is safe only because the instance is dedicated to eval;
this must never point at a shared or development database.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from actions_graph import ActionsGraph

    from .convert.longmemeval import SessionFixture

#: Marks the Session as awaiting distillation. Reconciliation sweeps for this.
PENDING = "pending"


@dataclass(frozen=True)
class Written:
    """What a batch injection wrote."""

    sessions: int
    turns: int


def inject_batch(fixtures: Iterable["SessionFixture"], *, graph: "ActionsGraph") -> Written:
    """Clear the eval graph, then load ``fixtures`` into it.

    Returns counts of what was written, so a caller can assert the batch landed
    rather than inferring it from the absence of an exception.
    """
    from actions_graph import MessageRole, Session

    fixtures = list(fixtures)

    # Validated before clearing: a blank session_id would collapse distinct
    # fixtures onto one node, silently merging sessions and destroying the
    # haystack. Failing first also avoids wiping the graph for a batch that was
    # never going to load.
    for fixture in fixtures:
        if not fixture.session_id:
            raise ValueError(f"fixture has no session_id: {fixture!r}")

    graph.clear()

    turns = 0
    for fixture in fixtures:
        graph.ensure_session(
            Session(
                session_id=fixture.session_id,
                started_at=fixture.date,
                # Deliberately not written: SessionFixture.holds_evidence. That
                # is corpus-side bookkeeping, and putting it in the graph would
                # hand retrieval the answer's location -- telling the thing
                # under test where to look.
                metadata={"origin": "eval-fixture"},
            )
        )
        for turn in fixture.turns:
            graph.record_message(
                session_id=fixture.session_id,
                role=MessageRole(turn.role),
                content=turn.content,
            )
            turns += 1

        _mark_pending(graph, fixture.session_id)

    return Written(sessions=len(fixtures), turns=turns)


def _mark_pending(graph: "ActionsGraph", session_id: str) -> None:
    graph._db.query(
        "MATCH (s:Session {session_id: $session_id}) SET s.reconciliation_status = $status",
        {"session_id": session_id, "status": PENDING},
    )
