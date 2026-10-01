"""Load a batch of session fixtures into the eval database.

Injection stages content only. It does not distil anything -- reconciliation is
a separate, LLM-backed pass the runner triggers afterwards, exactly as it would
run over a real harness session.

A batch is one graph holding many people: each question's haystack is its
own user's history (``SessionFixture.user_id``), and retrieval is scoped to the
asking user. The distractors a question has to get past are its own
haystack's, as LongMemEval frames them -- not other people's sessions.

The eval instance is cleared before each batch so every run starts from known,
fixed state -- otherwise a question could be answered from a previous run's
sessions rather than this batch's fixtures, and two runs would not be
comparable. Clearing is safe only because the instance is dedicated to eval;
this must never point at a shared or development database.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from actions_graph import ActionsGraph

    from .convert.longmemeval import SessionFixture

#: Marks the Session as awaiting distillation. Reconciliation sweeps for this.
PENDING = "pending"

#: LongMemEval's session date format, e.g. '2023/05/30 (Tue) 17:27'.
CORPUS_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


def corpus_time(date: str) -> datetime | None:
    """A corpus session date as a UTC datetime, or None if it isn't in CORPUS_DATE_FORMAT.

    The corpus gives no timezone; UTC is assumed, which only matters for
    arithmetic across sessions, and there every session shares it.
    """
    try:
        return datetime.strptime(date, CORPUS_DATE_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass(frozen=True)
class Written:
    """What a batch injection wrote."""

    sessions: int
    turns: int


def inject_batch(fixtures: Iterable["SessionFixture"], *, graph: "ActionsGraph") -> Written:
    """Clear the eval graph, then load ``fixtures`` into it.

    Each session is linked to its fixture's ``(:User)``. Fixtures are
    deduplicated by ``session_id``: writing a repeat again would append its
    turns a second time.

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

    deduped: dict[str, SessionFixture] = {}
    for fixture in fixtures:
        deduped.setdefault(fixture.session_id, fixture)

    _wipe(graph)

    turns = 0
    for fixture in deduped.values():
        started = corpus_time(fixture.date)
        graph.ensure_session(
            Session(
                session_id=fixture.session_id,
                started_at=started.isoformat() if started else fixture.date,
                # Deliberately not written: SessionFixture.holds_evidence. That
                # is corpus-side bookkeeping, and putting it in the graph would
                # hand retrieval the answer's location -- telling the thing
                # under test where to look.
                metadata={"origin": "eval-fixture"},
            )
        )
        for index, turn in enumerate(fixture.turns):
            # Stamped with the session's own date, not left at ingest time:
            # an extracted fact's valid_at is its source turn's timestamp
            # (#364), and ingest time would date every fact to the eval run.
            # One second apart because get_session_actions orders by
            # timestamp, so identical stamps would lose the turn order.
            stamp = {"timestamp": (started + timedelta(seconds=index)).isoformat()} if started else {}
            graph.record_message(
                session_id=fixture.session_id,
                role=MessageRole(turn.role),
                content=turn.content,
                **stamp,
            )
            turns += 1

        graph.db.query(
            "MERGE (u:User {user_id: $user_id}) WITH u MATCH (s:Session {session_id: $session_id}) "
            "MERGE (u)-[:HAD_SESSION]->(s)",
            {"user_id": fixture.user_id, "session_id": fixture.session_id},
        )
        _mark_pending(graph, fixture.session_id)

    return Written(sessions=len(deduped), turns=turns)


def _wipe(graph: "ActionsGraph") -> None:
    """Delete everything in the eval graph, vector indexes included.

    Deliberately not ``ActionsGraph.clear()``, which only removes
    ``Session|Agent|Action|Tool``. That leaves ``Chunk``, ``Entity``,
    ``Episode`` and ``Memory`` standing -- precisely what reconciliation
    produces. Relying on it would let the previous batch's *distilled memory*
    survive, so a question could be answered from the last run instead of this
    batch's fixtures: the leak #309 exists to prevent, and it would inflate
    scores invisibly.

    Deleting nodes is not enough on its own. lightrag-memgraph's vector
    storage creates its index once and treats a second ``CREATE VECTOR
    INDEX`` as a no-op "already exists" for *any* failure reason, dimension
    mismatch included (confirmed by reading ``vector_impl.py``'s
    ``_ensure_vector_index`` -- it logs and moves on rather than checking
    why creation failed). A prior batch's embedding model leaves its
    index's dimension behind otherwise, and the next batch's writes fail --
    or worse, silently mismatch -- against a leftover index sized for a
    different model. Dropping each existing vector index here means a batch
    that changes embedding model is exactly as safe as one that doesn't.

    Deleting the whole graph is only safe because the eval instance is
    dedicated. **This must never point at a shared or development database.**
    """
    graph.db.query("MATCH (n) DETACH DELETE n")
    for row in graph.db.query("SHOW VECTOR INDEX INFO"):
        graph.db.query(f"DROP VECTOR INDEX {row['index_name']}")


def _mark_pending(graph: "ActionsGraph", session_id: str) -> None:
    graph.db.query(
        "MATCH (s:Session {session_id: $session_id}) SET s.reconciliation_status = $status",
        {"session_id": session_id, "status": PENDING},
    )
