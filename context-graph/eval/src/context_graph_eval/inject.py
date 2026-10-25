"""Load session fixtures into an eval instance, resumably.

Injection stages content only. It does not distil anything -- reconciliation
is a separate pass the build runs afterwards, exactly as it would run over a
real harness session.

A graph holds many people: each question's haystack is its own user's
history (``SessionFixture.user_id``), and retrieval is scoped to the asking
user. The distractors a question has to get past are its own haystack's, as
LongMemEval frames them -- not other people's sessions.

Injection never clears the graph. A build that stopped part-way resumes:
a session already marked for reconciliation was written whole and is kept;
one present without that mark was cut off mid-write and is written again.
Clearing is :func:`wipe`'s job, which a build calls only when it starts over.
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
    """What an injection wrote, and what it found already there."""

    sessions: int
    turns: int
    #: Sessions a previous, interrupted injection had already written whole.
    kept: int = 0


def inject(fixtures: Iterable["SessionFixture"], *, graph: "ActionsGraph") -> Written:
    """Write every fixture session the graph doesn't already hold whole, each linked to its ``(:User)``.

    Fixtures are deduplicated by ``session_id``: writing a repeat again would
    append its turns a second time.

    Raises:
        ValueError: a fixture has no session_id. Checked before anything is
            written: a blank id would collapse distinct fixtures onto one node.
    """
    from actions_graph import MessageRole, Session

    deduped: dict[str, SessionFixture] = {}
    for fixture in fixtures:
        if not fixture.session_id:
            raise ValueError(f"fixture has no session_id: {fixture!r}")
        deduped.setdefault(fixture.session_id, fixture)

    whole = _drop_partial_sessions(graph, list(deduped))

    sessions = turns = 0
    for fixture in deduped.values():
        if fixture.session_id in whole:
            continue
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
            # The session's own date, since a fact's valid_at is its turn's
            # timestamp; one second apart to keep turn order.
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
        # Last, so its presence means every turn above landed.
        _mark_pending(graph, fixture.session_id)
        sessions += 1

    return Written(sessions=sessions, turns=turns, kept=len(whole))


def wipe(graph: "ActionsGraph") -> None:
    """Delete everything in the eval graph, vector indexes included.

    Not ``ActionsGraph.clear()``, which leaves distilled memory (Chunk,
    Entity, Memory) standing (#309). Vector indexes are dropped so an index
    sized for another embedding model can't survive into the next build.
    Safe only because the instance is dedicated to eval: **never point this
    at a shared or development database.**
    """
    graph.db.query("MATCH (n) DETACH DELETE n")
    for row in graph.db.query("SHOW VECTOR INDEX INFO"):
        graph.db.query(f"DROP VECTOR INDEX {row['index_name']}")


def _drop_partial_sessions(graph: "ActionsGraph", session_ids: list[str]) -> set[str]:
    """The ids already written whole; sessions cut off mid-write are deleted with their turns.

    A session carries ``reconciliation_status`` only once every turn was
    written (see :func:`inject`), so its absence marks an interrupted write.
    """
    rows = graph.db.query(
        "MATCH (s:Session) WHERE s.session_id IN $ids RETURN s.session_id AS id, s.reconciliation_status AS status",
        {"ids": session_ids},
    )
    partial = [row["id"] for row in rows if row["status"] is None]
    if partial:
        graph.db.query(
            "MATCH (s:Session) WHERE s.session_id IN $ids "
            "OPTIONAL MATCH (s)-[:HAS_ACTION]->(a:Action) DETACH DELETE a, s",
            {"ids": partial},
        )
    return {row["id"] for row in rows if row["status"] is not None}


def _mark_pending(graph: "ActionsGraph", session_id: str) -> None:
    graph.db.query(
        "MATCH (s:Session {session_id: $session_id}) SET s.reconciliation_status = $status",
        {"session_id": session_id, "status": PENDING},
    )
