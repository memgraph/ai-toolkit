"""Reconcile an eval graph's pending sessions: the pass a real session gets.

Local GLiNER2 under each session's user's adopted ontology version -- hygm's
default model until a learned build (``learned.py``) derives and adopts one.
No LLM. Separate from injection because extraction is what a build spends
its time on.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from memgraph_toolbox.api.memgraph import Memgraph

from .answering import setup_with_retry
from .inject import PENDING

#: What a build extracts under: "default", each user's adopted version
#: (hygm's default model, nothing derived); "learned", the same, then each
#: user's own model derived and adopted (learned.py).
ONTOLOGIES = ("default", "learned")

#: The class name sessions-graph persists as ``Session.extraction_backend``.
EXTRACTION_BACKEND = "GLiNER2Backend"


@dataclass(frozen=True)
class Reconciled:
    """Outcome of reconciling a graph's pending sessions."""

    reconciled: int
    failed: int
    errors: tuple[str, ...] = ()


def pending_sessions(db: "Memgraph", limit: int | None = None) -> list[str]:
    """Session ids awaiting reconciliation, in id order, so a resumed build makes deterministic progress."""
    query = (
        "MATCH (s:Session) WHERE s.reconciliation_status = $status "
        "RETURN s.session_id AS session_id ORDER BY s.session_id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [row["session_id"] for row in db.query(query, {"status": PENDING})]


async def reconcile_pending(db: "Memgraph", *, concurrency: int = 4, progress: bool = True) -> Reconciled:
    """Reconcile every pending session, reporting failures in the result rather than raising.

    Extraction runs in a worker thread per session, so sessions overlap;
    returns flatten past 4 since torch already uses several threads. Memgraph
    writes stay on the event loop thread, so no two MERGEs race.

    Raises:
        ValueError: ``concurrency`` below 1.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    from sessions_graph import SessionsGraph

    session_ids = pending_sessions(db)
    if not session_ids:
        return Reconciled(reconciled=0, failed=0)

    graph = SessionsGraph(memgraph=db)
    setup_with_retry(graph)

    reconciled = done = 0
    errors: list[str] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def one(session_id: str) -> None:
        nonlocal reconciled, done
        async with semaphore:
            try:
                summary = await graph.reconciler.reconcile(session_id, enforce_ontology=True)
            except Exception as exc:
                # One session must not end a build hours in; it stays pending for the next attempt.
                summary = None
                errors.append(f"{session_id}: {exc}")
        if summary is not None and summary.status == "completed":
            reconciled += 1
        elif summary is not None:
            errors.append(f"{session_id}: {summary.error}")
        done += 1
        # Silent until done is indistinguishable from hung.
        if progress:
            print(f"  reconciled {done}/{len(session_ids)} ({reconciled} ok, {len(errors)} failed)", flush=True)

    await asyncio.gather(*(one(session_id) for session_id in session_ids))
    return Reconciled(reconciled=reconciled, failed=len(errors), errors=tuple(errors))
