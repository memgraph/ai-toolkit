"""The shared-space reconcile worker: what members write, it extracts.

Members may only write curated memory into a shared space, so the derived
graph there (Chunks, entities, their relations, the memory passages recall
embeds) is written by one login, ``cg_service``, run by the org next to
``apply`` (map #501). Personal spaces keep reconciling locally, as before.

Each pass, per shared space in the spec:

1. Remove what deleted files left behind. A member's delete keeps its hands
   off Chunks (it can't touch them), so their file's extraction outlives it.
2. Reconcile every file marked ``extraction_status = 'pending'`` with
   sessions-graph's own per-file reconciliation (GLiNER2 under the space's
   adopted ontology, no LLM).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .provision import SERVICE_ACCOUNT, shared_database

if TYPE_CHECKING:
    from sessions_graph import SessionsGraph

    from memgraph_toolbox.api.memgraph import Memgraph

    from .spec import Spec


@dataclass
class SpaceReport:
    """One shared space's pass: files reconciled or failed, and leftovers removed."""

    space: str
    completed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    pruned: int = 0
    error: str | None = None


def reconcile_spaces(
    spec: Spec,
    *,
    url: str,
    password: str,
    extraction_backend: Any = None,
    limit: int = 100,
) -> list[SpaceReport]:
    """One pass over every shared space in *spec*, as the service login.

    A space that can't be reached is reported and skipped; the others still run.
    """
    from sessions_graph import SessionsGraph

    from memgraph_toolbox.api.memgraph import Memgraph

    reports = []
    for space in spec.spaces:
        report = SpaceReport(space.name)
        client = None
        try:
            client = Memgraph(
                url=url,
                username=SERVICE_ACCOUNT,
                password=password,
                database=shared_database(space.name),
                user_agent="context-graph-spaces-worker",
            )
            report.pruned = prune_deleted_extractions(client)
            asyncio.run(_reconcile_pending(SessionsGraph(client), report, extraction_backend, limit))
        # One space's failure is reported, and must not stop the others.
        except Exception as exc:
            report.error = f"{type(exc).__name__}: {exc}"
        finally:
            if client is not None:
                client.close()
        reports.append(report)
    return reports


async def _reconcile_pending(graph: SessionsGraph, report: SpaceReport, backend: Any, limit: int) -> None:
    for memory_id in graph.get_pending_memory_reconciliations(limit=limit):
        result = await graph.reconcile_memory(memory_id, extraction_backend=backend)
        if result.status == "completed":
            report.completed.append(result.path or memory_id)
        else:
            report.failed.append((result.path or memory_id, result.error or "unknown error"))


def prune_deleted_extractions(db: Memgraph) -> int:
    """Remove the extraction of memory files that no longer exist; returns how many relationships went.

    The same cleanup :func:`sessions_graph.memory_store.forget_extraction` does for a live file, keyed on
    the provenance ids that no longer name a ``(:Memory)``: in a shared space every source is a memory file.
    """
    live = [row["id"] for row in db.query("MATCH (m:Memory) RETURN m.memory_id AS id")]
    relations = db.query(
        "MATCH ()-[r]->() WHERE r.source_id IS NOT NULL AND NOT r.source_id IN $live "
        "WITH r DELETE r RETURN count(*) AS n",
        {"live": live},
    )
    db.query(
        """
        MATCH (c:Chunk)<-[mention:MENTIONED_IN]-(e)
        WITH c, mention, e, [s IN coalesce(mention.sources, []) WHERE s IN $live] AS kept
        SET mention.sources = kept
        WITH c, mention, e, kept
        WHERE size(kept) = 0
        DELETE mention
        WITH DISTINCT e
        WHERE NOT EXISTS { MATCH (e)-[:MENTIONED_IN]->() }
        DETACH DELETE e
        """,
        {"live": live},
    )
    db.query(
        """
        MATCH (c:Chunk)
        WHERE NOT EXISTS { MATCH (c)<-[:HAS_CHUNK]-() } AND NOT EXISTS { MATCH (c)<-[:MENTIONED_IN]-() }
        DETACH DELETE c
        """
    )
    return relations[0]["n"] if relations else 0
