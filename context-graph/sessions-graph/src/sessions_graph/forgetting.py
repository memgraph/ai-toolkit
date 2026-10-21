"""Removing what reconciliation derived from some sources, and nothing another source still backs.

A source is a session's Action or a memory file, named by its ``action_id`` or
``memory_id``. Reconciliation's provenance says exactly what each one gave:
extracted relationships carry its id as ``source_id``, entity mentions list it
in ``MENTIONED_IN.sources``, and ``(source)-[:HAS_CHUNK]->(:Chunk)`` links it
to the text it was read from. So a source's share can go while a Chunk or an
Entity that something else still backs stays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

    from memgraph_toolbox.api.memgraph import Memgraph


def chunks_of(memgraph: Memgraph, source_ids: Collection[str]) -> list[str]:
    """The hashes of the Chunks *source_ids* are linked to by ``HAS_CHUNK``."""
    rows = memgraph.query(
        """
        MATCH (source)-[:HAS_CHUNK]->(c:Chunk)
        WHERE coalesce(source.action_id, source.memory_id) IN $ids
        RETURN DISTINCT c.hash AS hash
        """,
        params={"ids": list(source_ids)},
    )
    return [row["hash"] for row in rows]


def forget_sources(memgraph: Memgraph, source_ids: Collection[str], chunk_hashes: Collection[str]) -> dict[str, int]:
    """Remove what *source_ids* gave the Chunks *chunk_hashes*, then whatever that leaves unbacked.

    Runs while the sources still exist or not: it matches their ids, never the
    nodes. A mention whose ``sources`` empties goes only if no other source
    links its Chunk; an Entity goes once it has no mention left; a Chunk goes
    once nothing links or mentions it.

    Returns:
        How many ``relations``, ``mentions``, ``entities`` and ``chunks`` went.
    """
    params = {"ids": list(source_ids), "hashes": list(chunk_hashes)}
    if not params["ids"] or not params["hashes"]:
        return {"relations": 0, "mentions": 0, "entities": 0, "chunks": 0}
    relations = memgraph.query(
        """
        MATCH (c:Chunk)<-[:MENTIONED_IN]-(e)
        WHERE c.hash IN $hashes
        MATCH (e)-[r]-()
        WHERE r.source_id IN $ids
        WITH DISTINCT r
        DELETE r
        RETURN count(*) AS n
        """,
        params=params,
    )
    mentions = memgraph.query(
        """
        MATCH (c:Chunk)<-[mention:MENTIONED_IN]-(e)
        WHERE c.hash IN $hashes
        SET mention.sources = [s IN coalesce(mention.sources, []) WHERE NOT s IN $ids]
        WITH c, mention, e
        WHERE size(mention.sources) = 0
          AND NOT EXISTS {
            MATCH (c)<-[:HAS_CHUNK]-(other)
            WHERE NOT coalesce(other.action_id, other.memory_id) IN $ids
          }
        DELETE mention
        RETURN count(*) AS n, collect(DISTINCT id(e)) AS entities
        """,
        params=params,
    )
    # An entity that lost a mention here may have had its others elsewhere, so it is
    # judged by every mention it has left, not by these chunks.
    entities = memgraph.query(
        """
        MATCH (e)
        WHERE id(e) IN $entities AND NOT EXISTS { MATCH (e)-[:MENTIONED_IN]->() }
        DETACH DELETE e
        RETURN count(*) AS n
        """,
        params={"entities": mentions[0]["entities"] if mentions else []},
    )
    memgraph.query(
        """
        MATCH (source)-[link:HAS_CHUNK]->(c:Chunk)
        WHERE c.hash IN $hashes AND coalesce(source.action_id, source.memory_id) IN $ids
        DELETE link
        """,
        params=params,
    )
    chunks = memgraph.query(
        """
        MATCH (c:Chunk)
        WHERE c.hash IN $hashes
          AND NOT EXISTS { MATCH (c)<-[:HAS_CHUNK]-() } AND NOT EXISTS { MATCH (c)<-[:MENTIONED_IN]-() }
        DETACH DELETE c
        RETURN count(*) AS n
        """,
        params=params,
    )
    return {
        "relations": _count(relations),
        "mentions": _count(mentions),
        "entities": _count(entities),
        "chunks": _count(chunks),
    }


def _count(rows: list[dict]) -> int:
    return rows[0]["n"] if rows else 0
