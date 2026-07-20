"""Shared Memgraph connection helpers for the LightRAG storage backends.

KV, vector and doc-status storages share one async driver per event loop (via
``memgraph_toolbox.api.memgraph.AsyncMemgraph``) instead of one per call.
Connection config comes entirely from the toolbox's ``memgraph_env``
(``MEMGRAPH_URL``/``USER``/``PASSWORD``/``DATABASE``); LightRAG's graph
backend uses different names (``MEMGRAPH_URI``/``USERNAME``), bridged once in
``core.py``.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING

from lightrag.utils import logger

from memgraph_toolbox.api.memgraph import AsyncMemgraph, memgraph_env

if TYPE_CHECKING:
    from neo4j import AsyncDriver, AsyncSession

# Keyed by event-loop id so tests spinning up a fresh loop per test don't
# reuse a driver bound to a closed loop.
_clients: dict[int, AsyncMemgraph] = {}


def _get_client() -> AsyncMemgraph:
    """Return the shared AsyncMemgraph client for the current event loop, creating it if needed."""
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is None:
        client = AsyncMemgraph(user_agent="lightrag-memgraph")
        _clients[loop_id] = client
        logger.info(f"Opened shared Memgraph driver for LightRAG storages at {memgraph_env()['MEMGRAPH_URL']}")
    return client


async def get_driver() -> AsyncDriver:
    """Return the shared async driver for the current event loop, creating it if needed."""
    return _get_client().driver


def get_database() -> str:
    """Return the database name bound to the current event loop's driver."""
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is not None:
        return client.database
    return memgraph_env()["MEMGRAPH_DATABASE"]


async def close_driver() -> None:
    """Close the shared driver bound to the current event loop, if any."""
    loop_id = id(asyncio.get_running_loop())
    client = _clients.pop(loop_id, None)
    if client is not None:
        await client.close()
        logger.info("Closed shared Memgraph driver for LightRAG storages")


def sanitize_label(value: str) -> str:
    """Escape for safe use as a backtick-quoted Cypher label (doubles backticks to prevent injection)."""
    return value.strip().replace("`", "``")


def sanitize_index_name(value: str) -> str:
    """Coerce into a bare identifier: Memgraph vector-index names are unquoted, so replace anything outside [A-Za-z0-9_]."""
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def resolve_workspace(workspace: str | None) -> str:
    """Resolve the effective workspace for a storage instance.

    ``MEMGRAPH_WORKSPACE`` wins over the constructor argument, which wins over
    the "base" default -- shared by all three storage backends so this
    precedence can't drift between them.
    """
    return os.environ.get("MEMGRAPH_WORKSPACE") or workspace or "base"


async def create_index(session: AsyncSession, *, label: str, prop: str, workspace: str) -> None:
    """CREATE INDEX on :`label`(prop), swallowing "already exists" (Memgraph has no IF NOT EXISTS)."""
    try:
        await (await session.run(f"CREATE INDEX ON :`{label}`({prop})")).consume()
    except Exception as e:  # index may already exist, which is not an error
        logger.warning(f"[{workspace}] Index creation on :`{label}`({prop}) may have failed or already exists: {e}")


class MemgraphCrudMixin:
    """Shared ``is_empty``/``filter_keys``/``delete``/``drop`` for the id-keyed,
    one-label-per-namespace storages (KV and doc-status).

    Requires the including class to set ``self._label`` (a backtick-quoted
    node label) and ``self.workspace`` in ``__post_init__``. Storages whose
    deletes need extra bookkeeping (``MemgraphVectorStorage``'s
    embedding-nulling) override ``delete``/``drop`` instead of using these.
    """

    _label: str
    workspace: str
    namespace: str

    async def is_empty(self) -> bool:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(f"MATCH (n:`{self._label}`) RETURN n LIMIT 1")
            record = await result.single()
            await result.consume()
            return record is None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        # Returns the subset of `keys` that do NOT exist in storage.
        if not keys:
            return set()
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                f"""
                UNWIND $keys AS k
                OPTIONAL MATCH (n:`{self._label}` {{id: k}})
                WITH k, n
                WHERE n IS NULL
                RETURN k
                """,
                keys=list(keys),
            )
            missing = {record["k"] async for record in result}
            await result.consume()
        return missing

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        driver = await get_driver()
        async with driver.session(database=get_database()) as session:
            await (
                await session.run(
                    f"""
                    UNWIND $ids AS target_id
                    MATCH (n:`{self._label}` {{id: target_id}})
                    DETACH DELETE n
                    """,
                    ids=list(ids),
                )
            ).consume()

    async def drop(self) -> dict[str, str]:
        try:
            driver = await get_driver()
            async with driver.session(database=get_database()) as session:
                await (await session.run(f"MATCH (n:`{self._label}`) DETACH DELETE n")).consume()
            logger.info(f"[{self.workspace}] Dropped Memgraph storage for {self.namespace}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping storage for {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
