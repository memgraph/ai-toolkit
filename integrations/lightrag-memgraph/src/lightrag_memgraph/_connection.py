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
import re
from typing import TYPE_CHECKING

from lightrag.utils import logger

from memgraph_toolbox.api.memgraph import AsyncMemgraph, memgraph_env

if TYPE_CHECKING:
    from neo4j import AsyncDriver

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
