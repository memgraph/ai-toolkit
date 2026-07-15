"""Shared Memgraph connection helpers for the LightRAG storage backends.

All three storage backends (KV, vector, doc-status) reuse a single async Bolt
driver per running event loop instead of opening one driver per storage
instance or per call. The driver comes from
``memgraph_toolbox.api.memgraph.AsyncMemgraph`` so this integration shares the
toolbox's connection contract instead of maintaining a bespoke driver.

Connection settings are owned entirely by the toolbox: ``AsyncMemgraph`` is
constructed with no explicit connection arguments and resolves
``MEMGRAPH_URL`` / ``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD`` /
``MEMGRAPH_DATABASE`` through ``memgraph_env`` (the single source of truth).
These are the canonical toolbox names. LightRAG's bundled graph backend
(``lightrag.kg.memgraph_impl.MemgraphStorage``) uses the alternative names
``MEMGRAPH_URI`` / ``MEMGRAPH_USERNAME``; that naming difference is bridged in
exactly one place (``core.py``), so this module does not read the LightRAG
aliases itself.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from lightrag.utils import logger

from memgraph_toolbox.api.memgraph import AsyncMemgraph, memgraph_env

if TYPE_CHECKING:
    from neo4j import AsyncDriver

# One shared AsyncMemgraph client per event loop, keyed by id(loop). Reusing the
# driver across storages avoids the overhead of a driver-per-call and keeps a
# single connection pool, while keying by loop id keeps async test suites (which
# spin up a fresh loop per test) from reusing a driver bound to a closed loop.
_clients: dict[int, AsyncMemgraph] = {}


def _get_client() -> AsyncMemgraph:
    """Return the shared AsyncMemgraph client for the current event loop, creating it if needed."""
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is None:
        # No explicit connection args: the toolbox resolves the canonical
        # MEMGRAPH_URL/USER/PASSWORD/DATABASE via memgraph_env, the single
        # source of truth for connection config.
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
    """Escape a value for safe use as a backtick-quoted Cypher label identifier.

    Backticks are doubled to prevent Cypher injection; all other characters are
    preserved. The result is intended to be wrapped in backticks, e.g.
    ``MATCH (n:`{label}`)``. Mirrors ``MemgraphStorage._get_workspace_label``.
    """
    return value.strip().replace("`", "``")


def sanitize_index_name(value: str) -> str:
    """Coerce a value into a bare (unquoted) identifier for a vector index name.

    Memgraph vector-index names are unquoted identifiers, so every character
    outside ``[A-Za-z0-9_]`` is replaced with an underscore.
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", value)
