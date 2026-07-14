"""Shared Memgraph connection helpers for the LightRAG storage backends.

All three storage backends (KV, vector, doc-status) reuse a single async Bolt
driver per running event loop instead of opening one driver per storage
instance or per call. The driver comes from
``memgraph_toolbox.api.memgraph.AsyncMemgraph`` so this integration shares the
toolbox's connection contract instead of maintaining a bespoke driver.

Connection settings mirror LightRAG's own graph backend
(``lightrag.kg.memgraph_impl.MemgraphStorage``), which reads
``MEMGRAPH_URI`` / ``MEMGRAPH_USERNAME`` / ``MEMGRAPH_PASSWORD`` /
``MEMGRAPH_DATABASE``, while the toolbox reads
``MEMGRAPH_URL`` / ``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD`` /
``MEMGRAPH_DATABASE``. To avoid a split-brain where the graph backend and these
storages point at different instances, both name variants are read here
(preferring the LightRAG names so we match the graph backend) and the resolved
values are passed EXPLICITLY to ``AsyncMemgraph`` so env-name drift cannot cause
a mismatch. The ``config.ini [memgraph]`` fallback is preserved.
"""

from __future__ import annotations

import asyncio
import configparser
import os
import re
from typing import TYPE_CHECKING

from lightrag.utils import logger

from memgraph_toolbox.api.memgraph import AsyncMemgraph

if TYPE_CHECKING:
    from neo4j import AsyncDriver

# Mirror lightrag.kg.memgraph_impl: allow a config.ini [memgraph] section to
# provide connection defaults when the environment variables are not set.
_config = configparser.ConfigParser()
_config.read("config.ini", "utf-8")

# One shared AsyncMemgraph client per event loop, keyed by id(loop). Reusing the
# driver across storages avoids the overhead of a driver-per-call and keeps a
# single connection pool, while keying by loop id keeps async test suites (which
# spin up a fresh loop per test) from reusing a driver bound to a closed loop.
_clients: dict[int, AsyncMemgraph] = {}


def _first_env(*names: str) -> str | None:
    """Return the first environment variable that is set (even to ""), else None."""
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def read_connection_config() -> tuple[str, str, str, str]:
    """Read Memgraph connection settings (env first, then config.ini fallback).

    Both the LightRAG env-var names (``MEMGRAPH_URI`` / ``MEMGRAPH_USERNAME``)
    and the toolbox names (``MEMGRAPH_URL`` / ``MEMGRAPH_USER``) are honoured,
    preferring the LightRAG names so these storages resolve to the same instance
    as LightRAG's built-in graph backend.

    Returns:
        Tuple of ``(uri, username, password, database)``.
    """
    uri = _first_env("MEMGRAPH_URI", "MEMGRAPH_URL")
    if uri is None:
        uri = _config.get("memgraph", "uri", fallback="bolt://localhost:7687")

    username = _first_env("MEMGRAPH_USERNAME", "MEMGRAPH_USER")
    if username is None:
        username = _config.get("memgraph", "username", fallback="")

    password = _first_env("MEMGRAPH_PASSWORD")
    if password is None:
        password = _config.get("memgraph", "password", fallback="")

    database = _first_env("MEMGRAPH_DATABASE")
    if database is None:
        database = _config.get("memgraph", "database", fallback="memgraph")

    return uri, username, password, database


def _get_client() -> AsyncMemgraph:
    """Return the shared AsyncMemgraph client for the current event loop, creating it if needed."""
    loop_id = id(asyncio.get_running_loop())
    client = _clients.get(loop_id)
    if client is None:
        uri, username, password, database = read_connection_config()
        # Pass values EXPLICITLY so AsyncMemgraph does not re-resolve from the
        # toolbox env-var names and risk a mismatch with the graph backend.
        client = AsyncMemgraph(
            url=uri,
            username=username,
            password=password,
            database=database,
            user_agent="lightrag-memgraph",
        )
        _clients[loop_id] = client
        logger.info(f"Opened shared Memgraph driver for LightRAG storages at {uri}")
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
    return read_connection_config()[3]


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
