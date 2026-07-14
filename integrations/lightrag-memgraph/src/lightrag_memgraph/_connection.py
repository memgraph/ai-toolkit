"""Shared Memgraph connection helpers for the LightRAG storage backends.

All three storage backends (KV, vector, doc-status) reuse a single async Bolt
driver per running event loop instead of opening one driver per storage
instance or per call. This mirrors the connection style of LightRAG's own
graph backend (``lightrag.kg.memgraph_impl.MemgraphStorage``): the same
``neo4j.AsyncGraphDatabase`` driver, the same environment variables
(``MEMGRAPH_URI`` / ``MEMGRAPH_USERNAME`` / ``MEMGRAPH_PASSWORD`` /
``MEMGRAPH_DATABASE``) and the same ``config.ini [memgraph]`` fallback.
"""

from __future__ import annotations

import asyncio
import configparser
import os
import re

from lightrag.utils import logger
from neo4j import AsyncDriver, AsyncGraphDatabase

# Mirror lightrag.kg.memgraph_impl: allow a config.ini [memgraph] section to
# provide connection defaults when the environment variables are not set.
_config = configparser.ConfigParser()
_config.read("config.ini", "utf-8")

# One shared driver per event loop, keyed by id(loop). Reusing the driver
# across storages avoids the overhead of a driver-per-call and keeps a single
# connection pool, while keying by loop id keeps async test suites (which spin
# up a fresh loop per test) from reusing a driver bound to a closed loop.
_drivers: dict[int, AsyncDriver] = {}
_databases: dict[int, str] = {}


def read_connection_config() -> tuple[str, str, str, str]:
    """Read Memgraph connection settings (env first, then config.ini fallback).

    Returns:
        Tuple of ``(uri, username, password, database)``.
    """
    uri = os.environ.get(
        "MEMGRAPH_URI",
        _config.get("memgraph", "uri", fallback="bolt://localhost:7687"),
    )
    username = os.environ.get("MEMGRAPH_USERNAME", _config.get("memgraph", "username", fallback=""))
    password = os.environ.get("MEMGRAPH_PASSWORD", _config.get("memgraph", "password", fallback=""))
    database = os.environ.get(
        "MEMGRAPH_DATABASE",
        _config.get("memgraph", "database", fallback="memgraph"),
    )
    return uri, username, password, database


async def get_driver() -> AsyncDriver:
    """Return the shared async driver for the current event loop, creating it if needed."""
    loop_id = id(asyncio.get_running_loop())
    driver = _drivers.get(loop_id)
    if driver is None:
        uri, username, password, database = read_connection_config()
        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        _drivers[loop_id] = driver
        _databases[loop_id] = database
        logger.info(f"Opened shared Memgraph driver for LightRAG storages at {uri}")
    return driver


def get_database() -> str:
    """Return the database name bound to the current event loop's driver."""
    loop_id = id(asyncio.get_running_loop())
    return _databases.get(loop_id, read_connection_config()[3])


async def close_driver() -> None:
    """Close the shared driver bound to the current event loop, if any."""
    loop_id = id(asyncio.get_running_loop())
    driver = _drivers.pop(loop_id, None)
    _databases.pop(loop_id, None)
    if driver is not None:
        await driver.close()
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
