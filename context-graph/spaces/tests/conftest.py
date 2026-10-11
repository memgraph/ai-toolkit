"""Fixtures for context-graph-spaces' end-to-end tests, against a real Enterprise-licensed Memgraph.

``./scripts/dev-memgraph.sh test-enterprise`` starts that Memgraph and sets
``ENTERPRISE_MEMGRAPH_URL``. Without it, or without a valid licence, the tests
skip and say why: nothing here is mocked, because grant syntax and semantics
are exactly what a mock would get wrong.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import neo4j
import pytest

from context_graph_spaces.cli import admin_query

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from context_graph_spaces.provision import Query

#: The admin these tests provision as. Deliberately unprefixed, so the provisioner treats it as not its own.
ADMIN = ("mgadmin", "mgadmin-test-password")


@pytest.fixture(scope="session")
def enterprise_url() -> str:
    url = os.environ.get("ENTERPRISE_MEMGRAPH_URL")
    if not url:
        pytest.skip("No Enterprise Memgraph: run ./scripts/dev-memgraph.sh test-enterprise")
    _bootstrap_admin(url)
    query, close = admin_query(url, *ADMIN)
    try:
        licence = {row["license info"]: row["value"] for row in query("SHOW LICENSE INFO", None, None)}
    finally:
        close()
    if licence.get("is_valid") is not True:
        pytest.skip(f"Memgraph at {url} has no valid Enterprise licence")
    return url


def _bootstrap_admin(url: str) -> None:
    """Create the admin on a fresh instance, where the first connection is anonymous."""
    driver = neo4j.GraphDatabase.driver(url, auth=("", ""))
    try:
        with driver.session() as session:
            session.run(f"CREATE USER IF NOT EXISTS {ADMIN[0]} IDENTIFIED BY '{ADMIN[1]}'").consume()
    except neo4j.exceptions.Neo4jError:
        pass  # Users exist already, so anonymous access is closed (Security.Unauthenticated); the admin is one of them.
    finally:
        driver.close()


@pytest.fixture
def admin(enterprise_url: str) -> Iterator[Query]:
    """An admin query function on a server cleared of every provisioner-owned object."""
    query, close = admin_query(enterprise_url, *ADMIN)
    _drop_owned(query)
    try:
        yield query
    finally:
        _drop_owned(query)
        close()


def _drop_owned(query: Query) -> None:
    owned = ("u_", "space_", "cg_")
    roles = [row["role"] for row in query("SHOW ROLES", None, None) if not row.get("builtin")]
    for row in query("SHOW USERS", None, None):
        user = row["user"]
        holds_owned_role = any(
            user in {r["users"] for r in query(f"SHOW USERS FOR ROLE {role}", None, None)}
            for role in roles
            if role.startswith(owned)
        )
        if user.startswith(owned) or holds_owned_role or user in TEST_LOGINS:
            query(f"DROP USER {user}", None, None)
    for role in roles:
        if role.startswith(owned):
            query(f"DROP ROLE {role}", None, None)
    for row in query("SHOW TENANT PROFILES", None, None):
        if row["profile"].startswith(owned):
            for database in filter(None, (name.strip() for name in str(row.get("databases") or "").split(","))):
                query(f"REMOVE TENANT PROFILE FROM DATABASE {database}", None, None)
            query(f"DROP TENANT PROFILE {row['profile']}", None, None)
    for row in query("SHOW DATABASES", None, None):
        if row["Name"].startswith(owned):
            query(f"DROP DATABASE {row['Name']} FORCE", None, None)


#: Every login the tests provision, so a failed run's leftovers are cleaned up too.
TEST_LOGINS = {"alice", "bob", "carol"}


@pytest.fixture
def connect(enterprise_url: str) -> Iterator[Callable[..., Any]]:
    """``connect(login, password, database=None)``: run statements as that login, a new connection each time."""
    drivers: list[neo4j.Driver] = []

    def run(login: str, password: str, statement: str, database: str | None = None, **params: Any) -> list[dict]:
        driver = neo4j.GraphDatabase.driver(enterprise_url, auth=(login, password))
        drivers.append(driver)
        with driver.session(database=database) as session:
            return [record.data() for record in session.run(cast("Any", statement), params)]

    yield run
    for driver in drivers:
        driver.close()
