"""Make a fresh licensed Memgraph ready for spaces: its first admin login, then whether it can host them.

``apply`` needs an admin login, and a fresh Memgraph has none: it accepts anonymous
connections until its first user exists, and that first user gets the built-in ``admin``
role. :func:`create_first_admin` makes that user. :func:`check_instance` then reports what
``apply`` and recall will need, so an instance missing them fails here rather than for
every member later.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .spec import RESERVED_PREFIXES, SpecError, _check_name

if TYPE_CHECKING:
    from .provision import Query

#: Memgraph's built-in role. A user with the same name can't be told apart from it in grants.
BUILTIN_ADMIN_ROLE = "admin"
#: The procedure recall embeds with; only the MAGE images ship it.
EMBEDDINGS_PROCEDURE = "embeddings.text"


def check_admin_name(login: str) -> None:
    """Refuse an admin login that would clash with Memgraph or with what the provisioner owns.

    Raises:
        SpecError: *login* isn't a valid name, is ``admin``, or starts with a reserved prefix.
    """
    _check_name(login, "--admin")
    if login == BUILTIN_ADMIN_ROLE:
        raise SpecError(f"--admin: {login!r} is Memgraph's built-in role name; pick another, e.g. cgadmin")
    if login.startswith(RESERVED_PREFIXES):
        # apply manages (and --prune drops) every object with these prefixes, its own admin included.
        raise SpecError(f"--admin: {login!r} starts with a reserved prefix ({', '.join(RESERVED_PREFIXES)})")


def create_first_admin(url: str, login: str, password: str | None = None) -> str | None:
    """Create *login* as the first user of the Memgraph at *url*, and return its password.

    The password is generated unless given. Returns None, and changes nothing, when the
    instance already has users: then it refuses anonymous connections, and its admin
    already exists.

    Raises:
        SpecError: *login* isn't usable as the admin (see :func:`check_admin_name`).
        neo4j.exceptions.ServiceUnavailable: nothing answers at *url*.
    """
    import neo4j

    check_admin_name(login)
    password = password or secrets.token_urlsafe(18)
    if "'" in password or "\\" in password:
        raise ValueError("the admin password can't contain quotes or backslashes")
    driver = neo4j.GraphDatabase.driver(url, auth=("", ""))
    try:
        with driver.session() as session:
            if list(session.run("SHOW USERS")):
                return None
            # Memgraph's CREATE USER takes no query parameters. login passed _NAME; the password is
            # token_urlsafe ([A-Za-z0-9_-]) or was just checked for quotes, so the literal can't be broken out of.
            session.run(cast("Any", f"CREATE USER {login} IDENTIFIED BY '{password}'")).consume()
    except neo4j.exceptions.ClientError as exc:
        if "Unauthenticated" in str(exc.code) or isinstance(exc, neo4j.exceptions.AuthError):
            return None
        raise
    finally:
        driver.close()
    return password


@dataclass(frozen=True)
class Check:
    """One thing spaces need from the instance: whether it holds, and what to do when it doesn't."""

    name: str
    ok: bool
    detail: str


def check_instance(query: Query, login: str) -> list[Check]:
    """What spaces need from the Memgraph behind *query*, connected as the admin *login*."""
    return [_licence(query), _admin_role(query, login), _embeddings(query), _memory_limit(query)]


def _licence(query: Query) -> Check:
    info = {row["license info"]: row["value"] for row in query("SHOW LICENSE INFO", None, None)}
    if info.get("is_valid") is True:
        return Check("Enterprise licence", True, f"{info.get('license_type')}, valid until {info.get('valid_until')}")
    return Check(
        "Enterprise licence",
        False,
        "missing or invalid; spaces need multi-tenancy and access control. Start Memgraph with "
        "MEMGRAPH_ENTERPRISE_LICENSE and MEMGRAPH_ORGANIZATION_NAME set, or run SET DATABASE SETTING.",
    )


def _admin_role(query: Query, login: str) -> Check:
    roles = {row["role"] for row in query("SHOW CURRENT ROLE", None, None)}
    if BUILTIN_ADMIN_ROLE in roles:
        return Check("Admin role", True, f"{login} holds Memgraph's built-in admin role")
    return Check(
        "Admin role",
        False,
        f"{login} lacks Memgraph's built-in admin role, which apply needs. As an admin, run: "
        f"SET ROLE FOR {login} TO {BUILTIN_ADMIN_ROLE};",
    )


def _embeddings(query: Query) -> Check:
    found = query(
        "CALL mg.procedures() YIELD name WHERE name = $name RETURN name", None, {"name": EMBEDDINGS_PROCEDURE}
    )
    if found:
        return Check("Embeddings", True, f"{EMBEDDINGS_PROCEDURE} is installed (MAGE)")
    return Check(
        "Embeddings",
        False,
        f"{EMBEDDINGS_PROCEDURE} is missing, so recall can't embed. Run Memgraph from the memgraph/memgraph-mage image.",
    )


def _memory_limit(query: Query) -> Check:
    info = {row["storage info"]: row["value"] for row in query("SHOW STORAGE INFO", None, None)}
    # Memgraph reads the host's RAM, not a container's cgroup limit: --memory-limit must sit below the container's,
    # or the container is killed instead of the query being refused. Only the operator knows the container's limit.
    return Check(
        "Memory limit",
        True,
        f"Memgraph refuses queries above {info.get('memory_limit')}; in a container, keep --memory-limit below its limit",
    )
