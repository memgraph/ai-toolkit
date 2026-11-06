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
#: What init and apply run as the admin: the licence and storage checks (CONFIG, STATS); logins, roles and
#: grants; databases and their memory caps; and, in each shared space, the memory store's schema and
#: membership. The built-in admin role implies all of them.
ADMIN_PRIVILEGES = (
    "CONFIG",
    "STATS",
    "AUTH",
    "MULTI_DATABASE_EDIT",
    "MULTI_DATABASE_USE",
    "PROFILE_RESTRICTION",
    "MATCH",
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "REMOVE",
    "INDEX",
    "CONSTRAINT",
)


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
    return [_licence(query), _admin_privileges(query, login), _embeddings(query), _memory_limit(query)]


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


def _admin_privileges(query: Query, login: str) -> Check:
    """Whether *login* can run init and apply: the built-in admin role, or every privilege they use on every database.

    An admin created before multi-tenancy often holds no role, only privileges granted directly; apply
    works for it, so the check asks for what they use, not which role grants it.
    """
    name = "Admin privileges"
    roles = {row["role"] for row in query("SHOW CURRENT ROLE", None, None)}
    if BUILTIN_ADMIN_ROLE in roles:
        return Check(name, True, f"{login} holds Memgraph's built-in admin role")
    missing = _missing_privileges(query, login)
    if not missing:
        return Check(name, True, f"{login} holds every privilege init and apply use, on every database")
    return Check(
        name,
        False,
        f"{login} can't provision: it lacks {', '.join(missing)}. As an admin, run: "
        f"SET ROLE FOR {login} TO {BUILTIN_ADMIN_ROLE}; or, to grant the privileges directly: "
        f"GRANT {', '.join(ADMIN_PRIVILEGES)} TO {login}; GRANT DATABASE * TO {login};",
    )


def _missing_privileges(query: Query, login: str) -> list[str]:
    """What of :data:`ADMIN_PRIVILEGES` and access to every database *login* lacks, as Memgraph names them."""
    import neo4j

    # Login names can't be query parameters; backticks, doubled inside, quote any name as one identifier.
    quoted = "`" + login.replace("`", "``") + "`"
    try:
        privileges = query(f"SHOW PRIVILEGES FOR {quoted} ON MAIN", None, None)
        databases = query(f"SHOW DATABASE PRIVILEGES FOR {quoted}", None, None)
    except neo4j.exceptions.ClientError:
        # Reading privileges needs AUTH, which apply needs too.
        return ["AUTH"]
    granted = {row["privilege"] for row in privileges if row["effective"] == "GRANT"}
    missing = [privilege for privilege in ADMIN_PRIVILEGES if privilege not in granted]
    if not any(row["grants"] == "*" and not row["denies"] for row in databases):
        missing.append("DATABASE *")
    return missing


def _embeddings(query: Query) -> Check:
    """Whether recall's model loads here: the procedure alone says nothing about a model it can't download."""
    from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL

    model = DEFAULT_EMBEDDING_MODEL

    found = query(
        "CALL mg.procedures() YIELD name WHERE name = $name RETURN name", None, {"name": EMBEDDINGS_PROCEDURE}
    )
    if not found:
        return Check(
            "Embeddings",
            False,
            f"{EMBEDDINGS_PROCEDURE} is missing, so recall can't embed. "
            "Run Memgraph from the memgraph/memgraph-mage image.",
        )
    rows = query(
        f"CALL {EMBEDDINGS_PROCEDURE}([$text], {{model_name: $model}}) YIELD success, embeddings "
        "RETURN success, embeddings",
        None,
        {"text": "context graph", "model": model},
    )
    if rows and rows[0]["success"] and rows[0]["embeddings"]:
        dimensions = len(rows[0]["embeddings"][0])
        return Check(
            "Embeddings",
            True,
            f"{model} loads ({dimensions} dimensions); on a cluster, every data instance needs it, or recall "
            "breaks at the next failover",
        )
    return Check("Embeddings", False, f"{EMBEDDINGS_PROCEDURE} can't load {model}: {_model_error(query, model)}")


def _model_error(query: Query, model: str) -> str:
    """Why *model* doesn't load, from the error ``embeddings.model_info`` raises, and how to fix it."""
    try:
        query("CALL embeddings.model_info({model_name: $model}) YIELD info RETURN info", None, {"model": model})
        cause = "the model loads for model_info but not for embedding"
    except Exception as exc:  # the procedure's own exception, whatever the model hub or the filesystem raised
        lines = [line.strip() for line in str(exc).splitlines() if line.strip() not in {"", "}"}]
        cause = lines[-1] if lines else type(exc).__name__
    if "Read-only file system" in cause:
        return (
            f"{cause}. The model cache is read-only (e.g. the Helm chart's readOnlyRootFilesystem). Point it at a "
            "writable mount on every data instance: HF_HOME=/var/lib/memgraph/huggingface (Helm: extraEnv.data)."
        )
    return (
        f"{cause}. The first call downloads the model, so each data instance needs outbound HTTPS to "
        "huggingface.co once, or a pre-seeded HF_HOME with HF_HUB_OFFLINE=1."
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
