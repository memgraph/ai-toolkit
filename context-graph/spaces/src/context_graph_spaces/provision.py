"""Provisioning: make Memgraph's databases, users, roles and grants match a :class:`~.spec.Spec`.

``plan`` reads Memgraph's current state and lists the changes; ``apply`` runs
them. Both are idempotent: once Memgraph matches the spec, ``plan`` is empty.

The shapes it maintains (map #501):

- Each member ``<login>`` has a personal database ``u_<login>`` as their main
  database, reached through a role that may do anything inside it.
- Each shared space ``<name>`` is a database ``space_<name>``. Its members hold
  a role, assigned on that database only, that may read everything there and
  create or change only the spec's labels and edge types.
- ``cg_service`` holds a worker role on every shared database; it writes what
  members can't, such as Chunks and entities.
- Each member's personal database holds ``(:SpaceMembership {space, database})``
  nodes naming the shared spaces they belong to: an unprivileged login can't
  list its own grants, and recall needs to know which spaces to read. Memgraph's
  grants stay the authority; a stale node only costs a permission error.

Roles are named after a hash of what they grant, so changing a space's labels
creates a new role, moves members onto it (new grant before the old revoke, so
access never lapses), and drops the old one. That keeps ``plan`` a structural
diff: Memgraph can list label and property grants, but revoking them unevenly
(a label wildcard revoke clears specific label rules, a property wildcard revoke
doesn't) makes editing a role in place unsafe.

Only objects with the provisioner's prefixes (``u_``, ``space_``, ``cg_``) are
ever changed; an admin's other users, roles and databases are left alone.
Removing a member or space revokes access but keeps its data: databases and
users are dropped only with ``prune``.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .spec import SharedSpace, Spec

SERVICE_ACCOUNT = "cg_service"
#: The label of a member's catalog of their shared spaces, in their personal database.
MEMBERSHIP_LABEL = "SpaceMembership"
_OWNED_PREFIXES = ("u_", "space_", "cg_")
_PRIVILEGES = "MATCH, CREATE, MERGE, SET, DELETE, REMOVE"
_OWNER_PRIVILEGES = _PRIVILEGES + ", INDEX, STATS, CONSTRAINT"
_EVERYTHING = (
    "GRANT * ON NODES CONTAINING LABELS * TO ROLE {role}",
    "GRANT * ON EDGES OF TYPE * TO ROLE {role}",
    "GRANT READ, SET PROPERTY {{*}} ON NODES CONTAINING LABELS * TO ROLE {role}",
    "GRANT READ, SET PROPERTY {{*}} ON EDGES OF TYPE * TO ROLE {role}",
)


def personal_database(login: str) -> str:
    """The database holding *login*'s personal space."""
    return f"u_{login}"


def shared_database(space: str) -> str:
    """The database holding shared space *space*."""
    return f"space_{space}"


def _owned(name: str) -> bool:
    return name.startswith(_OWNED_PREFIXES)


@dataclass(frozen=True)
class Role:
    """A role the provisioner owns: the databases it reaches and the grants it carries."""

    prefix: str
    databases: tuple[str, ...]
    grants: tuple[str, ...]

    @property
    def name(self) -> str:
        """``<prefix>_<hash of databases and grants>``: a different grant set is a different role."""
        digest = hashlib.sha256("\n".join((*self.databases, *self.grants)).encode()).hexdigest()[:8]
        return f"{self.prefix}_{digest}"

    def statements(self) -> list[str]:
        """Create the role and give it everything it carries."""
        return [
            f"CREATE ROLE {self.name}",
            *(f"GRANT DATABASE {database} TO ROLE {self.name}" for database in self.databases),
            *(grant.format(role=self.name) for grant in self.grants),
        ]


@dataclass(frozen=True)
class Assignment:
    """*user* holds *role* on *database*."""

    user: str
    role: str
    database: str


@dataclass
class State:
    """The provisioner's objects, as desired from a spec or as observed in Memgraph."""

    databases: set[str] = field(default_factory=set)
    users: set[str] = field(default_factory=set)
    roles: dict[str, Role | None] = field(default_factory=dict)
    assignments: set[Assignment] = field(default_factory=set)
    #: profile name -> (memory limit in MB, databases it's attached to)
    profiles: dict[str, tuple[int | None, frozenset[str]]] = field(default_factory=dict)
    #: login -> the shared spaces their catalog names, as (space, database)
    catalogs: dict[str, frozenset[tuple[str, str]]] = field(default_factory=dict)
    #: login -> the databases they can reach, directly or through a role (observed only)
    access: dict[str, frozenset[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    """One change: what it does, in words, and the statements that do it, each with the database to run it on."""

    summary: str
    statements: tuple[tuple[str, str | None], ...] = ()


@dataclass
class Plan:
    """The changes that make Memgraph match the spec, in the order they must run."""

    steps: list[Step] = field(default_factory=list)
    #: Logins ``apply`` will create, which need a generated password.
    new_users: list[str] = field(default_factory=list)
    #: What a removal keeps because ``prune`` wasn't asked for.
    kept: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """Whether Memgraph already matches the spec."""
        return not self.steps


def desired_state(spec: Spec) -> State:
    """What Memgraph should hold for *spec*."""
    state = State()
    for login in spec.members:
        database = personal_database(login)
        role = Role(
            f"cg_u_{login}",
            (database,),
            (f"GRANT {_OWNER_PRIVILEGES} TO ROLE {{role}}", *_EVERYTHING),
        )
        state.databases.add(database)
        state.users.add(login)
        state.roles[role.name] = role
        state.assignments.add(Assignment(login, role.name, database))
        state.catalogs[login] = frozenset((space.name, shared_database(space.name)) for space in spec.spaces_of(login))
    for space in spec.spaces:
        database = shared_database(space.name)
        role = _member_role(space)
        state.databases.add(database)
        state.roles[role.name] = role
        state.assignments.update(Assignment(login, role.name, database) for login in space.members)
    if spec.spaces:
        worker = Role(
            "cg_worker",
            tuple(sorted(shared_database(space.name) for space in spec.spaces)),
            (f"GRANT {_OWNER_PRIVILEGES} TO ROLE {{role}}", *_EVERYTHING),
        )
        state.users.add(SERVICE_ACCOUNT)
        state.roles[worker.name] = worker
        state.assignments.update(Assignment(SERVICE_ACCOUNT, worker.name, database) for database in worker.databases)
    limits: dict[int, set[str]] = {}
    if spec.personal_memory_limit_mb:
        limits.setdefault(spec.personal_memory_limit_mb, set()).update(map(personal_database, spec.members))
    for space in spec.spaces:
        if space.memory_limit_mb:
            limits.setdefault(space.memory_limit_mb, set()).add(shared_database(space.name))
    state.profiles = {f"cg_{mb}mb": (mb, frozenset(databases)) for mb, databases in limits.items()}
    return state


def _member_role(space: SharedSpace) -> Role:
    grants = [
        f"GRANT {_PRIVILEGES} TO ROLE {{role}}",
        "GRANT READ ON NODES CONTAINING LABELS * TO ROLE {role}",
        "GRANT READ ON EDGES OF TYPE * TO ROLE {role}",
        "GRANT READ {{*}} ON NODES CONTAINING LABELS * TO ROLE {role}",
        "GRANT READ {{*}} ON EDGES OF TYPE * TO ROLE {role}",
    ]
    for label in space.labels:
        grants += [
            f"GRANT * ON NODES CONTAINING LABELS :{label} TO ROLE {{role}}",
            f"GRANT READ, SET PROPERTY {{{{*}}}} ON NODES CONTAINING LABELS :{label} TO ROLE {{role}}",
        ]
    for edge_type in space.edge_types:
        grants += [
            f"GRANT * ON EDGES OF TYPE :{edge_type} TO ROLE {{role}}",
            f"GRANT READ, SET PROPERTY {{{{*}}}} ON EDGES OF TYPE :{edge_type} TO ROLE {{role}}",
        ]
    return Role(f"space_{space.name}_member", (shared_database(space.name),), tuple(grants))


#: Runs one statement as the admin: (statement, database or None for the admin's main one, parameters).
Query = Callable[[str, str | None, dict[str, Any] | None], list[dict[str, Any]]]


def observed_state(query: Query, members: Iterable[str] = ()) -> State:
    """What Memgraph holds of the provisioner's objects.

    *members* are logins to inspect even if they hold no role yet.
    """
    state = State()
    state.databases = {row["Name"] for row in query("SHOW DATABASES", None, None) if _owned(row["Name"])}
    all_users = {row["user"] for row in query("SHOW USERS", None, None)}
    state.roles = {
        row["role"]: None for row in query("SHOW ROLES", None, None) if not row.get("builtin") and _owned(row["role"])
    }
    holders = {SERVICE_ACCOUNT} & all_users
    for role in state.roles:
        holders.update(row["users"] for row in query(f"SHOW USERS FOR ROLE {role}", None, None))
    holders.update(set(members) & all_users)
    # A removed member holds no role any more; their personal database still marks them as ours.
    holders.update(user for user in all_users if personal_database(user) in state.databases)
    state.users = holders
    for user in holders:
        rows = query(f"SHOW DATABASE PRIVILEGES FOR USER {user}", None, None)
        state.access[user] = frozenset(name for row in rows for name in row.get("grants") or ())
    for user in holders:
        for database in state.databases:
            for row in query(f"SHOW ROLE FOR {user} ON DATABASE {database}", None, None):
                if row.get("role") in state.roles:
                    state.assignments.add(Assignment(user, row["role"], database))
    for row in query("SHOW TENANT PROFILES", None, None):
        if _owned(row["profile"]):
            databases = frozenset(name.strip() for name in str(row.get("databases") or "").split(",") if name.strip())
            state.profiles[row["profile"]] = (_megabytes(row.get("memory_limit")), databases)
    for user in holders - {SERVICE_ACCOUNT}:
        database = personal_database(user)
        if database in state.databases:
            rows = query(
                f"MATCH (m:{MEMBERSHIP_LABEL}) RETURN m.space AS space, m.database AS database", database, None
            )
            state.catalogs[user] = frozenset((row["space"], row["database"]) for row in rows)
    return state


def _megabytes(text: Any) -> int | None:
    """``"64.00MiB"`` -> 64; anything else (``"UNLIMITED"``) -> None."""
    value = str(text or "")
    if value.endswith("MiB"):
        return round(float(value[:-3]))
    if value.endswith("GiB"):
        return round(float(value[:-3]) * 1024)
    return None


def plan(spec: Spec, observed: State, *, prune: bool = False) -> Plan:
    """The steps that turn *observed* into *spec*'s desired state."""
    desired = desired_state(spec)
    result = Plan()
    steps = result.steps

    for database in sorted(desired.databases - observed.databases):
        steps.append(Step(f"create database {database}", ((f"CREATE DATABASE {database}", None),)))
    for name, (mb, databases) in sorted(desired.profiles.items()):
        current = observed.profiles.get(name)
        if current is None:
            steps.append(
                Step(
                    f"create tenant profile {name}",
                    ((f"CREATE TENANT PROFILE {name} LIMIT memory_limit {mb} MB", None),),
                )
            )
        for database in sorted(databases - (current[1] if current else frozenset())):
            steps.append(
                Step(f"limit {database} with {name}", ((f"SET TENANT PROFILE ON DATABASE {database} TO {name}", None),))
            )
    for name in sorted(set(desired.roles) - set(observed.roles)):
        role = desired.roles[name]
        assert role is not None
        steps.append(Step(f"create role {name}", tuple((statement, None) for statement in role.statements())))
    for login in sorted(desired.users - observed.users):
        result.new_users.append(login)
        # The password is generated at apply time, never shown by plan.
        steps.append(
            Step(f"create user {login}", (("CREATE USER IF NOT EXISTS {login} IDENTIFIED BY {password}", login),))
        )
    for login in sorted(desired.users - observed.users - {SERVICE_ACCOUNT}):
        steps.append(Step(f"make {personal_database(login)} {login}'s main database", _main_database(login)))
    for assignment in sorted(desired.assignments - observed.assignments, key=_order):
        steps.append(
            Step(
                f"give {assignment.user} {assignment.role} on {assignment.database}",
                ((f"GRANT ROLE {assignment.role} TO {assignment.user} ON {assignment.database}", None),),
            )
        )
    for login, spaces in sorted(desired.catalogs.items()):
        if observed.catalogs.get(login, frozenset()) != spaces:
            names = ", ".join(sorted(space for space, _ in spaces)) or "none"
            steps.append(Step(f"record {login}'s shared spaces: {names}", ((_CATALOG, login),)))

    for assignment in sorted(observed.assignments - desired.assignments, key=_order):
        steps.append(
            Step(
                f"take {assignment.role} on {assignment.database} from {assignment.user}",
                ((f"REVOKE ROLE {assignment.role} FROM {assignment.user} ON {assignment.database}", None),),
            )
        )
    for name in sorted(set(observed.roles) - set(desired.roles)):
        steps.append(Step(f"drop role {name}", ((f"DROP ROLE {name}", None),)))
    for login in sorted(observed.users - desired.users):
        database = personal_database(login)
        if database in observed.access.get(login, frozenset()):
            steps.append(
                Step(
                    f"revoke {login}'s access to {database}", ((f"REVOKE DATABASE {database} FROM USER {login}", None),)
                )
            )
        if prune:
            steps.append(Step(f"drop user {login}", ((f"DROP USER {login}", None),)))
        else:
            result.kept.append(f"user {login}")
    for name, (_, databases) in sorted(observed.profiles.items()):
        wanted = desired.profiles.get(name, (None, frozenset()))[1]
        for database in sorted(databases - wanted):
            steps.append(Step(f"unlimit {database}", ((f"REMOVE TENANT PROFILE FROM DATABASE {database}", None),)))
        if name not in desired.profiles:
            steps.append(Step(f"drop tenant profile {name}", ((f"DROP TENANT PROFILE {name}", None),)))
    for database in sorted(observed.databases - desired.databases):
        if prune:
            steps.append(Step(f"drop database {database}", ((f"DROP DATABASE {database} FORCE", None),)))
        else:
            result.kept.append(f"database {database}")
    return result


def _main_database(login: str) -> tuple[tuple[str, str | None], ...]:
    # The direct grant is what SET MAIN DATABASE needs: revoking it clears the main database even when a role
    # still reaches the database.
    database = personal_database(login)
    return ((f"GRANT DATABASE {database} TO USER {login}", None), (f"SET MAIN DATABASE {database} FOR {login}", None))


#: Catalog rewrite marker: run against a member's personal database with their shared spaces.
_CATALOG = "-- catalog --"


def _order(assignment: Assignment) -> tuple[str, str, str]:
    return (assignment.user, assignment.database, assignment.role)


@dataclass
class Applied:
    """What ``apply`` did: the steps run and the passwords of users it created (shown once, never stored)."""

    steps: list[Step]
    passwords: dict[str, str]
    kept: list[str]


def apply(spec: Spec, query: Query, *, prune: bool = False) -> Applied:
    """Make Memgraph match *spec*, then return what was done.

    Raises:
        neo4j.exceptions.Neo4jError: a statement Memgraph rejected; the steps before it stay applied, and
            re-running ``apply`` continues from where it stopped.
    """
    the_plan = plan(spec, observed_state(query, spec.members), prune=prune)
    desired = desired_state(spec)
    passwords = {login: secrets.token_urlsafe(18) for login in the_plan.new_users}
    for step in the_plan.steps:
        for statement, target in step.statements:
            if statement == _CATALOG:
                assert target is not None
                _write_catalog(query, target, desired.catalogs[target])
            elif "{password}" in statement:
                assert target is not None
                # token_urlsafe only yields [A-Za-z0-9_-], so the quoted literal can't be broken out of;
                # Memgraph's CREATE USER takes no query parameters.
                query(statement.format(login=target, password=f"'{passwords[target]}'"), None, None)
            else:
                query(statement, target, None)
    # Memgraph can't report a user's main database, so plan can't see it drift (a revoke clears it); re-assert it.
    for login in spec.members:
        for statement, target in _main_database(login):
            query(statement, target, None)
    return Applied(the_plan.steps, passwords, the_plan.kept)


def _write_catalog(query: Query, login: str, spaces: frozenset[tuple[str, str]]) -> None:
    rows = [{"space": space, "database": shared} for space, shared in sorted(spaces)]
    query(
        f"MATCH (m:{MEMBERSHIP_LABEL}) DETACH DELETE m "
        f"WITH count(*) AS _ UNWIND $rows AS row "
        f"CREATE (:{MEMBERSHIP_LABEL} {{space: row.space, database: row.database}})",
        personal_database(login),
        {"rows": rows},
    )
