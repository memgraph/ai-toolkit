"""``context-graph-spaces``: the admin's commands for a spaces server.

``compose`` prints a server to run, ``init`` makes its first admin login and checks it,
``plan``/``apply`` provision spaces from a spec, and ``reconcile`` is the shared-space worker.

The admin connection comes from flags, falling back to ``MEMGRAPH_URL`` /
``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD``. This is an interactive admin tool,
not a hook, so the config-file-only rule for hooks doesn't apply; the admin's
login is never written anywhere.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import TYPE_CHECKING, Any, cast

from .provision import SERVICE_ACCOUNT, ApplyError, Plan, Query, Step, apply, observed_state, plan
from .spec import SpecError, load_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .spec import Spec

#: Where the reconcile worker finds the service login's password.
SERVICE_PASSWORD_ENV = "CG_SERVICE_PASSWORD"


def main(argv: Sequence[str] | None = None) -> int:
    """Run one admin command; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="context-graph-spaces", description=__doc__)
    parser.add_argument("--url", default=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("MEMGRAPH_USER", ""), help="An admin login.")
    parser.add_argument("--password", default=os.environ.get("MEMGRAPH_PASSWORD", ""))
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "Show what apply would change."), ("apply", "Make Memgraph match the spec.")):
        command = commands.add_parser(name, help=text)
        command.add_argument("spec", help="The spaces spec (TOML).")
        command.add_argument("--prune", action="store_true", help="Also drop removed users and databases.")
    reconcile = commands.add_parser("reconcile", help="Reconcile shared spaces' changed memory files, as cg_service.")
    reconcile.add_argument("spec", help="The spaces spec (TOML): which shared spaces to reconcile.")
    reconcile.add_argument("--watch", action="store_true", help="Keep going, one pass every --interval seconds.")
    reconcile.add_argument("--interval", type=float, default=30.0)
    init = commands.add_parser(
        "init", help="Create a fresh Memgraph's first admin login, then check it can host spaces."
    )
    init.add_argument("--admin", default="cgadmin", help="The admin login to create on a fresh Memgraph.")
    compose = commands.add_parser("compose", help="Print a docker-compose file for a spaces server and its worker.")
    compose.add_argument("--bind", default="127.0.0.1", help="The host address Bolt listens on, e.g. a private IP.")
    compose.add_argument("--port", type=int, default=7687)
    compose.add_argument("--memory-gib", type=int, default=4, help="The Memgraph container's memory.")
    compose.add_argument("--spec", default="./spaces.toml", help="The spec's path on the server, for the worker.")
    args = parser.parse_args(argv)

    if args.command == "compose":
        return _compose(args.bind, args.port, args.memory_gib, args.spec)
    if args.command == "init":
        return _init(args.url, args.admin, args.user, args.password)
    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.command == "reconcile":
        return _reconcile(spec, args.url, watch=args.watch, interval=args.interval)
    query, close = admin_query(args.url, args.user, args.password)
    try:
        if args.command == "plan":
            _print_plan(plan(spec, observed_state(query, spec.members), prune=args.prune))
            return 0
        applied = apply(spec, query, prune=args.prune)
    except ApplyError as failed:
        _print_steps(failed.applied.steps, "Applied before the failure:")
        _print_passwords(failed.applied.passwords, args.url)
        advice = _describe(failed.cause, args.url, args.user)
        print(f"\napply stopped; run it again once this is fixed:\n{advice}", file=sys.stderr)
        return 1
    except Exception as exc:
        explained = _explain(exc, args.url, args.user)
        if explained is None:
            raise
        print(explained, file=sys.stderr)
        return 1
    finally:
        close()
    _print_steps(applied.steps, "Applied" if applied.steps else "Nothing to change; Memgraph matches the spec.")
    _print_kept(applied.kept)
    _print_passwords(applied.passwords, args.url)
    return 0


def _compose(bind: str, port: int, memory_gib: int, spec: str) -> int:
    from .compose import compose_file

    try:
        print(compose_file(bind=bind, port=port, memory_gib=memory_gib, spec=spec), end="")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


def _init(url: str, admin: str, user: str, password: str) -> int:
    """Create the first admin on a fresh Memgraph, or use the given one; then check the instance, exit 1 on a failure."""
    from .instance import check_instance, create_first_admin

    try:
        created = create_first_admin(url, admin)
    except SpecError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        explained = _explain(exc, url, admin)
        if explained is None:
            raise
        print(explained, file=sys.stderr)
        return 1
    if created is not None:
        user, password = admin, created
        print(f"Created the admin login {admin!r}; it holds Memgraph's built-in admin role. Shown once, never stored:")
        print(f"  export MEMGRAPH_URL='{url}' MEMGRAPH_USER='{admin}' MEMGRAPH_PASSWORD='{created}'")
    elif not user:
        print(
            f"Memgraph at {url} already has logins, so init created none. To check it, pass an admin login "
            "with --user and --password, or set MEMGRAPH_USER and MEMGRAPH_PASSWORD.",
            file=sys.stderr,
        )
        return 1
    query, close = admin_query(url, user, password)
    try:
        checks = check_instance(query, user)
    except Exception as exc:
        explained = _explain(exc, url, user)
        if explained is None:
            raise
        print(explained, file=sys.stderr)
        return 1
    finally:
        close()
    print("Checks:")
    for check in checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
        return 1
    print("Ready for spaces. Next: context-graph-spaces plan|apply <spec>")
    return 0


def _print_passwords(passwords: dict[str, str], url: str) -> None:
    """The one-time setup line for each login apply created; the passwords aren't stored anywhere else."""
    for login, password in sorted(passwords.items()):
        if login == SERVICE_ACCOUNT:
            print(f"\nNew service login {login!r} for the shared-space reconcile worker; it isn't stored anywhere.")
            print(f"  Give the worker: {SERVICE_PASSWORD_ENV}='{password}'")
            continue
        print(f"\nNew login {login!r}. Hand this to them once; it isn't stored anywhere:")
        print(f"  memgraph-context-graph join {url} {login} --password='{password}'")


def _reconcile(spec: Spec, url: str, *, watch: bool, interval: float) -> int:
    """Run the worker once, or every *interval* seconds with *watch*; exit 1 if the last pass had an error."""
    import time

    from .worker import reconcile_spaces

    password = os.environ.get(SERVICE_PASSWORD_ENV)
    if not password:
        print(f"Set {SERVICE_PASSWORD_ENV} to {SERVICE_ACCOUNT}'s password (apply printed it once).", file=sys.stderr)
        return 2
    while True:
        reports = reconcile_spaces(spec, url=url, password=password)
        for report in reports:
            if report.error:
                print(f"{report.space}: not reconciled: {report.error}", file=sys.stderr)
                continue
            failed = "".join(f"\n  failed {path}: {error}" for path, error in report.failed)
            # flush: under a service manager stdout is a pipe, block-buffered, and a pass's report would sit unseen.
            print(
                f"{report.space}: {len(report.completed)} reconciled, {report.pruned} leftovers removed{failed}",
                flush=True,
            )
        if not watch:
            return 1 if any(report.error or report.failed for report in reports) else 0
        time.sleep(interval)


def _explain(exc: Exception, url: str, user: str) -> str | None:
    """What to tell an admin whose connection or privileges fell short, or None for any other error."""
    import neo4j

    from .instance import ADMIN_PRIVILEGES

    if isinstance(exc, neo4j.exceptions.ServiceUnavailable):
        return f"Can't reach Memgraph at {url}. Is it running, and is the URL right (--url or MEMGRAPH_URL)?"
    if isinstance(exc, neo4j.exceptions.AuthError) or (
        isinstance(exc, neo4j.exceptions.ClientError) and "Unauthenticated" in str(exc.code)
    ):
        login = f"login {user!r}" if user else "(no login was given)"
        return (
            f"Memgraph at {url} refused the admin {login}. "
            "Pass an admin login with --user and --password, or set MEMGRAPH_USER and MEMGRAPH_PASSWORD."
        )
    if isinstance(exc, neo4j.exceptions.ClientError) and _UNPRIVILEGED.search(exc.message or ""):
        return (
            f"Memgraph refused the login {user!r}: {exc.message}\n"
            "Provisioning needs Memgraph's built-in admin role, or the privileges it uses granted directly. "
            "As an existing admin, run either:\n"
            f"  SET ROLE FOR {user} TO admin;\n"
            f"  GRANT {', '.join(ADMIN_PRIVILEGES)} TO {user}; GRANT DATABASE * TO {user};\n"
            "(On a fresh instance the first user created already holds the role.)"
        )
    return None


def _describe(exc: Exception, url: str, user: str) -> str:
    """:func:`_explain`'s advice when there is some, else the error itself."""
    return _explain(exc, url, user) or f"{type(exc).__name__}: {exc}"


#: How Memgraph words a refusal for missing privileges: a role-based one, a database grant, or a label permission.
_UNPRIVILEGED = re.compile(r"not authorized|missing \w+ permission")


def admin_query(url: str, user: str, password: str) -> tuple[Query, Any]:
    """A :data:`~.provision.Query` over one admin driver, and the function that closes it."""
    import neo4j

    driver = neo4j.GraphDatabase.driver(url, auth=(user, password))

    def query(statement: str, database: str | None, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        # Auth and multi-database DDL can't run in an explicit transaction, so each statement gets its own session.
        with driver.session(database=database) as session:
            # The driver's stubs want a LiteralString; admin DDL is built from names spec._NAME has validated.
            return [record.data() for record in session.run(cast("Any", statement), params or {})]

    return query, driver.close


def _print_plan(the_plan: Plan) -> None:
    _print_steps(the_plan.steps, "Plan:" if the_plan.steps else "Nothing to change; Memgraph matches the spec.")
    _print_kept(the_plan.kept)


def _print_steps(steps: list[Step], heading: str) -> None:
    print(heading)
    for step in steps:
        print(f"  + {step.summary}")


def _print_kept(kept: list[str]) -> None:
    if kept:
        print(f"Kept (pass --prune to drop): {', '.join(kept)}")
