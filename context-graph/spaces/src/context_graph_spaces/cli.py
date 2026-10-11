"""``context-graph-spaces``: the admin's commands for provisioning spaces from a spec.

The admin connection comes from flags, falling back to ``MEMGRAPH_URL`` /
``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD``. This is an interactive admin tool,
not a hook, so the config-file-only rule for hooks doesn't apply; the admin's
login is never written anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any, cast

from .provision import SERVICE_ACCOUNT, Plan, Query, Step, apply, observed_state, plan
from .spec import SpecError, load_spec

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Where the reconcile worker finds the service login's password.
SERVICE_PASSWORD_ENV = "CG_SERVICE_PASSWORD"


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``plan`` or ``apply``; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="context-graph-spaces", description=__doc__)
    parser.add_argument("--url", default=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("MEMGRAPH_USER", ""), help="An admin login.")
    parser.add_argument("--password", default=os.environ.get("MEMGRAPH_PASSWORD", ""))
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "Show what apply would change."), ("apply", "Make Memgraph match the spec.")):
        command = commands.add_parser(name, help=text)
        command.add_argument("spec", help="The spaces spec (TOML).")
        command.add_argument("--prune", action="store_true", help="Also drop removed users and databases.")
    args = parser.parse_args(argv)

    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(exc, file=sys.stderr)
        return 2
    query, close = admin_query(args.url, args.user, args.password)
    try:
        if args.command == "plan":
            _print_plan(plan(spec, observed_state(query, spec.members), prune=args.prune))
            return 0
        applied = apply(spec, query, prune=args.prune)
    finally:
        close()
    _print_steps(applied.steps, "Applied" if applied.steps else "Nothing to change; Memgraph matches the spec.")
    _print_kept(applied.kept)
    for login, password in sorted(applied.passwords.items()):
        if login == SERVICE_ACCOUNT:
            print(f"\nNew service login {login!r} for the shared-space reconcile worker; it isn't stored anywhere.")
            print(f"  Give the worker: {SERVICE_PASSWORD_ENV}='{password}'")
            continue
        print(f"\nNew login {login!r}. Hand this to them once; it isn't stored anywhere:")
        print(f"  agent-context-graph join {args.url} {login} --password '{password}'")
    return 0


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
