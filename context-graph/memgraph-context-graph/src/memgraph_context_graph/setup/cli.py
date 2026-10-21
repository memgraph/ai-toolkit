"""``memgraph-context-graph setup`` and ``uninstall``: flags, and the plain front end.

The interactive front end (the questions in ``prompts.py``) runs only when stdin and stdout are both
terminals. Under ``curl | bash`` without ``</dev/tty`` stdin is the script, and it would wait for keys that
can never arrive (#534).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from memgraph_context_graph.setup import engine, uninstall, update
from memgraph_context_graph.setup.agents import AGENTS
from memgraph_context_graph.setup.memgraph import CONTAINER

if TYPE_CHECKING:
    from collections.abc import Sequence

_RUNTIMES = [agent.runtime for agent in AGENTS]
_MARKS = {"ok": "OK  ", "skip": "SKIP", "warn": "WARN", "fail": "FAIL"}


def setup_main(argv: Sequence[str]) -> int:
    """Run ``setup``; 0 when nothing failed, 1 otherwise, 2 on bad arguments."""
    from memgraph_context_graph.adapters._identity import MEMORY_BACKENDS

    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph setup",
        description="Install Context Graph for your agents: Memgraph, identity, the agents' wiring, and a check.",
        epilog="Interactive on a terminal; otherwise (or with --no-tui) it prints the plan and runs it.",
    )
    parser.add_argument(
        "runtimes",
        nargs="*",
        metavar="runtime",
        help=f"one of {', '.join(_RUNTIMES)}; default: every agent found on PATH",
    )
    parser.add_argument(
        "--project-dir",
        action="append",
        type=Path,
        default=[],
        help="project to wire for per-project agents (repeatable); default: the current directory",
    )
    parser.add_argument("--memgraph-url", help="use this Memgraph instead of finding or starting one")
    parser.add_argument("--user-id", help="your identity in the graph; default: the configured one, else git user.name")
    parser.add_argument(
        "--memory-backend", choices=sorted(MEMORY_BACKENDS), help="whose memory the selected agents use"
    )
    parser.add_argument("--no-viewer", action="store_true", help="don't install the terminal viewer")
    parser.add_argument(
        "--memgraph-container", default=CONTAINER, help=f"the container to find or start (default {CONTAINER})"
    )
    parser.add_argument("--plan", action="store_true", help="print what setup would do, and change nothing")
    parser.add_argument("--no-tui", action="store_true", help="plain output even on a terminal")  # fmt: skip
    args = parser.parse_args(list(argv))
    _check_runtimes(parser, args.runtimes)

    interactive = not args.plan and not args.no_tui and _interactive()
    terminal = None
    banner: AbstractContextManager[object] = nullcontext()
    if interactive:
        from memgraph_context_graph.setup.prompts import Prompts, StdTerminal

        terminal = StdTerminal()
        # Animated while looking around, which asks each agent's CLI and can take several seconds.
        banner = Prompts(terminal).animated_banner()

    with banner:
        env = engine.look_around()
    choices = engine.default_choices(
        env,
        runtimes=args.runtimes,
        project_dirs=args.project_dir,
        user_id=args.user_id,
        memgraph_url=args.memgraph_url,
        memory_backend=args.memory_backend,
        viewer=not args.no_viewer,
        container=args.memgraph_container,
    )
    if args.plan:
        _print_plan(choices, env)
        return 0
    if interactive:
        from memgraph_context_graph.setup.prompts import run_setup_prompts

        # Agents named on the command line start ticked; otherwise the ones already set up, or on a first run
        # the plugin agents found.
        return run_setup_prompts(choices, env, terminal, preselect=bool(args.runtimes))
    _print_plan(choices, env)
    print()
    outcomes = engine.run(choices, env, log=_log, report=_report)
    return _finish(outcomes, choices, env)


def uninstall_main(argv: Sequence[str]) -> int:
    """Run ``uninstall``; 0 when nothing failed, 1 otherwise, 2 on bad arguments."""
    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph uninstall",
        description="Remove Context Graph from your agents, and on your say-so the graph, the config and the tool.",
        epilog="On a terminal it asks about each, every answer defaulting to keep; otherwise only the flags remove "
        "more than the wiring.",
    )
    parser.add_argument("runtimes", nargs="*", metavar="runtime", help="default: every agent found wired")
    parser.add_argument("--project-dir", action="append", type=Path, default=[], help="project to unwire for per-project agents (repeatable); default: the current directory")  # fmt: skip
    parser.add_argument("--graph", action="store_true", help="also delete the graph: setup's Memgraph container and its data, every session in it")  # fmt: skip
    parser.add_argument("--config", action="store_true", help="also delete the config file and the capture-failure log")  # fmt: skip
    parser.add_argument("--tool", action="store_true", help="also uninstall the memgraph-context-graph uv tool")
    parser.add_argument("--all", action="store_true", help="all three: remove everything setup put on this machine")
    parser.add_argument("--memgraph-container", default=CONTAINER, help=f"the container setup started (default: {CONTAINER})")  # fmt: skip
    parser.add_argument("--no-tui", action="store_true", help="don't ask; remove only the wiring and what the flags name")  # fmt: skip
    args = parser.parse_args(list(argv))
    _check_runtimes(parser, args.runtimes)

    project_dirs = [path.expanduser().resolve() for path in args.project_dir] or [Path.cwd().resolve()]
    runtimes = args.runtimes or uninstall.wired(project_dirs)
    flagged = args.graph or args.config or args.tool or args.all
    if not flagged and not args.no_tui and _interactive():
        from memgraph_context_graph.setup.prompts import run_uninstall_prompts

        available = uninstall.removable(args.memgraph_container)
        return run_uninstall_prompts(runtimes, project_dirs, available, args.memgraph_container)

    removals = uninstall.Removals(
        graph=args.graph or args.all, config=args.config or args.all, tool=args.tool or args.all
    )
    if not runtimes:
        print("Nothing is wired: no Context Graph plugin, and no hook files in " + ", ".join(map(str, project_dirs)))
    outcomes = uninstall.run(
        runtimes, project_dirs, log=_log, report=_report, removals=removals, container=args.memgraph_container
    )
    if kept := uninstall.leftovers(removals, args.memgraph_container):
        print()
        print("\n".join(kept))
    return 1 if engine.failed(outcomes) else 0


def update_main(argv: Sequence[str]) -> int:
    """Run ``update``; 0 when nothing failed, 1 otherwise, 2 on bad arguments."""
    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph update",
        description="Upgrade the package, then refresh what setup wrote once: plugins, hook files, MCP registrations "
        "and the graph's schema. Your data and config are untouched.",
    )
    parser.add_argument("runtimes", nargs="*", metavar="runtime", help="default: every agent found wired")
    parser.add_argument("--project-dir", action="append", type=Path, default=[], help="project to refresh for per-project agents (repeatable); default: the current directory")  # fmt: skip
    parser.add_argument("--no-upgrade", action="store_true", help="refresh the wiring only; the package is already current")  # fmt: skip
    args = parser.parse_args(list(argv))
    _check_runtimes(parser, args.runtimes)

    project_dirs = [path.expanduser().resolve() for path in args.project_dir] or [Path.cwd().resolve()]
    if not args.no_upgrade:
        outcome = update.upgrade_packages(_log)
        _report(outcome)
        if outcome.status == "fail":
            print(f"To fix: {outcome.fix}")
            return 1
        if outcome.status == "ok":
            # This process still runs the old code; the new executable refreshes the wiring.
            return update.rerun(args.runtimes, project_dirs)
    runtimes = args.runtimes or update.wired(project_dirs)
    if not runtimes:
        print("Nothing is wired: no Context Graph plugin, and no hook files in " + ", ".join(map(str, project_dirs)))
        print("Run memgraph-context-graph setup to install it.")
        return 0
    outcomes = update.run(runtimes, project_dirs, log=_log, report=_report)
    print()
    if engine.failed(outcomes):
        for outcome in outcomes:
            if outcome.status == "fail" and outcome.fix:
                print(f"To fix {outcome.step}: {outcome.fix}")
        return 1
    print(update.RESTART_NOTE)
    return 0


def _check_runtimes(parser: argparse.ArgumentParser, runtimes: Sequence[str]) -> None:
    # argparse's `choices` rejects an empty `nargs="*"`, so names are checked here.
    unknown = [runtime for runtime in runtimes if runtime not in _RUNTIMES]
    if unknown:
        parser.error(f"unknown runtime {', '.join(unknown)} (choose from {', '.join(_RUNTIMES)})")


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_plan(choices: engine.Choices, env: engine.Environment) -> None:
    print("Plan:")
    for line in engine.plan(choices, env):
        print(f"  {line}")


def _log(line: str) -> None:
    print(f"       {line}", flush=True)


def _report(outcome: engine.Outcome) -> None:
    print(f"{_MARKS[outcome.status]}  {outcome.step}: {outcome.summary}", flush=True)
    for line in outcome.lines:
        print(f"       {line}")


def _finish(outcomes: Sequence[engine.Outcome], choices: engine.Choices, env: engine.Environment) -> int:
    failures = [outcome for outcome in outcomes if outcome.status == "fail"]
    print()
    if not failures:
        print("Context Graph is set up.")
        print()
        steps = engine.next_steps(choices, env)
        print(steps[0])
        for line in steps[1:]:
            print(f"  {line}")
        return 0
    print(f"{len(failures)} step(s) failed. To fix:")
    for outcome in failures:
        for line in (outcome.fix or "memgraph-context-graph setup").splitlines():
            print(f"  {outcome.step}: {line}")
    return 1
