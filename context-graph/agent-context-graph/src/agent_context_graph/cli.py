"""Top-level CLI for agent-context-graph."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: agent-context-graph <command> [options]

Commands:
  doctor           Check runtime hook dependencies and Memgraph connectivity.
  setup <runtime>  Configure an agent runtime.
  hook <command>  Configure or run command hooks.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch top-level agent-context-graph commands."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        print(_HELP)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_HELP)
        return 0
    if args[0] in {"--version", "version"}:
        print(_package_version("agent-context-graph"))
        return 0

    command = args[0]
    if command == "doctor":
        return _doctor(args[1:])

    if command == "hook":
        from agent_context_graph.hooks.cli import main as hook_main

        return hook_main(args[1:])

    if command == "setup":
        from agent_context_graph.hooks.cli import main as hook_main

        return hook_main(["init", *args[1:]])

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP)
    return 2


def _doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check Agent Context Graph runtime hook setup.")
    parser.add_argument(
        "--runtime",
        choices=["codex", "claude-code"],
        default="claude-code",
        help="Runtime hook adapter to smoke test.",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Defaults to skills-graph.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    connectors = args.connector or ["skills-graph"]
    checks = [
        _check_cli(),
        _check_package("agent-context-graph"),
    ]
    for connector in connectors:
        checks.append(_check_connector(connector))
    checks.append(_check_runtime(args.runtime, connectors))

    ok = all(check["ok"] for check in checks)
    payload = {"ok": ok, "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_doctor(payload)
    return 0 if ok else 1


def _check_cli() -> dict[str, object]:
    path = shutil.which("agent-context-graph") or sys.argv[0]
    return {"name": "agent-context-graph executable", "ok": bool(path), "detail": path or "not found"}


def _check_package(package_name: str) -> dict[str, object]:
    version = _package_version(package_name)
    ok = version != "unknown"
    return {"name": package_name, "ok": ok, "detail": version if ok else "not installed"}


def _check_connector(connector_name: str) -> dict[str, object]:
    normalized = connector_name.strip().replace("-", "_")
    if normalized != "skills_graph":
        return {"name": f"connector:{connector_name}", "ok": False, "detail": "unsupported connector"}

    try:
        from skills_graph import SkillGraph

        graph = SkillGraph()
        version = _package_version("skills-graph")
        return {"name": "connector:skills-graph", "ok": True, "detail": f"installed={version}; memgraph=reachable"}
    except Exception as exc:
        return {"name": "connector:skills-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        driver = getattr(getattr(locals().get("graph", None), "_db", None), "driver", None)
        if driver is not None:
            driver.close()


def _check_runtime(runtime: str, connectors: list[str]) -> dict[str, object]:
    try:
        if runtime == "codex":
            from agent_context_graph.adapters.codex import CodexHooksAdapter, create_link

            adapter_cls = CodexHooksAdapter
        else:
            from agent_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter, create_link

            adapter_cls = ClaudeCodeHooksAdapter

        link = create_link(connectors)
        adapter = adapter_cls(link)
        adapter.handle_payload({"hook_event_name": "Stop", "session_id": "doctor"})
        return {"name": f"runtime:{runtime}", "ok": True, "detail": "strict hook smoke passed"}
    except Exception as exc:
        return {"name": f"runtime:{runtime}", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def _print_doctor(payload: dict[str, object]) -> None:
    for check in payload["checks"]:
        status = "OK" if check["ok"] else "FAIL"
        print(f"{status} {check['name']}: {check['detail']}")


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
