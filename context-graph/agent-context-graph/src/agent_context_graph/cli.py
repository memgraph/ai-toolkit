"""Top-level CLI for agent-context-graph."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: agent-context-graph <command> [options]

Commands:
  bootstrap        Install runtime dependencies and verify hook capture.
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
    if command == "bootstrap":
        return _bootstrap(args[1:])

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


def _bootstrap(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Install and verify Agent Context Graph runtime dependencies.")
    parser.add_argument(
        "--runtime",
        choices=["codex", "claude-code"],
        required=True,
        help="Runtime hook adapter to bootstrap.",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Defaults to skills-graph.",
    )
    parser.add_argument(
        "--memgraph-host",
        default=None,
        help="Memgraph host. Defaults to MEMGRAPH_HOST or localhost.",
    )
    parser.add_argument(
        "--memgraph-port",
        type=int,
        default=None,
        help="Memgraph Bolt port. Defaults to MEMGRAPH_PORT or 7687.",
    )
    parser.add_argument(
        "--no-reinstall",
        action="store_true",
        help="Do not pass --reinstall to uv tool install.",
    )
    args = parser.parse_args(argv)

    connectors = args.connector or ["skills-graph"]
    host = args.memgraph_host if args.memgraph_host is not None else os.environ.get("MEMGRAPH_HOST", "localhost")
    port = args.memgraph_port if args.memgraph_port is not None else int(os.environ.get("MEMGRAPH_PORT", "7687"))

    uv = shutil.which("uv")
    if not uv:
        print("FAIL uv: not found on PATH", file=sys.stderr)
        print("Install uv first:", file=sys.stderr)
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh", file=sys.stderr)
        print(
            "uv manages Python for the tool. If uv-managed Python downloads are blocked, install Python 3.10+.",
            file=sys.stderr,
        )
        return 1

    if not _memgraph_reachable(host, port):
        print(f"FAIL memgraph: bolt://{host}:{port} is not reachable", file=sys.stderr)
        print("Start Memgraph, then rerun bootstrap:", file=sys.stderr)
        print(f"  {_memgraph_command(port)}", file=sys.stderr)
        print("Or point bootstrap at an existing instance:", file=sys.stderr)
        print(
            f"  agent-context-graph bootstrap --runtime {args.runtime} "
            f"--connector {' --connector '.join(connectors)} --memgraph-host <host> --memgraph-port <port>",
            file=sys.stderr,
        )
        return 1

    print(f"OK uv: {uv}")
    print(f"OK memgraph: bolt://{host}:{port} reachable")

    install_cmd = [uv, "tool", "install", "agent-context-graph"]
    for connector in connectors:
        requirement = _connector_requirement(connector)
        if requirement is None:
            print(f"FAIL connector:{connector}: unsupported connector", file=sys.stderr)
            return 1
        install_cmd.extend(["--with", requirement])
    if not args.no_reinstall:
        install_cmd.append("--reinstall")

    try:
        subprocess.run(install_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"FAIL install: uv tool install exited with status {exc.returncode}", file=sys.stderr)
        return exc.returncode

    executable = shutil.which("agent-context-graph")
    if not executable:
        print("FAIL agent-context-graph: installed but not on PATH", file=sys.stderr)
        print("uv usually installs tools into ~/.local/bin.", file=sys.stderr)
        print("Add that directory to PATH, restart your shell, then rerun doctor.", file=sys.stderr)
        return 1

    print(f"OK agent-context-graph executable: {executable}")
    doctor_args = ["--runtime", args.runtime]
    for connector in connectors:
        doctor_args.extend(["--connector", connector])
    return _doctor(doctor_args)


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
        _check_memgraph(),
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


def _check_memgraph() -> dict[str, object]:
    url = os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687")
    try:
        from memgraph_toolbox.api.memgraph import Memgraph

        db = Memgraph()
        db.query("RETURN 1")
        return {"name": "memgraph", "ok": True, "detail": f"{url} reachable"}
    except Exception as exc:
        return {"name": "memgraph", "ok": False, "detail": f"{url} — {type(exc).__name__}: {exc}"}
    finally:
        driver = getattr(getattr(locals().get("db", None), "driver", None), "driver", None)
        if driver is not None:
            driver.close()


def _check_cli() -> dict[str, object]:
    path = shutil.which("agent-context-graph") or sys.argv[0]
    return {"name": "agent-context-graph executable", "ok": bool(path), "detail": path or "not found"}


def _check_package(package_name: str) -> dict[str, object]:
    version = _package_version(package_name)
    ok = version != "unknown"
    return {"name": package_name, "ok": ok, "detail": version if ok else "not installed"}


def _check_connector(connector_name: str) -> dict[str, object]:
    normalized = connector_name.strip().replace("-", "_")
    if normalized == "skills_graph":
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

    if normalized == "actions_graph":
        try:
            from actions_graph import ActionsGraph

            graph = ActionsGraph()
            version = _package_version("actions-graph")
            return {"name": "connector:actions-graph", "ok": True, "detail": f"installed={version}; memgraph=reachable"}
        except Exception as exc:
            return {"name": "connector:actions-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            driver = getattr(getattr(locals().get("graph", None), "_db", None), "driver", None)
            if driver is not None:
                driver.close()

    if normalized == "sessions_graph":
        try:
            from sessions_graph import SessionsGraph

            graph = SessionsGraph()
            version = _package_version("sessions-graph")
            return {
                "name": "connector:sessions-graph",
                "ok": True,
                "detail": f"installed={version}; memgraph=reachable",
            }
        except Exception as exc:
            return {"name": "connector:sessions-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            driver = getattr(getattr(locals().get("graph", None), "_db", None), "driver", None)
            if driver is not None:
                driver.close()

    else:
        return {"name": f"connector:{connector_name}", "ok": False, "detail": "unsupported connector"}


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


def _memgraph_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _memgraph_command(port: int) -> str:
    published = f"{port}:7687"
    return f"docker run --rm -p {published} memgraph/memgraph"


def _connector_requirement(connector: str) -> str | None:
    normalized = connector.strip().replace("_", "-")
    if normalized == "skills-graph":
        return "skills-graph[agent-context-graph]"
    if normalized == "actions-graph":
        return "actions-graph[agent-context-graph]"
    if normalized == "sessions-graph":
        return "sessions-graph[agent-context-graph]"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
