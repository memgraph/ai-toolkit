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
  config           Get or set persistent configuration values.
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

    if command == "config":
        return _config(args[1:])

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


def _config(argv: list[str]) -> int:
    """Get or set persistent config values in ~/.config/context-graph/config.toml."""
    from agent_context_graph.adapters._identity import (
        config_file_path,
        load_config,
        write_config,
    )

    config_path = config_file_path()

    _CONFIG_KEYS = {
        "identity.user_id": "user_id",
        "memgraph.url": "memgraph_url",
        "memgraph.user": "memgraph_user",
        "memgraph.password": "memgraph_password",
        "memgraph.database": "memgraph_database",
    }

    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: agent-context-graph config set <key> <value>")
        print("       agent-context-graph config get <key>")
        print("       agent-context-graph config show")
        print("")
        print("Supported keys:")
        for key in _CONFIG_KEYS:
            print(f"  {key}")
        print(f"\nConfig file: {config_path}")
        return 0

    action = argv[0]

    if action == "show":
        config = load_config()
        print(f"# {config_path}")
        print(f"identity.user_id = {config.user_id!r}")
        print(f"memgraph.url = {config.memgraph_url!r}")
        print(f"memgraph.user = {config.memgraph_user!r}")
        print(f"memgraph.password = {'***' if config.memgraph_password else repr('')}")
        print(f"memgraph.database = {config.memgraph_database!r}")
        return 0

    if action == "set":
        if len(argv) < 2:
            print("usage: agent-context-graph config set <key> <value>", file=sys.stderr)
            return 2
        key = argv[1]
        if key not in _CONFIG_KEYS:
            print(f"Unknown config key: {key}", file=sys.stderr)
            print(f"Supported keys: {', '.join(_CONFIG_KEYS)}", file=sys.stderr)
            return 2

        # Password: read from stdin if value not provided.
        if len(argv) < 3:
            if key == "memgraph.password":
                import getpass

                value = getpass.getpass("Enter memgraph password: ")
            else:
                print(f"usage: agent-context-graph config set {key} <value>", file=sys.stderr)
                return 2
        else:
            value = argv[2]

        write_config(**{_CONFIG_KEYS[key]: value})
        display_value = "***" if key == "memgraph.password" else repr(value)
        print(f"Wrote {key} = {display_value} to {config_path}")
        return 0

    if action == "get":
        if len(argv) < 2:
            print("usage: agent-context-graph config get <key>", file=sys.stderr)
            return 2
        key = argv[1]
        if key not in _CONFIG_KEYS:
            print(f"Unknown config key: {key}", file=sys.stderr)
            return 2

        config = load_config()
        attr = _CONFIG_KEYS[key]
        value = getattr(config, attr)
        if value is None:
            print(f"{key} is not set in {config_path}", file=sys.stderr)
            return 1
        # Don't print password to stdout unless explicitly requested.
        if key == "memgraph.password":
            print("***")
        else:
            print(value)
        return 0

    print(f"Unknown config action: {action}", file=sys.stderr)
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

    # Write hook configuration file (auto-write + inform).
    from agent_context_graph.adapters._identity import write_full_config

    memgraph_url = f"bolt://{host}:{port}"
    user_id = os.environ.get("AGENT_CONTEXT_GRAPH_USER_ID", "")
    memgraph_user = os.environ.get("MEMGRAPH_USER", "")
    memgraph_password = os.environ.get("MEMGRAPH_PASSWORD", "")
    memgraph_database = os.environ.get("MEMGRAPH_DATABASE", "memgraph")

    config_path = write_full_config(
        user_id=user_id,
        memgraph_url=memgraph_url,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        memgraph_database=memgraph_database,
    )
    print(f"OK config: wrote {config_path}")
    if user_id:
        print(f"   identity.user_id = {user_id!r}")
    else:
        print("   identity.user_id is empty — set it with: agent-context-graph config set identity.user_id <your-name>")
    print(f"   memgraph.url = {memgraph_url!r}")

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
        _check_config(),
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


def _check_config() -> dict[str, object]:
    from agent_context_graph.adapters._identity import config_file_path, load_config

    path = config_file_path()
    if not path.is_file():
        return {
            "name": "config",
            "ok": False,
            "detail": f"{path} not found — run: agent-context-graph bootstrap or agent-context-graph config set identity.user_id <name>",
        }
    config = load_config()
    parts = []
    if config.user_id:
        parts.append(f"user_id={config.user_id!r}")
    else:
        parts.append("user_id=NOT SET")
    parts.append(f"memgraph.url={config.memgraph_url!r}")
    return {"name": "config", "ok": bool(config.user_id), "detail": f"{path} — {'; '.join(parts)}"}


def _check_memgraph() -> dict[str, object]:
    from agent_context_graph.adapters._identity import resolve_memgraph_env

    env = resolve_memgraph_env()
    url = env["MEMGRAPH_URL"]
    try:
        from memgraph_toolbox.api.memgraph import Memgraph

        db = Memgraph(
            url=env["MEMGRAPH_URL"],
            username=env["MEMGRAPH_USER"],
            password=env["MEMGRAPH_PASSWORD"],
            database=env["MEMGRAPH_DATABASE"],
        )
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

            from agent_context_graph.adapters._identity import resolve_memgraph_env, resolve_user_id

            env = resolve_memgraph_env()
            graph = SessionsGraph(
                url=env["MEMGRAPH_URL"],
                username=env["MEMGRAPH_USER"],
                password=env["MEMGRAPH_PASSWORD"],
                database=env["MEMGRAPH_DATABASE"],
            )
            version = _package_version("sessions-graph")
            user_id = resolve_user_id({})
            if not user_id:
                return {
                    "name": "connector:sessions-graph",
                    "ok": False,
                    "detail": "user_id not resolved — run: agent-context-graph config set identity.user_id <your-name>",
                }
            return {
                "name": "connector:sessions-graph",
                "ok": True,
                "detail": f"installed={version}; memgraph=reachable; user_id={user_id!r}",
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
