"""Top-level CLI for memgraph-context-graph."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata
from typing import TYPE_CHECKING, Any, TypedDict

from memgraph_context_graph.connectors import (
    configured_connectors,
    connector_spec,
    known_connectors,
)
from memgraph_context_graph.tools import normalize_connector_name

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from memgraph_context_graph.config_schema import ConfigKey

_HELP = """usage: memgraph-context-graph <command> [options]

Commands:
  setup [runtime ...]  Install Context Graph for your agents (interactive on a terminal);
                   --memory-backend context-graph|native switches whose memory they use.
  uninstall [runtime ...]  Remove the agents' wiring; your graph and config stay.
  update [runtime ...]  Upgrade the package and refresh the agents' wiring and the graph's schema.
  config           Get or set persistent configuration values.
  doctor           Check runtime hook dependencies and Memgraph connectivity.
  join <url> <login>  Use an Enterprise Memgraph as <login> (edition = enterprise); --password or prompt.
  hook <command>  Configure or run command hooks.
  mcp              Serve the registered tools (recall, ...) over stdio MCP.
  recall <question>  Search your own past sessions; --json for the data.
  forget session <id> ... | project <dir>  Permanently delete sessions and what was derived from them.
  ui               Open the read-only terminal viewer (needs context-graph-viewer).
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch top-level memgraph-context-graph commands."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        print(_HELP)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_HELP)
        return 0
    if args[0] in {"--version", "version"}:
        print(_package_version("memgraph-context-graph"))
        return 0

    command = args[0]
    if command == "setup":
        from memgraph_context_graph.setup.cli import setup_main

        return setup_main(args[1:])

    if command == "uninstall":
        from memgraph_context_graph.setup.cli import uninstall_main

        return uninstall_main(args[1:])

    if command == "update":
        from memgraph_context_graph.setup.cli import update_main

        return update_main(args[1:])

    if command == "config":
        return _config(args[1:])

    if command == "doctor":
        return _doctor(args[1:])

    if command == "mcp":
        return _mcp(args[1:])

    if command == "recall":
        return _recall(args[1:])

    if command == "ui":
        return _ui(args[1:])

    if command == "forget":
        return _forget(args[1:])

    if command == "join":
        return _join(args[1:])

    if command == "hook":
        from memgraph_context_graph.hooks.cli import main as hook_main

        return hook_main(args[1:])

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP)
    return 2


def _forget(argv: list[str]) -> int:
    """``forget session <id>... | project <dir>``: show what matches, confirm, then delete it for good."""
    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph forget",
        description="Permanently delete sessions from your graph: what was captured, and what was derived from it. "
        "Shared entities, skills, resources and your memory files stay. A forgotten session is never captured again.",
    )
    parser.add_argument(
        "scope", choices=["session", "project"], help="Forget sessions by id, or every one in a directory."
    )
    parser.add_argument("targets", nargs="+", help="Session ids, or one project directory.")
    parser.add_argument("--dry-run", action="store_true", help="Show the matching sessions and delete nothing.")
    parser.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation.")
    args = parser.parse_args(argv)
    if args.scope == "project" and len(args.targets) != 1:
        parser.error("forget project takes one directory")

    from memgraph_context_graph.adapters._identity import resolve_memgraph_env
    from memgraph_context_graph.forget import describe, forget_session, sessions_in
    from memgraph_context_graph.hooks.runner import memgraph_kwargs
    from memgraph_toolbox.api.memgraph import Memgraph

    memgraph = Memgraph(**memgraph_kwargs(resolve_memgraph_env()))
    try:
        session_ids = sessions_in(memgraph, args.targets[0]) if args.scope == "project" else list(args.targets)
        found = describe(memgraph, session_ids)
        for summary in found:
            print(f"  {summary.session_id}  {summary.started_at or '-'}  {summary.working_directory or '-'}  "
                  f"{summary.actions} actions")  # fmt: skip
        missing = sorted(set(session_ids) - {summary.session_id for summary in found})
        for session_id in missing:
            print(f"  {session_id}  (no session in the graph; it will still never be captured again)")
        if not session_ids:
            print("No sessions match.")
            return 0
        if args.dry_run:
            print(f"Dry run: {len(session_ids)} session(s) would be forgotten.")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                print("Refusing to forget without --yes when not on a terminal.", file=sys.stderr)
                return 2
            answer = input(f"Permanently forget {len(session_ids)} session(s)? This cannot be undone. [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("Nothing forgotten.")
                return 1
        for session_id in session_ids:
            result = forget_session(memgraph, session_id)
            removed = ", ".join(f"{count} {kind}" for kind, count in sorted(result.removed.items()) if count)
            print(f"Forgot {session_id}: {removed or 'nothing left to remove'}")
            if result.unavailable:
                print(f"  Not installed, so not checked: {', '.join(result.unavailable)}")
        return 0
    finally:
        memgraph.close()


def _config(argv: list[str]) -> int:
    """Get or set persistent config values in ~/.config/context-graph/config.toml."""
    from memgraph_context_graph.adapters._identity import (
        config_file,
        load_config,
        set_config_value,
        stored_value,
    )
    from memgraph_context_graph.config_schema import CONFIG_KEYS, config_key

    config_path = config_file()
    usage = (
        "usage: memgraph-context-graph config set <key> <value>\n"
        "       memgraph-context-graph config get <key>\n"
        "       memgraph-context-graph config show"
    )

    if not argv or argv[0] in {"-h", "--help"}:
        print(usage)
        print("\nSupported keys:")
        for key in CONFIG_KEYS:
            print(f"  {key.name:32} {key.doc}")
        print(f"\nConfig file: {config_path}")
        return 0

    action = argv[0]

    if action == "show":
        config = load_config()
        print(f"# {config_path}")
        for problem in config.problems:
            print(f"# WARNING: malformed config, {problem}; values it meant to set are defaults below")
        for key in CONFIG_KEYS:
            if not key.applies_to(config.edition):
                continue
            value = stored_value(config, key)
            if value is None:
                print(f"{key.name} = unset ({key.unset})" if key.unset else f"{key.name} = unset")
            else:
                print(f"{key.name} = {_display(key, value)}")
        for name in config.redact_patterns:
            print(f"redact.{name} = {config.redact_patterns[name]!r}")
        return 0

    if action not in {"set", "get"}:
        print(f"Unknown config action: {action}", file=sys.stderr)
        return 2
    if len(argv) < 2:
        print(f"usage: memgraph-context-graph config {action} <key>{' <value>' if action == 'set' else ''}",
              file=sys.stderr)  # fmt: skip
        return 2
    key = config_key(argv[1])
    if key is None:
        print(f"Unknown config key: {argv[1]}", file=sys.stderr)
        print(f"Supported keys: {', '.join(k.name for k in CONFIG_KEYS)}", file=sys.stderr)
        return 2

    if action == "get":
        value = stored_value(load_config(), key)
        if value is None:
            print(f"{key.name} is not set in {config_path}", file=sys.stderr)
            return 1
        print("***" if key.secret else value)
        return 0

    if len(argv) >= 3:
        raw = argv[2]
    elif key.secret:
        import getpass

        raw = getpass.getpass(f"Enter {key.name}: ")
    else:
        print(f"usage: memgraph-context-graph config set {key.name} <value>", file=sys.stderr)
        return 2
    try:
        value = set_config_value(key.name, raw)
    except ValueError as exc:
        print(f"Invalid value for {key.name}: {raw!r} ({exc})", file=sys.stderr)
        return 2
    print(f"Wrote {key.name} = {_display(key, value)} to {config_path}")
    return 0


def _display(key: ConfigKey, value: str) -> str:
    """A stored value as ``config show/set`` print it: secrets masked, booleans bare, the rest quoted."""
    if key.secret:
        return "***" if value else repr("")
    if key.kind == "bool":
        return value
    return repr(value)


def _mcp(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph mcp", description="Serve the registered tools over stdio MCP."
    )
    parser.add_argument(
        "--session-context",
        action="store_true",
        help="send the session-start text (memory index, where you left off) as the server's instructions, "
        "for an agent whose hooks can't show the model anything at session start (Grok Build)",
    )
    args = parser.parse_args(argv)
    try:
        from memgraph_context_graph.mcp_server import serve
    except ImportError as exc:
        print(f"memgraph-context-graph mcp needs the mcp extra: uv tool install 'memgraph-context-graph[mcp]' ({exc})",
              file=sys.stderr)  # fmt: skip
        return 1
    return serve(session_context=args.session_context)


def _ui(argv: list[str]) -> int:
    try:
        from context_graph_viewer.cli import main as viewer_main
    except ImportError:
        print(
            "memgraph-context-graph ui needs the context-graph-viewer package, which isn't installed.\n"
            "Run setup again and say yes to the viewer:\n"
            "  memgraph-context-graph setup",
            file=sys.stderr,
        )
        return 1
    return viewer_main(argv)


def _recall(argv: list[str]) -> int:
    from memgraph_context_graph.adapters._identity import load_config
    from memgraph_context_graph.tools import ToolError, load_tools

    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph recall", description="Search your own past sessions, as the recall tool does."
    )
    parser.add_argument("question", nargs="+", help="What to look for, as you would ask it.")
    parser.add_argument("--json", action="store_true", help="Print the turns and facts as JSON.")
    args = parser.parse_args(argv)

    tool = load_tools().get("recall")
    if tool is None:
        print("No recall tool is installed: it comes with sessions-graph[memgraph-context-graph].", file=sys.stderr)
        return 1
    try:
        result = tool.call({"question": " ".join(args.question)}, load_config())
    except ToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result.structured, indent=2) if args.json else result.text)
    return 0


def _join(argv: list[str]) -> int:
    """Point the config file at an Enterprise Memgraph as one login, then prove the login works."""
    import getpass

    from memgraph_context_graph.adapters._identity import config_file, reload_config, write_config
    from memgraph_context_graph.spaces import EDITION_ENTERPRISE, SpaceError, spaces_for

    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph join", description="Use an Enterprise Memgraph's personal and shared spaces."
    )
    parser.add_argument("url", help="The Memgraph Bolt URL, e.g. bolt://memgraph.corp:7687.")
    parser.add_argument("login", help="Your Memgraph login; it is also who you are in Context Graph.")
    parser.add_argument("--password", default=None, help="Prompted for when omitted.")
    args = parser.parse_args(argv)
    password = args.password if args.password is not None else getpass.getpass(f"Password for {args.login}: ")
    _ensure_enterprise_package()

    write_config(
        edition=EDITION_ENTERPRISE, memgraph_url=args.url, memgraph_user=args.login, memgraph_password=password
    )
    try:
        spaces = spaces_for(reload_config())
        from memgraph_toolbox.api.memgraph import Memgraph

        client = Memgraph(**spaces.personal().target.client_kwargs())
        try:
            (row,) = client.query("SHOW CURRENT DATABASE")
        finally:
            client.close()
        shared = [space.name for space in spaces.readable() if space.shared]
    except (SpaceError, ValueError) as exc:
        print(f"Wrote {config_file()}, but the login doesn't work yet: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {config_file()}: edition = enterprise, login {args.login!r}.")
    print(f"Personal space: {row['Current']}. Shared spaces: {', '.join(shared) or 'none yet'}.")
    print("Next: memgraph-context-graph setup <runtime> (e.g. claude-code) if this machine isn't set up yet.")
    return 0


def _ensure_enterprise_package() -> None:
    """Install ``context-graph-spaces`` when this is the uv tool and it is missing; else the error says how."""
    from memgraph_context_graph.setup import tool
    from memgraph_context_graph.spaces import EDITION_ENTERPRISE, _edition_factory

    if _edition_factory(EDITION_ENTERPRISE) is not None or not tool.uv_tool_environment():
        return
    print("Enterprise spaces need one more package, with the local extraction models; installing it.", flush=True)
    print("This can take a minute.", flush=True)
    try:
        tool.install(viewer=False, spaces=True, log=lambda line: None)
    except tool.ToolError as exc:
        print(str(exc), file=sys.stderr)


def _doctor(argv: list[str]) -> int:
    from memgraph_context_graph.hooks.runtime_plugin import load_runtime_plugins

    parser = argparse.ArgumentParser(description="Check Memgraph Context Graph runtime hook setup.")
    parser.add_argument(
        "--runtime",
        choices=sorted(load_runtime_plugins()),
        default="claude-code",
        help="Runtime hook adapter to smoke test.",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to check, repeatable. Defaults to the config file's [capture] connectors.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    checks = doctor_checks([args.runtime], args.connector or configured_connectors())
    ok = all(check["ok"] for check in checks)
    payload: _DoctorPayload = {"ok": ok, "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_doctor(payload)
    return 0 if ok else 1


def doctor_checks(runtimes: Sequence[str], connectors: Sequence[str]) -> list[_CheckResult]:
    """Everything ``doctor`` checks: shared checks once, then each runtime's hook probe and memory backend."""
    checks = [
        _check_cli(),
        _check_package("memgraph-context-graph"),
        _check_config(),
    ]
    # Every later check connects through the edition's space resolver; when the config can't
    # produce one, the config check already says why, and the rest would only repeat it.
    if _edition_resolves():
        checks.append(_check_memgraph())
        for connector in connectors:
            checks.append(_check_connector(connector))
        if any(normalize_connector_name(connector) == "sessions-graph" for connector in connectors):
            checks.append(_check_embeddings())
            checks.append(_check_mcp())
        for runtime in runtimes:
            checks.append(_check_runtime(runtime, list(connectors)))
            if (memory_check := _check_memory_backend(runtime)) is not None:
                checks.append(memory_check)
    for runtime in runtimes:
        if (plugin_check := _check_plugin(runtime)) is not None:
            checks.append(plugin_check)
    checks.append(_check_capture_failures())
    return checks


class _RequiredCheckFields(TypedDict):
    name: str
    ok: bool
    detail: str


class _CheckResult(_RequiredCheckFields, total=False):
    #: Reporting only: shown, never part of pass/fail (its ``ok`` is always True).
    info: bool
    #: Structured detail for ``--json``.
    data: Any


class _DoctorPayload(TypedDict):
    ok: bool
    checks: list[_CheckResult]


def _edition_resolves() -> bool:
    from memgraph_context_graph.adapters._identity import load_config
    from memgraph_context_graph.spaces import SpaceError, spaces_for

    try:
        spaces_for(load_config())
    except SpaceError:
        return False
    return True


def _check_config() -> _CheckResult:
    from memgraph_context_graph.adapters._identity import config_file, load_config

    path = config_file()
    if not path.is_file():
        return {
            "name": "config",
            "ok": False,
            "detail": f"{path} not found — run: memgraph-context-graph setup",
        }
    from memgraph_context_graph.spaces import SpaceError, spaces_for

    config = load_config()
    if config.problems:
        # Whatever those lines meant to set fell back to defaults, which can point capture at the wrong Memgraph.
        return {"name": "config", "ok": False, "detail": f"{path} is malformed — {'; '.join(config.problems)}"}
    try:
        spaces = spaces_for(config)
    except SpaceError as exc:
        return {"name": "config", "ok": False, "detail": f"{path} — {exc}"}
    user_id = spaces.user_id()
    parts = [f"edition={spaces.edition}"]
    if user_id:
        parts.append(f"user_id={user_id!r}")
    else:
        parts.append("user_id=NOT SET")
    parts.append(f"memgraph.url={spaces.personal().target.url!r}")
    # Reconciliation extracts locally with GLiNER2; the key only lets it learn the schema.
    llm_key_set = bool(config.openai_api_key or config.anthropic_api_key)
    parts.append("llm_key=SET" if llm_key_set else "llm_key=NOT SET (needed only to learn the schema)")
    parts.append(f"auto_reconcile={'ON' if config.auto_reconcile else 'OFF'}")
    return {"name": "config", "ok": bool(user_id), "detail": f"{path} — {'; '.join(parts)}"}


def _check_memgraph() -> _CheckResult:
    from memgraph_context_graph.adapters._identity import resolve_memgraph_env

    env = resolve_memgraph_env()
    url = env["MEMGRAPH_URL"]
    db = None
    try:
        from memgraph_context_graph.hooks.runner import memgraph_kwargs
        from memgraph_toolbox.api.memgraph import Memgraph

        db = Memgraph(**memgraph_kwargs(env))
        db.query("RETURN 1")
        return {"name": "memgraph", "ok": True, "detail": f"{url} reachable"}
    except Exception as exc:
        return {"name": "memgraph", "ok": False, "detail": f"{url} — {type(exc).__name__}: {exc}"}
    finally:
        _close(db)


def _close(db: Any) -> None:
    if db is not None:
        db.close()


def _check_cli() -> _CheckResult:
    path = shutil.which("memgraph-context-graph") or sys.argv[0]
    return {"name": "memgraph-context-graph executable", "ok": bool(path), "detail": path or "not found"}


def _check_package(package_name: str) -> _CheckResult:
    version = _package_version(package_name)
    ok = version != "unknown"
    return {"name": package_name, "ok": ok, "detail": version if ok else "not installed"}


def _check_connector(connector_name: str) -> _CheckResult:
    normalized = normalize_connector_name(connector_name)
    name = f"connector:{normalized}"
    if normalized not in known_connectors():
        return {"name": f"connector:{connector_name}", "ok": False, "detail": "unsupported connector"}

    from memgraph_context_graph.adapters._identity import load_config, resolve_memgraph_env
    from memgraph_context_graph.hooks.runner import memgraph_kwargs

    graph = None
    try:
        spec = connector_spec(normalized)
        graph = spec.graph(**memgraph_kwargs(resolve_memgraph_env()))
        detail = f"installed={_package_version(normalized)}; memgraph=reachable"
        ok, more = spec.check(load_config()) if spec.check is not None else (True, "")
        return {"name": name, "ok": ok, "detail": f"{detail}; {more}" if ok and more else more or detail}
    except Exception as exc:
        return {"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        _close(graph and graph._db)


def _check_embeddings() -> _CheckResult:
    """Whether Memgraph can embed for recall: MAGE's ``embeddings`` module, with the configured model loading.

    The first run downloads the model inside Memgraph, so this can take a while once.
    """
    from memgraph_context_graph.adapters._identity import resolve_embedding_model, resolve_memgraph_env

    try:
        from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL, check_available

        from memgraph_toolbox.api.memgraph import Memgraph
    except ImportError as exc:
        return {"name": "embeddings", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    from memgraph_context_graph.hooks.runner import memgraph_kwargs

    model = resolve_embedding_model() or DEFAULT_EMBEDDING_MODEL
    db = None
    try:
        db = Memgraph(**memgraph_kwargs(resolve_memgraph_env()))
        dimension = check_available(db, model)
        return {"name": "embeddings", "ok": True, "detail": f"{model} ({dimension} dimensions) inside Memgraph"}
    except Exception as exc:
        return {
            "name": "embeddings",
            "ok": False,
            "detail": f"{model} — {exc}. Recall needs Memgraph with MAGE (memgraph/memgraph-mage, 2 GiB or more); "
            "without it, recall's vector lanes are off",
        }
    finally:
        _close(db)


def _check_mcp() -> _CheckResult:
    """Whether ``memgraph-context-graph mcp`` can start and serves recall, naming the tools the config turns on."""
    from memgraph_context_graph.adapters._identity import load_config
    from memgraph_context_graph.tools import is_available, load_tools

    try:
        import mcp  # noqa: F401 -- only whether the extra is installed
    except ImportError:
        return {
            "name": "mcp",
            "ok": False,
            "detail": "the mcp extra is missing, so the recall tool can't be served — "
            "run: uv tool install 'memgraph-context-graph[mcp]' --with 'sessions-graph[memgraph-context-graph]'",
        }
    tools = load_tools()
    if "recall" not in tools:
        return {"name": "mcp", "ok": False, "detail": f"no recall tool registered (tools: {sorted(tools) or 'none'})"}
    config = load_config()
    served = sorted(name for name, tool in tools.items() if is_available(tool, config))
    return {"name": "mcp", "ok": True, "detail": f"serves {', '.join(served)}"}


def _check_memory_backend(runtime: str) -> _CheckResult | None:
    """Which memory the runtime's model uses; fails when both are on. None for runtimes that can't switch."""
    from pathlib import Path

    from memgraph_context_graph.adapters._identity import load_config
    from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin
    from memgraph_context_graph.memory_backend import MemoryBackendError, native_memory_of

    native = native_memory_of(get_runtime_plugin(runtime))
    if native is None:
        return None
    try:
        native_on = native.enabled(Path.home())
    except MemoryBackendError as exc:
        return {"name": "memory", "ok": False, "detail": str(exc)}
    if not load_config().graph_memory:
        return {"name": "memory", "ok": True, "detail": f"{runtime}'s built-in memory (memory.backend native)"}
    if native_on:
        return {
            "name": "memory",
            "ok": False,
            "detail": f"memory.backend is context-graph but {runtime}'s built-in memory is still on, so the model "
            f"has two memories — run: memgraph-context-graph setup {runtime} --memory-backend context-graph",
        }
    try:
        override = _project_override(native)
    except MemoryBackendError as exc:
        return {"name": "memory", "ok": False, "detail": str(exc)}
    if override is not None:
        return {
            "name": "memory",
            "ok": False,
            "detail": f"memory.backend is context-graph but {override} turns {runtime}'s built-in memory back on "
            "in this project, so the model has two memories there — remove autoMemoryEnabled from it",
        }
    return {"name": "memory", "ok": True, "detail": f"Context Graph; {runtime}'s built-in memory is off"}


def _project_override(native: Any) -> Path | None:
    """The project settings file that turns *native* back on where doctor runs, for a harness that has them."""
    from pathlib import Path

    overridden_in = getattr(native, "overridden_in", None)
    if overridden_in is None:
        return None
    return next((path for project in _projects_of(Path.cwd()) if (path := overridden_in(project))), None)


def _projects_of(cwd: Path) -> list[Path]:
    """Where an agent started in *cwd* may read project settings from: *cwd* and its repository's root."""
    import subprocess
    from pathlib import Path

    try:
        root = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        root = ""
    return list(dict.fromkeys([cwd, *([Path(root)] if root else [])]))


def _check_plugin(runtime: str) -> _CheckResult | None:
    """Whether the agent will run the hooks at all: its Context Graph plugin installed and on.

    The runtime probe runs the connectors in doctor's own process, so it passes whether or not the agent ever
    calls them. None for per-project agents, whose wiring lives in each project's hook file.
    """
    from memgraph_context_graph.setup import agents

    agent = agents.BY_RUNTIME.get(runtime)
    if agent is None or agent.wiring != "plugin":
        return None
    name = f"plugin:{runtime}"
    fix = f"run: memgraph-context-graph setup {runtime}"
    if agent.binary() is None:
        return {"name": name, "ok": False, "detail": f"{' or '.join(agent.binaries)} isn't on PATH"}
    if not agents.plugin_wired(agent):
        return {"name": name, "ok": False, "detail": f"not installed, so {agent.label} records nothing — {fix}"}
    if not agents.plugin_enabled(agent):
        return {
            "name": name,
            "ok": False,
            "detail": f"installed but disabled, so {agent.label} records nothing — "
            f"run: {agent.binaries[0]} plugin enable {agents.PLUGIN_ID}",
        }
    return {"name": name, "ok": True, "detail": f"{agents.PLUGIN_ID} installed and enabled"}


def _check_runtime(runtime: str, connectors: list[str]) -> _CheckResult:
    """Replay the runtime's probe payload through real connectors, against the configured Memgraph.

    The probe runs under a throwaway session id and its Session is deleted
    afterwards, so ``doctor`` never leaves a session in the user's graph. It
    runs strict (no error handler) and writes nothing to the capture-failure
    log: doctor reports its own result.
    """
    import uuid

    from memgraph_context_graph.adapters._identity import resolve_memgraph_env

    probe_session_id = f"doctor-probe-{uuid.uuid4().hex}"
    memgraph_env: dict[str, str] | None = None
    try:
        from memgraph_context_graph.hooks.runner import create_link
        from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin

        plugin = get_runtime_plugin(runtime)
        memgraph_env = resolve_memgraph_env()
        link = create_link(connectors, memgraph_env=memgraph_env)
        adapter = plugin.adapter_class(link, session_id=probe_session_id)
        adapter.handle_payload(dict(plugin.probe_payload))
        return {"name": f"runtime:{runtime}", "ok": True, "detail": "connectors accept its probe event"}
    except Exception as exc:
        return {"name": f"runtime:{runtime}", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        if memgraph_env is not None:
            _delete_probe_session(memgraph_env, probe_session_id)


def _delete_probe_session(memgraph_env: dict[str, str], session_id: str) -> None:
    """Remove the probe's Session and the nodes only it owns; best effort, the probe result stands either way."""
    try:
        from memgraph_context_graph.hooks.runner import memgraph_kwargs
        from memgraph_toolbox.api.memgraph import Memgraph

        db = Memgraph(**memgraph_kwargs(memgraph_env))
    except Exception:
        return
    try:
        # The id is doctor's own uuid, so this can only ever match the probe's nodes.
        db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            OPTIONAL MATCH (s)-[:HAS_AGENT]->(agent:Agent)
            OPTIONAL MATCH (agent)-[:HAS_ACTION]->(nested:Action)
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(action:Action)
            OPTIONAL MATCH (touch:Touch)-[:IN_SESSION]->(s)
            DETACH DELETE nested, agent, action, touch, s
            """,
            params={"session_id": session_id},
        )
    except Exception:
        pass
    finally:
        _close(db)


def _check_capture_failures() -> _CheckResult:
    """What the hooks failed to capture in the last day, from the local capture-failure log. Reporting only.

    It never fails doctor: the runtime probe is the gate on health now, and the
    log is history -- a failure an hour ago that has since been fixed is not a
    reason to fail today.
    """
    from memgraph_context_graph import capture_log

    path = capture_log.path()
    groups = capture_log.recent_summary(hours=24)
    total = sum(group.count for group in groups)
    data = [
        {
            "stage": group.stage,
            "connector": group.connector,
            "error_type": group.error_type,
            "count": group.count,
            "last_at": group.last_at.isoformat(),
            "last_error": group.last_error,
            "level": group.level,
        }
        for group in groups
    ]
    if not groups:
        detail = f"none in the last 24h ({path})"
    else:
        last = groups[0]
        where = f"{last.stage}/{last.connector}" if last.connector else last.stage
        detail = (
            f"{total} in the last 24h; last: {where} {last.error_type} at {last.last_at.isoformat(timespec='seconds')}"
            f" — {last.last_error} ({path})"
        )
    return {"name": "capture-failures", "ok": True, "detail": detail, "info": True, "data": data}


def _print_doctor(payload: _DoctorPayload) -> None:
    for check in payload["checks"]:
        status = "INFO" if check.get("info") else "OK" if check["ok"] else "FAIL"
        print(f"{status} {check['name']}: {check['detail']}")


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
