"""Top-level CLI for memgraph-context-graph."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata
from typing import TYPE_CHECKING, Any, TypedDict, cast

from memgraph_context_graph.tools import normalize_connector_name

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: memgraph-context-graph <command> [options]

Commands:
  setup [runtime ...]  Install Context Graph for your agents (interactive on a terminal);
                   --memory-backend context-graph|native switches whose memory they use.
  uninstall [runtime ...]  Remove the agents' wiring; your graph and config stay.
  config           Get or set persistent configuration values.
  doctor           Check runtime hook dependencies and Memgraph connectivity.
  join <url> <login>  Use an Enterprise Memgraph as <login> (edition = enterprise); --password or prompt.
  hook <command>  Configure or run command hooks.
  mcp              Serve the registered tools (recall, ...) over stdio MCP.
  recall <question>  Search your own past sessions; --json for the data.
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

    if command == "join":
        return _join(args[1:])

    if command == "hook":
        from memgraph_context_graph.hooks.cli import main as hook_main

        return hook_main(args[1:])

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP)
    return 2


def _config(argv: list[str]) -> int:
    """Get or set persistent config values in ~/.config/context-graph/config.toml."""
    from memgraph_context_graph.adapters._identity import (
        FALSY_VALUES,
        MEMORY_BACKENDS,
        TRUTHY_VALUES,
        config_file,
        load_config,
        parse_bool_flag,
        write_config,
    )
    from memgraph_context_graph.spaces import EDITION_COMMUNITY, EDITIONS

    config_path = config_file()

    _CONFIG_KEYS = {
        "edition": "edition",
        "identity.user_id": "user_id",
        "memgraph.url": "memgraph_url",
        "memgraph.user": "memgraph_user",
        "memgraph.password": "memgraph_password",
        "memgraph.database": "memgraph_database",
        "llm.openai_api_key": "openai_api_key",
        "llm.anthropic_api_key": "anthropic_api_key",
        "reconcile.auto_reconcile": "auto_reconcile",
        "recall.embedding_model": "embedding_model",
        "ontology.path": "ontology_path",
        "ontology.derive": "ontology_derive",
        "github.token": "github_token",
        "memory.backend": "memory_backend",
    }
    _SECRET_KEYS = {"memgraph.password", "llm.openai_api_key", "llm.anthropic_api_key", "github.token"}
    _BOOL_KEYS = {"reconcile.auto_reconcile"}

    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: memgraph-context-graph config set <key> <value>")
        print("       memgraph-context-graph config get <key>")
        print("       memgraph-context-graph config show")
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
        for problem in config.problems:
            print(f"# WARNING: malformed config, {problem}; values it meant to set are defaults below")
        print(f"edition = {config.edition or EDITION_COMMUNITY!r}")
        print(f"identity.user_id = {config.user_id!r}")
        print(f"memgraph.url = {config.memgraph_url!r}")
        print(f"memgraph.user = {config.memgraph_user!r}")
        print(f"memgraph.password = {'***' if config.memgraph_password else repr('')}")
        print(f"memgraph.database = {config.memgraph_database!r}")
        print(f"llm.openai_api_key = {'***' if config.openai_api_key else repr('')}")
        print(f"llm.anthropic_api_key = {'***' if config.anthropic_api_key else repr('')}")
        if config.auto_reconcile is None:
            print("reconcile.auto_reconcile = unset (defaults to false)")
        else:
            print(f"reconcile.auto_reconcile = {'true' if config.auto_reconcile else 'false'}")
        if config.embedding_model is None:
            print("recall.embedding_model = unset (defaults to sessions-graph's model)")
        else:
            print(f"recall.embedding_model = {config.embedding_model!r}")
        print(f"ontology.path = {config.ontology_path!r}" if config.ontology_path else "ontology.path = unset")
        if config.ontology_derive is None:
            print("ontology.derive = unset (defaults to extend)")
        else:
            print(f"ontology.derive = {config.ontology_derive!r}")
        print(f"github.token = {'***' if config.github_token else 'unset'}")
        print(
            f"memory.backend = {config.memory_backend!r}"
            if config.memory_backend
            else "memory.backend = unset (native)"
        )
        return 0

    if action == "set":
        if len(argv) < 2:
            print("usage: memgraph-context-graph config set <key> <value>", file=sys.stderr)
            return 2
        key = argv[1]
        if key not in _CONFIG_KEYS:
            print(f"Unknown config key: {key}", file=sys.stderr)
            print(f"Supported keys: {', '.join(_CONFIG_KEYS)}", file=sys.stderr)
            return 2

        if len(argv) < 3:
            if key in _SECRET_KEYS:
                import getpass

                value = getpass.getpass(f"Enter {key}: ")
            else:
                print(f"usage: memgraph-context-graph config set {key} <value>", file=sys.stderr)
                return 2
        else:
            value = argv[2]

        write_value: str | bool = value
        if key in _BOOL_KEYS:
            normalized = value.strip().lower()
            if normalized not in TRUTHY_VALUES | FALSY_VALUES:
                print(f"Invalid value for {key}: {value!r} (expected true/false)", file=sys.stderr)
                return 2
            write_value = parse_bool_flag(normalized)
        if key == "edition" and value not in EDITIONS:
            print(f"Invalid value for {key}: {value!r} (expected {'/'.join(sorted(EDITIONS))})", file=sys.stderr)
            return 2
        if key == "memory.backend" and value not in MEMORY_BACKENDS:
            print(f"Invalid value for {key}: {value!r} (expected {'/'.join(sorted(MEMORY_BACKENDS))})", file=sys.stderr)
            return 2
        if key == "ontology.derive" and value not in {"extend", "off"}:
            print(f"Invalid value for {key}: {value!r} (expected extend/off)", file=sys.stderr)
            return 2

        # The key/value pairing is only known at runtime (bool iff key is in
        # _BOOL_KEYS, str otherwise); write_config's own parameters are precisely
        # typed per key, so the checker can't verify this dynamic dispatch itself.
        write_config(**cast("dict[str, Any]", {_CONFIG_KEYS[key]: write_value}))
        if key in _SECRET_KEYS:
            display_value = "***"
        elif key in _BOOL_KEYS:
            display_value = str(write_value).lower()
        else:
            display_value = repr(write_value)
        print(f"Wrote {key} = {display_value} to {config_path}")
        return 0

    if action == "get":
        if len(argv) < 2:
            print("usage: memgraph-context-graph config get <key>", file=sys.stderr)
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
        if key in _SECRET_KEYS:
            print("***")
        else:
            print(value)
        return 0

    print(f"Unknown config action: {action}", file=sys.stderr)
    return 2


def _mcp(argv: list[str]) -> int:
    argparse.ArgumentParser(
        prog="memgraph-context-graph mcp", description="Serve the registered tools over stdio MCP."
    ).parse_args(argv)
    try:
        from memgraph_context_graph.mcp_server import serve
    except ImportError as exc:
        print(f"memgraph-context-graph mcp needs the mcp extra: uv tool install 'memgraph-context-graph[mcp]' ({exc})",
              file=sys.stderr)  # fmt: skip
        return 1
    return serve()


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
        help="Graph connector to enable. Defaults to skills-graph.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    checks = doctor_checks([args.runtime], args.connector or ["skills-graph"])
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
    llm_key_set = bool(config.openai_api_key or config.anthropic_api_key)
    parts.append(
        "llm_key=SET" if llm_key_set else "llm_key=NOT SET (needed only for sessions-graph auto-reconciliation)"
    )
    auto_reconcile_on = bool(config.auto_reconcile)
    parts.append(f"auto_reconcile={'ON' if auto_reconcile_on else 'OFF'}")
    broken_auto_reconcile = auto_reconcile_on and not llm_key_set
    if broken_auto_reconcile:
        parts.append("WARNING: auto_reconcile is ON but no LLM key is set — reconciliation will fail")
    ok = bool(user_id) and not broken_auto_reconcile
    return {"name": "config", "ok": ok, "detail": f"{path} — {'; '.join(parts)}"}


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
    if normalized == "skills-graph":
        graph = None
        try:
            from skills_graph import SkillGraph

            graph = SkillGraph()
            version = _package_version("skills-graph")
            return {"name": "connector:skills-graph", "ok": True, "detail": f"installed={version}; memgraph=reachable"}
        except Exception as exc:
            return {"name": "connector:skills-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            _close(graph and graph._db)

    if normalized == "actions-graph":
        graph = None
        try:
            from actions_graph import ActionsGraph

            graph = ActionsGraph()
            version = _package_version("actions-graph")
            return {"name": "connector:actions-graph", "ok": True, "detail": f"installed={version}; memgraph=reachable"}
        except Exception as exc:
            return {"name": "connector:actions-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            _close(graph and graph._db)

    if normalized == "sessions-graph":
        graph = None
        try:
            from sessions_graph import SessionsGraph

            from memgraph_context_graph.adapters._identity import resolve_memgraph_env, resolve_user_id
            from memgraph_context_graph.hooks.runner import memgraph_kwargs

            graph = SessionsGraph(**memgraph_kwargs(resolve_memgraph_env()))
            version = _package_version("sessions-graph")
            user_id = resolve_user_id({})
            if not user_id:
                return {
                    "name": "connector:sessions-graph",
                    "ok": False,
                    "detail": "user_id not resolved — run: memgraph-context-graph config set identity.user_id <your-name>",
                }
            return {
                "name": "connector:sessions-graph",
                "ok": True,
                "detail": f"installed={version}; memgraph=reachable; user_id={user_id!r}",
            }
        except Exception as exc:
            return {"name": "connector:sessions-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            _close(graph and graph._db)

    if normalized == "resources-graph":
        graph = None
        try:
            from memgraph_context_graph.adapters._identity import load_config, resolve_memgraph_env
            from memgraph_context_graph.hooks.runner import memgraph_kwargs
            from resources_graph import ResourcesGraph

            graph = ResourcesGraph(**memgraph_kwargs(resolve_memgraph_env()))
            version = _package_version("resources-graph")
            # Touches are recorded without a token; the Sweep falls back to the user's gh login.
            token = "set" if load_config().github_token else "unset (the Sweep uses the gh login)"
            return {
                "name": "connector:resources-graph",
                "ok": True,
                "detail": f"installed={version}; memgraph=reachable; github.token {token}",
            }
        except Exception as exc:
            return {"name": "connector:resources-graph", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        finally:
            _close(graph and graph._db)

    else:
        return {"name": f"connector:{connector_name}", "ok": False, "detail": "unsupported connector"}


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
    """Whether ``memgraph-context-graph mcp`` can start and serves recall."""
    from memgraph_context_graph.tools import load_tools

    try:
        import mcp  # noqa: F401 -- only whether the extra is installed
    except ImportError:
        return {
            "name": "mcp",
            "ok": False,
            "detail": "the mcp extra is missing, so the recall tool can't be served — "
            "run: uv tool install 'memgraph-context-graph[mcp]' --with 'sessions-graph[memgraph-context-graph]'",
        }
    names = sorted(load_tools())
    if "recall" not in names:
        return {"name": "mcp", "ok": False, "detail": f"no recall tool registered (tools: {names or 'none'})"}
    return {"name": "mcp", "ok": True, "detail": f"serves {', '.join(names)}"}


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
    return {"name": "memory", "ok": True, "detail": f"Context Graph; {runtime}'s built-in memory is off"}


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
        return {"name": f"runtime:{runtime}", "ok": True, "detail": "strict hook smoke passed"}
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
