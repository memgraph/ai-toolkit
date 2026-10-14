"""Command-hook runner shared by every registered runtime: stdin payload, connectors, adapter, response."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import TYPE_CHECKING, Any, NoReturn, TypedDict

from memgraph_context_graph import capture_log
from memgraph_context_graph.events import ToolStartEvent
from memgraph_context_graph.link import AgentLink, ConnectorFailuresError
from memgraph_context_graph.protocols import GraphConnector
from memgraph_context_graph.tools import normalize_connector_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from memgraph_context_graph.events import Event
    from memgraph_context_graph.hooks.runtime_plugin import RuntimeCLIPlugin
    from memgraph_toolbox.api.memgraph import Memgraph

    #: Told about one connector failing: its name, the event (None while setting it up), and why.
    ConnectorFailureHandler = Callable[[str, Event | None, Exception], None]

#: Seconds a hook waits to reach Memgraph. Must stay well under the runtimes' hook timeout
#: (30 s in our plugins): the driver's own 30 s default let a blackholed host outlast the hook,
#: so the runtime killed it before the failure could be logged (#378).
HOOK_CONNECT_TIMEOUT = 5.0


def load_payload(stream: Any | None = None) -> dict[str, Any]:
    """Read one JSON hook payload from a text stream (stdin by default)."""
    if stream is None:
        stream = sys.stdin
    raw = stream.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        msg = "Hook payload must be a JSON object"
        raise TypeError(msg)
    return payload


def create_link(
    connector_names: Iterable[str] = (),
    *,
    memgraph_env: dict[str, str] | None = None,
    on_error: ConnectorFailureHandler | None = None,
) -> AgentLink:
    """Create an AgentLink with optional connectors named by CLI/config.

    Runtime-agnostic: connectors (skills-graph, actions-graph, sessions-graph, resources-graph)
    are graph components, not tied to any particular runtime adapter.

    With *memgraph_env*, every connector shares one Memgraph connection with a
    short connect timeout (:data:`HOOK_CONNECT_TIMEOUT`); failing to reach
    Memgraph raises, since no connector could run. Without it each connector
    connects on its own, as library callers always have.

    Args:
        connector_names: Connectors to enable, by distribution name.
        memgraph_env: Resolved Memgraph settings (``resolve_memgraph_env``).
        on_error: When given, a connector that can't be set up, or fails an
            event, is reported here and the others still run. Without it the
            first failure raises.

    Raises:
        ValueError: if a connector name is not supported, or Memgraph can't be reached.
    """
    names = supported_connectors(connector_names)

    def report(connector: GraphConnector, event: Event, exc: Exception) -> None:
        if on_error is not None:
            on_error(connector_name(connector), event, exc)

    link = AgentLink(on_error=report if on_error is not None else None)
    shared = _hook_memgraph(memgraph_env) if names and memgraph_env is not None else None
    for name in names:
        try:
            _CONNECTOR_ADDERS[name](link, memgraph_env, memgraph=shared)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(name, None, exc)
    return link


def supported_connectors(connector_names: Iterable[str]) -> list[str]:
    """*connector_names* normalized to distribution form (``actions-graph``), blanks dropped.

    Raises:
        ValueError: if a name is not a supported connector.
    """
    names = [normalize_connector_name(name) for name in connector_names if name.strip()]
    for name in names:
        if name not in _CONNECTOR_ADDERS:
            msg = f"Unsupported connector: {name}"
            raise ValueError(msg)
    return names


def connector_name(connector: GraphConnector) -> str:
    """The distribution a connector comes from (``actions-graph``), which is how failures name it."""
    return type(connector).__module__.split(".", 1)[0].replace("_", "-")


def _hook_memgraph(memgraph_env: dict[str, str]) -> Memgraph:
    from memgraph_toolbox.api.memgraph import Memgraph

    return Memgraph(**memgraph_kwargs(memgraph_env), driver_config={"connection_timeout": HOOK_CONNECT_TIMEOUT})


_SCHEMA_COMPONENTS = {
    "skills-graph": ("skills_graph", "SkillGraph"),
    "actions-graph": ("actions_graph", "ActionsGraph"),
    "sessions-graph": ("sessions_graph", "SessionsGraph"),
    "resources-graph": ("resources_graph", "ResourcesGraph"),
}


def setup_connector_schemas(connector_names: Iterable[str], *, memgraph_env: dict[str, str] | None = None) -> None:
    """Create the Memgraph schema (indexes, constraints) for each named connector.

    Raises:
        ValueError: if a connector name is not supported.
    """
    kwargs = memgraph_kwargs(memgraph_env)
    for connector_name in connector_names:
        normalized = normalize_connector_name(connector_name)
        if not normalized:
            continue
        if normalized not in _SCHEMA_COMPONENTS:
            raise ValueError(f"Unsupported connector: {connector_name}")
        module_name, class_name = _SCHEMA_COMPONENTS[normalized]
        component_class = getattr(importlib.import_module(module_name), class_name)
        component_class(**kwargs).setup()


def _env_prefix(runtime_name: str) -> str:
    return "MEMGRAPH_CONTEXT_GRAPH_" + runtime_name.strip().replace("-", "_").upper()


def _connectors_from_env(runtime_name: str) -> list[str]:
    value = os.environ.get(f"{_env_prefix(runtime_name)}_CONNECTORS", "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _debug_log(runtime_name: str, message: str) -> None:
    if os.environ.get(f"{_env_prefix(runtime_name)}_DEBUG") == "1":
        print(message, file=sys.stderr)


class _HookArgumentError(Exception):
    """A hook's flags don't parse. Raised instead of argparse's exit 2, which blocks a Claude Code tool call."""


class _HookArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _HookArgumentError(message)


def _hook_parser(plugin: RuntimeCLIPlugin) -> argparse.ArgumentParser:
    parser = _HookArgumentParser(description=f"Bridge {plugin.name} hooks to memgraph-context-graph.")
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Currently supported: skills-graph, actions-graph, sessions-graph, resources-graph.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Override the session id from the hook payload.",
    )
    parser.add_argument(
        "--event-name",
        default=None,
        help="Supply the event name when the runtime payload omits it.",
    )
    parser.add_argument("--memgraph-url", default=None, help="Memgraph Bolt URL. Overrides config file value.")
    parser.add_argument("--memgraph-user", default=None, help="Memgraph username. Overrides config file value.")
    parser.add_argument("--memgraph-password", default=None, help="Memgraph password. Overrides config file value.")
    parser.add_argument("--memgraph-database", default=None, help="Memgraph database. Overrides config file value.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status if the hook payload cannot be recorded.",
    )
    return parser


class _FailureReport:
    """What one hook run was capturing, so each failure is logged with where it happened and for what."""

    def __init__(self, runtime: str) -> None:
        self.runtime = runtime
        self.started = time.monotonic()
        self.stage = "args"
        self.payload: dict[str, Any] = {}
        self.session_id: str | None = None
        self.connector_failures: list[tuple[str, BaseException]] = []

    def capturing(self, payload: dict[str, Any], session_id: str | None) -> None:
        self.payload = payload
        self.session_id = session_id

    def record(self, stage: str, exc: BaseException | None = None, **fields: Any) -> None:
        fields.setdefault("hook_event", _text(self.payload.get("hook_event_name")))
        fields.setdefault("tool_name", _text(self.payload.get("tool_name")))
        fields.setdefault("cwd", _text(self.payload.get("cwd")))
        fields.setdefault("session_id", self.session_id)
        capture_log.record_failure(stage, exc, runtime=self.runtime, started=self.started, **fields)

    def connector_failed(self, name: str, event: Event | None, exc: Exception) -> None:
        self.connector_failures.append((name, exc))
        if event is None:
            self.record("setup", exc, connector=name)
        else:
            self.record("emit", exc, connector=name, event_type=event.event_type.value, session_id=event.session_id)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def run_hook(plugin: RuntimeCLIPlugin, argv: Sequence[str] | None = None) -> int:
    """Read one hook payload from stdin, run it through *plugin*, print a response.

    This is what ``memgraph-context-graph hook run <runtime>`` invokes: parse
    the shared CLI flags, resolve Memgraph config, translate the payload via
    ``plugin.adapter_class``, and write back whatever ``plugin.response_for_payload``
    says the runtime expects -- identical flow for every registered runtime.

    It never fails the session it observes: every failure, from a bad flag to
    one connector's write, goes to the capture-failure log
    (:mod:`memgraph_context_graph.capture_log`), the runtime still gets its
    normal response, and the exit status is 0. ``--strict`` (or the
    ``_STRICT`` env flag) re-raises after logging, for developers; a connector
    failure then raises :class:`ConnectorFailuresError` once every connector has run.
    """
    report = _FailureReport(plugin.name)
    strict = os.environ.get(f"{_env_prefix(plugin.name)}_STRICT") == "1"
    payload: dict[str, Any] = {}
    connector_names: list[str] = []
    try:
        args = _hook_parser(plugin).parse_args(argv)
        strict = strict or args.strict
        connector_names = args.connector if args.connector is not None else _connectors_from_env(plugin.name)

        from memgraph_context_graph.adapters._identity import load_config, resolve_memgraph_env

        report.stage = "config"
        # Inside the guard: an edition the config can't resolve skips the hook like any other failure.
        memgraph_env = resolve_memgraph_env(
            url=args.memgraph_url,
            user=args.memgraph_user,
            password=args.memgraph_password,
            database=args.memgraph_database,
        )
        report.stage = "payload"
        payload = load_payload()
        if args.event_name is not None:
            payload.setdefault("hook_event_name", args.event_name)
        report.capturing(payload, args.session_id or _payload_session_id(payload))
        if problems := load_config().problems:
            report.record("config", level="warning", error_type="MalformedConfig", error="; ".join(problems))

        report.stage = "setup"
        supported_connectors(connector_names)
        report.stage = "connect"
        link = create_link(connector_names, memgraph_env=memgraph_env, on_error=report.connector_failed)
        tool_starts = _ToolStarts()
        link.add_connector(tool_starts)
        report.stage = "translate"
        adapter = plugin.adapter_class(link, session_id=args.session_id)
        adapter.handle_payload(payload)
        report.stage = "respond"
        _print_response(plugin, payload, connector_names, before_tool=_context_before_tool(link, tool_starts.events))
    except Exception as exc:
        if report.stage in {"args", "config"}:
            # Failed before stdin was read: read it now, so the runtime still gets the response its event expects
            # and the failure names the session.
            payload = _payload_or_empty()
            report.capturing(payload, _payload_session_id(payload))
        # Logged first, so a crash below (in the response, or a strict re-raise) can't lose the record.
        report.record(report.stage, exc, error_type="HookArgumentError" if report.stage == "args" else None)
        if strict:
            raise
        _print_response(plugin, payload, connector_names)
        _debug_log(plugin.name, f"memgraph-context-graph {plugin.name} hook skipped: {exc}")
        return 0
    if strict and report.connector_failures:
        raise ConnectorFailuresError(report.connector_failures)
    return 0


def _payload_or_empty() -> dict[str, Any]:
    try:
        return load_payload()
    except Exception:
        return {}


def _payload_session_id(payload: dict[str, Any]) -> str | None:
    """The session id under the keys runtimes use, only to label a failure; adapters resolve it properly."""
    for key in ("session_id", "sessionId", "conversationId"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _print_response(
    plugin: RuntimeCLIPlugin, payload: dict[str, Any], connector_names: list[str], *, before_tool: Sequence[str] = ()
) -> None:
    response = plugin.response_for_payload(payload) or {}
    if payload.get("hook_event_name") == "SessionStart":
        response.update(session_start_context(connector_names, payload))
    render = getattr(plugin, "context_before_tool_response", None)
    if before_tool and render is not None:
        response.update(render("\n".join(before_tool)) or {})
    if response:
        print(json.dumps(response))


class _ToolStarts(GraphConnector):
    """Keeps the tool-start events one payload produced, to ask the connectors about afterwards."""

    def __init__(self) -> None:
        self.events: list[ToolStartEvent] = []

    def supports(self, event: Event) -> bool:
        return isinstance(event, ToolStartEvent)

    def on_event(self, event: Event) -> None:
        if isinstance(event, ToolStartEvent):
            self.events.append(event)


def _context_before_tool(link: AgentLink, events: list[ToolStartEvent]) -> list[str]:
    """Every connector's line for the tools about to run; a connector that fails adds nothing."""
    lines: list[str] = []
    for event in events:
        for connector in link.connectors:
            try:
                line = connector.context_before_tool(event)
            except Exception:
                line = None
            if line:
                lines.append(line)
    return lines


def session_start_context(connector_names: list[str], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """The SessionStart output telling the model which memory tools it has; empty when none.

    Recall is announced in one line and never pushes retrieved content: the
    model calls it when a question needs memory (#394). The ``memory`` tool
    adds its index of the user's memory files instead, standing in for the
    harness's own memory it replaces (#484). Claude Code and Codex read the
    same shape.
    """
    from memgraph_context_graph.tools import session_hints

    try:
        hints = session_hints(connector_names, payload=payload)
    except Exception as exc:  # A tool that fails to load must not fail the hook; the model just isn't told.
        capture_log.record_failure(
            "respond",
            exc,
            level="warning",
            hook_event="SessionStart",
            session_id=_payload_session_id(payload or {}),
        )
        return {}
    if not hints:
        return {}
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "\n".join(hints)}}


def _add_skills_graph_connector(
    link: AgentLink, memgraph_env: dict[str, str] | None = None, *, memgraph: Memgraph | None = None
) -> None:
    try:
        from skills_graph import SkillGraph
        from skills_graph.connector import SkillGraphConnector
    except ImportError as exc:
        msg = "skills-graph is required for the skills-graph connector"
        raise ImportError(msg) from exc

    kwargs = memgraph_kwargs(memgraph_env)
    graph = SkillGraph(memgraph) if memgraph is not None else SkillGraph(**kwargs)
    link.add_connector(SkillGraphConnector(graph))


def _add_sessions_graph_connector(
    link: AgentLink, memgraph_env: dict[str, str] | None = None, *, memgraph: Memgraph | None = None
) -> None:
    try:
        from sessions_graph import SessionsGraph
        from sessions_graph.connector import SessionsGraphConnector
    except ImportError as exc:
        msg = "sessions-graph is required for the sessions-graph connector"
        raise ImportError(msg) from exc

    from memgraph_context_graph.adapters._identity import resolve_auto_reconcile

    kwargs = memgraph_kwargs(memgraph_env)
    graph = SessionsGraph(memgraph) if memgraph is not None else SessionsGraph(**kwargs)
    link.add_connector(SessionsGraphConnector(graph, auto_reconcile=resolve_auto_reconcile()))


def _add_actions_graph_connector(
    link: AgentLink, memgraph_env: dict[str, str] | None = None, *, memgraph: Memgraph | None = None
) -> None:
    try:
        from actions_graph import ActionsGraph
        from actions_graph.connector import ActionsGraphConnector
    except ImportError as exc:
        msg = "actions-graph is required for the actions-graph connector"
        raise ImportError(msg) from exc

    kwargs = memgraph_kwargs(memgraph_env)
    graph = ActionsGraph(memgraph) if memgraph is not None else ActionsGraph(**kwargs)
    link.add_connector(ActionsGraphConnector(graph))


def _add_resources_graph_connector(
    link: AgentLink, memgraph_env: dict[str, str] | None = None, *, memgraph: Memgraph | None = None
) -> None:
    try:
        from resources_graph import ResourcesGraph
        from resources_graph.connector import ResourcesGraphConnector
    except ImportError as exc:
        msg = "resources-graph is required for the resources-graph connector"
        raise ImportError(msg) from exc

    kwargs = memgraph_kwargs(memgraph_env)
    graph = ResourcesGraph(memgraph) if memgraph is not None else ResourcesGraph(**kwargs)
    link.add_connector(ResourcesGraphConnector(graph))


_CONNECTOR_ADDERS: dict[str, Callable[..., None]] = {
    "skills-graph": _add_skills_graph_connector,
    "actions-graph": _add_actions_graph_connector,
    "sessions-graph": _add_sessions_graph_connector,
    "resources-graph": _add_resources_graph_connector,
}


# A closed key set (never `memgraph`) so `Component(**kwargs)` below can't be
# read as colliding with that parameter's `Memgraph | None` type, unlike a
# plain `dict[str, str]`.
class _MemgraphKwargs(TypedDict, total=False):
    url: str
    username: str
    password: str
    database: str


def memgraph_kwargs(memgraph_env: dict[str, str] | None) -> _MemgraphKwargs:
    """Map a :func:`resolve_memgraph_env` dict onto ``Memgraph``'s keyword arguments; ``None`` gives none."""
    if memgraph_env is None:
        return {}
    return {
        "url": memgraph_env["MEMGRAPH_URL"],
        "username": memgraph_env["MEMGRAPH_USER"],
        "password": memgraph_env["MEMGRAPH_PASSWORD"],
        "database": memgraph_env["MEMGRAPH_DATABASE"],
    }
