"""Command-hook runner shared by every registered runtime: stdin payload, connectors, adapter, response."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, NoReturn, TypedDict

from memgraph_context_graph import capture_log
from memgraph_context_graph.capture_policy import CapturePolicy, policy_from_config
from memgraph_context_graph.connectors import (
    INSTALLABLE,
    configured_connectors,
    connector_spec,
    supported_connectors,
)
from memgraph_context_graph.events import ToolStartEvent
from memgraph_context_graph.link import AgentLink, ConnectorFailuresError
from memgraph_context_graph.protocols import GraphConnector

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

#: Seconds a hook runs before it logs where it was stuck, answers the runtime, and exits, for the 30 s hook
#: timeout our plugins declare. A query waiting on Memgraph's storage access never raises, so without a deadline
#: of its own the hook is killed at the runtime's timeout and nothing is logged.
HOOK_DEADLINE = 25.0


def deadline_for(timeout: float) -> float:
    """The ``--deadline`` for a hook the runtime kills after *timeout* seconds: enough margin to log and answer."""
    return max(timeout - 5.0, timeout / 2)


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
    policy: CapturePolicy | None = None,
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
        policy: The capture policy every event passes before a connector sees it.

    Raises:
        ValueError: if a connector name is not supported, or Memgraph can't be reached.
    """
    names = supported_connectors(connector_names)

    def report(connector: GraphConnector, event: Event, exc: Exception) -> None:
        if on_error is not None:
            on_error(connector_name(connector), event, exc)

    link = AgentLink(on_error=report if on_error is not None else None, policy=policy)
    shared = _hook_memgraph(memgraph_env) if names and memgraph_env is not None else None
    for name in names:
        try:
            link.add_connector(_build_connector(name, memgraph_env, memgraph=shared))
        except Exception as exc:
            if on_error is None:
                raise
            on_error(name, None, exc)
    return link


def connector_name(connector: GraphConnector) -> str:
    """The distribution a connector comes from (``actions-graph``), which is how failures name it."""
    return type(connector).__module__.split(".", 1)[0].replace("_", "-")


def _hook_memgraph(memgraph_env: dict[str, str]) -> Memgraph:
    from memgraph_toolbox.api.memgraph import Memgraph

    return Memgraph(**memgraph_kwargs(memgraph_env), driver_config={"connection_timeout": HOOK_CONNECT_TIMEOUT})


def setup_connector_schemas(connector_names: Iterable[str], *, memgraph_env: dict[str, str] | None = None) -> None:
    """Create the Memgraph schema (indexes, constraints) for each named connector.

    Raises:
        ValueError: if a connector name is not supported.
    """
    kwargs = memgraph_kwargs(memgraph_env)
    for name in supported_connectors(connector_names):
        connector_spec(name).graph(**kwargs).setup()


def _env_prefix(runtime_name: str) -> str:
    return "MEMGRAPH_CONTEXT_GRAPH_" + runtime_name.strip().replace("-", "_").upper()


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
        help=f"Graph connector to enable, repeatable; overrides the config file's [capture] connectors. "
        f"Supported: {', '.join(INSTALLABLE)}, or any other installed component's.",
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
        "--deadline",
        type=float,
        default=HOOK_DEADLINE,
        help=f"Seconds before the hook gives up, logs why, and answers the runtime (default: {HOOK_DEADLINE:g}). "
        "Keep it under the runtime's hook timeout.",
    )
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


class _Deadline:
    """Answers the runtime and ends the process once a hook has run too long, logging the stage it was stuck in.

    Every response goes through :meth:`respond`, so the runtime reads one
    answer whichever of the hook and the deadline gets there first.
    """

    def __init__(self, plugin: RuntimeCLIPlugin, report: _FailureReport) -> None:
        self._plugin = plugin
        self._report = report
        self._lock = threading.Lock()
        self._responded = False
        self._timer: threading.Timer | None = None

    def start(self, seconds: float) -> None:
        self._timer = threading.Timer(seconds, self._expire, args=(seconds,))
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def respond(self, response: dict[str, Any]) -> None:
        with self._lock:
            if self._responded:
                return
            self._responded = True
            if response:
                print(json.dumps(response))

    def _expire(self, seconds: float) -> None:
        # Never released once answering here: the hook's own thread may still be blocked and must not answer too.
        self._lock.acquire()
        if self._responded:
            self._lock.release()
            return
        self._report.record(
            self._report.stage, error_type="HookDeadline", error=f"hook still running after {seconds:g} s"
        )
        # The plain response: the session-start text would query the graph that is holding the hook up.
        response = self._plugin.response_for_payload(self._report.payload) or {}
        if response:
            print(json.dumps(response))
        sys.stdout.flush()
        os._exit(0)


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

    A hook that blocks, on a query waiting for Memgraph's storage access, never
    raises: at ``--deadline`` it logs a ``HookDeadline`` failure at the stage it
    was stuck in, prints the runtime's plain response, and exits 0 itself,
    before the runtime's own timeout would kill it unlogged.
    """
    report = _FailureReport(plugin.name)
    deadline = _Deadline(plugin, report)
    strict = os.environ.get(f"{_env_prefix(plugin.name)}_STRICT") == "1"
    payload: dict[str, Any] = {}
    connector_names: list[str] = []
    try:
        args = _hook_parser(plugin).parse_args(argv)
        strict = strict or args.strict
        deadline.start(args.deadline)

        from memgraph_context_graph.adapters._identity import load_config, resolve_memgraph_env

        report.stage = "config"
        connector_names = args.connector if args.connector is not None else configured_connectors()
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
        policy, policy_problems = policy_from_config(load_config())
        if problems := [*load_config().problems, *policy_problems]:
            report.record("config", level="warning", error_type="MalformedConfig", error="; ".join(problems))
        if policy.excludes_project(_with_cwd(payload)["cwd"]):
            deadline.respond(_response(plugin, payload, connector_names))
            return 0

        report.stage = "setup"
        supported_connectors(connector_names)
        report.stage = "connect"
        link = create_link(connector_names, memgraph_env=memgraph_env, on_error=report.connector_failed, policy=policy)
        tool_starts = _ToolStarts()
        link.add_connector(tool_starts)
        report.stage = "translate"
        adapter = plugin.adapter_class(link, session_id=args.session_id)
        adapter.handle_payload(payload)
        report.stage = "respond"
        deadline.respond(
            _response(plugin, payload, connector_names, before_tool=_context_before_tool(link, tool_starts.events))
        )
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
        deadline.respond(_response(plugin, payload, connector_names))
        _debug_log(plugin.name, f"memgraph-context-graph {plugin.name} hook skipped: {exc}")
        return 0
    finally:
        deadline.cancel()
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


def _response(
    plugin: RuntimeCLIPlugin, payload: dict[str, Any], connector_names: list[str], *, before_tool: Sequence[str] = ()
) -> dict[str, Any]:
    response = plugin.response_for_payload(payload) or {}
    render_session = getattr(plugin, "session_context_response", None)
    if render_session is not None:
        response.update(render_session(payload, lambda: session_start_text(connector_names, payload)) or {})
    render = getattr(plugin, "context_before_tool_response", None)
    if before_tool and render is not None:
        response.update(render("\n".join(before_tool)) or {})
    return response


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


def session_start_text(connector_names: list[str], payload: dict[str, Any] | None = None) -> str | None:
    """The session-start text telling the model which memory tools it has; None when there are none.

    Recall announces itself and adds where the user left off in this project,
    quoted from the last sessions, but never a ranked recall result: the model
    calls recall when a question needs memory (#394). The ``memory`` tool adds
    its index of the user's memory files, standing in for the harness's own
    memory it replaces (#484). Each runtime wraps the text in the shape its
    session-start hook output takes (``RuntimeSpec.session_context``).
    """
    from memgraph_context_graph.tools import session_hints

    try:
        hints = session_hints(connector_names, payload=_with_cwd(payload or {}))
    except Exception as exc:  # A tool that fails to load must not fail the hook; the model just isn't told.
        capture_log.record_failure(
            "respond",
            exc,
            level="warning",
            hook_event="SessionStart",
            session_id=_payload_session_id(payload or {}),
        )
        return None
    return "\n".join(hints) or None


def _with_cwd(payload: dict[str, Any]) -> dict[str, Any]:
    """*payload* with a ``cwd``: Antigravity names its workspace ``workspacePaths`` and Cursor ``workspace_roots``;
    every runtime starts its hooks in the project, so the hook's own directory is the last resort."""
    if isinstance(payload.get("cwd"), str) and payload["cwd"]:
        return payload
    for key in ("workspacePaths", "workspace_roots"):
        paths = payload.get(key)
        if isinstance(paths, list) and paths and isinstance(paths[0], str):
            return {**payload, "cwd": paths[0]}
    return {**payload, "cwd": os.getcwd()}


def _build_connector(
    name: str, memgraph_env: dict[str, str] | None = None, *, memgraph: Memgraph | None = None
) -> GraphConnector:
    from memgraph_context_graph.adapters._identity import load_config

    spec = connector_spec(name)
    graph = spec.graph(memgraph) if memgraph is not None else spec.graph(**memgraph_kwargs(memgraph_env))
    return spec.build(graph, load_config())


# A closed key set (never `memgraph`) so `spec.graph(**kwargs)` above can't be
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
