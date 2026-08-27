"""Generic command-hook CLI runner, shared by every registered runtime plugin.

Every command-hook adapter (``adapters/codex.py``, ``adapters/claude_code.py``)
used to duplicate this exact stdin-loading / connector-construction / run
sequence, differing only in which adapter class to instantiate and a handful
of env var names. This collapses that into one implementation driven by a
``RuntimeCLIPlugin`` (see ``runtime_plugin.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any, TypedDict

from agent_context_graph.link import AgentLink

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from agent_context_graph.hooks.runtime_plugin import RuntimeCLIPlugin


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


def create_link(connector_names: Iterable[str] = (), *, memgraph_env: dict[str, str] | None = None) -> AgentLink:
    """Create an AgentLink with optional connectors named by CLI/config.

    Runtime-agnostic: connectors (skills-graph, actions-graph, sessions-graph)
    are graph components, not tied to any particular runtime adapter.
    """
    link = AgentLink()
    for connector_name in connector_names:
        normalized = connector_name.strip().replace("-", "_")
        if not normalized:
            continue
        if normalized == "skills_graph":
            _add_skills_graph_connector(link, memgraph_env)
        elif normalized == "actions_graph":
            _add_actions_graph_connector(link, memgraph_env)
        elif normalized == "sessions_graph":
            _add_sessions_graph_connector(link, memgraph_env)
        else:
            msg = f"Unsupported connector: {connector_name}"
            raise ValueError(msg)
    return link


def _env_prefix(runtime_name: str) -> str:
    return "AGENT_CONTEXT_GRAPH_" + runtime_name.strip().replace("-", "_").upper()


def _connectors_from_env(runtime_name: str) -> list[str]:
    value = os.environ.get(f"{_env_prefix(runtime_name)}_CONNECTORS", "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _debug_log(runtime_name: str, message: str) -> None:
    if os.environ.get(f"{_env_prefix(runtime_name)}_DEBUG") == "1":
        print(message, file=sys.stderr)


def run_hook(plugin: RuntimeCLIPlugin, argv: Sequence[str] | None = None) -> int:
    """Read one hook payload from stdin, run it through *plugin*, print a response.

    This is what ``agent-context-graph hook run <runtime>`` invokes: parse
    the shared CLI flags, resolve Memgraph config, translate the payload via
    ``plugin.adapter_class``, and write back whatever ``plugin.response_for_payload``
    says the runtime expects -- identical flow for every registered runtime.
    """
    parser = argparse.ArgumentParser(description=f"Bridge {plugin.name} hooks to agent-context-graph.")
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Currently supported: skills-graph, actions-graph, sessions-graph.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Override the session id from the hook payload.",
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
    args = parser.parse_args(argv)

    connector_names = args.connector
    if connector_names is None:
        connector_names = _connectors_from_env(plugin.name)

    from agent_context_graph.adapters._identity import resolve_memgraph_env

    memgraph_env = resolve_memgraph_env(
        url=args.memgraph_url,
        user=args.memgraph_user,
        password=args.memgraph_password,
        database=args.memgraph_database,
    )

    payload: dict[str, Any] = {}
    try:
        payload = load_payload()
        link = create_link(connector_names, memgraph_env=memgraph_env)
        adapter = plugin.adapter_class(link, session_id=args.session_id)
        adapter.handle_payload(payload)
        response = plugin.response_for_payload(payload)
        if response is not None:
            print(json.dumps(response))
    except Exception as exc:
        strict_env = os.environ.get(f"{_env_prefix(plugin.name)}_STRICT") == "1"
        if args.strict or strict_env:
            raise
        response = plugin.response_for_payload(payload)
        if response is not None:
            print(json.dumps(response))
        _debug_log(plugin.name, f"agent-context-graph {plugin.name} hook skipped: {exc}")
    return 0


def _add_skills_graph_connector(link: AgentLink, memgraph_env: dict[str, str] | None = None) -> None:
    try:
        from skills_graph import SkillGraph
        from skills_graph.connector import SkillGraphConnector
    except ImportError as exc:
        msg = "skills-graph is required for the skills-graph connector"
        raise ImportError(msg) from exc

    kwargs = _memgraph_kwargs(memgraph_env)
    graph = SkillGraph(**kwargs)
    link.add_connector(SkillGraphConnector(graph))


def _add_sessions_graph_connector(link: AgentLink, memgraph_env: dict[str, str] | None = None) -> None:
    try:
        from sessions_graph import SessionsGraph
        from sessions_graph.connector import SessionsGraphConnector
    except ImportError as exc:
        msg = "sessions-graph is required for the sessions-graph connector"
        raise ImportError(msg) from exc

    from agent_context_graph.adapters._identity import resolve_auto_reconcile

    kwargs = _memgraph_kwargs(memgraph_env)
    graph = SessionsGraph(**kwargs)
    link.add_connector(SessionsGraphConnector(graph, auto_reconcile=resolve_auto_reconcile()))


def _add_actions_graph_connector(link: AgentLink, memgraph_env: dict[str, str] | None = None) -> None:
    try:
        from actions_graph import ActionsGraph
        from actions_graph.connector import ActionsGraphConnector
    except ImportError as exc:
        msg = "actions-graph is required for the actions-graph connector"
        raise ImportError(msg) from exc

    kwargs = _memgraph_kwargs(memgraph_env)
    graph = ActionsGraph(**kwargs)
    link.add_connector(ActionsGraphConnector(graph))


# A closed key set (never `memgraph`) so `Component(**kwargs)` below can't be
# read as colliding with that parameter's `Memgraph | None` type, unlike a
# plain `dict[str, str]`.
class _MemgraphKwargs(TypedDict, total=False):
    url: str
    username: str
    password: str
    database: str


def _memgraph_kwargs(memgraph_env: dict[str, str] | None) -> _MemgraphKwargs:
    """Convert resolved memgraph env dict to kwargs for graph component constructors."""
    if memgraph_env is None:
        return {}
    return {
        "url": memgraph_env["MEMGRAPH_URL"],
        "username": memgraph_env["MEMGRAPH_USER"],
        "password": memgraph_env["MEMGRAPH_PASSWORD"],
        "database": memgraph_env["MEMGRAPH_DATABASE"],
    }
