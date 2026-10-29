"""Whether this client may write or read a graph that a newer Context Graph may have reshaped.

A later release may migrate a component's part of the graph to a new shape.
Clients installed before it keep running: on other machines, against a shared
Enterprise server, or as a harness's long-lived MCP server. They would write
the old shape into the new graph, or read it and find nothing. So the graph
says, per component, the oldest shape a client must know to write or read it:

    (:ContextGraphSchema {component, epoch, min_writer_epoch, min_reader_epoch})

and every client checks that before it writes or reads. This rule is the one
part of graph versioning frozen into every installed client, so it stays this
small. The migrations, their ledger and ``epoch`` itself belong to the release
that writes these nodes. A component with no node has no constraint, so every
graph written before this module allows everything.

``memgraph-context-graph`` is the component of the shared Session Node, which
every connector writes through, so a constraint on it holds for all of them.

A client also records itself on each check as
``(:ContextGraphClient {client_id, version, baseline_epoch, epochs, last_seen})``.
Before a migration removes an old shape, it can then see whether clients that
still write that shape are about.

Checks are cached per graph for :data:`CHECK_INTERVAL` in a state file beside
the capture-failure log, so a hook pays for one at most every few minutes. A
check that can't reach the graph allows everything and isn't cached: the
connectors then fail and log on their own.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from memgraph_context_graph import capture_log

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from contextlib import AbstractContextManager
    from pathlib import Path

#: The component that owns the Session Node; a constraint on it covers every connector.
HUB = "memgraph-context-graph"
#: The shape every component is at in a graph no migration has touched.
BASELINE_EPOCH = 1
#: The epoch of each component's shape this release writes and reads, where it differs from the baseline.
EPOCHS: dict[str, int] = {}
#: How long a check holds before the graph is asked again.
CHECK_INTERVAL = timedelta(minutes=10)

UPDATE_COMMAND = "memgraph-context-graph update"


class GraphTooNewError(Exception):
    """The graph's shape for a component is newer than this client can safely write or read."""


def known_epoch(component: str) -> int:
    """The epoch of *component*'s shape this release writes and reads."""
    return EPOCHS.get(_normalize(component), BASELINE_EPOCH)


@dataclass(frozen=True)
class Rules:
    """What the graph asks of its clients: per component, the oldest epoch that may write and read it."""

    min_writer: Mapping[str, int] = field(default_factory=dict)
    min_reader: Mapping[str, int] = field(default_factory=dict)

    def may_write(self, component: str) -> bool:
        """Whether this client may write *component*'s part of the graph."""
        return not self._newer(self.min_writer, component)

    def may_read(self, component: str) -> bool:
        """Whether this client may read *component*'s part of the graph."""
        return not self._newer(self.min_reader, component)

    def refuse_write(self, component: str) -> GraphTooNewError:
        """The error to report for a write :meth:`may_write` refuses."""
        return GraphTooNewError(_refusal("write", self._newer(self.min_writer, component)))

    def refuse_read(self, component: str) -> GraphTooNewError:
        """The error to report for a read :meth:`may_read` refuses."""
        return GraphTooNewError(_refusal("read", self._newer(self.min_reader, component)))

    def notice(self) -> str | None:
        """One line for the model's session-start context while anything is refused, else None."""
        writes = sorted(c for c, epoch in self.min_writer.items() if epoch > known_epoch(c))
        reads = sorted(c for c, epoch in self.min_reader.items() if epoch > known_epoch(c))
        if not writes and not reads:
            return None
        paused = ", ".join(dict.fromkeys(writes + reads))
        return (
            f"Context Graph is paused for {paused}: a newer Context Graph upgraded this graph, and the "
            f"version installed here can't safely use it. Ask the user to run `{UPDATE_COMMAND}`."
        )

    def _newer(self, required: Mapping[str, int], component: str) -> list[str]:
        """The components among *component* and the hub whose required epoch is above what this client knows."""
        return [c for c in dict.fromkeys((HUB, _normalize(component))) if required.get(c, 0) > known_epoch(c)]


def _refusal(action: str, components: list[str]) -> str:
    return (
        f"this graph needs a newer Context Graph to {action} {', '.join(components)} than the one installed "
        f"here; run {UPDATE_COMMAND}"
    )


def setup(db: Any) -> None:
    """Create the constraints and indexes the check and the client record use; safe to repeat."""
    db.query("CREATE CONSTRAINT ON (s:ContextGraphSchema) ASSERT s.component IS UNIQUE;")
    db.query("CREATE INDEX ON :ContextGraphSchema;")
    db.query("CREATE CONSTRAINT ON (c:ContextGraphClient) ASSERT c.client_id IS UNIQUE;")
    db.query("CREATE INDEX ON :ContextGraphClient(client_id);")


def setup_at(memgraph_env: dict[str, str]) -> None:
    """:func:`setup` on the graph a resolved Memgraph environment (``resolve_memgraph_env``) names."""
    from memgraph_context_graph.hooks.runner import memgraph_kwargs
    from memgraph_toolbox.api.memgraph import Memgraph

    with contextlib.closing(Memgraph(**memgraph_kwargs(memgraph_env))) as db:
        setup(db)


def check(
    connect: Callable[[], AbstractContextManager[Any]],
    *,
    url: str,
    database: str = "",
    fresh: bool = False,
    now: datetime | None = None,
) -> Rules:
    """The graph's rules for this client, from the cache while it holds, else from the graph.

    Args:
        connect: Opens a client for the graph, called only when the cache doesn't hold.
        url: The graph's Bolt URL; with *database*, what the cache keys on.
        database: The Memgraph database, empty for the default one.
        fresh: Ask the graph even while the cache holds.
        now: The current time, for tests.

    Returns:
        The rules; no rules (everything allowed) when the graph can't be asked.
    """
    now = now or datetime.now(timezone.utc)
    state = _load_state()
    key = _graph_key(url, database)
    cached = state["graphs"].get(key)
    if not fresh and cached is not None and _holds(cached, now):
        return _rules_from(cached.get("rules"))
    try:
        with connect() as db:
            rules = read_rules(db)
            # A client that can read but not write the graph (a shared space) still gets its rules.
            with contextlib.suppress(Exception):
                _record_client(db, state["client_id"], now)
    except Exception:
        return Rules()
    state["graphs"][key] = {"checked_at": now.isoformat(), "rules": _rules_to(rules)}
    _save_state(state)
    return rules


def cached_rules(url: str, database: str = "") -> Rules:
    """The rules this client last read from the graph, without asking it; none when it never has."""
    cached = _load_state()["graphs"].get(_graph_key(url, database))
    return _rules_from(cached.get("rules")) if cached is not None else Rules()


def check_configured(*, fresh: bool = False) -> Rules:
    """:func:`check` for the graph the config file names, connecting only when the cache doesn't hold."""
    from memgraph_context_graph.adapters._identity import resolve_memgraph_env
    from memgraph_context_graph.hooks.runner import memgraph_kwargs
    from memgraph_toolbox.api.memgraph import Memgraph

    try:
        env = resolve_memgraph_env()
    except Exception:
        return Rules()
    return check(
        lambda: contextlib.closing(Memgraph(**memgraph_kwargs(env))),
        url=env["MEMGRAPH_URL"],
        database=env["MEMGRAPH_DATABASE"],
        fresh=fresh,
    )


def configured_notice() -> str | None:
    """The session-start notice for the graph the config file names, from the cache alone; never raises."""
    from memgraph_context_graph.adapters._identity import resolve_memgraph_env

    try:
        env = resolve_memgraph_env()
        return cached_rules(env["MEMGRAPH_URL"], env["MEMGRAPH_DATABASE"]).notice()
    except Exception:
        return None


def read_rules(db: Any) -> Rules:
    """The graph's rules, read from its ``(:ContextGraphSchema)`` nodes."""
    rows = db.query(
        "MATCH (s:ContextGraphSchema) "
        "RETURN s.component AS component, s.min_writer_epoch AS min_writer, s.min_reader_epoch AS min_reader"
    )
    min_writer: dict[str, int] = {}
    min_reader: dict[str, int] = {}
    for row in rows:
        component = row.get("component")
        if not isinstance(component, str):
            continue
        if isinstance(row.get("min_writer"), int):
            min_writer[_normalize(component)] = row["min_writer"]
        if isinstance(row.get("min_reader"), int):
            min_reader[_normalize(component)] = row["min_reader"]
    return Rules(min_writer, min_reader)


def _client_version() -> str:
    from memgraph_context_graph import __version__

    return __version__


def _record_client(db: Any, client_id: str, now: datetime) -> None:
    from memgraph_toolbox.api.memgraph import merge_with_retry

    merge_with_retry(
        db,
        """
        MERGE (c:ContextGraphClient {client_id: $client_id})
        SET c.version = $version, c.baseline_epoch = $baseline, c.epochs = $epochs, c.last_seen = $now
        """,
        params={
            "client_id": client_id,
            "version": _client_version(),
            "baseline": BASELINE_EPOCH,
            "epochs": dict(EPOCHS),
            "now": now.isoformat(),
        },
    )


def _holds(cached: dict[str, Any], now: datetime) -> bool:
    try:
        checked_at = datetime.fromisoformat(cached["checked_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return timedelta(0) <= now - checked_at < CHECK_INTERVAL


def _rules_to(rules: Rules) -> dict[str, dict[str, int]]:
    return {"min_writer": dict(rules.min_writer), "min_reader": dict(rules.min_reader)}


def _rules_from(raw: Any) -> Rules:
    if not isinstance(raw, dict):
        return Rules()

    def epochs(name: str) -> dict[str, int]:
        value = raw.get(name)
        if not isinstance(value, dict):
            return {}
        return {str(c): e for c, e in value.items() if isinstance(e, int)}

    return Rules(epochs("min_writer"), epochs("min_reader"))


def _graph_key(url: str, database: str) -> str:
    return f"{url}#{database}"


def _normalize(component: str) -> str:
    from memgraph_context_graph.tools import normalize_connector_name

    return normalize_connector_name(component)


def state_path() -> Path:
    """The cache file: beside the capture-failure log, so a config chosen with ``CONTEXT_GRAPH_CONFIG`` gets its own."""
    log = capture_log.path()
    return log.with_name(log.name.removesuffix("capture-failures.jsonl") + "graph-version.json")


def _load_state() -> dict[str, Any]:
    try:
        state = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    if not isinstance(state.get("client_id"), str):
        state["client_id"] = uuid.uuid4().hex
    if not isinstance(state.get("graphs"), dict):
        state["graphs"] = {}
    return state


def _save_state(state: dict[str, Any]) -> None:
    """Write the cache whole, through a rename, so a hook running at the same moment never reads half of it."""
    target = state_path()
    with contextlib.suppress(OSError):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        os.replace(temporary, target)
