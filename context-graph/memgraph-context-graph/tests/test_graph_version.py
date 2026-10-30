"""The graph-version check: what a client refuses once a newer Context Graph reshapes the graph.

The rules are tested as plain data; reading them, recording the client and
the cache run against a live Memgraph and skip when none is reachable. Each
test names its own component and removes only its own nodes, so pointing it
at a Memgraph that holds real data neither blocks nor touches a real client.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

import pytest

from memgraph_context_graph import graph_version
from memgraph_context_graph.graph_version import HUB, GraphTooNewError, Rules
from memgraph_context_graph.hooks import runner
from memgraph_context_graph.tools import ToolResult
from memgraph_toolbox.api.memgraph import Memgraph


def test_a_graph_without_rules_allows_everything():
    rules = Rules()

    assert rules.may_write("sessions-graph")
    assert rules.may_read("sessions-graph")
    assert rules.notice() is None


def test_a_component_newer_than_this_client_refuses_only_that_component():
    rules = Rules(min_writer={"skills-graph": 2})

    assert not rules.may_write("skills_graph")
    assert rules.may_write("actions-graph")
    assert rules.may_read("skills-graph")


def test_a_rule_at_the_epoch_this_client_knows_allows_it():
    rules = Rules(min_writer={"skills-graph": graph_version.BASELINE_EPOCH})

    assert rules.may_write("skills-graph")


def test_a_newer_session_node_refuses_every_connector():
    rules = Rules(min_writer={HUB: 2})

    assert not rules.may_write("actions-graph")
    assert not rules.may_write("sessions-graph")
    assert rules.may_read("sessions-graph")


def test_readers_and_writers_are_refused_separately():
    rules = Rules(min_reader={"sessions-graph": 2})

    assert rules.may_write("sessions-graph")
    assert not rules.may_read("sessions-graph")


def test_the_refusal_and_the_notice_tell_the_user_to_update():
    rules = Rules(min_writer={"skills-graph": 2})

    assert "memgraph-context-graph update" in str(rules.refuse_write("skills-graph"))
    notice = rules.notice()
    assert notice is not None
    assert "skills-graph" in notice
    assert "memgraph-context-graph update" in notice


@pytest.fixture
def db():
    try:
        client = Memgraph(user_agent="memgraph-context-graph-tests")
        client.query("RETURN 1")
    except Exception as exc:
        pytest.skip(f"No live Memgraph reachable: {exc}")
    graph_version.setup(client)
    yield client
    client.close()


@pytest.fixture
def component(db):
    """A component of this test's own, so its rule can't refuse a real client sharing the Memgraph."""
    name = f"test-component-{uuid.uuid4().hex}"
    yield name
    db.query("MATCH (s:ContextGraphSchema {component: $c}) DELETE s", {"c": name})


@pytest.fixture
def client_id(db):
    """This test's client id, saved before the test runs; its client record is removed afterwards."""
    state = graph_version._load_state()
    graph_version._save_state(state)
    yield state["client_id"]
    db.query("MATCH (c:ContextGraphClient {client_id: $id}) DELETE c", {"id": state["client_id"]})


def _require(db, component: str, *, writer: int | None = None, reader: int | None = None) -> None:
    db.query(
        "MERGE (s:ContextGraphSchema {component: $c}) SET s.epoch = 2, s.min_writer_epoch = $w, s.min_reader_epoch = $r",
        {"c": component, "w": writer, "r": reader},
    )


def _check(db, **kwargs: Any) -> Rules:
    return graph_version.check(lambda: contextlib.nullcontext(db), url="bolt://test", **kwargs)


def test_setup_can_run_again(db):
    graph_version.setup(db)


def test_check_reads_the_graphs_rules(db, component, client_id):
    _require(db, component, writer=2)

    rules = _check(db)

    assert not rules.may_write(component)
    assert rules.may_read(component)


def test_check_records_this_client(db, client_id):
    _check(db)

    rows = db.query(
        "MATCH (c:ContextGraphClient {client_id: $id}) RETURN c.version AS version, c.baseline_epoch AS baseline",
        {"id": client_id},
    )
    assert rows == [{"version": graph_version._client_version(), "baseline": graph_version.BASELINE_EPOCH}]


def test_check_answers_from_the_cache_until_it_expires(db, component, client_id):
    _require(db, component, writer=2)
    now = datetime.now(timezone.utc)
    _check(db, now=now)
    db.query("MATCH (s:ContextGraphSchema {component: $c}) DELETE s", {"c": component})

    assert not _check(db, now=now + timedelta(minutes=1)).may_write(component)
    assert _check(db, now=now + graph_version.CHECK_INTERVAL).may_write(component)


def test_a_fresh_check_skips_the_cache(db, component, client_id):
    _check(db)
    _require(db, component, writer=2)

    assert not _check(db, fresh=True).may_write(component)


def test_a_graph_that_cant_be_asked_allows_everything_and_isnt_cached():
    def unreachable():
        raise ConnectionError("no route")

    rules = graph_version.check(unreachable, url="bolt://nowhere")

    assert rules == Rules()
    assert graph_version._load_state()["graphs"] == {}


def test_create_link_leaves_out_a_connector_the_graph_refuses(db, component, client_id, monkeypatch):
    _require(db, component, writer=2)
    built: list[str] = []
    monkeypatch.setattr(runner, "supported_connectors", lambda names: list(names))
    monkeypatch.setattr(runner, "_hook_memgraph", lambda env: db)
    monkeypatch.setattr(runner, "_build_connector", lambda name, env, memgraph: built.append(name) or _Quiet())
    failures: list[tuple[str, Exception]] = []

    runner.create_link(
        [component, "other-component"],
        memgraph_env={"MEMGRAPH_URL": "bolt://test", "MEMGRAPH_DATABASE": ""},
        on_error=lambda name, event, exc: failures.append((name, exc)),
    )

    assert built == ["other-component"]
    assert [(name, type(exc)) for name, exc in failures] == [(component, GraphTooNewError)]


class _Quiet:
    def supports(self, event: Any) -> bool:
        return False


def test_session_start_text_leads_with_the_notice(db, component, client_id, monkeypatch):
    _require(db, component, writer=2)
    rules = _check(db)
    monkeypatch.setattr(graph_version, "cached_rules", lambda url, database="": rules)

    text = runner.session_start_text([])

    assert text is not None
    assert text.startswith("Context Graph is paused for " + component)


@dataclass
class _Tool:
    connector: str
    name: str = "memory"
    writes_graph: bool = False
    description: str = "A stand-in."
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    session_hint: str | None = None

    def call(self, arguments: dict[str, Any], config: Any) -> ToolResult:
        return ToolResult(text="called")


@pytest.mark.asyncio
async def test_mcp_refuses_a_writing_tool_but_still_serves_a_reading_one(db, component, client_id, monkeypatch):
    pytest.importorskip("mcp")
    from memgraph_context_graph.mcp_server import build_server

    _require(db, component, writer=2)
    monkeypatch.setattr(graph_version, "check_configured", lambda **_: _check(db))
    server = build_server(
        {
            "memory": _Tool(connector=component, writes_graph=True),
            "recall": _Tool(connector=component, name="recall"),
        }
    )

    async with _connected(server) as client:
        written = await client.call_tool("memory", {})
        read = await client.call_tool("recall", {})

    assert written.isError
    assert "memgraph-context-graph update" in written.content[0].text
    assert not read.isError
    assert read.content[0].text == "called"


@contextlib.asynccontextmanager
async def _connected(server):
    """A client session talking to ``server`` in memory, the same way under mcp 1.x and 2.x."""
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(lambda: server.run(*server_streams, server.create_initialization_options()))
        async with ClientSession(*client_streams) as client:
            await client.initialize()
            yield client
        tg.cancel_scope.cancel()
