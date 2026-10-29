"""The spool of refused events: what a refused connector's events become, and how ``update`` replays them.

The format is tested as plain data. Spooling through ``create_link`` and
replaying run against a live Memgraph, which holds the rule that refuses, and
skip when none is reachable; the connector they replay into is a stand-in
that records what it receives. Each test names its own component and removes
only its own nodes.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import uuid

import pytest

from memgraph_context_graph import capture_log, graph_version, refused_events
from memgraph_context_graph.events import MessageEvent, ToolStartEvent
from memgraph_context_graph.hooks import runner
from memgraph_context_graph.protocols import GraphConnector
from memgraph_context_graph.refused_events import ReplayResult, Spool
from memgraph_toolbox.api.memgraph import Memgraph

_TOOL_START = ToolStartEvent(
    session_id="s-1",
    timestamp="2026-10-29T10:00:00+00:00",
    tool_name="Bash",
    tool_input={"command": "ls"},
    tool_use_id="t-1",
)


def _spooled() -> list[dict]:
    spool = refused_events.path()
    if not spool.exists():
        return []
    return [json.loads(line) for line in spool.read_text(encoding="utf-8").splitlines()]


def test_an_event_round_trips_with_its_timestamp():
    decoded = refused_events.decode(json.loads(json.dumps(refused_events.encode(_TOOL_START))))

    assert decoded == _TOOL_START


def test_decode_drops_fields_it_does_not_know_and_types_it_cannot_build():
    raw = refused_events.encode(_TOOL_START)
    raw["fields"]["added_by_a_later_release"] = 1

    assert refused_events.decode(raw) == _TOOL_START
    assert refused_events.decode({"type": "EventFromTheFuture", "fields": {}}) is None
    assert refused_events.decode({"type": "Path", "fields": {}}) is None


def test_the_spool_is_private_and_sits_beside_the_capture_log():
    assert refused_events.append(_TOOL_START, ["skills-graph"])

    spool = refused_events.path()
    assert spool.parent == capture_log.path().parent
    assert stat.S_IMODE(os.stat(spool).st_mode) == 0o600
    (entry,) = _spooled()
    assert entry["v"] == refused_events.FORMAT_VERSION
    assert entry["connectors"] == ["skills-graph"]
    assert entry["attempts"] == 0


def test_a_full_spool_drops_the_event_and_logs_it(monkeypatch):
    monkeypatch.setattr(refused_events, "MAX_BYTES", 10)

    Spool(["skills-graph"]).on_event(_TOOL_START)

    assert _spooled() == []
    (failure,) = capture_log.read_recent()
    assert failure.error_type == "RefusedEventSpoolFull"
    assert failure.connector == "skills-graph"


class _Recorder(GraphConnector):
    """Stands in for a connector: records what it receives, or fails while ``failing``."""

    def __init__(self) -> None:
        self.events: list = []
        self.failing = False

    def supports(self, event) -> bool:
        return True

    def on_event(self, event) -> None:
        if self.failing:
            raise RuntimeError("the connector failed")
        self.events.append(event)


@pytest.fixture
def db():
    try:
        client = Memgraph(user_agent="memgraph-context-graph-tests")
        client.query("RETURN 1")
    except Exception as exc:
        pytest.skip(f"No live Memgraph reachable: {exc}")
    graph_version.setup(client)
    yield client
    with contextlib.suppress(Exception):
        client.query(
            "MATCH (c:ContextGraphClient {client_id: $id}) DELETE c", {"id": graph_version._load_state()["client_id"]}
        )
    client.close()


@pytest.fixture
def component(db):
    name = f"test-component-{uuid.uuid4().hex}"
    yield name
    db.query("MATCH (s:ContextGraphSchema {component: $c}) DELETE s", {"c": name})


@pytest.fixture
def env():
    return {
        "MEMGRAPH_URL": os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        "MEMGRAPH_USER": os.environ.get("MEMGRAPH_USER", ""),
        "MEMGRAPH_PASSWORD": os.environ.get("MEMGRAPH_PASSWORD", ""),
        "MEMGRAPH_DATABASE": os.environ.get("MEMGRAPH_DATABASE", ""),
    }


@pytest.fixture
def recorder(db, monkeypatch) -> _Recorder:
    """Every connector create_link builds is this one recorder, sharing the test's Memgraph."""
    built = _Recorder()
    monkeypatch.setattr(runner, "supported_connectors", lambda names: list(names))
    monkeypatch.setattr(runner, "_hook_memgraph", lambda memgraph_env: db)
    monkeypatch.setattr(runner, "_build_connector", lambda name, memgraph_env, memgraph: built)
    return built


def _refuse(db, component: str) -> None:
    db.query("MERGE (s:ContextGraphSchema {component: $c}) SET s.epoch = 2, s.min_writer_epoch = 2", {"c": component})


def _allow(db, component: str) -> None:
    db.query("MATCH (s:ContextGraphSchema {component: $c}) DELETE s", {"c": component})


def test_create_link_spools_what_a_refused_connector_would_have_captured(db, component, env, recorder):
    _refuse(db, component)

    link = runner.create_link([component], memgraph_env=env, on_error=lambda *_: None)
    link.emit(_TOOL_START)

    assert recorder.events == []
    (entry,) = _spooled()
    assert entry["connectors"] == [component]
    assert refused_events.decode(entry["event"]) == _TOOL_START


def test_the_spool_keeps_events_as_the_capture_policy_left_them(db, component, env, recorder):
    from memgraph_context_graph.capture_policy import CapturePolicy

    _refuse(db, component)
    link = runner.create_link([component], memgraph_env=env, on_error=lambda *_: None, policy=CapturePolicy())

    link.emit(MessageEvent(session_id="s-1", role="user", content="my key is sk-ant-api03-" + "a" * 40))

    (entry,) = _spooled()
    assert "sk-ant-api03" not in json.dumps(entry)


def test_replay_adds_the_events_once_the_graph_allows_the_connector(db, component, env, recorder):
    _refuse(db, component)
    runner.create_link([component], memgraph_env=env, on_error=lambda *_: None).emit(_TOOL_START)
    _allow(db, component)

    result = refused_events.replay(env)

    assert result == ReplayResult(replayed=1)
    assert recorder.events == [_TOOL_START]
    assert not refused_events.pending()


def test_replay_keeps_what_the_graph_still_refuses(db, component, env, recorder):
    _refuse(db, component)
    refused_events.append(_TOOL_START, [component])

    assert refused_events.replay(env) == ReplayResult(waiting=1)
    assert recorder.events == []
    assert [entry["connectors"] for entry in _spooled()] == [[component]]


def test_replay_retries_a_failing_connector_then_drops_the_event(db, component, env, recorder):
    recorder.failing = True
    refused_events.append(_TOOL_START, [component])

    for _ in range(refused_events.MAX_ATTEMPTS - 1):
        assert refused_events.replay(env) == ReplayResult(waiting=1)
    assert refused_events.replay(env) == ReplayResult(dropped=1)

    assert not refused_events.pending()
    assert [failure.error_type for failure in capture_log.read_recent()][-1] == "RefusedEventDropped"


def test_replay_leaves_the_events_when_the_graph_cannot_be_reached():
    refused_events.append(_TOOL_START, ["skills-graph"])
    unreachable = {
        "MEMGRAPH_URL": "bolt://127.0.0.1:1",
        "MEMGRAPH_USER": "",
        "MEMGRAPH_PASSWORD": "",
        "MEMGRAPH_DATABASE": "",
    }

    with pytest.raises(Exception):  # noqa: B017 -- the driver's own connection error type
        refused_events.replay(unreachable)

    assert refused_events.pending()
