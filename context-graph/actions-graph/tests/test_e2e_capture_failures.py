"""Capture failures against a real Memgraph: one broken connector costs nothing else, and is logged.

Also the capture-side fixes that only show on a real graph: ``doctor`` leaves no
Session behind, a Session keeps no tool payload, and a subagent's actions stay
under its Agent when its start event is missing.
"""

import contextlib
import io
import json
import os

import pytest

from actions_graph import ActionsGraph
from actions_graph.connector import ActionsGraphConnector
from memgraph_context_graph import AgentLink, capture_log
from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter
from memgraph_context_graph.hooks import runner
from memgraph_context_graph.hooks.cli import main as hook_main
from memgraph_context_graph.link import ConnectorFailuresError
from memgraph_context_graph.protocols import GraphConnector


@pytest.fixture
def graph():
    g = ActionsGraph()
    with contextlib.suppress(Exception):
        g.setup()
    g.clear()
    yield g
    g.clear()


@pytest.fixture
def config(monkeypatch, tmp_path):
    """A config naming the test Memgraph, and a capture log of the test's own."""
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_full_config(user_id="tester", memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    _identity._reset_cache()
    yield
    _identity._reset_cache()


def _log_lines():
    log = capture_log.path()
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


class _Exploding(GraphConnector):
    def supports(self, event):
        return True

    def on_event(self, event):
        raise RuntimeError("connector bug")


def _run_hook(monkeypatch, argv, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return hook_main(["run", "claude-code", *argv])


_PRE_TOOL_USE = {
    "hook_event_name": "PreToolUse",
    "session_id": "cf-1",
    "tool_name": "Bash",
    "tool_use_id": "tu-1",
    "tool_input": {"command": "echo hi"},
    "cwd": "/repo",
}


def _actions(graph, session_id):
    return graph._db.query(
        "MATCH (:Session {session_id: $id})-[:HAS_ACTION]->(a:Action) RETURN a.tool_name AS tool",
        params={"id": session_id},
    )


def test_a_failing_connector_does_not_drop_the_others(monkeypatch, graph, config):
    monkeypatch.setitem(
        runner._CONNECTOR_ADDERS, "skills-graph", lambda link, env, *, memgraph: link.add_connector(_Exploding())
    )

    assert _run_hook(monkeypatch, ["--connector", "skills-graph", "--connector", "actions-graph"], _PRE_TOOL_USE) == 0

    assert _actions(graph, "cf-1") == [{"tool": "Bash"}]
    (line,) = _log_lines()
    assert (line["stage"], line["event_type"], line["session_id"]) == ("emit", "tool_start", "cf-1")
    assert line["error"] == "connector bug"


def test_strict_mode_raises_only_after_every_connector_ran(monkeypatch, graph, config):
    monkeypatch.setitem(
        runner._CONNECTOR_ADDERS, "skills-graph", lambda link, env, *, memgraph: link.add_connector(_Exploding())
    )

    with pytest.raises(ConnectorFailuresError):
        _run_hook(
            monkeypatch, ["--connector", "skills-graph", "--connector", "actions-graph", "--strict"], _PRE_TOOL_USE
        )

    assert _actions(graph, "cf-1") == [{"tool": "Bash"}]


def test_a_connector_that_cannot_be_set_up_does_not_drop_the_others(monkeypatch, graph, config):
    def missing(link, env, *, memgraph):
        raise ImportError("skills-graph is required for the skills-graph connector")

    monkeypatch.setitem(runner._CONNECTOR_ADDERS, "skills-graph", missing)

    assert _run_hook(monkeypatch, ["--connector", "skills-graph", "--connector", "actions-graph"], _PRE_TOOL_USE) == 0

    assert _actions(graph, "cf-1") == [{"tool": "Bash"}]
    (line,) = _log_lines()
    assert (line["stage"], line["connector"], line["error_type"]) == ("setup", "skills-graph", "ImportError")


def test_connectors_share_one_connection(monkeypatch, graph, config):
    built, handed = [], []
    original = runner._hook_memgraph

    def counting(env):
        built.append(env)
        return original(env)

    def second_connector(link, env, *, memgraph):
        # Stands in for skills-graph, which isn't installed beside actions-graph: what matters is the
        # connection it is handed.
        handed.append(memgraph)

    monkeypatch.setattr(runner, "_hook_memgraph", counting)
    monkeypatch.setitem(runner._CONNECTOR_ADDERS, "skills-graph", second_connector)
    assert _run_hook(monkeypatch, ["--connector", "actions-graph", "--connector", "skills-graph"], _PRE_TOOL_USE) == 0
    assert len(built) == 1
    assert len(handed) == 1 and handed[0] is not None
    assert _actions(graph, "cf-1") == [{"tool": "Bash"}]
    assert _log_lines() == []


def test_doctor_probe_leaves_no_session(monkeypatch, graph, config):
    from memgraph_context_graph import cli

    # cursor's probe is a session end, which actions-graph records as a Session.
    assert cli._check_runtime("cursor", ["actions-graph"])["ok"] is True
    assert graph._db.query("MATCH (s:Session) RETURN count(s) AS c") == [{"c": 0}]

    # Without the cleanup the probe would have left one: the check above is not vacuous.
    monkeypatch.setattr(cli, "_delete_probe_session", lambda env, session_id: None)
    assert cli._check_runtime("cursor", ["actions-graph"])["ok"] is True
    (row,) = graph._db.query("MATCH (s:Session) RETURN s.session_id AS id")
    assert row["id"].startswith("doctor-probe-")


def test_session_metadata_keeps_no_tool_payload(graph):
    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
    ClaudeCodeHooksAdapter(link).handle_payload(
        {**_PRE_TOOL_USE, "tool_input": {"command": "export TOKEN=s3cret"}, "permission_mode": "default"}
    )

    (row,) = graph._db.query("MATCH (s:Session {session_id: 'cf-1'}) RETURN s.metadata AS metadata")
    metadata = json.loads(row["metadata"])
    assert metadata == {"cwd": "/repo", "permission_mode": "default", "source_sdk": "claude-code"}


def test_subagent_actions_stay_under_the_agent_when_its_start_is_missing(graph):
    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
    adapter = ClaudeCodeHooksAdapter(link)
    adapter.handle_payload({**_PRE_TOOL_USE, "tool_use_id": "parent-call", "tool_name": "Agent"})
    # The subagent's first tool call arrives before (here: without) its SubagentStart.
    adapter.handle_payload({**_PRE_TOOL_USE, "tool_use_id": "sub-1", "agent_id": "agent-7", "agent_type": "Explore"})

    nested = graph._db.query(
        "MATCH (:Session {session_id: 'cf-1'})-[:HAS_AGENT]->(a:Agent)-[:HAS_ACTION]->(x:Action) "
        "RETURN a.agent_id AS agent, a.agent_type AS type, x.tool_use_id AS call"
    )
    assert nested == [{"agent": "agent-7", "type": "Explore", "call": "sub-1"}]
    direct = graph._db.query(
        "MATCH (:Session {session_id: 'cf-1'})-[:HAS_ACTION]->(x:Action) RETURN x.tool_use_id AS call"
    )
    assert direct == [{"call": "parent-call"}]

    # When the start does arrive late, it fills in the same Agent rather than making a second one.
    adapter.handle_payload(
        {"hook_event_name": "SubagentStart", "session_id": "cf-1", "agent_id": "agent-7", "agent_type": "Explore"}
    )
    assert graph._db.query("MATCH (a:Agent) RETURN count(a) AS c") == [{"c": 1}]


def test_top_level_agent_names_stay_on_the_session(graph):
    """An agent name with no runtime ``agent_id`` (the SDK adapters name every agent) is not a subagent."""
    from memgraph_context_graph.events import ToolStartEvent

    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
    link.emit(ToolStartEvent(session_id="cf-2", tool_name="Read", tool_use_id="t1", agent_name="main"))

    assert graph._db.query("MATCH (a:Agent) RETURN count(a) AS c") == [{"c": 0}]
    assert _actions(graph, "cf-2") == [{"tool": "Read"}]
