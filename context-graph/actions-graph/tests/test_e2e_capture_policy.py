"""The capture policy against a real Memgraph: what a hook run stores once the config file's policy has applied."""

import contextlib
import io
import json
import os

import pytest

from actions_graph import ActionsGraph
from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.hooks.cli import main as hook_main

# Assembled, so this file holds no string a secret scanner would flag.
_GITHUB = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


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
    """A config naming the test Memgraph, with a capture policy set."""
    path = tmp_path / "config.toml"
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(path))
    _identity._reset_cache()
    _identity.write_config(user_id="tester", memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    path.write_text(
        path.read_text()
        + '\n[capture]\nconnectors = "actions-graph"\nexclude_tools = "TodoWrite"\n'
        + f'exclude_projects = "{tmp_path / "excluded"}"\n'
        + "\n[redact]\nticket = 'TICKET-[0-9]+'\n"
    )
    _identity._reset_cache()
    yield tmp_path
    _identity._reset_cache()


def _run_hook(monkeypatch, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return hook_main(["run", "claude-code", "--strict"])


def _stored(graph, session_id):
    """Every property of every node the session reached, as one string to search."""
    rows = graph._db.query(
        "MATCH (s:Session {session_id: $id}) OPTIONAL MATCH (s)-[*1..2]->(n) "
        "RETURN properties(s) AS session, collect(properties(n)) AS nodes",
        params={"id": session_id},
    )
    return json.dumps(rows, default=str)


def test_secrets_and_private_text_never_reach_the_graph(monkeypatch, graph, config):
    session = {"session_id": "cp-1", "cwd": str(config / "repo")}
    _run_hook(
        monkeypatch,
        {**session, "hook_event_name": "UserPromptSubmit", "prompt": "fix TICKET-7 <private>pw is hunter2</private>"},
    )
    _run_hook(
        monkeypatch,
        {
            **session,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_use_id": "tu-1",
            "tool_input": {"command": f"gh auth login --with-token {_GITHUB}"},
            "tool_response": {"stdout": f"logged in with {_GITHUB}"},
        },
    )

    stored = _stored(graph, "cp-1")
    assert "Bash" in stored and "fix" in stored, "the session was captured"
    assert _GITHUB not in stored and "hunter2" not in stored and "TICKET-7" not in stored
    assert "[REDACTED:github_token]" in stored and "[REDACTED:ticket]" in stored


def test_a_prompt_that_is_only_private_is_not_stored(monkeypatch, graph, config):
    session = {"session_id": "cp-2", "cwd": str(config / "repo")}
    _run_hook(monkeypatch, {**session, "hook_event_name": "UserPromptSubmit", "prompt": "<private>my diary</private>"})
    _run_hook(monkeypatch, {**session, "hook_event_name": "UserPromptSubmit", "prompt": "now the release notes"})

    stored = _stored(graph, "cp-2")
    assert "release notes" in stored, "prompts in this session are captured"
    assert "diary" not in stored


def test_excluded_tools_leave_no_action(monkeypatch, graph, config):
    session = {"session_id": "cp-3", "cwd": str(config / "repo")}
    for tool in ("TodoWrite", "Read"):
        _run_hook(
            monkeypatch,
            {**session, "hook_event_name": "PreToolUse", "tool_name": tool, "tool_use_id": tool, "tool_input": {}},
        )

    rows = graph._db.query("MATCH (:Session {session_id: 'cp-3'})-[:HAS_ACTION]->(a:Action) RETURN a.tool_name AS tool")
    assert rows == [{"tool": "Read"}]


def test_an_excluded_project_captures_nothing(monkeypatch, graph, config):
    payload = {
        "session_id": "cp-4",
        "cwd": str(config / "excluded" / "src"),
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_use_id": "tu-1",
        "tool_input": {"command": "ls"},
    }

    assert _run_hook(monkeypatch, payload) == 0
    assert graph._db.query("MATCH (s:Session {session_id: 'cp-4'}) RETURN count(s) AS c") == [{"c": 0}]
