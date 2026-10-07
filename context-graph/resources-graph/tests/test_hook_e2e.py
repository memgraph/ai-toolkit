"""The real hook path: a raw Claude Code payload on stdin, ``hook run claude-code --connector resources-graph``."""

from __future__ import annotations

import io
import json
import os

import pytest

from agent_context_graph.adapters import _identity
from agent_context_graph.hooks.cli import main as hook_main


@pytest.fixture()
def hook_config(monkeypatch, tmp_path):
    """A config file pointing at the test Memgraph, selected the way a parent process would (ADR 0003)."""
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_full_config(
        user_id="ante",
        memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        memgraph_user=os.environ.get("MEMGRAPH_USER", ""),
        memgraph_password=os.environ.get("MEMGRAPH_PASSWORD", ""),
    )
    yield
    _identity._reset_cache()


def run_hook(monkeypatch, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return hook_main(["run", "claude-code", "--connector", "resources-graph", "--strict"])


def test_pre_tool_use_payload_becomes_a_pending_touch(graph, hook_config, monkeypatch):
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": "cc-1",
        "tool_name": "Bash",
        "tool_input": {"command": "gh pr view 3600 --repo memgraph/memgraph"},
        "tool_use_id": "toolu_cc_1",
    }

    assert run_hook(monkeypatch, payload) == 0

    rows = graph._db.query(
        "MATCH (t:Touch)-[:IN_SESSION]->(:Session {session_id: 'cc-1'}) "
        "RETURN t.address AS address, t.status AS status, t.tool_use_id AS tool_use_id"
    )
    assert rows == [{"address": "github:memgraph/memgraph#3600", "status": "pending", "tool_use_id": "toolu_cc_1"}]


def test_user_prompt_payload_becomes_prompted_touches(graph, hook_config, monkeypatch):
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "cc-1",
        "prompt": "Summarise https://github.com/memgraph/memgraph/issues/2000 and memgraph/mage#12",
    }

    assert run_hook(monkeypatch, payload) == 0

    rows = graph._db.query("MATCH (t:Touch {provenance: 'PROMPTED'}) RETURN t.address AS address ORDER BY address")
    assert [row["address"] for row in rows] == ["github:memgraph/mage#12", "github:memgraph/memgraph#2000"]


def test_session_start_tells_the_model_about_the_resource_tool(graph, hook_config, monkeypatch, capsys):
    assert run_hook(monkeypatch, {"hook_event_name": "SessionStart", "session_id": "cc-1"}) == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "`resource` tool" in context
