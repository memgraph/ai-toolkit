"""The real hook path: a raw payload on stdin, ``hook run <runtime> --connector resources-graph``."""

from __future__ import annotations

import io
import json
import os

import pytest

from agent_context_graph.adapters import _identity
from agent_context_graph.hooks.cli import main as hook_main
from resources_graph.address import parse_address
from resources_graph.models import FETCHED
from resources_graph.sweep import sweep


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


def run_hook(monkeypatch, payload, runtime="claude-code"):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return hook_main(["run", runtime, "--connector", "resources-graph", "--strict"])


def pre_tool_use(command, tool_use_id="toolu_n"):
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "cc-1",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_use_id": tool_use_id,
    }


@pytest.fixture()
def remembered(graph, source):
    """memgraph/memgraph#2000 and memgraph/gqlalchemy's open issues, already in memory."""
    for text in ("memgraph/memgraph#2000", "gh issue list -R memgraph/gqlalchemy --limit 100"):
        address = parse_address(text)
        assert address is not None
        graph.record_touch("earlier", address, FETCHED, discriminator=text)
    sweep(graph, source)


@pytest.mark.parametrize("runtime", ["claude-code", "codex"])
def test_the_nudge_tells_the_model_and_the_fetch_still_runs(
    graph, hook_config, remembered, monkeypatch, capsys, runtime
):
    assert run_hook(monkeypatch, pre_tool_use("gh issue view 2000 -R memgraph/memgraph"), runtime) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": output["hookSpecificOutput"]["additionalContext"],
        }
    }
    nudge = output["hookSpecificOutput"]["additionalContext"]
    assert "github:memgraph/memgraph#2000" in nudge and "fetched" in nudge and "`resource`" in nudge
    assert "permissionDecision" not in output["hookSpecificOutput"]  # never a block or a rewrite
    rows = graph._db.query("MATCH (t:Touch {tool_use_id: 'toolu_n'}) RETURN t.status AS status")
    assert rows == [{"status": "pending"}]  # the real fetch is recorded as usual


def test_a_narrower_listing_is_nudged_from_the_broader_one(graph, hook_config, remembered, monkeypatch, capsys):
    assert run_hook(monkeypatch, pre_tool_use("gh issue list -R memgraph/gqlalchemy --label bug")) == 0

    nudge = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "github:memgraph/gqlalchemy/issues?labels=bug&state=open (filtered from a broader stored listing)" in nudge


def test_no_nudge_for_what_memory_does_not_hold(graph, hook_config, remembered, monkeypatch, capsys):
    assert run_hook(monkeypatch, pre_tool_use("gh issue view 1 -R memgraph/memgraph")) == 0

    assert capsys.readouterr().out.strip() == ""


def test_harnesses_that_cannot_add_pre_tool_context_get_no_nudge(graph, hook_config, remembered, monkeypatch, capsys):
    payload = {
        "hook_event_name": "preToolUse",
        "sessionId": "copilot-1",
        "toolName": "bash",
        "toolArgs": json.dumps({"command": "gh issue view 2000 -R memgraph/memgraph"}),
    }

    assert run_hook(monkeypatch, payload, "copilot-cli") == 0

    assert "additionalContext" not in capsys.readouterr().out
    touched = graph._db.query(
        "MATCH (t:Touch)-[:IN_SESSION]->(:Session {session_id: 'copilot-1'}) RETURN t.address AS a"
    )
    assert touched == [{"a": "github:memgraph/memgraph#2000"}]  # the Touch is still recorded


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
