"""Failures of the detached children a hook starts reach the capture-failure log, not nowhere.

A hook starts ``sessions-graph embed``/``reconcile`` with stderr discarded, so
the log is the only place a failure before the Session can be updated shows up.
"""

import json

import pytest
from sessions_graph import connector as connector_module
from sessions_graph.cli import main

from memgraph_context_graph import capture_log
from memgraph_context_graph.adapters import _identity


def _log_lines():
    log = capture_log.path()
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_a_child_that_cannot_start_is_logged(monkeypatch):
    # The one branch a real Memgraph can't trigger: the OS refusing to start the process.
    def refuse(*args, **kwargs):
        raise FileNotFoundError("no such executable")

    monkeypatch.setattr(connector_module.subprocess, "Popen", refuse)

    connector_module._spawn_detached(["embed", "--session", "s-1"], env={})

    (line,) = _log_lines()
    assert (line["stage"], line["connector"], line["session_id"]) == ("spawn", "sessions-graph", "s-1")
    assert line["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize("command", ["embed", "reconcile"])
def test_a_child_that_cannot_reach_memgraph_is_logged(monkeypatch, command):
    _identity.write_config(user_id="u", memgraph_url="bolt://127.0.0.1:1")
    _identity._reset_cache()
    # Explicit, so the env the CLI fills from config is restored after the test.
    monkeypatch.setenv("MEMGRAPH_URL", "bolt://127.0.0.1:1")

    with pytest.raises(ValueError, match="Could not connect"):
        main([command, "--session", "s-2"])

    (line,) = _log_lines()
    assert (line["stage"], line["connector"], line["session_id"]) == (command, "sessions-graph", "s-2")
    assert line["cause_type"].startswith("neo4j.exceptions.")
