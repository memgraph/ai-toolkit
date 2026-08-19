"""Tests that the sessions-graph connector wiring resolves auto_reconcile from
persistent config (agent_context_graph.adapters._identity), not just the
SESSIONS_GRAPH_AUTO_RECONCILE env var read internally by SessionsGraphConnector.
"""

import sys
from types import ModuleType

import pytest

from agent_context_graph.adapters import _identity
from agent_context_graph.hooks.runner import create_link


@pytest.fixture(autouse=True)
def _reset_cache():
    _identity._reset_cache()
    yield
    _identity._reset_cache()


@pytest.fixture()
def config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "context-graph"
    config_file = config_dir / "config.toml"
    monkeypatch.setattr(_identity, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(_identity, "_CONFIG_FILE", config_file)
    return config_dir


@pytest.fixture()
def fake_sessions_graph(monkeypatch):
    """Stub the sessions_graph package so runner.py's import succeeds without it installed."""
    constructed = {}

    class _SessionsGraph:
        def __init__(self, **kwargs):
            pass

    class _SessionsGraphConnector:
        def __init__(self, graph, *, auto_reconcile=None):
            constructed["auto_reconcile"] = auto_reconcile

    fake_core = ModuleType("sessions_graph")
    fake_core.SessionsGraph = _SessionsGraph
    fake_connector = ModuleType("sessions_graph.connector")
    fake_connector.SessionsGraphConnector = _SessionsGraphConnector

    monkeypatch.setitem(sys.modules, "sessions_graph", fake_core)
    monkeypatch.setitem(sys.modules, "sessions_graph.connector", fake_connector)
    return constructed


def test_sessions_graph_connector_defaults_auto_reconcile_off(config_dir, fake_sessions_graph):
    create_link(["sessions_graph"])
    assert fake_sessions_graph["auto_reconcile"] is False


def test_sessions_graph_connector_reads_auto_reconcile_from_config(config_dir, fake_sessions_graph):
    _identity.write_full_config(auto_reconcile=True)
    _identity._reset_cache()

    create_link(["sessions_graph"])
    assert fake_sessions_graph["auto_reconcile"] is True
