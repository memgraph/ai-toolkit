"""Unit tests for lightrag_memgraph.core's env-name bridging.

These don't need a live Memgraph: they only check what _bridge_lightrag_env_names
writes into os.environ.
"""

from __future__ import annotations

import os

import pytest

from lightrag_memgraph.core import _bridge_lightrag_env_names

_MEMGRAPH_ENV_NAMES = ("MEMGRAPH_URL", "MEMGRAPH_URI", "MEMGRAPH_USER", "MEMGRAPH_USERNAME")


@pytest.fixture(autouse=True)
def _clear_memgraph_env(monkeypatch):
    """Isolate every test from both the ambient environment and each other.

    _bridge_lightrag_env_names writes to os.environ directly (not through
    monkeypatch), so teardown must scrub it explicitly or a value set by one
    test would leak into the next.
    """
    for name in _MEMGRAPH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in _MEMGRAPH_ENV_NAMES:
        os.environ.pop(name, None)


def test_bridge_defaults_uri_to_local_memgraph_when_nothing_set():
    """Regression test for #217: with zero Memgraph env vars set, MEMGRAPH_URI
    must still end up pointing at the local default Memgraph, matching the
    behaviour on main (which defaulted it unconditionally at import time).
    """
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_URI"] == "bolt://localhost:7687"


def test_bridge_mirrors_url_onto_uri(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_URL", "bolt://example:1234")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_URI"] == "bolt://example:1234"


def test_bridge_never_overwrites_explicit_uri(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_URL", "bolt://example:1234")
    monkeypatch.setenv("MEMGRAPH_URI", "bolt://explicit:9999")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_URI"] == "bolt://explicit:9999"


def test_bridge_mirrors_user_onto_username(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_USER", "alice")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_USERNAME"] == "alice"


def test_bridge_never_overwrites_explicit_username(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_USER", "alice")
    monkeypatch.setenv("MEMGRAPH_USERNAME", "bob")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_USERNAME"] == "bob"
