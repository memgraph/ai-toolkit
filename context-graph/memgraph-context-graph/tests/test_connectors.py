"""The connector registry: entry-point registrations, and the names a hook may enable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from memgraph_context_graph import connectors
from memgraph_context_graph.connectors import INSTALLABLE, ConnectorSpec

if TYPE_CHECKING:
    from memgraph_context_graph.protocols import GraphConnector


def _never_built(graph, config) -> GraphConnector:
    raise AssertionError("no connector is built in these tests")


def _registered(monkeypatch, *names: str) -> None:
    specs = {name: ConnectorSpec(name=name, graph=object, build=_never_built) for name in names}
    monkeypatch.setattr(connectors, "load_connectors", lambda: specs)


def test_installable_names_are_supported_without_scanning_entry_points(monkeypatch):
    def scan():
        raise AssertionError("scanned entry points")

    monkeypatch.setattr(connectors, "load_connectors", scan)

    assert connectors.supported_connectors(["actions_graph", " ", "sessions-graph"]) == [
        "actions-graph",
        "sessions-graph",
    ]


def test_another_installed_components_connector_is_supported(monkeypatch):
    _registered(monkeypatch, "my-graph")

    assert connectors.supported_connectors(["my-graph"]) == ["my-graph"]
    assert connectors.known_connectors() == [*INSTALLABLE, "my-graph"]


def test_an_unknown_connector_is_rejected_naming_the_known_ones(monkeypatch):
    _registered(monkeypatch, "my-graph")

    with pytest.raises(ValueError, match=r"Unsupported connector: nope; choose from .*my-graph"):
        connectors.supported_connectors(["nope"])


def test_a_connector_no_installed_component_registers_says_what_to_install(monkeypatch):
    _registered(monkeypatch)

    with pytest.raises(ImportError, match="sessions-graph is required for the sessions-graph connector"):
        connectors.connector_spec("sessions-graph")
