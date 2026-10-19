"""The graph connectors hooks can enable, and which of them a hook enables.

One table, so the hook runner, schema setup, ``doctor`` and ``setup`` agree on
what each connector is and which run by default.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.tools import normalize_connector_name

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class Component:
    """A graph component's import path: its graph class and the connector that writes through it."""

    module: str
    graph_class: str
    connector_module: str
    connector_class: str

    def graph(self) -> type[Any]:
        """The graph class (``SessionsGraph``)."""
        return getattr(importlib.import_module(self.module), self.graph_class)

    def connector(self) -> type[Any]:
        """The connector class (``SessionsGraphConnector``)."""
        return getattr(importlib.import_module(self.connector_module), self.connector_class)


COMPONENTS: dict[str, Component] = {
    "skills-graph": Component("skills_graph", "SkillGraph", "skills_graph.connector", "SkillGraphConnector"),
    "actions-graph": Component("actions_graph", "ActionsGraph", "actions_graph.connector", "ActionsGraphConnector"),
    "sessions-graph": Component(
        "sessions_graph", "SessionsGraph", "sessions_graph.connector", "SessionsGraphConnector"
    ),
    "resources-graph": Component(
        "resources_graph", "ResourcesGraph", "resources_graph.connector", "ResourcesGraphConnector"
    ),
}

#: What a hook enables when neither ``--connector`` nor ``[capture] connectors`` names any,
#: and what setup installs and wires.
DEFAULT_CONNECTORS: tuple[str, ...] = ("skills-graph", "actions-graph", "sessions-graph")


def supported_connectors(connector_names: Iterable[str]) -> list[str]:
    """*connector_names* normalized to distribution form (``actions-graph``), blanks dropped.

    Raises:
        ValueError: if a name is not a supported connector.
    """
    names = [normalize_connector_name(name) for name in connector_names if name.strip()]
    for name in names:
        if name not in COMPONENTS:
            raise ValueError(f"Unsupported connector: {name}; choose from {', '.join(COMPONENTS)}")
    return names


def configured_connectors() -> list[str]:
    """The config file's ``[capture] connectors``, else :data:`DEFAULT_CONNECTORS`."""
    from memgraph_context_graph.adapters._identity import load_config

    configured = load_config().connectors
    return list(configured if configured is not None else DEFAULT_CONNECTORS)
