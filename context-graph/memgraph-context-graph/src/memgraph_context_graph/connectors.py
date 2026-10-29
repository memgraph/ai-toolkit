"""The graph connectors hooks can enable, and which of them a hook enables.

A component registers its connector under the ``memgraph_context_graph.connectors``
entry-point group, in its own ``pyproject.toml``::

    [project.entry-points."memgraph_context_graph.connectors"]
    sessions-graph = "sessions_graph.connector:CONNECTOR"

The value is a :class:`ConnectorSpec`: how to build the component's graph and
connector, and what ``doctor`` checks beyond a reachable Memgraph. The hook
runner, schema setup and ``doctor`` read only that.

What the hub has to know before a component is installed stays here:
:data:`INSTALLABLE`, the requirement ``setup`` installs each connector from,
and :data:`DEFAULT_CONNECTORS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.tools import normalize_connector_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from memgraph_context_graph.adapters._identity import HookConfig
    from memgraph_context_graph.protocols import GraphConnector

_ENTRY_POINT_GROUP = "memgraph_context_graph.connectors"


@dataclass(frozen=True)
class ConnectorSpec:
    """What a component registers about its connector.

    Attributes:
        name: The distribution name the connector is enabled by (``sessions-graph``).
        graph: The graph class, called with a Memgraph client or with its
            connection keyword arguments.
        build: Makes the connector over a graph, from the config file's values.
        check: What ``doctor`` checks beyond a reachable Memgraph: ``(ok, detail)``,
            where detail is added to the check line, or says why it fails.
        settings: The config sections whose keys the component reads, each with the parser it
            reads them with; ``config set`` runs it on the would-be section and refuses a value
            it raises ValueError on, so a bad value is caught when it is typed, not when a hook runs.
    """

    name: str
    graph: Callable[..., Any]
    build: Callable[[Any, HookConfig], GraphConnector]
    check: Callable[[HookConfig], tuple[bool, str]] | None = None
    settings: Mapping[str, Callable[[dict[str, str]], object]] = field(default_factory=dict)


#: The requirement each connector ``setup`` can install is installed from.
#: Kept in step with the plugins' bootstrap scripts and install.sh, which install the default connectors first.
INSTALLABLE: dict[str, str] = {
    "skills-graph": "skills-graph[memgraph-context-graph]>=0.4.0",
    "actions-graph": "actions-graph[memgraph-context-graph]>=0.7.1",
    "sessions-graph": "sessions-graph[memgraph-context-graph]>=0.12.2",
    "resources-graph": "resources-graph[memgraph-context-graph]>=0.3.0",
}

#: What a hook enables when neither ``--connector`` nor ``[capture] connectors`` names any,
#: and what setup installs and wires.
DEFAULT_CONNECTORS: tuple[str, ...] = ("skills-graph", "actions-graph", "sessions-graph")


def load_connectors() -> dict[str, ConnectorSpec]:
    """Every installed component's registered connector, keyed by name."""
    specs: dict[str, ConnectorSpec] = {}
    for entry_point in metadata.entry_points(group=_ENTRY_POINT_GROUP):
        spec = entry_point.load()
        specs[spec.name] = spec
    return specs


def connector_spec(name: str) -> ConnectorSpec:
    """The registered connector *name*.

    Raises:
        ImportError: no installed component registers it.
    """
    spec = load_connectors().get(name)
    if spec is None:
        raise ImportError(f"{name} is required for the {name} connector")
    return spec


def known_connectors() -> list[str]:
    """Every connector a name may refer to: the installable ones, then any other installed component's."""
    return [*INSTALLABLE, *(name for name in load_connectors() if name not in INSTALLABLE)]


def supported_connectors(connector_names: Iterable[str]) -> list[str]:
    """*connector_names* normalized to distribution form (``actions-graph``), blanks dropped.

    Raises:
        ValueError: if a name is not a known connector.
    """
    names = [normalize_connector_name(name) for name in connector_names if name.strip()]
    # Only an unfamiliar name pays for scanning the installed packages' entry points.
    if any(name not in INSTALLABLE for name in names):
        known = known_connectors()
        for name in names:
            if name not in known:
                raise ValueError(f"Unsupported connector: {name}; choose from {', '.join(known)}")
    return names


def check_settings(section: str, settings: dict[str, str]) -> None:
    """Parse *settings* as every installed component that reads ``[section]`` would.

    Raises:
        ValueError: a component rejects them; the message is its own.
    """
    for spec in load_connectors().values():
        parse = spec.settings.get(section)
        if parse is not None:
            parse(settings)


def configured_connectors() -> list[str]:
    """The config file's ``[capture] connectors``, else :data:`DEFAULT_CONNECTORS`."""
    from memgraph_context_graph.adapters._identity import load_config

    configured = load_config().connectors
    return list(configured if configured is not None else DEFAULT_CONNECTORS)
