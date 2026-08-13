"""Typed result shapes for the graph-projection tool.

FastMCP derives the tool's ``outputSchema`` and populates ``structuredContent``
from these return-type annotations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """A node in a graph projection."""

    id: str
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """A relationship in a graph projection."""

    id: str
    type: str
    start: str | None = None
    end: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphResult:
    """A normalized graph projection: deduplicated nodes and relationships."""

    nodes: list[GraphNode] = field(default_factory=list)
    relationships: list[GraphRelationship] = field(default_factory=list)
    error: str | None = None
