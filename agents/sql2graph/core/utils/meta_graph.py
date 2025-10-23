"""Shared helpers for HyGM meta graph summaries and keys."""

from __future__ import annotations

from typing import Any, Dict, List


def node_key(node: Any) -> str:
    """Generate a stable key for a graph node definition."""
    source = getattr(node, "source", None)
    source_name = getattr(source, "name", None)
    if source_name:
        return f"source::{source_name}"
    labels = sorted(getattr(node, "labels", []))
    return "labels::" + "|".join(labels)


def relationship_key(rel: Any) -> str:
    """Generate a stable key for a graph relationship."""
    source = getattr(rel, "source", None)
    source_name = getattr(source, "name", None)
    if source_name:
        return f"source::{source_name}"
    edge_type = getattr(rel, "edge_type", "")
    start = "|".join(sorted(getattr(rel, "start_node_labels", [])))
    end = "|".join(sorted(getattr(rel, "end_node_labels", [])))
    return f"edge::{edge_type}:{start}->{end}"


def summarize_node(node: Any) -> Dict[str, Any]:
    """Create a JSON-serializable summary for a node definition."""
    properties = sorted(
        {prop.key for prop in getattr(node, "properties", []) if hasattr(prop, "key")}
    )
    node_source = getattr(node, "source", None)
    mapping: Dict[str, Any] = {}
    source_name = None
    id_field = None
    if node_source:
        mapping = dict(getattr(node_source, "mapping", {}) or {})
        source_name = getattr(node_source, "name", None)
        id_field = mapping.get("id_field")
    return {
        "labels": sorted(getattr(node, "labels", [])),
        "properties": properties,
        "id_field": id_field,
        "source": source_name,
        "mapping": mapping,
    }


def summarize_relationship(rel: Any) -> Dict[str, Any]:
    """Create a JSON-serializable summary for a relationship."""
    rel_source = getattr(rel, "source", None)
    mapping: Dict[str, Any] = {}
    source_name = None
    source_type = None
    if rel_source:
        mapping = dict(getattr(rel_source, "mapping", {}) or {})
        source_name = getattr(rel_source, "name", None)
        source_type = getattr(rel_source, "type", None)
    start_labels = sorted(getattr(rel, "start_node_labels", []))
    end_labels = sorted(getattr(rel, "end_node_labels", []))
    start_table = mapping.get("from_table")
    end_table = mapping.get("to_table")
    if not start_table and mapping.get("start_node"):
        start_table = str(mapping["start_node"]).split(".")[0]
    if not end_table and mapping.get("end_node"):
        end_table = str(mapping["end_node"]).split(".")[0]
    return {
        "edge_type": getattr(rel, "edge_type", ""),
        "start": start_labels,
        "end": end_labels,
        "source": source_name,
        "source_type": source_type,
        "mapping": mapping,
        "start_table": start_table,
        "end_table": end_table,
        "join_table": mapping.get("join_table"),
    }


def summarize_nodes(nodes: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Build summaries for all provided nodes."""
    summaries: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        summaries[node_key(node)] = summarize_node(node)
    return summaries


def summarize_relationships(
    relationships: List[Any],
) -> Dict[str, Dict[str, Any]]:
    """Build summaries for all provided relationships."""
    summaries: Dict[str, Dict[str, Any]] = {}
    for relationship in relationships:
        summaries[relationship_key(relationship)] = summarize_relationship(relationship)
    return summaries


__all__ = [
    "node_key",
    "relationship_key",
    "summarize_node",
    "summarize_relationship",
    "summarize_nodes",
    "summarize_relationships",
]
