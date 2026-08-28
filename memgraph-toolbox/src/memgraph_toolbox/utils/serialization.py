"""
Serialization utilities for Neo4j date/time types.
"""

from collections.abc import Iterable
from typing import Any

from neo4j.graph import Node, Path, Relationship
from neo4j.spatial import Point


def serialize_neo4j_types(value: Any) -> Any:
    """
    Convert Neo4j date/time types to JSON-serializable strings.

    Args:
        value: The value to serialize

    Returns:
        The serialized value with Neo4j types converted to strings
    """
    try:
        class_name = value.__class__.__name__
        module_name = getattr(value.__class__, "__module__", "")

        # Check if it's a Neo4j temporal type
        if "neo4j" in module_name and hasattr(value, "iso_format"):
            neo4j_temporal_types = [
                "Date",
                "Time",
                "DateTime",
                "LocalTime",
                "LocalDateTime",
            ]
            if class_name in neo4j_temporal_types:
                return value.iso_format()

        # Handle Neo4j Duration type
        if "neo4j" in module_name and class_name == "Duration":
            return str(value)

    except (AttributeError, TypeError):
        # If we can't access the class name or module, return as-is
        pass

    return value


def serialize_record_data(record_data: dict) -> dict:
    """
    Serialize a single record's data, handling Neo4j types recursively.

    Args:
        record_data: Dictionary from a Neo4j record.data()

    Returns:
        Dictionary with Neo4j types serialized to JSON-safe values
    """
    serialized = {}
    for key, value in record_data.items():
        if isinstance(value, dict):
            serialized[key] = serialize_record_data(value)
        elif isinstance(value, list):
            serialized[key] = [serialize_neo4j_types(item) for item in value]
        else:
            serialized[key] = serialize_neo4j_types(value)

    return serialized


def serialize_node(node: Node) -> dict:
    """
    Project a neo4j Node into a JSON-safe node record.

    Preserves the identity that ``Record.data()`` discards: the stable
    ``element_id`` and ``labels``. Property values reuse
    :func:`serialize_record_data` so temporal/duration types stay JSON-safe.
    """
    return {
        "_type": "node",
        "id": node.element_id,
        "labels": sorted(node.labels),
        "properties": serialize_record_data(dict(node)),
    }


def serialize_relationship(rel: Relationship) -> dict:
    """
    Project a neo4j Relationship into a JSON-safe edge record.

    Keeps the type and the start/end endpoints, which ``Record.data()`` drops.
    ``start``/``end`` are the endpoint nodes' ``element_id``s, or ``None`` when
    the driver did not hydrate an endpoint.
    """
    start = rel.start_node
    end = rel.end_node
    return {
        "_type": "relationship",
        "id": rel.element_id,
        "type": rel.type,
        "start": start.element_id if start is not None else None,
        "end": end.element_id if end is not None else None,
        "properties": serialize_record_data(dict(rel)),
    }


def serialize_path(path: Path) -> dict:
    """Project a neo4j Path into a JSON-safe record of its nodes and relationships."""
    return {
        "_type": "path",
        "nodes": [serialize_node(node) for node in path.nodes],
        "relationships": [serialize_relationship(rel) for rel in path.relationships],
    }


def serialize_point(point: Point) -> dict:
    """Project a neo4j spatial Point into a JSON-safe record."""
    return {
        "_type": "point",
        "srid": point.srid,
        "coordinates": [float(c) for c in point],
    }


def serialize_value(value: Any) -> Any:
    """Serialize a single Cypher result value, preserving its type.

    The type-preserving counterpart to ``Record.data()`` (which flattens graph
    entities into bare property maps): nodes, relationships and paths become
    ``_type``-tagged objects with their identity and topology intact; points and
    temporals become JSON-safe structures; lists and maps recurse; primitives
    pass through.
    """
    if isinstance(value, Node):
        return serialize_node(value)
    if isinstance(value, Relationship):
        return serialize_relationship(value)
    if isinstance(value, Path):
        return serialize_path(value)
    if isinstance(value, Point):
        return serialize_point(value)
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return serialize_neo4j_types(value)


def serialize_records(records: Iterable[Any]) -> list[dict]:
    """Serialize raw neo4j records into type-preserving rows (one dict per row).

    Each row keeps its ``RETURN`` column names; each value is serialized with
    :func:`serialize_value`, so nodes/edges/paths retain their identity instead
    of collapsing to bare property maps.

    Accepts anything iterable of records (a plain ``list``, or a neo4j
    ``Result``/``AsyncResult`` consumed lazily) -- the body only ever iterates
    once and calls ``.items()`` per record, it never needs list operations.
    """
    return [{key: serialize_value(value) for key, value in record.items()} for record in records]
