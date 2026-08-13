"""
Serialization utilities for Neo4j date/time types.
"""

from typing import Any

from neo4j.graph import Node, Path, Relationship


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
        "id": rel.element_id,
        "type": rel.type,
        "start": start.element_id if start is not None else None,
        "end": end.element_id if end is not None else None,
        "properties": serialize_record_data(dict(rel)),
    }


def project_graph(records: list[Any]) -> dict[str, list[dict]]:
    """
    Walk raw neo4j records and return deduplicated nodes and relationships.

    Nodes and relationships are keyed on the driver's stable ``element_id``, so
    the same entity appearing in several rows is emitted once. Explicitly
    returned nodes take precedence over the lightweight endpoint stubs a
    relationship may carry, so a node returned in full keeps its labels and
    properties regardless of iteration order. Nodes, relationships and paths
    nested inside lists or maps are traversed too.
    """
    nodes: dict[str, dict] = {}
    relationships: dict[str, dict] = {}

    def add_node(node: Node) -> None:
        nodes[node.element_id] = serialize_node(node)

    def add_relationship(rel: Relationship) -> None:
        if rel.element_id not in relationships:
            relationships[rel.element_id] = serialize_relationship(rel)
        for endpoint in (rel.start_node, rel.end_node):
            if isinstance(endpoint, Node):
                nodes.setdefault(endpoint.element_id, serialize_node(endpoint))

    def visit(value: Any) -> None:
        if isinstance(value, Node):
            add_node(value)
        elif isinstance(value, Relationship):
            add_relationship(value)
        elif isinstance(value, Path):
            for node in value.nodes:
                add_node(node)
            for rel in value.relationships:
                add_relationship(rel)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for record in records:
        for value in record.values():
            visit(value)

    return {
        "nodes": list(nodes.values()),
        "relationships": list(relationships.values()),
    }
