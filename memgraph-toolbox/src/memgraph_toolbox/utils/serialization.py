"""
Serialization utilities for Neo4j date/time types.
"""

from typing import Any


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
