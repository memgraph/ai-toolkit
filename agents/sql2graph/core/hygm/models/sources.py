"""
Source tracking models for Hypothetical Graph Modeling (HyGM).

These models track the origin and mapping of graph elements to their source data.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PropertySource:
    """Source information for a property."""

    field: str  # Source field name (e.g., "users.name")
    transformation: str | None = None  # Any transformation applied


@dataclass
class NodeSource:
    """Source information for a node."""

    type: str  # "table", "view", "file", "api", "manual"
    name: str  # Source name (e.g., table name)
    location: str  # Full location path
    mapping: dict[str, Any]  # Mapping details for labels, id_field, etc.


@dataclass
class RelationshipSource:
    """Source information for a relationship."""

    type: str  # "table", "view", "many_to_many", "derived"
    name: str  # Source name
    location: str  # Full location path
    mapping: dict[str, Any]  # Mapping for start_node, end_node, edge_type


@dataclass
class IndexSource:
    """Source information for an index."""

    origin: str  # "source_database_index", "migration_requirement", etc.
    reason: str  # Why this index was created
    created_by: str  # Who/what created this index
    index_name: str | None = None  # Original index name if applicable
    migrated_from: str | None = None  # Source location


@dataclass
class ConstraintSource:
    """Source information for a constraint."""

    origin: str  # "source_database_constraint", "migration_requirement", etc.
    constraint_name: str | None = None  # Original constraint name
    migrated_from: str | None = None  # Source location
    reason: str | None = None  # Why this constraint exists
    created_by: str | None = None  # Who/what created this constraint


@dataclass
class EnumSource:
    """Source information for an enum."""

    origin: str  # "source_database_enum", "detected_from_data", etc.
    enum_name: str | None = None  # Original enum name if applicable
    migrated_from: str | None = None  # Source location
    created_by: str | None = None  # Who/what created this enum
