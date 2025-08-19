"""
Source tracking models for Hypothetical Graph Modeling (HyGM).

These models track the origin and mapping of graph elements to their source data.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PropertySource:
    """Source information for a property."""

    field: str  # Source field name (e.g., "users.name")
    transformation: Optional[str] = None  # Any transformation applied


@dataclass
class NodeSource:
    """Source information for a node."""

    type: str  # "table", "view", "file", "api", "manual"
    name: str  # Source name (e.g., table name)
    location: str  # Full location path
    mapping: Dict[str, Any]  # Mapping details for labels, id_field, etc.


@dataclass
class RelationshipSource:
    """Source information for a relationship."""

    type: str  # "table", "view", "junction_table", "derived"
    name: str  # Source name
    location: str  # Full location path
    mapping: Dict[str, Any]  # Mapping for start_node, end_node, edge_type


@dataclass
class IndexSource:
    """Source information for an index."""

    origin: str  # "source_database_index", "migration_requirement", etc.
    reason: str  # Why this index was created
    created_by: str  # Who/what created this index
    index_name: Optional[str] = None  # Original index name if applicable
    migrated_from: Optional[str] = None  # Source location


@dataclass
class ConstraintSource:
    """Source information for a constraint."""

    origin: str  # "source_database_constraint", "migration_requirement", etc.
    constraint_name: Optional[str] = None  # Original constraint name
    migrated_from: Optional[str] = None  # Source location
    reason: Optional[str] = None  # Why this constraint exists
    created_by: Optional[str] = None  # Who/what created this constraint


@dataclass
class EnumSource:
    """Source information for an enum."""

    origin: str  # "source_database_enum", "detected_from_data", etc.
    enum_name: Optional[str] = None  # Original enum name if applicable
    migrated_from: Optional[str] = None  # Source location
    created_by: Optional[str] = None  # Who/what created this enum
