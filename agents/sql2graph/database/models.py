"""
Database data models.

This module contains all the data structures used to represent
database schema information in a standardized way.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional


class TableType(Enum):
    """Enumeration of table types."""

    ENTITY = "entity"
    JOIN = "join"
    VIEW = "view"
    LOOKUP = "lookup"


@dataclass
class ColumnInfo:
    """Standardized column information across different database systems."""

    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    default_value: Optional[Any] = None
    auto_increment: bool = False
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None

    def to_hygm_format(self) -> Dict[str, Any]:
        """Convert to HyGM-compatible format."""
        # Determine key type
        key_type = ""
        if self.is_primary_key:
            key_type = "PRI"
        elif self.is_foreign_key:
            key_type = "MUL"  # MySQL convention for foreign keys

        # Determine null constraint
        null_constraint = "NO" if not self.is_nullable else "YES"

        # Build type string with length/precision info
        type_str = self.data_type
        if self.max_length:
            type_str += f"({self.max_length})"
        elif self.precision and self.scale:
            type_str += f"({self.precision},{self.scale})"
        elif self.precision:
            type_str += f"({self.precision})"

        # Build extra field
        extra = ""
        if self.auto_increment:
            extra = "auto_increment"

        return {
            "field": self.name,
            "type": type_str,
            "null": null_constraint,
            "key": key_type,
            "default": self.default_value,
            "extra": extra,
        }


@dataclass
class ForeignKeyInfo:
    """Standardized foreign key information."""

    column_name: str
    referenced_table: str
    referenced_column: str
    constraint_name: Optional[str] = None

    def to_hygm_format(self) -> Dict[str, Any]:
        """Convert to HyGM-compatible format."""
        hygm_fk = {
            "column": self.column_name,
            "referenced_table": self.referenced_table,
            "referenced_column": self.referenced_column,
        }
        if self.constraint_name:
            hygm_fk["constraint_name"] = self.constraint_name
        return hygm_fk


@dataclass
class TableInfo:
    """Standardized table information."""

    name: str
    table_type: TableType
    columns: List[ColumnInfo]
    foreign_keys: List[ForeignKeyInfo]
    row_count: int
    primary_keys: List[str]
    indexes: List[Dict[str, Any]]

    def to_hygm_format(self) -> Dict[str, Any]:
        """Convert to HyGM-compatible format."""
        # Format columns for HyGM
        schema = [col.to_hygm_format() for col in self.columns]

        # Format foreign keys for HyGM
        foreign_keys = [fk.to_hygm_format() for fk in self.foreign_keys]

        return {
            "schema": schema,
            "foreign_keys": foreign_keys,
            "type": self.table_type.value,
            "row_count": self.row_count,
            "primary_keys": self.primary_keys,
            "indexes": self.indexes,
        }


@dataclass
class RelationshipInfo:
    """Standardized relationship information."""

    relationship_type: str  # "one_to_many", "many_to_many", "one_to_one"
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    join_table: Optional[str] = None
    join_from_column: Optional[str] = None
    join_to_column: Optional[str] = None
    additional_properties: Optional[List[str]] = None

    def to_hygm_format(self) -> Dict[str, Any]:
        """Convert to HyGM-compatible format."""
        hygm_rel = {
            "type": self.relationship_type,
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
        }

        # Add many-to-many specific fields
        if self.relationship_type == "many_to_many":
            hygm_rel["join_table"] = self.join_table
            hygm_rel["join_from_column"] = self.join_from_column
            hygm_rel["join_to_column"] = self.join_to_column
            hygm_rel["additional_properties"] = self.additional_properties or []

        return hygm_rel


@dataclass
class DatabaseStructure:
    """Standardized database structure representation."""

    tables: Dict[str, TableInfo]
    entity_tables: Dict[str, TableInfo]
    join_tables: Dict[str, TableInfo]
    view_tables: Dict[str, TableInfo]
    relationships: List[RelationshipInfo]
    sample_data: Dict[str, List[Dict[str, Any]]]
    table_counts: Dict[str, int]
    database_name: str
    database_type: str

    def to_hygm_format(self) -> Dict[str, Any]:
        """
        Convert to HyGM-compatible format.

        This replaces the need for DatabaseDataInterface.get_hygm_data_structure()
        """
        # Convert tables to HyGM format
        hygm_tables = {}
        hygm_entity_tables = {}
        hygm_join_tables = {}
        hygm_views = {}

        for table_name, table_info in self.tables.items():
            hygm_table = table_info.to_hygm_format()
            hygm_tables[table_name] = hygm_table

            # Categorize based on existing categorization
            if table_name in self.view_tables:
                hygm_views[table_name] = hygm_table
            elif table_name in self.join_tables:
                hygm_join_tables[table_name] = hygm_table
            else:
                hygm_entity_tables[table_name] = hygm_table

        # Convert relationships to HyGM format
        hygm_relationships = [rel.to_hygm_format() for rel in self.relationships]

        return {
            "tables": hygm_tables,
            "entity_tables": hygm_entity_tables,
            "join_tables": hygm_join_tables,
            "views": hygm_views,
            "relationships": hygm_relationships,
            "sample_data": self.sample_data,
            "table_counts": self.table_counts,
            "database_type": self.database_type,
            "database_name": self.database_name,
        }
