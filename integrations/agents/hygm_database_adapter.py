"""
Database Data Interface for HyGM consumption.

This module provides a clean data interface between database analyzers
and HyGM, enabling HyGM to work with any SQL database system through
standardized data structures.
"""

from typing import Dict, Any
from database_analyzer_interface import (
    DatabaseStructure,
    TableInfo,
    ColumnInfo,
)


class DatabaseDataInterface:
    """
    Data interface for HyGM to consume database structures.

    This interface provides clean, standardized data structures that HyGM
    can naturally consume to create graph models from any SQL database system.
    """

    @staticmethod
    def get_hygm_data_structure(
        db_structure: DatabaseStructure,
    ) -> Dict[str, Any]:
        """
        Get database structure in HyGM's expected data format.

        Args:
            db_structure: DatabaseStructure from any database analyzer

        Returns:
            Dictionary in the format HyGM expects for graph modeling
        """
        # Prepare tables in HyGM data format
        tables = {}
        entity_tables = {}
        join_tables = {}
        views = {}

        for table_name, table_info in db_structure.tables.items():
            hygm_table = DatabaseDataInterface._format_table_for_hygm(table_info)
            tables[table_name] = hygm_table

            # Categorize tables based on type
            if table_info.table_type.value == "view":
                views[table_name] = hygm_table
            elif table_info.table_type.value == "join":
                join_tables[table_name] = hygm_table
            else:
                entity_tables[table_name] = hygm_table

        # Prepare relationships in HyGM data format
        relationships = []
        for rel in db_structure.relationships:
            hygm_rel = DatabaseDataInterface._format_relationship_for_hygm(rel)
            relationships.append(hygm_rel)

        return {
            "tables": tables,
            "entity_tables": entity_tables,
            "join_tables": join_tables,
            "views": views,
            "relationships": relationships,
            "sample_data": db_structure.sample_data,
            "table_counts": db_structure.table_counts,
            "database_type": db_structure.database_type,
            "database_name": db_structure.database_name,
        }

    @staticmethod
    def connect_and_get_hygm_data(
        database_analyzer, connect_and_analyze: bool = True
    ) -> Dict[str, Any]:
        """
        Connect to database and get data structure for HyGM consumption.

        This is the primary interface for HyGM to get database structures
        from any SQL database system.

        Args:
            database_analyzer: Any DatabaseAnalyzer instance
            connect_and_analyze: Whether to connect and analyze immediately

        Returns:
            Data structure ready for HyGM graph modeling
        """
        if connect_and_analyze:
            if not database_analyzer.connect():
                raise ConnectionError(
                    f"Failed to connect to {database_analyzer.database_type} "
                    f"database"
                )

        # Get standardized structure
        db_structure = database_analyzer.get_database_structure()

        # Format for HyGM consumption
        hygm_data = DatabaseDataInterface.get_hygm_data_structure(db_structure)

        return hygm_data

    @staticmethod
    def _format_table_for_hygm(table_info: TableInfo) -> Dict[str, Any]:
        """Format TableInfo for HyGM data consumption."""
        # Format columns for HyGM
        schema = []
        for col in table_info.columns:
            hygm_col = DatabaseDataInterface._format_column_for_hygm(col)
            schema.append(hygm_col)

        # Format foreign keys for HyGM
        foreign_keys = []
        for fk in table_info.foreign_keys:
            hygm_fk = {
                "column": fk.column_name,
                "referenced_table": fk.referenced_table,
                "referenced_column": fk.referenced_column,
            }
            if fk.constraint_name:
                hygm_fk["constraint_name"] = fk.constraint_name
            foreign_keys.append(hygm_fk)

        return {
            "schema": schema,
            "foreign_keys": foreign_keys,
            "type": table_info.table_type.value,
            "row_count": table_info.row_count,
            "primary_keys": table_info.primary_keys,
            "indexes": table_info.indexes,
        }

    @staticmethod
    def _format_column_for_hygm(col: ColumnInfo) -> Dict[str, Any]:
        """Format ColumnInfo for HyGM data consumption."""
        # Determine key type
        key_type = ""
        if col.is_primary_key:
            key_type = "PRI"
        elif col.is_foreign_key:
            key_type = "MUL"  # MySQL convention for foreign keys

        # Determine null constraint
        null_constraint = "NO" if not col.is_nullable else "YES"

        # Build type string with length/precision info
        type_str = col.data_type
        if col.max_length:
            type_str += f"({col.max_length})"
        elif col.precision and col.scale:
            type_str += f"({col.precision},{col.scale})"
        elif col.precision:
            type_str += f"({col.precision})"

        # Build extra field
        extra = ""
        if col.auto_increment:
            extra = "auto_increment"

        return {
            "field": col.name,
            "type": type_str,
            "null": null_constraint,
            "key": key_type,
            "default": col.default_value,
            "extra": extra,
        }

    @staticmethod
    def _format_relationship_for_hygm(rel) -> Dict[str, Any]:
        """Format RelationshipInfo for HyGM data consumption."""
        hygm_rel = {
            "type": rel.relationship_type,
            "from_table": rel.from_table,
            "from_column": rel.from_column,
            "to_table": rel.to_table,
            "to_column": rel.to_column,
        }

        # Add many-to-many specific fields
        if rel.relationship_type == "many_to_many":
            hygm_rel.update(
                {
                    "join_table": rel.join_table,
                    "join_from_column": rel.join_from_column,
                    "join_to_column": rel.join_to_column,
                    "additional_properties": rel.additional_properties or [],
                }
            )

        return hygm_rel

    @staticmethod
    def create_connection_config_for_migration(analyzer) -> Dict[str, str]:
        """
        Create connection config dictionary for migration tools.

        This method extracts connection information in a format suitable
        for migration tools that may expect specific configuration formats.

        Args:
            analyzer: DatabaseAnalyzer instance

        Returns:
            Dictionary with connection configuration
        """
        config = analyzer.connection_config.copy()

        # Ensure all values are strings for compatibility
        migration_config = {}
        for key, value in config.items():
            if key == "password" and value is None:
                migration_config[key] = ""
            else:
                migration_config[key] = str(value)

        return migration_config

    @staticmethod
    def prepare_for_migration_agent(
        db_structure: DatabaseStructure, analyzer
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """
        Prepare database structure and analyzer for use with migration agent.

        Args:
            db_structure: Standardized database structure
            analyzer: Database analyzer instance

        Returns:
            Tuple of (hygm_compatible_structure, migration_compatible_config)
        """
        hygm_structure = DatabaseDataInterface.get_hygm_data_structure(db_structure)
        migration_config = DatabaseDataInterface.create_connection_config_for_migration(
            analyzer
        )

        return hygm_structure, migration_config
