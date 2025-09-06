"""
Abstract analyzer interface for database systems.

This module defines the abstract base class that all database analyzers
must implement to ensure compatibility with the migration system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .models import (
    DatabaseStructure,
    TableInfo,
    ColumnInfo,
    ForeignKeyInfo,
    RelationshipInfo,
    TableType,
)


class DatabaseAnalyzer(ABC):
    """
    Abstract base class for database analyzers.

    All database-specific analyzers must implement this interface to ensure
    compatibility with HyGM and the migration system.
    """

    def __init__(self, connection_config: Dict[str, Any]):
        """
        Initialize the database analyzer.

        Args:
            connection_config: Database-specific connection configuration
        """
        self.connection_config = connection_config
        self.connection = None
        self.database_type = self._get_database_type()

    @abstractmethod
    def _get_database_type(self) -> str:
        """Return the type of database (e.g., 'mysql', 'postgresql')."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.

        Returns:
            List of table names
        """
        pass

    @abstractmethod
    def get_table_schema(self, table_name: str) -> List[ColumnInfo]:
        """
        Get schema information for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            List of ColumnInfo objects describing the table schema
        """
        pass

    @abstractmethod
    def get_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """
        Get foreign key relationships for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of ForeignKeyInfo objects
        """
        pass

    @abstractmethod
    def get_table_data(
        self, table_name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get data from a specific table.

        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return

        Returns:
            List of dictionaries representing rows
        """
        pass

    @abstractmethod
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        pass

    @abstractmethod
    def is_view(self, table_name: str) -> bool:
        """
        Check if a table is actually a view.

        Args:
            table_name: Name of the table

        Returns:
            True if the table is a view, False otherwise
        """
        pass

    @abstractmethod
    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get indexes for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            List of dictionaries representing the table's indexes
        """
        pass

    def is_connected(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            True if connected, False otherwise
        """
        return self.connection is not None

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information (excluding sensitive data like passwords).

        Returns:
            Dictionary with connection information
        """
        safe_config = self.connection_config.copy()
        if "password" in safe_config:
            safe_config["password"] = "***"
        return {
            "database_type": self.database_type,
            "config": safe_config,
            "connected": self.is_connected(),
        }

    def get_migration_config(self) -> Dict[str, str]:
        """
        Get connection config formatted for migration tools.

        Returns:
            Dictionary with string values suitable for migration tools
        """
        config = self.connection_config.copy()

        # Ensure all values are strings for compatibility
        migration_config = {}
        for key, value in config.items():
            if key == "password" and value is None:
                migration_config[key] = ""
            else:
                migration_config[key] = str(value)

        return migration_config

    def is_join_table(self, table_info: TableInfo) -> bool:
        """
        Determine if a table is a join table (many-to-many).

        This implementation is database-agnostic and can be overridden
        if database-specific logic is needed.

        Args:
            table_info: TableInfo object

        Returns:
            True if the table is a join table, False otherwise
        """
        # A join table typically has:
        # 1. Only foreign key columns (and maybe an ID or timestamp)
        # 2. At least 2 foreign keys
        # 3. Small number of total columns

        if len(table_info.foreign_keys) < 2:
            return False

        # Count non-FK columns (excluding common metadata columns)
        non_fk_columns = []
        fk_column_names = {fk.column_name for fk in table_info.foreign_keys}
        metadata_columns = {
            "id",
            "created_at",
            "updated_at",
            "created_on",
            "updated_on",
            "timestamp",
        }

        for col in table_info.columns:
            field_name = col.name.lower()
            if col.name not in fk_column_names and field_name not in metadata_columns:
                non_fk_columns.append(col.name)

        # If most columns are foreign keys, it's likely a join table
        total_columns = len(table_info.columns)
        fk_ratio = len(table_info.foreign_keys) / total_columns

        # Consider it a join table if:
        # - At least 2 FKs and FK ratio > 0.5, OR
        # - All columns are FKs or metadata columns
        return (len(table_info.foreign_keys) >= 2 and fk_ratio > 0.5) or len(
            non_fk_columns
        ) == 0

    def determine_table_type(self, table_info: TableInfo) -> TableType:
        """
        Determine the type of table.

        Args:
            table_info: TableInfo object

        Returns:
            TableType enum value
        """
        # Check if it's a view first
        if self.is_view(table_info.name):
            return TableType.VIEW

        if self.is_join_table(table_info):
            return TableType.JOIN
        elif len(table_info.foreign_keys) == 0:
            return TableType.ENTITY  # Pure entity table with no references
        else:
            return TableType.ENTITY  # Entity table with references

    def get_database_structure(self) -> DatabaseStructure:
        """
        Get complete database structure including tables, schemas, and relationships.

        This method provides a standardized database structure that works
        with HyGM regardless of the underlying database system.

        Returns:
            DatabaseStructure object containing all database information
        """
        tables = {}
        entity_tables = {}
        join_tables = {}
        view_tables = {}
        relationships = []
        sample_data = {}
        table_counts = {}

        # Get all tables
        all_table_names = self.get_tables()

        # First pass: collect table information
        for table_name in all_table_names:
            columns = self.get_table_schema(table_name)
            foreign_keys = self.get_foreign_keys(table_name)
            row_count = self.get_table_row_count(table_name)

            # Get primary keys
            primary_keys = [col.name for col in columns if col.is_primary_key]

            # Get indexes for this table
            try:
                table_indexes = self.get_indexes(table_name)
            except (NotImplementedError, AttributeError):
                # If get_indexes is not implemented, use empty list
                table_indexes = []

            # Create TableInfo object
            table_info = TableInfo(
                name=table_name,
                table_type=TableType.ENTITY,  # Will be determined later
                columns=columns,
                foreign_keys=foreign_keys,
                row_count=row_count,
                primary_keys=primary_keys,
                indexes=table_indexes,
            )

            # Determine table type
            table_info.table_type = self.determine_table_type(table_info)

            tables[table_name] = table_info
            table_counts[table_name] = row_count

            # Categorize tables
            if table_info.table_type == TableType.VIEW:
                view_tables[table_name] = table_info
            elif table_info.table_type == TableType.JOIN:
                join_tables[table_name] = table_info
            else:
                entity_tables[table_name] = table_info

            # Get sample data (limit to 3 rows for performance)
            try:
                sample_data[table_name] = self.get_table_data(table_name, limit=3)
            except Exception:
                sample_data[table_name] = []

        # Second pass: create relationships
        for table_name, table_info in tables.items():
            if table_info.table_type == TableType.JOIN:
                # Handle join tables as many-to-many relationships
                fks = table_info.foreign_keys
                if len(fks) >= 2:
                    # Create a many-to-many relationship
                    fk1, fk2 = fks[0], fks[1]

                    # Get additional properties from non-FK columns
                    fk_columns = {fk.column_name for fk in fks}
                    additional_properties = []
                    metadata_columns = {
                        "id",
                        "created_at",
                        "updated_at",
                        "created_on",
                        "updated_on",
                        "timestamp",
                    }

                    for col in table_info.columns:
                        if (
                            col.name not in fk_columns
                            and col.name.lower() not in metadata_columns
                        ):
                            additional_properties.append(col.name)

                    relationships.append(
                        RelationshipInfo(
                            relationship_type="many_to_many",
                            from_table=fk1.referenced_table,
                            from_column=fk1.referenced_column,
                            to_table=fk2.referenced_table,
                            to_column=fk2.referenced_column,
                            join_table=table_name,
                            join_from_column=fk1.column_name,
                            join_to_column=fk2.column_name,
                            additional_properties=additional_properties,
                        )
                    )
            else:
                # Handle regular foreign key relationships
                for fk in table_info.foreign_keys:
                    relationships.append(
                        RelationshipInfo(
                            relationship_type="one_to_many",
                            from_table=table_name,
                            from_column=fk.column_name,
                            to_table=fk.referenced_table,
                            to_column=fk.referenced_column,
                        )
                    )

        return DatabaseStructure(
            tables=tables,
            entity_tables=entity_tables,
            join_tables=join_tables,
            view_tables=view_tables,
            relationships=relationships,
            sample_data=sample_data,
            table_counts=table_counts,
            database_name=self.connection_config.get("database", "unknown"),
            database_type=self.database_type,
        )
