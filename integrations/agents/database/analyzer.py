"""
Abstract analyzer interface for database systems.

This module defines the abstract base class that all database analyzers
must implement to ensure compatibility with the migration system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .models import DatabaseStructure, TableInfo, ColumnInfo, ForeignKeyInfo, TableType


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
