"""
Database analyzer factory for creating database-specific analyzers.

This module provides a factory pattern for creating appropriate database
analyzers based on the database type or connection parameters.
"""

from typing import Dict, Any, Type
from database_analyzer_interface import DatabaseAnalyzer
from mysql_analyzer import MySQLAnalyzer


class DatabaseAnalyzerFactory:
    """Factory for creating database-specific analyzers."""

    # Registry of available analyzers
    _analyzers: Dict[str, Type[DatabaseAnalyzer]] = {
        "mysql": MySQLAnalyzer,
        # Future database types can be added here:
        # "postgresql": PostgreSQLAnalyzer,
        # "duckdb": DuckDBAnalyzer,
        # "oracle": OracleAnalyzer,
        # "sqlserver": SQLServerAnalyzer,
    }

    @classmethod
    def create_analyzer(
        cls, database_type: str, **connection_params
    ) -> DatabaseAnalyzer:
        """
        Create a database analyzer for the specified database type.

        Args:
            database_type: Type of database (mysql, postgresql, etc.)
            **connection_params: Database-specific connection parameters

        Returns:
            DatabaseAnalyzer instance

        Raises:
            ValueError: If database type is not supported
        """
        database_type = database_type.lower()

        if database_type not in cls._analyzers:
            supported_types = ", ".join(cls._analyzers.keys())
            raise ValueError(
                f"Unsupported database type: {database_type}. "
                f"Supported types: {supported_types}"
            )

        analyzer_class = cls._analyzers[database_type]

        # Create analyzer with appropriate parameters based on database type
        if database_type == "mysql":
            return analyzer_class(
                host=connection_params.get("host", "localhost"),
                user=connection_params.get("user", "root"),
                password=connection_params.get("password", ""),
                database=connection_params.get("database"),
                port=connection_params.get("port", 3306),
            )
        # Future database types can be handled here:
        # elif database_type == "postgresql":
        #     return analyzer_class(
        #         host=connection_params.get("host", "localhost"),
        #         user=connection_params.get("user", "postgres"),
        #         password=connection_params.get("password", ""),
        #         database=connection_params.get("database"),
        #         port=connection_params.get("port", 5432),
        #     )
        # elif database_type == "duckdb":
        #     return analyzer_class(
        #         database_path=connection_params.get("database_path"),
        #     )

        # This should never be reached due to the check above
        raise ValueError(f"No implementation for database type: {database_type}")

    @classmethod
    def create_from_uri(cls, database_uri: str) -> DatabaseAnalyzer:
        """
        Create a database analyzer from a database URI.

        Args:
            database_uri: Database connection URI (e.g., mysql://user:pass@host/db)

        Returns:
            DatabaseAnalyzer instance

        Raises:
            ValueError: If URI format is invalid or database type unsupported
        """
        try:
            # Parse the URI
            if "://" not in database_uri:
                raise ValueError("Invalid URI format: missing protocol")

            protocol, rest = database_uri.split("://", 1)
            database_type = protocol.lower()

            # Parse connection parameters based on database type
            if database_type == "mysql":
                return cls._parse_mysql_uri(rest)
            # elif database_type == "postgresql":
            #     return cls._parse_postgresql_uri(rest)
            # elif database_type == "duckdb":
            #     return cls._parse_duckdb_uri(rest)
            else:
                raise ValueError(f"Unsupported database type in URI: {database_type}")

        except Exception as e:
            raise ValueError(f"Failed to parse database URI: {e}")

    @classmethod
    def _parse_mysql_uri(cls, uri_part: str) -> MySQLAnalyzer:
        """Parse MySQL URI and create analyzer."""
        # Format: user:password@host:port/database
        if "@" not in uri_part:
            raise ValueError("Invalid MySQL URI: missing credentials")

        credentials, host_db = uri_part.split("@", 1)

        # Parse credentials
        if ":" in credentials:
            user, password = credentials.split(":", 1)
        else:
            user = credentials
            password = ""

        # Parse host, port, and database
        if "/" not in host_db:
            raise ValueError("Invalid MySQL URI: missing database name")

        host_port, database = host_db.rsplit("/", 1)

        if ":" in host_port:
            host, port_str = host_port.split(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 3306

        return MySQLAnalyzer(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )

    @classmethod
    def get_supported_databases(cls) -> list[str]:
        """
        Get list of supported database types.

        Returns:
            List of supported database type strings
        """
        return list(cls._analyzers.keys())

    @classmethod
    def register_analyzer(
        cls, database_type: str, analyzer_class: Type[DatabaseAnalyzer]
    ) -> None:
        """
        Register a new database analyzer.

        This allows for extending the factory with new database types
        without modifying the core factory code.

        Args:
            database_type: String identifier for the database type
            analyzer_class: DatabaseAnalyzer subclass for this database type
        """
        cls._analyzers[database_type.lower()] = analyzer_class
