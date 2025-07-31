"""
MySQL-specific database analyzer implementation.

This module provides MySQL-specific implementation of the DatabaseAnalyzer interface.
"""

import mysql.connector
from typing import Dict, List, Any, Optional
import logging
from ..interface import (
    DatabaseAnalyzer,
    ColumnInfo,
    ForeignKeyInfo,
    TableInfo,
    TableType,
)

logger = logging.getLogger(__name__)


class MySQLAnalyzer(DatabaseAnalyzer):
    """MySQL-specific implementation of DatabaseAnalyzer."""

    def __init__(
        self, host: str, user: str, password: str, database: str, port: int = 3306
    ):
        """
        Initialize MySQL analyzer.

        Args:
            host: MySQL server hostname
            user: MySQL username
            password: MySQL password
            database: Database name
            port: MySQL port (default: 3306)
        """
        connection_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
        }
        super().__init__(connection_config)

    def _get_database_type(self) -> str:
        """Return the database type."""
        return "mysql"

    def connect(self) -> bool:
        """Establish connection to MySQL database."""
        try:
            self.connection = mysql.connector.connect(**self.connection_config)
            logger.info("Successfully connected to MySQL database")
            return True
        except mysql.connector.Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False

    def disconnect(self) -> None:
        """Close MySQL connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        return tables

    def get_table_schema(self, table_name: str) -> List[ColumnInfo]:
        """Get schema information for a specific table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute(f"DESCRIBE {table_name}")

        columns = []
        for row in cursor.fetchall():
            field_name = row[0]
            data_type = row[1]
            is_nullable = row[2] == "YES"
            key_type = row[3]
            default_value = row[4]
            extra = row[5]

            # Determine if it's a primary key
            is_primary_key = key_type == "PRI"

            # Determine if it's a foreign key (will be checked separately)
            is_foreign_key = False

            # Check for auto increment
            auto_increment = "auto_increment" in extra.lower()

            # Parse data type for length, precision, scale
            max_length = None
            precision = None
            scale = None

            if "(" in data_type:
                type_part = data_type.split("(")[0].lower()
                params_part = data_type.split("(")[1].rstrip(")")

                try:
                    if "," in params_part:
                        # Decimal type with precision and scale
                        precision, scale = map(int, params_part.split(","))
                    elif type_part in ("varchar", "char", "varbinary", "binary", "bit"):
                        # Types that have numeric length parameters
                        max_length = int(params_part)
                    elif type_part in ("decimal", "numeric", "float", "double"):
                        # Numeric types that might have precision
                        if params_part.isdigit():
                            precision = int(params_part)
                    # For enum, set, and other types, we don't parse the
                    # parameters as integers
                    # They will be handled as part of the type definition
                except (ValueError, TypeError) as e:
                    # If we can't parse the parameters as integers, it's
                    # likely an enum, set, etc.
                    # Keep the full type definition including parameters
                    logger.debug(
                        "Could not parse type parameters for " "%s: %s", data_type, e
                    )
                    # Don't modify data_type in this case, keep it as is
                    continue

                # Only update data_type if we successfully parsed parameters
                length_types = ("varchar", "char", "varbinary", "binary", "bit")
                numeric_types = ("decimal", "numeric", "float", "double")
                if type_part in length_types or type_part in numeric_types:
                    data_type = type_part

            columns.append(
                ColumnInfo(
                    name=field_name,
                    data_type=data_type,
                    is_nullable=is_nullable,
                    is_primary_key=is_primary_key,
                    is_foreign_key=is_foreign_key,
                    default_value=default_value,
                    auto_increment=auto_increment,
                    max_length=max_length,
                    precision=precision,
                    scale=scale,
                )
            )

        cursor.close()

        # Now check for foreign keys and update the columns
        foreign_keys = self.get_foreign_keys(table_name)
        fk_column_names = {fk.column_name for fk in foreign_keys}

        for column in columns:
            if column.name in fk_column_names:
                column.is_foreign_key = True

        return columns

    def get_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Get foreign key relationships for a table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        query = """
        SELECT
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME,
            CONSTRAINT_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        cursor.execute(query, (self.connection_config["database"], table_name))

        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append(
                ForeignKeyInfo(
                    column_name=row[0],
                    referenced_table=row[1],
                    referenced_column=row[2],
                    constraint_name=row[3],
                )
            )
        cursor.close()
        return foreign_keys

    def get_table_data(
        self, table_name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get data from a specific table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor(dictionary=True)
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        return data

    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def is_view(self, table_name: str) -> bool:
        """Check if a table is actually a view."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        query = """
        SELECT TABLE_TYPE 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = %s
        """
        cursor.execute(query, (self.connection_config["database"], table_name))
        result = cursor.fetchone()
        cursor.close()

        if result:
            return result[0] == "VIEW"
        return False

    def get_tables_excluding_views(self) -> List[str]:
        """Get list of all tables in the database, excluding views."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_TYPE = 'BASE TABLE'
        """
        cursor.execute(query, (self.connection_config["database"],))
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        return tables

    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get index information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of index information dictionaries
        """
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        query = """
        SELECT 
            INDEX_NAME,
            COLUMN_NAME,
            NON_UNIQUE,
            INDEX_TYPE
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        ORDER BY INDEX_NAME, SEQ_IN_INDEX
        """
        cursor.execute(query, (self.connection_config["database"], table_name))

        indexes = {}
        for row in cursor.fetchall():
            index_name = row[0]
            column_name = row[1]
            is_unique = row[2] == 0
            index_type = row[3]

            if index_name not in indexes:
                indexes[index_name] = {
                    "name": index_name,
                    "columns": [],
                    "is_unique": is_unique,
                    "type": index_type,
                }

            indexes[index_name]["columns"].append(column_name)

        cursor.close()
        return list(indexes.values())
