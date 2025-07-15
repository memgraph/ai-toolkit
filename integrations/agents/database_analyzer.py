"""
Database analyzer module for extracting schema and data from MySQL databases.
"""

import mysql.connector
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MySQLAnalyzer:
    """Analyzes MySQL database structure and extracts data."""

    def __init__(
        self, host: str, user: str, password: str, database: str, port: int = 3306
    ):
        """Initialize MySQL connection."""
        self.connection_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
        }
        self.connection = None

    def connect(self) -> bool:
        """Establish connection to MySQL database."""
        try:
            self.connection = mysql.connector.connect(**self.connection_config)
            logger.info("Successfully connected to MySQL database")
            return True
        except mysql.connector.Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False

    def disconnect(self):
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

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute(f"DESCRIBE {table_name}")
        columns = []
        for row in cursor.fetchall():
            columns.append(
                {
                    "field": row[0],
                    "type": row[1],
                    "null": row[2],
                    "key": row[3],
                    "default": row[4],
                    "extra": row[5],
                }
            )
        cursor.close()
        return columns

    def get_foreign_keys(self, table_name: str) -> List[Dict[str, str]]:
        """Get foreign key relationships for a table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        query = """
        SELECT
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        cursor.execute(query, (self.connection_config["database"], table_name))

        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append(
                {
                    "column": row[0],
                    "referenced_table": row[1],
                    "referenced_column": row[2],
                }
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

    def get_database_structure(self) -> Dict[str, Any]:
        """Get complete database structure including tables, schemas,
        and relationships."""
        structure = {"tables": {}, "relationships": []}

        tables = self.get_tables()

        for table in tables:
            structure["tables"][table] = {
                "schema": self.get_table_schema(table),
                "foreign_keys": self.get_foreign_keys(table),
            }

            # Add relationships
            for fk in structure["tables"][table]["foreign_keys"]:
                structure["relationships"].append(
                    {
                        "from_table": table,
                        "from_column": fk["column"],
                        "to_table": fk["referenced_table"],
                        "to_column": fk["referenced_column"],
                    }
                )

        return structure

    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        if not self.connection:
            raise ConnectionError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
