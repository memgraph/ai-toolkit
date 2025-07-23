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

    def is_join_table(
        self,
        table_name: str,
        schema: List[Dict[str, Any]],
        foreign_keys: List[Dict[str, str]],
    ) -> bool:
        """Determine if a table is a join table (many-to-many)."""
        # A join table typically has:
        # 1. Only foreign key columns (and maybe an ID or timestamp)
        # 2. At least 2 foreign keys
        # 3. Small number of total columns

        if len(foreign_keys) < 2:
            return False

        # Count non-FK columns (excluding common metadata columns)
        non_fk_columns = []
        fk_column_names = {fk["column"] for fk in foreign_keys}
        metadata_columns = [
            "id",
            "created_at",
            "updated_at",
            "created_on",
            "updated_on",
            "timestamp",
        ]

        for col in schema:
            field_name = col["field"].lower()
            if (
                col["field"] not in fk_column_names
                and field_name not in metadata_columns
            ):
                non_fk_columns.append(col["field"])

        # If most columns are foreign keys, it's likely a join table
        total_columns = len(schema)
        fk_ratio = len(foreign_keys) / total_columns

        # Consider it a join table if:
        # - At least 2 FKs and FK ratio > 0.5, OR
        # - All columns are FKs or metadata columns
        return (len(foreign_keys) >= 2 and fk_ratio > 0.5) or len(non_fk_columns) == 0

    def get_table_type(self, table_name: str) -> str:
        """Determine the type of table: 'entity', 'join', 'view', or 'lookup'."""
        # Check if it's a view first
        if self.is_view(table_name):
            return "view"

        schema = self.get_table_schema(table_name)
        foreign_keys = self.get_foreign_keys(table_name)

        if self.is_join_table(table_name, schema, foreign_keys):
            return "join"
        elif len(foreign_keys) == 0:
            return "entity"  # Pure entity table with no references
        else:
            return "entity"  # Entity table with references

    def get_database_structure(self) -> Dict[str, Any]:
        """Get complete database structure including tables, schemas,
        and relationships."""
        structure = {
            "tables": {},
            "relationships": [],
            "join_tables": {},
            "entity_tables": {},
            "views": {},
        }

        # Get all tables (including views) for completeness
        all_tables = self.get_tables()
        # Get only real tables for migration
        tables = self.get_tables_excluding_views()

        # First pass: categorize tables and collect basic info
        for table in all_tables:
            schema = self.get_table_schema(table)
            foreign_keys = self.get_foreign_keys(table)
            table_type = self.get_table_type(table)

            structure["tables"][table] = {
                "schema": schema,
                "foreign_keys": foreign_keys,
                "type": table_type,
                "row_count": self.get_table_row_count(table),
            }

            if table_type == "view":
                structure["views"][table] = structure["tables"][table]
                logger.info(f"Skipping view table: {table}")
            elif table_type == "join":
                structure["join_tables"][table] = structure["tables"][table]
            else:
                structure["entity_tables"][table] = structure["tables"][table]

        # Second pass: create relationships
        for table_name, table_info in structure["tables"].items():
            if table_info["type"] == "join":
                # Handle join tables as many-to-many relationships
                fks = table_info["foreign_keys"]
                if len(fks) >= 2:
                    # Create a many-to-many relationship
                    # For now, take first two FKs as the main relationship
                    fk1, fk2 = fks[0], fks[1]

                    # Get additional properties from non-FK columns
                    fk_columns = {fk["column"] for fk in fks}
                    additional_properties = []
                    metadata_columns = [
                        "id",
                        "created_at",
                        "updated_at",
                        "created_on",
                        "updated_on",
                        "timestamp",
                    ]
                    for col in table_info["schema"]:
                        if (
                            col["field"] not in fk_columns
                            and col["field"].lower() not in metadata_columns
                        ):
                            additional_properties.append(col["field"])

                    structure["relationships"].append(
                        {
                            "type": "many_to_many",
                            "join_table": table_name,
                            "from_table": fk1["referenced_table"],
                            "from_column": fk1["referenced_column"],
                            "to_table": fk2["referenced_table"],
                            "to_column": fk2["referenced_column"],
                            "join_from_column": fk1["column"],
                            "join_to_column": fk2["column"],
                            "additional_properties": additional_properties,
                        }
                    )
            else:
                # Handle regular foreign key relationships
                for fk in table_info["foreign_keys"]:
                    structure["relationships"].append(
                        {
                            "type": "one_to_many",
                            "from_table": table_name,
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
