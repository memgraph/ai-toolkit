"""
Schema transformation utilities for SQL to graph migration.
Provides utilities for transforming SQL schema elements to graph equivalents.
"""

import logging

logger = logging.getLogger(__name__)


class SchemaUtilities:
    """Utilities for schema transformation and naming conventions."""

    @staticmethod
    def table_name_to_label(table_name: str) -> str:
        """Convert table name to Cypher node label.

        Args:
            table_name: SQL table name (e.g., 'user_profiles')

        Returns:
            Cypher label in PascalCase (e.g., 'UserProfiles')
        """
        return "".join(word.capitalize() for word in table_name.split("_"))

    @staticmethod
    def column_name_to_property(column_name: str) -> str:
        """Convert column name to graph property name.

        Args:
            column_name: SQL column name

        Returns:
            Property name (currently unchanged, but extensible)
        """
        return column_name

    @staticmethod
    def generate_relationship_name(to_table: str, join_table: str | None = None) -> str:
        """Generate semantic relationship name.

        Args:
            to_table: Target table name
            join_table: Join table name for many-to-many relationships

        Returns:
            Relationship name in UPPER_CASE
        """
        if join_table:
            return SchemaUtilities.table_name_to_label(join_table).upper()
        else:
            to_label = SchemaUtilities.table_name_to_label(to_table).upper()
            return f"HAS_{to_label}"

    @staticmethod
    def is_metadata_column(column_name: str) -> bool:
        """Check if a column is a metadata/system column.

        Args:
            column_name: Name of the column to check

        Returns:
            True if it's a metadata column, False otherwise
        """
        metadata_columns = {
            "id",
            "created_at",
            "updated_at",
            "created_on",
            "updated_on",
            "timestamp",
            "version",
            "deleted_at",
            "modified_at",
        }
        return column_name.lower() in metadata_columns

    @staticmethod
    def is_foreign_key_column(column_name: str) -> bool:
        """Check if a column name suggests it's a foreign key.

        Args:
            column_name: Name of the column to check

        Returns:
            True if it appears to be a foreign key, False otherwise
        """
        return column_name.lower().endswith("_id") and column_name.lower() != "id"
