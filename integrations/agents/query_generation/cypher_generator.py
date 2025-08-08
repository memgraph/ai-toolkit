"""
Cypher query generation utilities for SQL to Memgraph migration.
Provides label naming, relationship naming, and index generation.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class CypherGenerator:
    """Utilities for Cypher query generation in SQL to Memgraph migration."""

    def __init__(self):
        """Initialize the Cypher query generator."""

    def generate_index_queries(
        self, table_name: str, schema: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate index creation queries."""
        queries = []
        label = self._table_name_to_label(table_name)

        for col in schema:
            if col["key"] in ["PRI", "UNI", "MUL"]:
                query = f"CREATE INDEX ON :{label}({col['field']})"
                queries.append(query.strip())

        return queries

    def generate_constraint_queries(
        self, table_name: str, schema: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate constraint creation queries."""
        queries = []
        label = self._table_name_to_label(table_name)

        # Primary key constraints
        primary_keys = [col["field"] for col in schema if col["key"] == "PRI"]
        for pk in primary_keys:
            query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{pk} IS UNIQUE"
            queries.append(query)

        # Unique constraints
        unique_keys = [col["field"] for col in schema if col["key"] == "UNI"]
        for uk in unique_keys:
            query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{uk} IS UNIQUE"
            queries.append(query)

        return queries

    def _table_name_to_label(self, table_name: str) -> str:
        """Convert table name to Cypher label."""
        # Convert to PascalCase
        return "".join(word.capitalize() for word in table_name.split("_"))

    def generate_relationship_type(
        self, from_table: str, to_table: str, join_table: str = None
    ) -> str:
        """Generate relationship type based on table names.

        Args:
            from_table: Source table name (unused, kept for compatibility)
            to_table: Target table name
            join_table: Join table name (for many-to-many relationships)

        Returns:
            Relationship type in UPPER_CASE format
        """
        # pylint: disable=unused-argument
        # Table-based naming strategy
        if join_table:
            return self._table_name_to_label(join_table).upper()
        else:
            return f"HAS_{self._table_name_to_label(to_table).upper()}"
