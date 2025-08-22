"""
Cypher query generation utilities for SQL to Memgraph migration.
Provides label naming, relationship naming, and index generation.
"""

from typing import Dict, List, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from core.hygm.models.graph_models import GraphIndex, GraphConstraint

logger = logging.getLogger(__name__)


class CypherGenerator:
    """Utilities for Cypher query generation in SQL to Memgraph migration."""

    def __init__(self):
        """Initialize the Cypher query generator."""

    def generate_index_queries_from_hygm(
        self, hygm_indexes: List["GraphIndex"]
    ) -> List[str]:
        """Generate index creation queries from HyGM graph model indexes."""
        queries = []

        for graph_index in hygm_indexes:
            # Handle node indexes
            if graph_index.labels:
                label = graph_index.labels[0]  # Use first label
                for prop in graph_index.properties:
                    query = f"CREATE INDEX ON :{label}({prop})"
                    queries.append(query.strip())

            # Handle edge indexes (if supported in future)
            elif graph_index.edge_type:
                # Edge indexes are not commonly used in current versions
                # but we can add support here if needed
                logger.info("Skipping edge index for %s", graph_index.edge_type)

        return queries

    def generate_constraint_queries_from_hygm(
        self, hygm_constraints: List["GraphConstraint"]
    ) -> List[str]:
        """Generate constraint creation queries from HyGM graph model."""
        queries = []

        for graph_constraint in hygm_constraints:
            # Handle node constraints
            if graph_constraint.labels:
                label = graph_constraint.labels[0]  # Use first label

                if graph_constraint.type == "unique":
                    for prop in graph_constraint.properties:
                        query = (
                            f"CREATE CONSTRAINT ON (n:{label}) "
                            f"ASSERT n.{prop} IS UNIQUE"
                        )
                        queries.append(query)

                # Add support for other constraint types if needed
                elif graph_constraint.type == "existence":
                    for prop in graph_constraint.properties:
                        query = (
                            f"CREATE CONSTRAINT ON (n:{label}) "
                            f"ASSERT exists(n.{prop})"
                        )
                        queries.append(query)

        return queries

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
        self, from_table: str, to_table: str, join_table: str | None = None
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
