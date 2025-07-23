"""
Schema utilities for MySQL to Memgraph migration.
Provides label naming, relationship naming, and index generation.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class SchemaUtilities:
    """Utilities for schema conversion and naming in MySQL to Memgraph
    migration."""

    def __init__(self, relationship_naming_strategy: str = "table_based"):
        """Initialize the schema utilities.

        Args:
            relationship_naming_strategy: Strategy for naming relationships.
                - "table_based": Use table names directly (default)
                - "llm": Use LLM to generate meaningful names (requires LLM)
        """
        self.relationship_naming_strategy = relationship_naming_strategy
        self.llm = None  # Will be set if using LLM strategy

    def set_llm(self, llm):
        """Set LLM for relationship naming strategy."""
        self.llm = llm

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

    def _generate_relationship_type(
        self, from_table: str, to_table: str, join_table: str = None
    ) -> str:
        """Generate relationship type name."""
        if self.relationship_naming_strategy == "llm" and self.llm:
            return self._generate_relationship_type_with_llm(
                from_table, to_table, join_table
            )
        else:
            # Table-based naming strategy (default)
            if join_table:
                return self._table_name_to_label(join_table).upper()
            else:
                return f"HAS_{self._table_name_to_label(to_table).upper()}"

    def _generate_relationship_type_with_llm(
        self, from_table: str, to_table: str, join_table: str = None
    ) -> str:
        """Generate relationship type using LLM."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            if join_table:
                prompt = f"""
                Given a many-to-many relationship between tables '{from_table}'
                and '{to_table}' via join table '{join_table}', suggest a
                meaningful relationship name in UPPER_CASE format.

                Examples:
                - film_actor -> ACTED_IN
                - customer_rental -> RENTED
                - user_role -> HAS_ROLE

                Return only the relationship name, nothing else.
                """
            else:
                prompt = f"""
                Given a one-to-many relationship from table '{from_table}'
                to table '{to_table}', suggest a meaningful relationship name
                in UPPER_CASE format.

                Examples:
                - customer -> order: PLACED
                - order -> order_item: CONTAINS
                - film -> language: IN_LANGUAGE

                Return only the relationship name, nothing else.
                """

            messages = [
                SystemMessage(content="You are a database modeling expert."),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            relationship_name = response.content.strip().upper()

            # Validate the response
            if relationship_name and relationship_name.replace("_", "").isalpha():
                return relationship_name
            else:
                # Fallback to table-based naming
                if join_table:
                    return self._table_name_to_label(join_table).upper()
                else:
                    return f"HAS_{self._table_name_to_label(to_table).upper()}"

        except (ImportError, AttributeError, ValueError) as e:
            logger.warning("LLM relationship naming failed: %s", e)
            # Fallback to table-based naming
            if join_table:
                return self._table_name_to_label(join_table).upper()
            else:
                return f"HAS_{self._table_name_to_label(to_table).upper()}"


# For backward compatibility, alias the new class name
CypherGenerator = SchemaUtilities
