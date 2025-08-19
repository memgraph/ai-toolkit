"""
Deterministic modeling strategy for Hypothetical Graph Modeling (HyGM).

This strategy creates graph models using rule-based approaches without AI.
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.hygm.models.graph_models import GraphModel

try:
    from .base import BaseModelingStrategy
except ImportError:
    from core.hygm.strategies.base import BaseModelingStrategy

logger = logging.getLogger(__name__)


class DeterministicStrategy(BaseModelingStrategy):
    """Deterministic graph modeling strategy using rule-based approaches."""

    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "deterministic"

    def create_model(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,  # noqa: ARG002
    ) -> "GraphModel":
        """
        Create a basic graph model deterministically from database structure.
        This method creates a straightforward mapping without AI assistance.
        """
        logger.info("Creating deterministic graph model...")

        # Import here to avoid circular imports
        from core.hygm.models.graph_models import (
            GraphModel,
            GraphNode,
            GraphRelationship,
            GraphProperty,
            GraphIndex,
            GraphConstraint,
        )
        from core.hygm.models.sources import (
            NodeSource,
            PropertySource,
            RelationshipSource,
            IndexSource,
            ConstraintSource,
        )

        nodes = []
        relationships = []
        node_indexes = []
        node_constraints = []

        # Convert entity tables to nodes
        entity_tables = database_structure.get("entity_tables", {})
        for table_name, table_info in entity_tables.items():
            # Create source information
            source = NodeSource(
                type="table",
                name=table_name,
                location=f"database.schema.{table_name}",
                mapping={
                    "labels": [table_name.title()],
                    "id_field": self._find_primary_key(table_info),
                },
            )

            # Extract properties
            properties = []
            node_props = self._extract_node_properties_from_table(table_info)
            for prop_name in node_props:
                prop_source = PropertySource(field=f"{table_name}.{prop_name}")
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            # Create node
            node = GraphNode(
                labels=[table_name.title()], properties=properties, source=source
            )
            nodes.append(node)

            # Create indexes for this node
            for index_prop in self._extract_indexes_from_table(table_info):
                index_source = IndexSource(
                    origin="migration_requirement",
                    reason="performance_optimization",
                    created_by="migration_agent",
                    index_name=None,
                    migrated_from=None,
                )
                graph_index = GraphIndex(
                    labels=[table_name.title()],
                    properties=[index_prop],
                    type="label+property",
                    source=index_source,
                )
                node_indexes.append(graph_index)

            # Create constraints for this node
            constraints = self._extract_constraints_from_table(table_info)
            for constraint_str in constraints:
                if "UNIQUE" in constraint_str.upper():
                    prop_name = constraint_str.replace("UNIQUE(", "")
                    prop_name = prop_name.replace(")", "")
                    constraint_source = ConstraintSource(
                        origin="source_database_constraint",
                        constraint_name=f"{table_name}_{prop_name}_unique",
                        migrated_from=f"database.schema.{table_name}",
                    )
                    graph_constraint = GraphConstraint(
                        type="unique",
                        labels=[table_name.title()],
                        properties=[prop_name],
                        source=constraint_source,
                    )
                    node_constraints.append(graph_constraint)

        # Convert relationships
        relationships_data = database_structure.get("relationships", [])
        for rel_data in relationships_data:
            rel_name = self._generate_relationship_name(rel_data)

            # Find source and target node labels
            from_table = rel_data.get("from_table", "")
            to_table = rel_data.get("to_table", "")

            start_labels = [from_table.title()]
            end_labels = [to_table.title()]

            # Create relationship source
            rel_source = RelationshipSource(
                type="table",
                name=rel_data.get("constraint_name", rel_name),
                location=f"database.schema.{from_table}",
                mapping={
                    "start_node": (f"{from_table}.{rel_data.get('from_column', 'id')}"),
                    "end_node": (f"{to_table}.{rel_data.get('to_column', 'id')}"),
                    "edge_type": rel_name,
                },
            )

            # Create relationship
            relationship = GraphRelationship(
                edge_type=rel_name,
                start_node_labels=start_labels,
                end_node_labels=end_labels,
                properties=[],
                source=rel_source,
                directionality="directed",
            )
            relationships.append(relationship)

        return GraphModel(
            nodes=nodes,
            edges=relationships,
            node_indexes=node_indexes,
            node_constraints=node_constraints,
        )

    def _extract_node_properties_from_table(
        self, table_info: Dict[str, Any]
    ) -> List[str]:
        """Extract properties that should be included in the node."""
        properties = []

        # Get schema from table_info - it's a list of column dictionaries
        schema_list = table_info.get("schema", [])
        for col_info in schema_list:
            col_name = col_info.get("field")
            if not col_name:
                continue

            # Include all columns except foreign key columns (ending with _id
            # but not primary keys)
            if not (col_name.endswith("_id") and col_info.get("key") != "PRI"):
                properties.append(col_name)
        return properties

    def _extract_indexes_from_table(self, table_info: Dict[str, Any]) -> List[str]:
        """Extract properties that should have indexes."""
        indexes = []
        for col_name, col_info in table_info.get("columns", {}).items():
            # Add indexes for primary keys and unique columns
            if col_info.get("key") in ["PRI", "UNI"]:
                indexes.append(col_name)
        return indexes

    def _extract_constraints_from_table(self, table_info: Dict[str, Any]) -> List[str]:
        """Extract constraint definitions from table info."""
        constraints = []
        for col_name, col_info in table_info.get("columns", {}).items():
            if col_info.get("key") == "UNI":
                constraints.append(f"UNIQUE({col_name})")
        return constraints

    def _generate_relationship_name(self, rel_data: Dict[str, Any]) -> str:
        """Generate a semantic relationship name from relationship data."""
        constraint_name = rel_data.get("constraint_name", "")
        if constraint_name:
            # Extract meaningful name from constraint
            if "_fk" in constraint_name:
                join_table = constraint_name.split("_fk")[0]
            else:
                join_table = constraint_name

            if join_table:
                return join_table.upper().replace("_", "_")
            else:
                return "CONNECTS"
        else:
            from_table = rel_data.get("from_table", "")
            to_table = rel_data.get("to_table", "")
            return f"{from_table.upper()}_TO_{to_table.upper()}"

    def _find_primary_key(self, table_info: Dict[str, Any]) -> str:
        """Find the primary key column for a table."""
        for col_name, col_info in table_info.get("columns", {}).items():
            if col_info.get("key") == "PRI":
                return col_name
        return "id"  # Fallback
