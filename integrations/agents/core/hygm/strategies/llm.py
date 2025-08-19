"""
LLM-powered modeling strategy for Hypothetical Graph Modeling (HyGM).

This strategy uses AI/LLM models to create sophisticated graph models.
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.hygm.models.graph_models import GraphModel

try:
    from .base import BaseModelingStrategy
except ImportError:
    from core.hygm.strategies.base import BaseModelingStrategy

logger = logging.getLogger(__name__)


class LLMStrategy(BaseModelingStrategy):
    """LLM-powered graph modeling strategy using AI for intelligent mapping."""

    def __init__(
        self, llm_client=None, model_name: str = "gpt-4", temperature: float = 0.1
    ):
        """
        Initialize LLM strategy.

        Args:
            llm_client: OpenAI client instance
            model_name: Model to use for graph generation
            temperature: Temperature for generation (lower=more deterministic)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature

    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "llm"

    def create_model(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> "GraphModel":
        """
        Create a sophisticated graph model using LLM analysis.

        Args:
            database_structure: Database schema and structure
            domain_context: Optional domain context for better modeling

        Returns:
            GraphModel: AI-generated graph model
        """
        logger.info("Creating LLM-powered graph model...")

        if not self.llm_client:
            logger.warning("No LLM client provided, falling back to basic")
            return self._create_basic_model(database_structure)

        try:
            return self._create_llm_model(database_structure, domain_context)
        except Exception as e:  # noqa: BLE001
            logger.error("LLM model creation failed: %s", e)
            logger.info("Falling back to basic model creation")
            return self._create_basic_model(database_structure)

    def _create_llm_model(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> "GraphModel":
        """Create model using LLM structured output."""
        from core.hygm.models.llm_models import LLMGraphModel

        # Prepare the prompt
        prompt = self._build_modeling_prompt(database_structure, domain_context)

        # Use LangChain's structured output instead of direct OpenAI API
        llm_with_structure = self.llm_client.with_structured_output(LLMGraphModel)

        # Call LLM with structured output
        system_message = (
            "You are an expert database architect specializing "
            "in converting relational schemas to graph models."
        )

        llm_model = llm_with_structure.invoke(
            [("system", system_message), ("user", prompt)]
        )

        # Convert LLM response to internal graph model
        return self._convert_llm_to_graph_model(llm_model, database_structure)

    def _build_modeling_prompt(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> str:
        """Build the prompt for LLM graph modeling."""

        prompt_parts = [
            "Convert this relational database schema to a graph model:",
            "",
            "Database Structure:",
            str(database_structure),
            "",
            "Guidelines:",
            "- Create meaningful node labels (not just table names)",
            "- Identify semantic relationships between entities",
            "- Consider domain-specific modeling patterns",
            "- Optimize for query performance with appropriate indexes",
            "- Add constraints to maintain data integrity",
        ]

        if domain_context:
            prompt_parts.extend(
                [
                    "",
                    f"Domain Context: {domain_context}",
                    "Use this context to create more semantically meaningful "
                    "models.",
                ]
            )

        return "\n".join(prompt_parts)

    def _convert_llm_to_graph_model(
        self, llm_model, database_structure: Dict[str, Any]
    ) -> "GraphModel":
        """Convert LLM response to internal GraphModel format."""
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

        # Convert nodes
        nodes = []
        for llm_node in llm_model.nodes:
            source = NodeSource(
                type="llm_generated",
                name=llm_node.name,
                location="ai_analysis",
                mapping={"labels": [llm_node.label]},
            )

            properties = []
            for prop_name in llm_node.properties:
                prop_source = PropertySource(
                    field=f"{llm_node.source_table}.{prop_name}"
                )
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            node = GraphNode(
                labels=[llm_node.label], properties=properties, source=source
            )
            nodes.append(node)

        # Convert relationships with proper database mapping
        relationships = []
        actual_relationships = database_structure.get("relationships", [])

        logger.info("LLM generated %d relationships", len(llm_model.relationships))
        logger.info("Database has %d relationships", len(actual_relationships))

        for llm_rel in llm_model.relationships:
            # Find matching database relationship
            db_rel = self._find_matching_database_relationship(
                llm_rel, actual_relationships
            )

            if db_rel:
                logger.info("Matched LLM relationship %s to database", llm_rel.name)
                # Use actual database relationship mapping
                rel_source = RelationshipSource(
                    type="database_foreign_key",
                    name=llm_rel.name,
                    location=f"database.relationships.{llm_rel.name}",
                    mapping=self._create_relationship_mapping(db_rel),
                )
            else:
                logger.warning(
                    "No database match for LLM relationship %s " "(%s -> %s)",
                    llm_rel.name,
                    llm_rel.from_node,
                    llm_rel.to_node,
                )
                # Fallback to LLM-generated mapping
                rel_source = RelationshipSource(
                    type="llm_generated",
                    name=llm_rel.name,
                    location="ai_analysis",
                    mapping={
                        "edge_type": llm_rel.name,
                        "directionality": llm_rel.directionality,
                    },
                )

            properties = []
            for prop_name in llm_rel.properties:
                prop_source = PropertySource(field=f"relationship.{prop_name}")
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            relationship = GraphRelationship(
                edge_type=llm_rel.name,
                start_node_labels=[llm_rel.from_node],
                end_node_labels=[llm_rel.to_node],
                properties=properties,
                source=rel_source,
                directionality=llm_rel.directionality,
            )
            relationships.append(relationship)

        # Convert indexes from node definitions
        indexes = []
        for llm_node in llm_model.nodes:
            for index_prop in llm_node.indexes:
                index_source = IndexSource(
                    origin="llm_recommendation",
                    reason=f"Index recommended by AI for {llm_node.name}",
                    created_by="ai_analysis",
                    index_name=None,
                    migrated_from=None,
                )
                graph_index = GraphIndex(
                    labels=[llm_node.label],
                    properties=[index_prop],
                    source=index_source,
                )
                indexes.append(graph_index)

        # Convert constraints from node definitions
        constraints = []
        for llm_node in llm_model.nodes:
            for constraint_prop in llm_node.constraints:
                constraint_source = ConstraintSource(
                    origin="llm_recommendation",
                    reason=f"Constraint recommended by AI for {llm_node.name}",
                    created_by="ai_analysis",
                    constraint_name=None,
                    migrated_from=None,
                )
                graph_constraint = GraphConstraint(
                    type="unique",
                    labels=[llm_node.label],
                    properties=[constraint_prop],
                    source=constraint_source,
                )
                constraints.append(graph_constraint)

        # Create and return the complete graph model
        return GraphModel(
            nodes=nodes,
            edges=relationships,
            node_indexes=indexes,
            node_constraints=constraints,
        )

    def _find_matching_database_relationship(self, llm_rel, actual_relationships):
        """Find the database relationship that matches the LLM relationship."""
        # Convert node names to likely table names
        from_table = self._node_to_table_name(llm_rel.from_node)
        to_table = self._node_to_table_name(llm_rel.to_node)

        # Look for relationships between these tables
        for db_rel in actual_relationships:
            if isinstance(db_rel, dict):
                # Check if this relationship connects the right tables
                if (
                    db_rel.get("from_table") == from_table
                    and db_rel.get("to_table") == to_table
                ) or (
                    db_rel.get("from_table") == to_table
                    and db_rel.get("to_table") == from_table
                ):
                    return db_rel

                # For many-to-many, check if either table is involved
                if db_rel.get("type") == "many_to_many" and (
                    from_table in [db_rel.get("from_table"), db_rel.get("to_table")]
                    or to_table in [db_rel.get("from_table"), db_rel.get("to_table")]
                ):
                    return db_rel

        return None

    def _node_to_table_name(self, node_name: str) -> str:
        """Convert LLM node name to likely table name."""
        # Convert CamelCase/PascalCase to snake_case
        import re

        table_name = re.sub(r"(?<!^)(?=[A-Z])", "_", node_name).lower()

        # Handle pluralization (basic rules)
        if not table_name.endswith("s"):
            if table_name.endswith("y"):
                table_name = table_name[:-1] + "ies"  # category -> categories
            else:
                table_name = table_name + "s"  # customer -> customers

        return table_name

    def _create_relationship_mapping(self, db_rel: Dict[str, Any]) -> Dict[str, Any]:
        """Create relationship mapping from database relationship info."""
        mapping = {
            "edge_type": db_rel.get("type", "RELATED_TO"),
        }

        if db_rel.get("type") == "many_to_many":
            # Many-to-many relationship via junction table
            mapping.update(
                {
                    "join_table": db_rel.get("join_table"),
                    "from_table": db_rel.get("from_table"),
                    "to_table": db_rel.get("to_table"),
                    "join_from_column": db_rel.get("join_from_column"),
                    "join_to_column": db_rel.get("join_to_column"),
                    "from_column": db_rel.get("from_column"),
                    "to_column": db_rel.get("to_column"),
                }
            )
        else:
            # One-to-many relationship (foreign key)
            from_table = db_rel.get("from_table")
            from_column = db_rel.get("from_column")
            to_table = db_rel.get("to_table")
            to_column = db_rel.get("to_column")

            mapping.update(
                {
                    "start_node": f"{from_table}.{from_column}",
                    "end_node": f"{to_table}.{to_column}",
                }
            )

        return mapping

    def _create_basic_model(self, database_structure: Dict[str, Any]):
        """Fallback to basic model creation when LLM is unavailable."""
        from core.hygm.strategies.deterministic import DeterministicStrategy

        fallback_strategy = DeterministicStrategy()
        return fallback_strategy.create_model(database_structure)
