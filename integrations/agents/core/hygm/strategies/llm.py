"""
LLM-powered modeling strategy for Hypothetical Graph Modeling (HyGM).

This strategy uses AI/LLM models to create sophisticated graph models.
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
        self._database_structure = {}

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

        Raises:
            ValueError: If no LLM client is provided
            Exception: If LLM model creation fails
        """
        logger.info("Creating LLM-powered graph model...")

        # Store database structure for relationship mapping
        self._database_structure = database_structure

        if not self.llm_client:
            raise ValueError(
                "No LLM client provided for LLM-powered modeling. "
                "Please configure OpenAI API key or use deterministic strategy."
            )

        return self._create_llm_model(database_structure, domain_context)

    def _create_llm_model(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> "GraphModel":
        """Create model using LLM structured output."""
        logger.info("Using LLM to generate graph model...")

        try:
            # Import the structured output model
            from core.hygm.models.llm_models import LLMGraphModel

            # Prepare the prompt
            prompt = self._build_modeling_prompt(database_structure, domain_context)

            # Call LLM with LangChain's structured output support
            # LangChain's ChatOpenAI supports structured output via with_structured_output()
            if not self.llm_client:
                raise ValueError("No LLM client configured")

            # Create structured output chain using the Pydantic model
            structured_llm = self.llm_client.with_structured_output(LLMGraphModel)

            # Create system message for graph modeling
            system_message = (
                "You are an expert database architect specializing "
                "in converting relational schemas to graph models. "
                "Analyze the provided database structure and create an optimal "
                "graph model that preserves relationships and enables efficient querying."
            )

            # Generate the structured output directly as LLMGraphModel
            llm_model = structured_llm.invoke(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            )

            logger.info(
                f"LLM generated {len(llm_model.nodes)} nodes and {len(llm_model.relationships)} relationships"
            )

            # Extract and convert LLM response to internal graph model
            return self._convert_llm_to_graph_model(llm_model)

        except ImportError as e:
            error_msg = f"Missing LLM models module: {e}"
            logger.error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"LLM model creation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _build_modeling_prompt(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> str:
        """Build the prompt for LLM graph modeling."""

        prompt_parts = [
            "Convert this relational database schema to an optimal graph model.",
            "Analyze the database structure and create nodes and relationships that:",
            "",
            "Database Structure:",
            str(database_structure),
            "",
            "Requirements:",
            "- Create semantic node labels that represent entities (not just table names)",
            "- Identify meaningful relationships between entities based on foreign keys",
            "- Include relevant properties from source tables on nodes",
            "- Set appropriate primary keys, indexes, and constraints",
            "- Use descriptive relationship names (e.g., 'OWNS', 'BELONGS_TO')",
            "- Consider one-to-many, many-to-many, and one-to-one cardinalities",
            "- Optimize for both data integrity and query performance",
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

    def _convert_llm_to_graph_model(self, llm_model) -> "GraphModel":
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

        # Convert nodes - preserve original table mapping
        nodes = []
        for llm_node in llm_model.nodes:
            source = NodeSource(
                type="table",  # Keep as table source for migration
                name=llm_node.source_table,  # Use actual source table name
                location=f"database.schema.{llm_node.source_table}",
                mapping={"labels": llm_node.labels},
            )

            properties = []
            for prop_name in llm_node.properties:
                field_path = f"{llm_node.source_table}.{prop_name}"
                prop_source = PropertySource(field=field_path)
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            node = GraphNode(
                labels=llm_node.labels, properties=properties, source=source
            )
            nodes.append(node)

        # Convert relationships - preserve database structure mapping
        relationships = []
        for llm_rel in llm_model.relationships:
            # Find the source database relationship information
            db_rel_info = self._find_database_relationship(llm_rel)

            if db_rel_info:
                # Use actual database structure for migration mapping
                from_table = db_rel_info.get("from_table", "")
                location = f"database.schema.{from_table}"
                rel_source = RelationshipSource(
                    type=db_rel_info.get("source_type", "table"),
                    name=db_rel_info.get("constraint_name", llm_rel.name),
                    location=location,
                    mapping=db_rel_info.get("mapping", {}),
                )
            else:
                # Fallback for relationships without database mapping
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
                prop_source = PropertySource(field=f"llm.{prop_name}")
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            # Map LLM node names to actual node labels
            start_labels = self._map_node_name_to_labels(llm_rel.from_node, llm_model)
            end_labels = self._map_node_name_to_labels(llm_rel.to_node, llm_model)

            relationship = GraphRelationship(
                edge_type=llm_rel.name,
                start_node_labels=start_labels,
                end_node_labels=end_labels,
                properties=properties,
                source=rel_source,
                directionality=llm_rel.directionality,
            )
            relationships.append(relationship)

        # Convert indexes from node-level index specifications
        indexes = []
        for llm_node in llm_model.nodes:
            for index_prop in llm_node.indexes:
                index_source = IndexSource(
                    origin="llm_recommendation",
                    reason=f"Index recommended by LLM for {llm_node.name}.{index_prop}",
                    created_by="ai_analysis",
                    index_name=None,
                    migrated_from=None,
                )

                graph_index = GraphIndex(
                    labels=llm_node.labels,
                    properties=[index_prop],
                    type="btree",  # Default index type
                    source=index_source,
                )
                indexes.append(graph_index)

        # Convert constraints from node-level constraint specifications
        constraints = []
        for llm_node in llm_model.nodes:
            for constraint_prop in llm_node.constraints:
                constraint_source = ConstraintSource(
                    origin="llm_recommendation",
                    constraint_name=f"ai_unique_constraint_{llm_node.name}_{constraint_prop}",
                    migrated_from="ai_analysis",
                )

                graph_constraint = GraphConstraint(
                    type="unique",  # Assume unique constraints
                    labels=llm_node.labels,
                    properties=[constraint_prop],
                    source=constraint_source,
                )
                constraints.append(graph_constraint)

        return GraphModel(
            nodes=nodes,
            edges=relationships,
            node_indexes=indexes,
            node_constraints=constraints,
        )

    def _find_database_relationship(self, llm_rel) -> Dict[str, Any]:
        """
        Find the original database relationship information for an LLM
        relationship.

        This method matches LLM-generated relationships back to the original
        database structure to preserve technical migration details.
        """
        if not hasattr(self, "_database_structure"):
            return {}

        relationships = self._database_structure.get("relationships", [])

        # Try to match by node names and relationship semantics
        for db_rel in relationships:
            from_table = db_rel.get("from_table", "").lower()
            to_table = db_rel.get("to_table", "").lower()

            # Check if this database relationship matches the LLM relationship
            if self._tables_match_nodes(
                from_table, llm_rel.from_node
            ) and self._tables_match_nodes(to_table, llm_rel.to_node):
                # Build the mapping information needed for migration
                from_col = db_rel.get("from_column", "id")
                to_col = db_rel.get("to_column", "id")
                mapping = {
                    "start_node": f"{from_table}.{from_col}",
                    "end_node": f"{to_table}.{to_col}",
                    "edge_type": llm_rel.name,
                }

                # Add many-to-many specific information if available
                if db_rel.get("relationship_type") == "many_to_many":
                    mapping.update(
                        {
                            "join_table": db_rel.get("join_table"),
                            "join_from_column": db_rel.get("join_from_column"),
                            "join_to_column": db_rel.get("join_to_column"),
                            "from_table": from_table,
                            "to_table": to_table,
                            "from_column": db_rel.get("from_column"),
                            "to_column": db_rel.get("to_column"),
                        }
                    )

                # Determine source type
                rel_type = db_rel.get("relationship_type")
                source_type = (
                    "junction_table" if rel_type == "many_to_many" else "table"
                )

                return {
                    "source_type": source_type,
                    "constraint_name": db_rel.get("constraint_name", llm_rel.name),
                    "from_table": from_table,
                    "to_table": to_table,
                    "mapping": mapping,
                }

        return {}

    def _tables_match_nodes(self, table_name: str, node_name: str) -> bool:
        """Check if a table name matches a node name (flexible matching)."""
        table_lower = table_name.lower()
        node_lower = node_name.lower()

        # Direct match
        if table_lower == node_lower:
            return True

        # Singularized match (e.g., "users" table -> "User" node)
        if table_lower.rstrip("s") == node_lower:
            return True

        # Capitalized match (e.g., "user" table -> "User" node)
        if table_lower == node_lower.lower():
            return True

        return False

    def _map_node_name_to_labels(self, node_name: str, llm_model) -> List[str]:
        """Map LLM node name to the actual labels defined in the model."""
        for llm_node in llm_model.nodes:
            if llm_node.name.lower() == node_name.lower() or any(
                label.lower() == node_name.lower() for label in llm_node.labels
            ):
                return llm_node.labels

        # Fallback to the node name itself as label
        return [node_name]
