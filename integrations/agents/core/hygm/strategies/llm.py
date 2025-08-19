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
        from core.hygm.models.llm_models import GraphModelingStrategy

        # Prepare the prompt
        prompt = self._build_modeling_prompt(database_structure, domain_context)

        # Call LLM with structured output
        completion = self.llm_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert database architect specializing "
                        "in converting relational schemas to graph models."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=GraphModelingStrategy,
            temperature=self.temperature,
        )

        # Extract and convert LLM response to internal graph model
        llm_model = completion.choices[0].message.parsed
        return self._convert_llm_to_graph_model(llm_model)

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

        # Convert nodes
        nodes = []
        for llm_node in llm_model.nodes:
            source = NodeSource(
                type="llm_generated",
                name=llm_node.reasoning,
                location="ai_analysis",
                mapping={"labels": llm_node.labels},
            )

            properties = []
            for llm_prop in llm_node.properties:
                prop_source = PropertySource(field=f"llm.{llm_prop.key}")
                graph_prop = GraphProperty(key=llm_prop.key, source=prop_source)
                properties.append(graph_prop)

            node = GraphNode(
                labels=llm_node.labels, properties=properties, source=source
            )
            nodes.append(node)

        # Convert relationships
        relationships = []
        for llm_rel in llm_model.relationships:
            rel_source = RelationshipSource(
                type="llm_generated",
                name=llm_rel.reasoning,
                location="ai_analysis",
                mapping={
                    "edge_type": llm_rel.edge_type,
                    "directionality": llm_rel.directionality,
                },
            )

            properties = []
            for llm_prop in llm_rel.properties:
                prop_source = PropertySource(field=f"llm.{llm_prop.key}")
                graph_prop = GraphProperty(key=llm_prop.key, source=prop_source)
                properties.append(graph_prop)

            relationship = GraphRelationship(
                edge_type=llm_rel.edge_type,
                start_node_labels=llm_rel.start_node_labels,
                end_node_labels=llm_rel.end_node_labels,
                properties=properties,
                source=rel_source,
                directionality=llm_rel.directionality,
            )
            relationships.append(relationship)

        # Convert indexes
        indexes = []
        for llm_index in llm_model.indexes:
            index_source = IndexSource(
                origin="llm_recommendation",
                reason=llm_index.reasoning,
                created_by="ai_analysis",
                index_name=None,
                migrated_from=None,
            )

            graph_index = GraphIndex(
                labels=llm_index.labels,
                properties=llm_index.properties,
                type=llm_index.type,
                source=index_source,
            )
            indexes.append(graph_index)

        # Convert constraints
        constraints = []
        for llm_constraint in llm_model.constraints:
            constraint_source = ConstraintSource(
                origin="llm_recommendation",
                constraint_name=f"ai_{llm_constraint.type}_constraint",
                migrated_from="ai_analysis",
            )

            graph_constraint = GraphConstraint(
                type=llm_constraint.type,
                labels=llm_constraint.labels,
                properties=llm_constraint.properties,
                source=constraint_source,
            )
            constraints.append(graph_constraint)

        return GraphModel(
            nodes=nodes,
            edges=relationships,
            node_indexes=indexes,
            node_constraints=constraints,
        )

    def _create_basic_model(self, database_structure: Dict[str, Any]):
        """Fallback to basic model creation when LLM is unavailable."""
        from core.hygm.strategies.deterministic import DeterministicStrategy

        fallback_strategy = DeterministicStrategy()
        return fallback_strategy.create_model(database_structure)
