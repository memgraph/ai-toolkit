"""
LLM-specific models for Hypothetical Graph Modeling (HyGM).

These models define the structure for LLM input/output and interactive
operations.
"""

from enum import Enum
from typing import List, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .graph_models import GraphModel


class GraphModelingStrategy(Enum):
    """Graph modeling strategies available."""

    DETERMINISTIC = "deterministic"  # Rule-based graph creation
    LLM_POWERED = "llm_powered"  # LLM generates the graph model


class ModelingMode(Enum):
    """Modeling modes available."""

    AUTOMATIC = "automatic"  # Generate model without user interaction
    INTERACTIVE = "interactive"  # Interactive mode with user feedback


# Structured output models for LLM graph generation
class LLMGraphNode(BaseModel):
    """Node definition for LLM-generated graph models."""

    name: str = Field(description="Unique identifier for the node")
    labels: List[str] = Field(
        description="Cypher labels for the node (e.g., ['User'], ['Product'])"
    )
    properties: List[str] = Field(
        description="List of properties to include from source table"
    )
    primary_key: str = Field(description="Primary key property name")
    indexes: List[str] = Field(description="Properties that should have indexes")
    constraints: List[str] = Field(
        description="Properties that should have uniqueness constraints"
    )
    source_table: str = Field(description="Source SQL table name")

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"


class LLMGraphRelationship(BaseModel):
    """Relationship definition for LLM-generated graph models."""

    name: str = Field(description="Relationship type name (e.g., 'OWNS', 'BELONGS_TO')")
    type: Literal["one_to_many", "many_to_many", "one_to_one"] = Field(
        description="Cardinality of the relationship"
    )
    from_node: str = Field(description="Source node name")
    to_node: str = Field(description="Target node name")
    properties: List[str] = Field(
        description="Properties to include on the relationship (if any)", default=[]
    )
    directionality: Literal["directed", "undirected"] = Field(
        description="Whether the relationship has direction"
    )

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"


class LLMGraphModel(BaseModel):
    """Complete graph model structure for LLM generation."""

    nodes: List[LLMGraphNode] = Field(description="All nodes in the graph model")
    relationships: List[LLMGraphRelationship] = Field(
        description="All relationships in the graph model"
    )

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"

    def to_graph_model(self) -> "GraphModel":
        """Convert LLMGraphModel to GraphModel for schema export."""
        from .graph_models import (
            GraphModel,
            GraphNode,
            GraphRelationship,
            GraphProperty,
        )
        from .sources import NodeSource, RelationshipSource, PropertySource

        # Convert nodes
        graph_nodes = []
        for llm_node in self.nodes:
            # Create properties with proper GraphProperty objects
            properties = []
            for prop_name in llm_node.properties:
                graph_prop = GraphProperty(
                    key=prop_name,
                    count=1,
                    filling_factor=100.0,
                    types=[{"type": "String", "count": 1, "examples": [""]}],
                    source=PropertySource(field=f"{llm_node.source_table}.{prop_name}"),
                )
                properties.append(graph_prop)

            # Create node source
            node_source = NodeSource(
                type="table",
                name=llm_node.source_table,
                location=f"database.schema.{llm_node.source_table}",
                mapping={
                    "labels": llm_node.labels,
                    "id_field": (f"{llm_node.source_table}.{llm_node.primary_key}"),
                },
            )

            graph_node = GraphNode(
                labels=llm_node.labels,
                count=1,
                properties=properties,
                examples=[{"gid": 0}],
                source=node_source,
            )
            graph_nodes.append(graph_node)

        # Convert relationships
        graph_relationships = []
        for llm_rel in self.relationships:
            # Find source/target node labels
            from_labels = []
            to_labels = []
            for node in self.nodes:
                if node.name == llm_rel.from_node:
                    from_labels = node.labels
                if node.name == llm_rel.to_node:
                    to_labels = node.labels

            # Create relationship properties
            rel_properties = []
            for prop_name in llm_rel.properties:
                rel_prop = GraphProperty(
                    key=prop_name,
                    count=1,
                    filling_factor=100.0,
                    types=[{"type": "String", "count": 1, "examples": [""]}],
                )
                rel_properties.append(rel_prop)

            # Create relationship source
            rel_source = RelationshipSource(
                type="derived",
                name=f"{llm_rel.from_node}_{llm_rel.name}_{llm_rel.to_node}",
                location=f"derived.{llm_rel.name}",
                mapping={
                    "start_node": llm_rel.from_node,
                    "end_node": llm_rel.to_node,
                    "edge_type": llm_rel.name,
                },
            )

            graph_rel = GraphRelationship(
                edge_type=llm_rel.name,
                start_node_labels=from_labels,
                end_node_labels=to_labels,
                count=1,
                properties=rel_properties,
                examples=[{}],
                source=rel_source,
                directionality=llm_rel.directionality,
            )
            graph_relationships.append(graph_rel)

        return GraphModel(nodes=graph_nodes, edges=graph_relationships)
