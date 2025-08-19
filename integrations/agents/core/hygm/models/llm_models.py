"""
LLM-specific models for Hypothetical Graph Modeling (HyGM).

These models define the structure for LLM input/output and interactive operations.
"""

from enum import Enum
from typing import List, Literal
from pydantic import BaseModel, Field


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
    label: str = Field(
        description="Cypher label for the node (e.g., 'User', 'Product')"
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
