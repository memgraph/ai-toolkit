"""
HyGM Models - Core data models for Hypothetical Graph Modeling.

This package contains all the data models used in HyGM:
- LLM models for AI-powered generation
- Core graph models for representing the graph structure
- Operation models for interactive modifications
- Source models for tracking data lineage
"""

from .graph_models import (
    GraphConstraint,
    GraphEnum,
    GraphIndex,
    GraphModel,
    GraphNode,
    GraphProperty,
    GraphRelationship,
)
from .llm_models import (
    GraphModelingStrategy,
    LLMGraphModel,
    LLMGraphNode,
    LLMGraphRelationship,
    ModelingMode,
)
from .operations import (
    AddIndexOperation,
    AddPropertyOperation,
    ChangeNodeLabelOperation,
    ChangeRelationshipNameOperation,
    DropIndexOperation,
    DropPropertyOperation,
    DropRelationshipOperation,
    ModelModifications,
    ModelOperation,
    RenamePropertyOperation,
)
from .sources import (
    ConstraintSource,
    EnumSource,
    IndexSource,
    NodeSource,
    PropertySource,
    RelationshipSource,
)

__all__ = [
    "AddIndexOperation",
    "AddPropertyOperation",
    "ChangeNodeLabelOperation",
    "ChangeRelationshipNameOperation",
    "ConstraintSource",
    "DropIndexOperation",
    "DropPropertyOperation",
    "DropRelationshipOperation",
    "EnumSource",
    "GraphConstraint",
    "GraphEnum",
    "GraphIndex",
    "GraphModel",
    # Enums and strategies
    "GraphModelingStrategy",
    "GraphNode",
    # Core graph models
    "GraphProperty",
    "GraphRelationship",
    "IndexSource",
    "LLMGraphModel",
    # LLM models
    "LLMGraphNode",
    "LLMGraphRelationship",
    "ModelModifications",
    # Operation models
    "ModelOperation",
    "ModelingMode",
    "NodeSource",
    # Source models
    "PropertySource",
    "RelationshipSource",
    "RenamePropertyOperation",
]
