"""
HyGM Models - Core data models for Hypothetical Graph Modeling.

This package contains all the data models used in HyGM:
- LLM models for AI-powered generation
- Core graph models for representing the graph structure
- Operation models for interactive modifications
- Source models for tracking data lineage
"""

from .llm_models import (
    GraphModelingStrategy,
    ModelingMode,
    LLMGraphNode,
    LLMGraphRelationship,
    LLMGraphModel,
)
from .operations import (
    ModelOperation,
    ChangeNodeLabelOperation,
    RenamePropertyOperation,
    DropPropertyOperation,
    AddPropertyOperation,
    ChangeRelationshipNameOperation,
    DropRelationshipOperation,
    AddIndexOperation,
    DropIndexOperation,
    ModelModifications,
)
from .sources import (
    PropertySource,
    NodeSource,
    RelationshipSource,
    IndexSource,
    ConstraintSource,
    EnumSource,
)
from .graph_models import (
    GraphProperty,
    GraphNode,
    GraphRelationship,
    GraphIndex,
    GraphConstraint,
    GraphEnum,
    GraphModel,
)

__all__ = [
    # Enums and strategies
    "GraphModelingStrategy",
    "ModelingMode",
    # LLM models
    "LLMGraphNode",
    "LLMGraphRelationship",
    "LLMGraphModel",
    # Operation models
    "ModelOperation",
    "ChangeNodeLabelOperation",
    "RenamePropertyOperation",
    "DropPropertyOperation",
    "AddPropertyOperation",
    "ChangeRelationshipNameOperation",
    "DropRelationshipOperation",
    "AddIndexOperation",
    "DropIndexOperation",
    "ModelModifications",
    # Source models
    "PropertySource",
    "NodeSource",
    "RelationshipSource",
    "IndexSource",
    "ConstraintSource",
    "EnumSource",
    # Core graph models
    "GraphProperty",
    "GraphNode",
    "GraphRelationship",
    "GraphIndex",
    "GraphConstraint",
    "GraphEnum",
    "GraphModel",
]
