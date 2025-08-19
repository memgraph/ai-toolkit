"""
HyGM - Hypothetical Graph Modeling Package

A modular system for converting relational database schemas to graph models.
Supports multiple modeling strategies and interactive refinement.
"""

from .hygm import HyGM, ModelingMode, GraphModelingStrategy
from .models.graph_models import (
    GraphModel,
    GraphNode,
    GraphRelationship,
    GraphProperty,
    GraphIndex,
    GraphConstraint,
)
from .models.llm_models import LLMGraphNode, LLMGraphRelationship, LLMGraphModel
from .models.operations import (
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
from .models.sources import (
    PropertySource,
    NodeSource,
    RelationshipSource,
    IndexSource,
    ConstraintSource,
)
from .strategies import BaseModelingStrategy, DeterministicStrategy, LLMStrategy

__all__ = [
    # Main class
    "HyGM",
    "ModelingMode",
    "GraphModelingStrategy",
    # Core graph models
    "GraphModel",
    "GraphNode",
    "GraphRelationship",
    "GraphProperty",
    "GraphIndex",
    "GraphConstraint",
    # LLM models
    "LLMGraphNode",
    "LLMGraphRelationship",
    "LLMGraphModel",
    # Operations
    "ChangeNodeLabelOperation",
    "RenamePropertyOperation",
    "DropPropertyOperation",
    "AddPropertyOperation",
    "ChangeRelationshipNameOperation",
    "DropRelationshipOperation",
    "AddIndexOperation",
    "DropIndexOperation",
    "ModelModifications",
    # Sources
    "PropertySource",
    "NodeSource",
    "RelationshipSource",
    "IndexSource",
    "ConstraintSource",
    # Strategies
    "BaseModelingStrategy",
    "DeterministicStrategy",
    "LLMStrategy",
]
