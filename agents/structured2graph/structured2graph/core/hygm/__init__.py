"""
HyGM - Hypothetical Graph Modeling Package

A modular system for converting relational database schemas to graph models.
Supports multiple modeling strategies and interactive refinement.
"""

from .hygm import GraphModelingStrategy, HyGM, ModelingMode
from .models.graph_models import (
    GraphConstraint,
    GraphIndex,
    GraphModel,
    GraphNode,
    GraphProperty,
    GraphRelationship,
)
from .models.llm_models import LLMGraphModel, LLMGraphNode, LLMGraphRelationship
from .models.operations import (
    AddIndexOperation,
    AddPropertyOperation,
    ChangeNodeLabelOperation,
    ChangeRelationshipNameOperation,
    DropIndexOperation,
    DropPropertyOperation,
    DropRelationshipOperation,
    ModelModifications,
    RenamePropertyOperation,
)
from .models.sources import (
    ConstraintSource,
    IndexSource,
    NodeSource,
    PropertySource,
    RelationshipSource,
)
from .strategies import BaseModelingStrategy, DeterministicStrategy, LLMStrategy

__all__ = [
    "AddIndexOperation",
    "AddPropertyOperation",
    # Strategies
    "BaseModelingStrategy",
    # Operations
    "ChangeNodeLabelOperation",
    "ChangeRelationshipNameOperation",
    "ConstraintSource",
    "DeterministicStrategy",
    "DropIndexOperation",
    "DropPropertyOperation",
    "DropRelationshipOperation",
    "GraphConstraint",
    "GraphIndex",
    # Core graph models
    "GraphModel",
    "GraphModelingStrategy",
    "GraphNode",
    "GraphProperty",
    "GraphRelationship",
    # Main class
    "HyGM",
    "IndexSource",
    "LLMGraphModel",
    # LLM models
    "LLMGraphNode",
    "LLMGraphRelationship",
    "LLMStrategy",
    "ModelModifications",
    "ModelingMode",
    "NodeSource",
    # Sources
    "PropertySource",
    "RelationshipSource",
    "RenamePropertyOperation",
]
