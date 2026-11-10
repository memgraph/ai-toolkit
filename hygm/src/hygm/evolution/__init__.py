"""Layer 6: how a model changes (docs/contract.md, section 11).

Operations are pure functions from a model to a model, with a JSON form;
`apply` runs a list of them, `diff` finds the list between two models. What
moving the data takes is each consumer's: hygm says what changed.
"""

from .apply import apply
from .diff import diff
from .operations import (
    OPERATIONS,
    AddNodeType,
    AddProperty,
    AddRelationType,
    MergeNodeTypes,
    MergeRelationTypes,
    Operation,
    OrderNodeTypes,
    OrderProperties,
    OrderRelationTypes,
    RemoveNodeType,
    RemoveProperty,
    RemoveRelationType,
    RenameNodeType,
    RenameProperty,
    RenameRelationType,
    SetDescription,
    SetEndpoints,
    SetIri,
    SetKey,
    SetParents,
    SetProfile,
    SetProperty,
    operation_from_mapping,
    operation_to_mapping,
)

__all__ = [
    "OPERATIONS",
    "AddNodeType",
    "AddProperty",
    "AddRelationType",
    "MergeNodeTypes",
    "MergeRelationTypes",
    "Operation",
    "OrderNodeTypes",
    "OrderProperties",
    "OrderRelationTypes",
    "RemoveNodeType",
    "RemoveProperty",
    "RemoveRelationType",
    "RenameNodeType",
    "RenameProperty",
    "RenameRelationType",
    "SetDescription",
    "SetEndpoints",
    "SetIri",
    "SetKey",
    "SetParents",
    "SetProfile",
    "SetProperty",
    "apply",
    "diff",
    "operation_from_mapping",
    "operation_to_mapping",
]
