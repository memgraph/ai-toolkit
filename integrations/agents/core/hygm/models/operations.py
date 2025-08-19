"""
Operation models for interactive graph model modifications.

These models define the structure for operations that can be applied to graph models
during interactive sessions.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelOperation(BaseModel):
    """Base class for model operations."""

    operation_type: str = Field(description="Type of operation to perform")
    description: str = Field(description="Human-readable description of the operation")

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"


class ChangeNodeLabelOperation(ModelOperation):
    """Operation to change a node's label."""

    operation_type: str = "change_node_label"
    old_label: str = Field(description="Current node label")
    new_label: str = Field(description="New node label")


class RenamePropertyOperation(ModelOperation):
    """Operation to rename a property."""

    operation_type: str = "rename_property"
    node_label: str = Field(description="Node label containing the property")
    old_property: str = Field(description="Current property name")
    new_property: str = Field(description="New property name")


class DropPropertyOperation(ModelOperation):
    """Operation to drop a property."""

    operation_type: str = "drop_property"
    node_label: str = Field(description="Node label containing the property")
    property_name: str = Field(description="Property name to drop")


class AddPropertyOperation(ModelOperation):
    """Operation to add a property."""

    operation_type: str = "add_property"
    node_label: str = Field(description="Node label to add property to")
    property_name: str = Field(description="Property name to add")


class ChangeRelationshipNameOperation(ModelOperation):
    """Operation to change a relationship name."""

    operation_type: str = "change_relationship_name"
    old_name: str = Field(description="Current relationship name")
    new_name: str = Field(description="New relationship name")


class DropRelationshipOperation(ModelOperation):
    """Operation to drop a relationship."""

    operation_type: str = "drop_relationship"
    relationship_name: str = Field(description="Relationship name to drop")


class AddIndexOperation(ModelOperation):
    """Operation to add an index."""

    operation_type: str = "add_index"
    node_label: str = Field(description="Node label for the index")
    property_name: str = Field(description="Property name for the index")


class DropIndexOperation(ModelOperation):
    """Operation to drop an index."""

    operation_type: str = "drop_index"
    node_label: str = Field(description="Node label for the index")
    property_name: str = Field(description="Property name for the index")


class ModelModifications(BaseModel):
    """Container for multiple model operations."""

    operations: List[ModelOperation] = Field(
        description="List of operations to apply to the graph model"
    )
    reasoning: str = Field(
        description="Explanation of why these changes improve the model"
    )

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"
