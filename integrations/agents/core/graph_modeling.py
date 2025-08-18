"""
Hypothetical Graph Modeling (HyGM) Module

This module uses LLM to analyze database schemas and provide intelligent
graph modeling suggestions for optimal MySQL to Memgraph migration.
Supports both automatic and interactive modeling modes.
"""

import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class GraphModelingStrategy(Enum):
    """Graph modeling strategies available."""

    DETERMINISTIC = "deterministic"  # Rule-based graph creation
    LLM_POWERED = "llm_powered"  # LLM generates the graph model


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
    """Complete LLM-generated graph model."""

    nodes: List[LLMGraphNode] = Field(description="All nodes in the graph model")
    relationships: List[LLMGraphRelationship] = Field(
        description="All relationships in the graph model"
    )

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"  # This ensures additionalProperties = false


# Structured output models for LLM operations
class ModelOperation(BaseModel):
    """Base model for graph model operations."""

    type: Literal[
        "change_node_label",
        "rename_property",
        "drop_property",
        "add_property",
        "change_relationship_name",
        "drop_relationship",
        "add_index",
        "drop_index",
    ]


class ChangeNodeLabelOperation(ModelOperation):
    """Change a node's label."""

    type: Literal["change_node_label"] = "change_node_label"
    target: str = Field(description="Current node label to change")
    new_value: str = Field(description="New label for the node")


class RenamePropertyOperation(ModelOperation):
    """Rename a property in a node."""

    type: Literal["rename_property"] = "rename_property"
    node: str = Field(description="Node label containing the property")
    target: str = Field(description="Current property name")
    new_value: str = Field(description="New property name")


class DropPropertyOperation(ModelOperation):
    """Drop a property from a node."""

    type: Literal["drop_property"] = "drop_property"
    node: str = Field(description="Node label containing the property")
    target: str = Field(description="Property name to drop")


class AddPropertyOperation(ModelOperation):
    """Add a property to a node."""

    type: Literal["add_property"] = "add_property"
    node: str = Field(description="Node label to add property to")
    new_value: str = Field(description="New property name to add")


class ChangeRelationshipNameOperation(ModelOperation):
    """Change a relationship's name."""

    type: Literal["change_relationship_name"] = "change_relationship_name"
    target: str = Field(description="Current relationship name")
    new_value: str = Field(description="New relationship name")


class DropRelationshipOperation(ModelOperation):
    """Drop a relationship."""

    type: Literal["drop_relationship"] = "drop_relationship"
    target: str = Field(description="Relationship name to drop")


class AddIndexOperation(ModelOperation):
    """Add an index to a node property."""

    type: Literal["add_index"] = "add_index"
    node: str = Field(description="Node label")
    property: str = Field(description="Property name to index")


class DropIndexOperation(ModelOperation):
    """Drop an index from a node property."""

    type: Literal["drop_index"] = "drop_index"
    node: str = Field(description="Node label")
    property: str = Field(description="Property name to remove index from")


class ModelModifications(BaseModel):
    """Container for all model modification operations."""

    operations: List[
        ChangeNodeLabelOperation
        | RenamePropertyOperation
        | DropPropertyOperation
        | AddPropertyOperation
        | ChangeRelationshipNameOperation
        | DropRelationshipOperation
        | AddIndexOperation
        | DropIndexOperation
    ] = Field(description="List of operations to apply to the graph model")


class ModelingMode(Enum):
    """Modeling modes for HyGM."""

    AUTOMATIC = "automatic"
    INTERACTIVE = "interactive"


@dataclass
class PropertySource:
    """Source information for a property."""

    field: str  # Source field name
    transformation: Optional[str] = None  # Any transformation applied


@dataclass
class NodeSource:
    """Source information for a node."""

    type: str  # "table", "view", "file", "api", "manual"
    name: str  # Source name (e.g., table name)
    location: str  # Full location path
    mapping: Dict[str, Any]  # Mapping details for labels, id_field, etc.


@dataclass
class RelationshipSource:
    """Source information for a relationship."""

    type: str  # "table", "view", "junction_table", "derived"
    name: str  # Source name
    location: str  # Full location path
    mapping: Dict[str, Any]  # Mapping for start_node, end_node, edge_type


@dataclass
class IndexSource:
    """Source information for an index."""

    origin: str  # "migration_requirement", "performance_optimization", etc.
    reason: str  # Why this index was created
    created_by: str  # Who/what created it


@dataclass
class ConstraintSource:
    """Source information for a constraint."""

    origin: str  # "source_database_constraint", "business_rule", etc.
    constraint_name: Optional[str] = None  # Original constraint name
    migrated_from: Optional[str] = None  # Source location
    reason: Optional[str] = None  # Business reason
    created_by: Optional[str] = None  # Creator


@dataclass
class EnumSource:
    """Source information for an enum."""

    origin: str  # "source_database_enum", "manual", etc.
    enum_name: Optional[str] = None  # Original enum name
    migrated_from: Optional[str] = None  # Source location


@dataclass
class GraphProperty:
    """Represents a property with full schema format details."""

    key: str
    count: int = 1
    filling_factor: float = 100.0
    types: List[Dict[str, Any]] = None
    source: Optional[PropertySource] = None

    def __post_init__(self):
        if self.types is None:
            self.types = [{"type": "String", "count": 1, "examples": [""]}]


@dataclass
class GraphNode:
    """Represents a node in the graph model aligned with schema format."""

    labels: List[str]  # Node labels
    count: int = 1
    properties: List[GraphProperty] = None
    examples: List[Dict[str, Any]] = None
    source: Optional[NodeSource] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = []
        if self.examples is None:
            self.examples = [{"gid": 0}]

    @property
    def primary_label(self) -> str:
        """Get the primary (first) label for backward compatibility."""
        return self.labels[0] if self.labels else "Unknown"


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph model aligned with schema format."""

    edge_type: str
    start_node_labels: List[str]
    end_node_labels: List[str]
    count: int = 1
    properties: List[GraphProperty] = None
    examples: List[Dict[str, Any]] = None
    source: Optional[RelationshipSource] = None
    directionality: str = "directed"

    def __post_init__(self):
        if self.properties is None:
            self.properties = []
        if self.examples is None:
            self.examples = [{}]


@dataclass
class GraphIndex:
    """Represents an index aligned with schema format."""

    labels: Optional[List[str]] = None  # For node indexes
    edge_type: Optional[str] = None  # For edge indexes
    properties: List[str] = None
    count: int = 0
    examples: List[Dict[str, Any]] = None
    type: str = "label+property"  # Index type
    source: Optional[IndexSource] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = []
        if self.examples is None:
            self.examples = [{}]


@dataclass
class GraphConstraint:
    """Represents a constraint aligned with schema format."""

    type: str  # "unique", "existence", "data_type"
    labels: Optional[List[str]] = None  # For node constraints
    edge_type: Optional[str] = None  # For edge constraints
    properties: List[str] = None
    data_type: Optional[str] = None
    source: Optional[ConstraintSource] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = []


@dataclass
class GraphEnum:
    """Represents an enum aligned with schema format."""

    name: str
    values: List[str]
    source: Optional[EnumSource] = None


@dataclass
class GraphModel:
    """Complete graph model aligned with schema format."""

    nodes: List[GraphNode]
    edges: List[GraphRelationship]
    node_indexes: List[GraphIndex] = None
    edge_indexes: List[GraphIndex] = None
    node_constraints: List[GraphConstraint] = None
    edge_constraints: List[GraphConstraint] = None
    enums: List[GraphEnum] = None

    def __post_init__(self):
        if self.node_indexes is None:
            self.node_indexes = []
        if self.edge_indexes is None:
            self.edge_indexes = []
        if self.node_constraints is None:
            self.node_constraints = []
        if self.edge_constraints is None:
            self.edge_constraints = []
        if self.enums is None:
            self.enums = []

    @classmethod
    def from_schema_format(cls, schema_dict: Dict[str, Any]) -> "GraphModel":
        """
        Create a GraphModel from schema format dictionary.

        Args:
            schema_dict: Dictionary in the comprehensive schema format

        Returns:
            GraphModel instance
        """
        # Convert nodes
        nodes = []
        for node_dict in schema_dict.get("nodes", []):
            # Convert properties
            properties = []
            for prop_dict in node_dict.get("properties", []):
                prop_source = None
                if "source" in prop_dict:
                    prop_source = PropertySource(
                        field=prop_dict["source"]["field"],
                        transformation=prop_dict["source"].get("transformation"),
                    )

                graph_prop = GraphProperty(
                    key=prop_dict["key"],
                    count=prop_dict.get("count", 1),
                    filling_factor=prop_dict.get("filling_factor", 100.0),
                    types=prop_dict.get("types", []),
                    source=prop_source,
                )
                properties.append(graph_prop)

            # Convert source
            node_source = None
            if "source" in node_dict:
                node_source = NodeSource(
                    type=node_dict["source"]["type"],
                    name=node_dict["source"]["name"],
                    location=node_dict["source"]["location"],
                    mapping=node_dict["source"]["mapping"],
                )

            node = GraphNode(
                labels=node_dict["labels"],
                count=node_dict.get("count", 1),
                properties=properties,
                examples=node_dict.get("examples", [{"gid": 0}]),
                source=node_source,
            )
            nodes.append(node)

        # Convert edges
        edges = []
        for edge_dict in schema_dict.get("edges", []):
            # Convert properties
            properties = []
            for prop_dict in edge_dict.get("properties", []):
                prop_source = None
                if "source" in prop_dict:
                    prop_source = PropertySource(
                        field=prop_dict["source"]["field"],
                        transformation=prop_dict["source"].get("transformation"),
                    )

                graph_prop = GraphProperty(
                    key=prop_dict["key"],
                    count=prop_dict.get("count", 1),
                    filling_factor=prop_dict.get("filling_factor", 100.0),
                    types=prop_dict.get("types", []),
                    source=prop_source,
                )
                properties.append(graph_prop)

            # Convert source
            edge_source = None
            if "source" in edge_dict:
                edge_source = RelationshipSource(
                    type=edge_dict["source"]["type"],
                    name=edge_dict["source"]["name"],
                    location=edge_dict["source"]["location"],
                    mapping=edge_dict["source"]["mapping"],
                )

            edge = GraphRelationship(
                edge_type=edge_dict["edge_type"],
                start_node_labels=edge_dict["start_node_labels"],
                end_node_labels=edge_dict["end_node_labels"],
                count=edge_dict.get("count", 1),
                properties=properties,
                examples=edge_dict.get("examples", [{}]),
                source=edge_source,
            )
            edges.append(edge)

        # Convert indexes
        node_indexes = []
        for index_dict in schema_dict.get("node_indexes", []):
            index_source = None
            if "source" in index_dict:
                index_source = IndexSource(
                    origin=index_dict["source"]["origin"],
                    reason=index_dict["source"]["reason"],
                    created_by=index_dict["source"]["created_by"],
                )

            index = GraphIndex(
                labels=index_dict.get("labels"),
                properties=index_dict.get("properties", []),
                count=index_dict.get("count", 0),
                examples=index_dict.get("examples", [{}]),
                type=index_dict.get("type", "label+property"),
                source=index_source,
            )
            node_indexes.append(index)

        edge_indexes = []
        for index_dict in schema_dict.get("edge_indexes", []):
            index_source = None
            if "source" in index_dict:
                index_source = IndexSource(
                    origin=index_dict["source"]["origin"],
                    reason=index_dict["source"]["reason"],
                    created_by=index_dict["source"]["created_by"],
                )

            index = GraphIndex(
                edge_type=index_dict.get("edge_type"),
                properties=index_dict.get("properties", []),
                count=index_dict.get("count", 0),
                examples=index_dict.get("examples", [{}]),
                type=index_dict.get("type", "edge_type+property"),
                source=index_source,
            )
            edge_indexes.append(index)

        # Convert constraints
        node_constraints = []
        for constraint_dict in schema_dict.get("node_constraints", []):
            constraint_source = None
            if "source" in constraint_dict:
                constraint_source = ConstraintSource(
                    origin=constraint_dict["source"]["origin"],
                    constraint_name=constraint_dict["source"].get("constraint_name"),
                    migrated_from=constraint_dict["source"].get("migrated_from"),
                    reason=constraint_dict["source"].get("reason"),
                    created_by=constraint_dict["source"].get("created_by"),
                )

            constraint = GraphConstraint(
                type=constraint_dict["type"],
                labels=constraint_dict.get("labels"),
                properties=constraint_dict.get("properties", []),
                data_type=constraint_dict.get("data_type"),
                source=constraint_source,
            )
            node_constraints.append(constraint)

        edge_constraints = []
        for constraint_dict in schema_dict.get("edge_constraints", []):
            constraint_source = None
            if "source" in constraint_dict:
                constraint_source = ConstraintSource(
                    origin=constraint_dict["source"]["origin"],
                    constraint_name=constraint_dict["source"].get("constraint_name"),
                    migrated_from=constraint_dict["source"].get("migrated_from"),
                    reason=constraint_dict["source"].get("reason"),
                    created_by=constraint_dict["source"].get("created_by"),
                )

            constraint = GraphConstraint(
                type=constraint_dict["type"],
                edge_type=constraint_dict.get("edge_type"),
                properties=constraint_dict.get("properties", []),
                data_type=constraint_dict.get("data_type"),
                source=constraint_source,
            )
            edge_constraints.append(constraint)

        # Convert enums
        enums = []
        for enum_dict in schema_dict.get("enums", []):
            enum_source = None
            if "source" in enum_dict:
                enum_source = EnumSource(
                    origin=enum_dict["source"]["origin"],
                    enum_name=enum_dict["source"].get("enum_name"),
                    migrated_from=enum_dict["source"].get("migrated_from"),
                )

            enum = GraphEnum(
                name=enum_dict["name"], values=enum_dict["values"], source=enum_source
            )
            enums.append(enum)

        return cls(
            nodes=nodes,
            edges=edges,
            node_indexes=node_indexes,
            edge_indexes=edge_indexes,
            node_constraints=node_constraints,
            edge_constraints=edge_constraints,
            enums=enums,
        )

    def to_schema_format(
        self, sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Convert the GraphModel to the comprehensive schema format.

        This method now directly returns the schema format since the internal
        structure already matches it.

        Args:
            sample_data: Optional sample data dictionary with
                table_name -> list of rows

        Returns:
            Dictionary in the specified schema format with nodes, edges,
            indexes, constraints, and enums
        """
        # Convert internal structures to schema format
        schema_nodes = []
        for node in self.nodes:
            schema_node = self._node_to_schema_dict(node, sample_data)
            schema_nodes.append(schema_node)

        schema_edges = []
        for edge in self.edges:
            schema_edge = self._edge_to_schema_dict(edge, sample_data)
            schema_edges.append(schema_edge)

        schema_node_indexes = []
        for index in self.node_indexes:
            schema_index = self._index_to_schema_dict(index)
            schema_node_indexes.append(schema_index)

        schema_edge_indexes = []
        for index in self.edge_indexes:
            schema_index = self._index_to_schema_dict(index)
            schema_edge_indexes.append(schema_index)

        schema_node_constraints = []
        for constraint in self.node_constraints:
            schema_constraint = self._constraint_to_schema_dict(constraint)
            schema_node_constraints.append(schema_constraint)

        schema_edge_constraints = []
        for constraint in self.edge_constraints:
            schema_constraint = self._constraint_to_schema_dict(constraint)
            schema_edge_constraints.append(schema_constraint)

        schema_enums = []
        for enum in self.enums:
            schema_enum = self._enum_to_schema_dict(enum)
            schema_enums.append(schema_enum)

        return {
            "nodes": schema_nodes,
            "edges": schema_edges,
            "node_indexes": schema_node_indexes,
            "edge_indexes": schema_edge_indexes,
            "node_constraints": schema_node_constraints,
            "edge_constraints": schema_edge_constraints,
            "enums": schema_enums,
        }

    def _node_to_schema_dict(
        self, node: GraphNode, sample_data: Optional[Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """Convert GraphNode to schema dictionary format."""
        # Convert properties to schema format
        schema_properties = []
        for prop in node.properties:
            if isinstance(prop, GraphProperty):
                prop_dict = {
                    "key": prop.key,
                    "count": prop.count,
                    "filling_factor": prop.filling_factor,
                    "types": prop.types,
                }
                if prop.source:
                    prop_dict["source"] = {
                        "field": prop.source.field,
                        "transformation": prop.source.transformation,
                    }
                schema_properties.append(prop_dict)
            else:
                # Handle legacy string properties
                prop_dict = self._convert_property_to_schema(prop, [])
                schema_properties.append(prop_dict)

        schema_node = {
            "labels": node.labels,
            "count": node.count,
            "properties": schema_properties,
            "examples": node.examples,
        }

        if node.source:
            schema_node["source"] = {
                "type": node.source.type,
                "name": node.source.name,
                "location": node.source.location,
                "mapping": node.source.mapping,
            }

        return schema_node

    def _edge_to_schema_dict(
        self,
        edge: GraphRelationship,
        sample_data: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """Convert GraphRelationship to schema dictionary format."""
        # Convert properties to schema format
        schema_properties = []
        for prop in edge.properties:
            if isinstance(prop, GraphProperty):
                prop_dict = {
                    "key": prop.key,
                    "count": prop.count,
                    "filling_factor": prop.filling_factor,
                    "types": prop.types,
                }
                if prop.source:
                    prop_dict["source"] = {
                        "field": prop.source.field,
                        "transformation": prop.source.transformation,
                    }
                schema_properties.append(prop_dict)
            else:
                # Handle legacy string properties
                prop_dict = self._convert_property_to_schema(prop, [])
                schema_properties.append(prop_dict)

        schema_edge = {
            "edge_type": edge.edge_type,
            "start_node_labels": edge.start_node_labels,
            "end_node_labels": edge.end_node_labels,
            "count": edge.count,
            "properties": schema_properties,
            "examples": edge.examples,
        }

        if edge.source:
            schema_edge["source"] = {
                "type": edge.source.type,
                "name": edge.source.name,
                "location": edge.source.location,
                "mapping": edge.source.mapping,
            }

        return schema_edge

    def _index_to_schema_dict(self, index: GraphIndex) -> Dict[str, Any]:
        """Convert GraphIndex to schema dictionary format."""
        schema_index = {
            "properties": index.properties,
            "count": index.count,
            "examples": index.examples,
            "type": index.type,
        }

        if index.labels:
            schema_index["labels"] = index.labels
        if index.edge_type:
            schema_index["edge_type"] = index.edge_type

        if index.source:
            schema_index["source"] = {
                "origin": index.source.origin,
                "reason": index.source.reason,
                "created_by": index.source.created_by,
            }

        return schema_index

    def _constraint_to_schema_dict(self, constraint: GraphConstraint) -> Dict[str, Any]:
        """Convert GraphConstraint to schema dictionary format."""
        schema_constraint = {
            "type": constraint.type,
            "properties": constraint.properties,
        }

        if constraint.labels:
            schema_constraint["labels"] = constraint.labels
        if constraint.edge_type:
            schema_constraint["edge_type"] = constraint.edge_type
        if constraint.data_type:
            schema_constraint["data_type"] = constraint.data_type

        if constraint.source:
            source_dict = {
                "origin": constraint.source.origin,
            }
            if constraint.source.constraint_name:
                source_dict["constraint_name"] = constraint.source.constraint_name
            if constraint.source.migrated_from:
                source_dict["migrated_from"] = constraint.source.migrated_from
            if constraint.source.reason:
                source_dict["reason"] = constraint.source.reason
            if constraint.source.created_by:
                source_dict["created_by"] = constraint.source.created_by
            schema_constraint["source"] = source_dict

        return schema_constraint

    def _enum_to_schema_dict(self, enum: GraphEnum) -> Dict[str, Any]:
        """Convert GraphEnum to schema dictionary format."""
        schema_enum = {
            "name": enum.name,
            "values": enum.values,
        }

        if enum.source:
            source_dict = {
                "origin": enum.source.origin,
            }
            if enum.source.enum_name:
                source_dict["enum_name"] = enum.source.enum_name
            if enum.source.migrated_from:
                source_dict["migrated_from"] = enum.source.migrated_from
            schema_enum["source"] = source_dict

        return schema_enum

    def _convert_node_to_schema(
        self, node: GraphNode, sample_data: Optional[Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """Convert a GraphNode to the schema node format."""
        node_schema = {
            "labels": [node.label],
            "count": 1,  # Default, would need actual data
            "properties": [],
            "examples": [{"gid": 0}],
        }

        # Get sample data for this node's source table
        table_samples = sample_data.get(node.source_table, []) if sample_data else []

        # Convert properties
        for prop in node.properties:
            prop_schema = self._convert_property_to_schema(prop, table_samples)
            node_schema["properties"].append(prop_schema)

        # Add examples from sample data
        if table_samples:
            node_schema["examples"] = [
                {"gid": i, **{k: v for k, v in row.items() if k in node.properties}}
                for i, row in enumerate(table_samples[:3])
            ]

        return node_schema

    def _convert_relationship_to_schema(
        self,
        relationship: GraphRelationship,
        sample_data: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """Convert a GraphRelationship to the schema edge format."""
        # Find the node labels for start and end nodes
        start_labels = [relationship.from_node]
        end_labels = [relationship.to_node]

        # Try to find actual labels from nodes
        for node in self.nodes:
            if (
                node.source_table == relationship.from_node
                or node.name == relationship.from_node
            ):
                start_labels = [node.label]
            if (
                node.source_table == relationship.to_node
                or node.name == relationship.to_node
            ):
                end_labels = [node.label]

        edge_schema = {
            "edge_type": relationship.name,
            "start_node_labels": start_labels,
            "end_node_labels": end_labels,
            "count": 1,  # Default, would need actual data
            "properties": [],
            "examples": [{}],
        }

        # Convert relationship properties if any
        if relationship.properties:
            for prop in relationship.properties:
                prop_schema = self._convert_property_to_schema(prop, [])
                edge_schema["properties"].append(prop_schema)

        return edge_schema

    def _convert_property_to_schema(
        self, prop_name: str, sample_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert a property to schema format with type detection."""
        prop_schema = {
            "key": prop_name,
            "count": 1,
            "filling_factor": 100.00,
            "types": [],
        }

        # Analyze sample data to determine types
        if sample_rows:
            type_counts = {}
            examples_by_type = {}

            for row in sample_rows:
                value = row.get(prop_name)
                detected_type = self._detect_value_type(value)

                current_count = type_counts.get(detected_type, 0)
                type_counts[detected_type] = current_count + 1
                if detected_type not in examples_by_type:
                    examples_by_type[detected_type] = []
                if len(examples_by_type[detected_type]) < 3:
                    examples_by_type[detected_type].append(value)

            # Create type entries
            for type_name, count in type_counts.items():
                type_entry = {
                    "type": type_name,
                    "count": count,
                    "examples": examples_by_type[type_name][:3],
                }
                prop_schema["types"].append(type_entry)
        else:
            # Default type when no sample data
            prop_schema["types"] = [{"type": "String", "count": 1, "examples": [""]}]

        return prop_schema

    def _detect_value_type(self, value: Any) -> str:
        """Detect the type of a value and return the schema type name."""
        if value is None:
            return "Null"
        elif isinstance(value, bool):
            return "Boolean"
        elif isinstance(value, int):
            return "Integer"
        elif isinstance(value, float):
            return "Float"
        elif isinstance(value, list):
            return "List"
        elif isinstance(value, dict):
            return "Map"
        elif isinstance(value, str):
            # Check for special string types
            if self._is_date_string(value):
                return "Date"
            elif self._is_datetime_string(value):
                return "LocalDateTime"
            elif self._is_time_string(value):
                return "LocalTime"
            else:
                return "String"
        else:
            return "String"  # Default fallback

    def _is_date_string(self, value: str) -> bool:
        """Check if string looks like a date."""
        import re

        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
            r"^\d{2}/\d{2}/\d{4}$",  # MM/DD/YYYY
            r"^\d{2}-\d{2}-\d{4}$",  # MM-DD-YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)

    def _is_datetime_string(self, value: str) -> bool:
        """Check if string looks like a datetime."""
        import re

        datetime_patterns = [
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # YYYY-MM-DD HH:MM:SS
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format
        ]
        return any(re.match(pattern, value) for pattern in datetime_patterns)

    def _is_time_string(self, value: str) -> bool:
        """Check if string looks like a time."""
        import re

        return bool(re.match(r"^\d{2}:\d{2}:\d{2}$", value))

    def _detect_enums_from_sample_data(
        self, sample_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect enum-like columns from sample data."""
        enums = []

        for table_name, rows in sample_data.items():
            if not rows:
                continue

            # Analyze each column for enum-like behavior
            for column in rows[0].keys():
                values = set()
                total_values = 0

                for row in rows:
                    value = row.get(column)
                    if isinstance(value, str) and value:
                        values.add(value)
                        total_values += 1

                # Check if few unique values compared to total (potential enum)
                if len(values) <= 10 and total_values > len(values) * 2:
                    enum_schema = {
                        "name": f"{table_name}_{column}",
                        "values": sorted(list(values)),
                    }
                    enums.append(enum_schema)

        return enums


class HyGM:
    """
    Uses LLM to create intelligent graph models from relational schemas.

    Supports two modes:
    - AUTOMATIC: Creates graph model without user interaction
    - INTERACTIVE: Interactive mode with user feedback via terminal input
    """

    def __init__(
        self,
        llm,
        mode: ModelingMode = ModelingMode.AUTOMATIC,
        strategy: GraphModelingStrategy = GraphModelingStrategy.DETERMINISTIC,
    ):
        """Initialize with an LLM instance, modeling mode, and strategy."""
        self.llm = llm
        self.mode = mode
        self.strategy = strategy
        self.current_graph_model = None
        self.iteration_count = 0
        self.database_structure = None

    def create_graph_model(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        strategy: Optional[GraphModelingStrategy] = None,
    ) -> GraphModel:
        """
        Create a graph model using the specified strategy.

        Args:
            database_structure: Database structure from data_interface
            domain_context: Optional domain context for better modeling
            strategy: Override the default strategy for this call

        Returns:
            GraphModel created using the specified strategy
        """
        used_strategy = strategy or self.strategy

        logger.info(f"Creating graph model using {used_strategy.value} strategy...")

        if used_strategy == GraphModelingStrategy.LLM_POWERED:
            return self._llm_powered_modeling(database_structure, domain_context)
        else:
            # Use existing deterministic approach
            return self.model_graph(database_structure, domain_context)

    def _llm_powered_modeling(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> GraphModel:
        """
        Use LLM to generate the complete graph model from scratch.

        This method sends the database schema and sample data to the LLM
        and asks it to design the optimal graph model.
        """
        logger.info("Using LLM to generate graph model...")

        try:
            # Create system prompt for graph modeling
            system_prompt = self._create_llm_modeling_system_prompt()

            # Create human prompt with database structure and sample data
            human_prompt = self._create_llm_modeling_human_prompt(
                database_structure, domain_context
            )

            # Get structured response from LLM
            structured_llm = self.llm.with_structured_output(LLMGraphModel)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            llm_model = structured_llm.invoke(messages)

            # Convert LLM model to standard GraphModel format
            graph_model = self._convert_llm_model_to_graph_model(
                llm_model, database_structure
            )

            logger.info("LLM-powered graph modeling completed")
            return graph_model

        except Exception as e:
            logger.error(f"LLM-powered modeling failed: {e}")
            logger.info("Falling back to deterministic modeling...")
            return self.model_graph(database_structure, domain_context)

    def _create_llm_modeling_system_prompt(self) -> str:
        """Create system prompt for LLM graph modeling."""
        return """You are an expert graph database architect specializing in
converting relational database schemas into optimal graph models for Memgraph.

Your task is to analyze the provided SQL database schema and sample data, then
design the best possible graph model that:

1. PRESERVES DATA INTEGRITY: All important data relationships are maintained
2. MAPS TO ACTUAL SCHEMA: Use the exact table and column names provided
3. OPTIMIZES FOR GRAPH QUERIES: Structure enables efficient traversals
4. FOLLOWS GRAPH BEST PRACTICES: Uses appropriate node/relationship patterns
5. CONSIDERS DOMAIN SEMANTICS: Understands the business meaning of the data

CRITICAL REQUIREMENTS:
- For nodes: Use the exact table names as 'source_table', create semantic 'label' names
- For node properties: Use the exact column names from the schema
- For node primary keys: Use the exact primary key column name from the schema
- For relationships: Reference nodes using their exact 'source_table' names
- For relationship mapping: Base relationships ONLY on the foreign keys shown in the schema
- For many-to-many: Use join tables exactly as defined in the schema

Key principles:
- Entity tables become nodes with meaningful labels but preserve source_table
- Foreign key relationships become directed edges with proper column mapping
- Many-to-many relationships use the join table properties on edges
- Choose descriptive relationship names (OWNS, BELONGS_TO, etc.) but map to actual FKs
- Primary keys and column names must match the database schema exactly
- Index frequently queried properties using their actual column names

DO NOT create relationships that don't correspond to actual foreign keys in the database."""

    def _create_llm_modeling_human_prompt(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> str:
        """Create human prompt with database details for LLM modeling."""

        # Build schema description
        schema_parts = ["DATABASE SCHEMA:"]
        schema_parts.append(
            "IMPORTANT: Use exact table and column names in your model."
        )
        schema_parts.append(
            "For relationships, reference nodes by their source_table names."
        )

        entity_tables = database_structure.get("entity_tables", {})
        join_tables = database_structure.get("join_tables", {})
        relationships = database_structure.get("relationships", [])

        # Describe entity tables
        if entity_tables:
            schema_parts.append("\nENTITY TABLES:")
            for table_name, table_info in entity_tables.items():
                schema_parts.append(f"\n{table_name}:")
                schema_parts.append(f"  Rows: {table_info.get('row_count', 0)}")

                # Add column information
                schema_list = table_info.get("schema", [])
                if schema_list:
                    schema_parts.append("  Columns:")
                    for col in schema_list:
                        col_desc = f"    - {col.get('field', 'unknown')}"
                        col_desc += f" ({col.get('type', 'unknown')})"
                        if col.get("key") == "PRI":
                            col_desc += " [PRIMARY KEY]"
                        if col.get("key") == "MUL":
                            col_desc += " [FOREIGN KEY]"
                        schema_parts.append(col_desc)

        # Describe join tables
        if join_tables:
            schema_parts.append("\nJOIN TABLES:")
            for table_name, table_info in join_tables.items():
                schema_parts.append(f"\n{table_name}:")
                schema_parts.append(f"  Rows: {table_info.get('row_count', 0)}")

                # Show foreign keys
                fks = table_info.get("foreign_keys", [])
                if fks:
                    schema_parts.append("  Foreign Keys:")
                    for fk in fks:
                        fk_desc = f"    - {fk.get('column')} -> "
                        fk_desc += f"{fk.get('referenced_table')}."
                        fk_desc += f"{fk.get('referenced_column')}"
                        schema_parts.append(fk_desc)

        # Describe relationships with detailed foreign key mapping
        if relationships:
            schema_parts.append("\nFOREIGN KEY RELATIONSHIPS:")
            schema_parts.append("These are the ONLY relationships you should model:")
            for rel in relationships:
                rel_desc = f"  - {rel.get('from_table')}.{rel.get('from_column')} "
                rel_desc += f"-> {rel.get('to_table')}.{rel.get('to_column')}"
                rel_desc += f" [constraint: {rel.get('constraint_name', 'unnamed')}]"
                schema_parts.append(rel_desc)

            schema_parts.append("\nFor relationships:")
            schema_parts.append("- Use from_node = source table name")
            schema_parts.append("- Use to_node = target table name")
            schema_parts.append("- Create semantic relationship names")
            schema_parts.append("- Map to the exact foreign key constraints above")

        # Describe relationships
        if relationships:
            schema_parts.append("\nRELATIONSHIPS:")
            for rel in relationships:
                rel_desc = f"  - {rel.get('type', 'unknown')}: "
                rel_desc += f"{rel.get('from_table')}."
                rel_desc += f"{rel.get('from_column')} "
                rel_desc += f"-> {rel.get('to_table')}."
                rel_desc += f"{rel.get('to_column')}"
                schema_parts.append(rel_desc)

        # Add sample data if available
        sample_data = database_structure.get("sample_data", {})
        if sample_data:
            schema_parts.append("\nSAMPLE DATA:")
            for table_name, samples in sample_data.items():
                if samples:
                    schema_parts.append(f"\n{table_name} (first few rows):")
                    for i, row in enumerate(samples[:3]):
                        schema_parts.append(f"  Row {i+1}: {row}")

        # Add domain context if provided
        if domain_context:
            schema_parts.append(f"\nDOMAIN CONTEXT: {domain_context}")

        schema_parts.append(
            "\nCreate an optimal graph model for this database structure."
        )

        return "\n".join(schema_parts)

    def _convert_llm_model_to_graph_model(
        self, llm_model: LLMGraphModel, database_structure: Dict[str, Any]
    ) -> GraphModel:
        """Convert LLM-generated model to standard GraphModel format."""

        # Convert nodes
        nodes = []
        for llm_node in llm_model.nodes:
            # Create source information
            source = NodeSource(
                type="table",
                name=llm_node.source_table,
                location=f"database.schema.{llm_node.source_table}",
                mapping={"labels": [llm_node.label], "id_field": llm_node.primary_key},
            )

            # Convert properties to GraphProperty objects
            graph_properties = []
            for prop_name in llm_node.properties:
                prop_source = PropertySource(
                    field=f"{llm_node.source_table}.{prop_name}"
                )
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                graph_properties.append(graph_prop)

            node = GraphNode(
                labels=[llm_node.label], properties=graph_properties, source=source
            )
            nodes.append(node)

        # Convert relationships
        relationships = []
        for llm_rel in llm_model.relationships:
            # Find source info from database structure
            source_info = self._find_relationship_source_info(
                llm_rel, database_structure
            )

            # Create relationship source
            rel_source = RelationshipSource(
                type="table"
                if source_info.get("type") != "many_to_many"
                else "junction_table",
                name=source_info.get("constraint_name", llm_rel.name),
                location=f"database.schema.{source_info.get('from_table', llm_rel.from_node)}",
                mapping={
                    "start_node": f"{source_info.get('from_table', llm_rel.from_node)}.{source_info.get('from_column', 'id')}",
                    "end_node": f"{source_info.get('to_table', llm_rel.to_node)}.{source_info.get('to_column', 'id')}",
                    "edge_type": llm_rel.name,
                },
            )

            # Find actual node labels
            start_labels = [self._find_node_label_by_table(llm_rel.from_node, nodes)]
            end_labels = [self._find_node_label_by_table(llm_rel.to_node, nodes)]

            # Convert relationship properties to GraphProperty objects
            graph_properties = []
            for prop_name in llm_rel.properties:
                prop_source = PropertySource(
                    field=f"{source_info.get('from_table', llm_rel.from_node)}.{prop_name}"
                )
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                graph_properties.append(graph_prop)

            relationship = GraphRelationship(
                edge_type=llm_rel.name,
                start_node_labels=start_labels,
                end_node_labels=end_labels,
                properties=graph_properties,
                source=rel_source,
                directionality=llm_rel.directionality,
            )
            relationships.append(relationship)

        # Create indexes from node information
        node_indexes = []
        for llm_node in llm_model.nodes:
            for index_prop in llm_node.indexes:
                index_source = IndexSource(
                    origin="migration_requirement",
                    reason="performance_optimization",
                    created_by="migration_agent",
                )

                graph_index = GraphIndex(
                    labels=[llm_node.label],
                    properties=[index_prop],
                    type="label+property",
                    source=index_source,
                )
                node_indexes.append(graph_index)

        # Create constraints from node information
        node_constraints = []
        for llm_node in llm_model.nodes:
            for constraint_str in llm_node.constraints:
                if "UNIQUE" in constraint_str.upper():
                    prop_name = constraint_str.replace("UNIQUE(", "").replace(")", "")
                    constraint_source = ConstraintSource(
                        origin="source_database_constraint",
                        constraint_name=f"{llm_node.source_table}_{prop_name}_unique",
                        migrated_from=f"database.schema.{llm_node.source_table}",
                    )

                    graph_constraint = GraphConstraint(
                        type="unique",
                        labels=[llm_node.label],
                        properties=[prop_name],
                        source=constraint_source,
                    )
                    node_constraints.append(graph_constraint)

        return GraphModel(
            nodes=nodes,
            edges=relationships,
            node_indexes=node_indexes,
            node_constraints=node_constraints,
        )

    def _find_node_label_by_table(self, table_name: str, nodes: List[GraphNode]) -> str:
        """Find node label by source table name."""
        for node in nodes:
            if node.source and node.source.name == table_name:
                return node.primary_label
        return table_name.title()  # Fallback

    def _find_relationship_source_info(
        self, llm_rel: LLMGraphRelationship, database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find source relationship info from database structure."""

        # Try to match with existing relationships - handle case variations
        for rel in database_structure.get("relationships", []):
            # Check exact match first
            if (
                rel.get("from_table") == llm_rel.from_node
                and rel.get("to_table") == llm_rel.to_node
            ):
                return rel

            # Check case-insensitive match
            from_match = rel.get("from_table", "").lower() == llm_rel.from_node.lower()
            to_match = rel.get("to_table", "").lower() == llm_rel.to_node.lower()
            if from_match and to_match:
                return rel

            # Check reversed relationship (LLM might infer wrong direction)
            rev_from_match = (
                rel.get("from_table", "").lower() == llm_rel.to_node.lower()
            )
            rev_to_match = rel.get("to_table", "").lower() == llm_rel.from_node.lower()
            if rev_from_match and rev_to_match:
                # Return with swapped direction to match LLM's inference
                reversed_rel = rel.copy()
                reversed_rel["from_table"] = llm_rel.from_node
                reversed_rel["to_table"] = llm_rel.to_node
                logger.info(
                    f"Found reversed relationship for "
                    f"{llm_rel.from_node} -> {llm_rel.to_node}"
                )
                return reversed_rel

        # Log when relationship is not found
        rel_info = f"{llm_rel.from_node} -> {llm_rel.to_node}"
        logger.warning(
            f"Could not find relationship source info for "
            f"{rel_info} (type: {llm_rel.type})"
        )

        available_rels = [
            (
                r.get("from_table"),
                r.get("to_table"),
                r.get("constraint_name", "unnamed"),
            )
            for r in database_structure.get("relationships", [])
        ]
        logger.warning(f"Available relationships: {available_rels}")

        # Return minimal info if not found
        return {
            "from_table": llm_rel.from_node,
            "to_table": llm_rel.to_node,
            "type": llm_rel.type,
        }

    def model_graph(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> GraphModel:
        """
        Main entry point for graph modeling.

        Args:
            database_structure: Database structure from data_interface
            domain_context: Optional domain context for better modeling

        Returns:
            GraphModel with intelligent modeling decisions
        """
        logger.info(f"Starting graph modeling in {self.mode.value} mode...")

        self.database_structure = database_structure

        if self.mode == ModelingMode.AUTOMATIC:
            return self._automatic_modeling(database_structure, domain_context)
        else:  # INTERACTIVE mode
            return self._interactive_modeling(database_structure, domain_context)

    def _automatic_modeling(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> GraphModel:
        """
        Automatic graph modeling without user interaction.
        """
        logger.info("Performing automatic graph modeling...")

        # Generate initial model
        graph_model = self._generate_initial_model(database_structure, domain_context)

        # Validate the model
        validation_result = self.validate_graph_model(graph_model, database_structure)

        if not validation_result["is_valid"]:
            logger.warning(
                "Generated model has validation issues, attempting to fix..."
            )
            graph_model = self._fix_validation_issues(graph_model, validation_result)

        logger.info("Automatic graph modeling completed")
        return graph_model

    def _interactive_modeling(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> GraphModel:
        """
        Interactive graph modeling with user feedback via terminal input.
        """
        logger.info("Starting interactive graph modeling...")

        # Generate initial model
        self.current_graph_model = self._generate_initial_model(
            database_structure, domain_context
        )
        self.iteration_count = 0

        # Interactive feedback loop
        while True:
            # Present current model to user
            model_presentation = self._get_model_presentation()

            # Display the model to the user
            print("\n" + "=" * 60)
            print("📊 CURRENT GRAPH MODEL")
            print("=" * 60)
            print(model_presentation)
            print("\n" + "=" * 60)

            # Get user feedback via terminal input
            print(
                "\n🔄 Interactive Graph Modeling - Iteration", self.iteration_count + 1
            )
            print("\nOptions:")
            print("  • Type 'approve' to accept the current model")
            print("  • Type 'quit' to exit interactive mode")
            print("  • Provide natural language feedback to modify the model")
            print("\nExamples of feedback:")
            print("  - 'Change Customer label to Person'")
            print("  - 'Add an index on email property for User nodes'")
            print("  - 'Create a LIVES_IN relationship between Person and Address'")

            try:
                user_feedback = input(
                    "\n💭 Your feedback (or 'approve' to continue): "
                ).strip()
            except KeyboardInterrupt:
                print("\n⚠️ Interactive modeling cancelled by user")
                return self.current_graph_model
            except EOFError:
                print("\n⚠️ End of input - accepting current model")
                return self.current_graph_model

            if not user_feedback or user_feedback.lower() in [
                "approve",
                "accept",
                "done",
            ]:
                print("✅ Graph model approved!")
                break
            elif user_feedback.lower() in ["quit", "exit", "cancel"]:
                print("❌ Interactive modeling cancelled")
                break

            # Apply user feedback
            print(f"\n🔄 Applying feedback: {user_feedback}")
            self._apply_natural_language_feedback(user_feedback)
            self.iteration_count += 1

            # Validate after changes
            validation_result = self.validate_graph_model(
                self.current_graph_model, database_structure
            )
            if not validation_result["is_valid"]:
                print("⚠️ Warning: Model has validation issues after your changes:")
                for issue in validation_result["issues"]:
                    print(f"  - {issue}")

        logger.info(
            f"Interactive modeling completed after {self.iteration_count} "
            f"iterations"
        )
        return self.current_graph_model

    def validate_graph_model(
        self, graph_model: GraphModel, database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate graph model against SQL-to-Graph modeling best practices.
        """
        logger.info("Validating graph model...")

        issues = []
        warnings = []

        # 1. Check all entity tables are represented as nodes
        entity_tables = set(database_structure.get("entity_tables", {}).keys())
        model_source_tables = set()

        for node in graph_model.nodes:
            if node.source and hasattr(node.source, "name"):
                model_source_tables.add(node.source.name)

        missing_tables = entity_tables - model_source_tables
        if missing_tables:
            issues.append(f"Missing nodes for entity tables: {list(missing_tables)}")

        # 2. Validate node properties exist in source tables
        for node in graph_model.nodes:
            source_table = node.source.name if node.source else None
            if source_table in database_structure.get("entity_tables", {}):
                table_info = database_structure["entity_tables"][source_table]
                available_columns = self._get_table_columns(table_info)

                # Check GraphProperty objects
                invalid_props = []
                for prop in node.properties:
                    if isinstance(prop, GraphProperty):
                        prop_name = prop.key
                    else:
                        prop_name = prop

                    if prop_name not in available_columns:
                        invalid_props.append(prop_name)

                if invalid_props:
                    node_label = node.primary_label
                    issues.append(
                        f"Node {node_label} has invalid properties: {invalid_props}"
                    )

        # 3. Validate relationships
        node_labels_set = set()
        for node in graph_model.nodes:
            if node.labels:
                node_labels_set.update([label.lower() for label in node.labels])

        for rel in graph_model.edges:
            # Check start node labels
            if rel.start_node_labels:
                for label in rel.start_node_labels:
                    if label.lower() not in node_labels_set:
                        issues.append(
                            f"Relationship {rel.edge_type} references unknown start node label: {label}"
                        )

            # Check end node labels
            if rel.end_node_labels:
                for label in rel.end_node_labels:
                    if label.lower() not in node_labels_set:
                        issues.append(
                            f"Relationship {rel.edge_type} references unknown end node label: {label}"
                        )

        # 4. Check for graph modeling best practices
        for node in graph_model.nodes:
            source_table = node.source.name if node.source else None
            if node.labels and source_table:
                primary_label = node.primary_label
                if primary_label.lower() == source_table.lower():
                    warnings.append(
                        f"Node label '{primary_label}' is same as table name, "
                        f"consider more semantic naming"
                    )

        is_valid = len(issues) == 0

        result = {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "summary": f"Found {len(issues)} issues and {len(warnings)} warnings",
        }

        if issues:
            logger.warning(f"Validation failed: {len(issues)} issues found")
        else:
            logger.info("Graph model validation passed")

        return result

    def _apply_natural_language_feedback(self, feedback: str) -> None:
        """
        Apply natural language feedback using structured output.
        """
        logger.info("Processing natural language feedback with structured output...")

        system_message = SystemMessage(
            content="""
You are a graph modeling expert that processes natural language feedback
to modify graph models.

Parse the user's feedback and return specific modifications as structured operations.

Available operations:
- change_node_label: Change a node's label
- rename_property: Rename a property in a node  
- drop_property: Remove a property from a node
- add_property: Add a property to a node
- change_relationship_name: Change a relationship's name
- drop_relationship: Remove a relationship
- add_index: Add an index to a node property
- drop_index: Remove an index from a node property

Analyze the feedback carefully and return the appropriate operations.
"""
        )

        current_model_summary = self._get_model_summary()

        human_message = HumanMessage(
            content=f"""
Current graph model:
{current_model_summary}

User feedback: "{feedback}"

Parse this feedback into specific operations to modify the graph model.
"""
        )

        try:
            # Use structured output if the LLM supports it
            if hasattr(self.llm, "with_structured_output"):
                structured_llm = self.llm.with_structured_output(ModelModifications)
                response = structured_llm.invoke([system_message, human_message])
                operations = [op.model_dump() for op in response.operations]
            else:
                # Fallback to JSON parsing for LLMs without structured output
                logger.warning(
                    "LLM doesn't support structured output, falling back to JSON parsing"
                )
                response = self.llm.invoke([system_message, human_message])

                # Try to extract JSON from response
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()

                response_data = json.loads(content)
                operations = response_data.get("operations", [])

            self._execute_model_operations(operations)
            logger.info(f"Successfully applied {len(operations)} operations")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            print(f"❌ Failed to parse feedback: {e}")
        except Exception as e:
            logger.error(f"Error processing natural language feedback: {e}")
            print(f"❌ Error processing feedback: {e}")

    def _execute_model_operations(self, operations: List[Dict[str, Any]]) -> None:
        """Execute a list of model modification operations."""

        for op in operations:
            op_type = op.get("type")

            if op_type == "change_node_label":
                self._change_node_label(op["target"], op["new_value"])
            elif op_type == "rename_property":
                self._rename_node_property(op["node"], op["target"], op["new_value"])
            elif op_type == "drop_property":
                self._drop_node_property(op["node"], op["target"])
            elif op_type == "add_property":
                self._add_node_property(op["node"], op["new_value"])
            elif op_type == "change_relationship_name":
                self._change_relationship_name(op["target"], op["new_value"])
            elif op_type == "drop_relationship":
                self._drop_relationship(op["target"])
            elif op_type == "add_index":
                self._add_node_index(op["node"], op["property"])
            elif op_type == "drop_index":
                self._drop_node_index(op["node"], op["property"])
            else:
                logger.warning(f"Unknown operation type: {op_type}")

        logger.info(f"Executed {len(operations)} model operations")

    def _change_node_label(self, old_label: str, new_label: str) -> None:
        """Change a node's label."""
        for node in self.current_graph_model.nodes:
            if node.primary_label == old_label:
                logger.info(f"Changing node label: {old_label} -> {new_label}")
                node.labels = [new_label] + node.labels[1:]  # Replace primary label
                break

    def _rename_node_property(
        self, node_label: str, old_prop: str, new_prop: str
    ) -> None:
        """Rename a property in a node."""
        for node in self.current_graph_model.nodes:
            if node.primary_label == node_label:
                for prop in node.properties:
                    if isinstance(prop, GraphProperty) and prop.key == old_prop:
                        logger.info(
                            f"Renaming property in {node_label}: {old_prop} -> {new_prop}"
                        )
                        prop.key = new_prop
                        break
                break

    def _drop_node_property(self, node_label: str, prop_name: str) -> None:
        """Drop a property from a node."""
        for node in self.current_graph_model.nodes:
            if node.primary_label == node_label:
                node.properties = [
                    prop
                    for prop in node.properties
                    if not (isinstance(prop, GraphProperty) and prop.key == prop_name)
                ]
                logger.info(f"Dropping property {prop_name} from {node_label}")
                break

    def _add_node_property(self, node_label: str, prop_name: str) -> None:
        """Add a property to a node."""
        for node in self.current_graph_model.nodes:
            if node.primary_label == node_label:
                # Check if property already exists
                existing = any(
                    isinstance(prop, GraphProperty) and prop.key == prop_name
                    for prop in node.properties
                )
                if not existing:
                    logger.info(f"Adding property {prop_name} to {node_label}")
                    new_prop = GraphProperty(key=prop_name)
                    node.properties.append(new_prop)
                break

    def _change_relationship_name(self, old_name: str, new_name: str) -> None:
        """Change a relationship's name."""
        for rel in self.current_graph_model.edges:
            if rel.edge_type == old_name:
                logger.info(f"Changing relationship name: {old_name} -> {new_name}")
                rel.edge_type = new_name
                break

    def _drop_relationship(self, rel_name: str) -> None:
        """Drop a relationship."""
        self.current_graph_model.edges = [
            rel for rel in self.current_graph_model.edges if rel.edge_type != rel_name
        ]
        logger.info(f"Dropped relationship: {rel_name}")

    def _add_node_index(self, node_label: str, prop_name: str) -> None:
        """Add an index to a node property."""
        # Find the node
        target_node = None
        for node in self.current_graph_model.nodes:
            if node.primary_label == node_label:
                target_node = node
                break

        if target_node:
            # Check if index already exists
            existing = any(
                index.labels == target_node.labels and prop_name in index.properties
                for index in self.current_graph_model.node_indexes
            )
            if not existing:
                logger.info(f"Adding index on {node_label}.{prop_name}")
                index_source = IndexSource(
                    origin="user_request",
                    reason="performance_optimization",
                    created_by="interactive_session",
                )
                new_index = GraphIndex(
                    labels=target_node.labels,
                    properties=[prop_name],
                    type="label+property",
                    source=index_source,
                )
                self.current_graph_model.node_indexes.append(new_index)

    def _drop_node_index(self, node_label: str, prop_name: str) -> None:
        """Drop an index from a node property."""
        # Find the node
        target_node = None
        for node in self.current_graph_model.nodes:
            if node.primary_label == node_label:
                target_node = node
                break

        if target_node:
            self.current_graph_model.node_indexes = [
                index
                for index in self.current_graph_model.node_indexes
                if not (
                    index.labels == target_node.labels and prop_name in index.properties
                )
            ]
            logger.info(f"Dropping index on {node_label}.{prop_name}")

    def _get_model_presentation(self) -> str:
        """Get formatted presentation of current model for user review."""
        if not self.current_graph_model:
            return "No model available"

        presentation = []
        presentation.append("NODES:")
        for i, node in enumerate(self.current_graph_model.nodes, 1):
            source_name = node.source.name if node.source else "Unknown"
            presentation.append(f"{i}. {node.primary_label} (from: {source_name})")

            prop_names = [
                prop.key if isinstance(prop, GraphProperty) else str(prop)
                for prop in node.properties
            ]
            presentation.append(f"   Properties: {', '.join(prop_names)}")

            # Show indexes
            node_indexes = [
                index
                for index in self.current_graph_model.node_indexes
                if index.labels == node.labels
            ]
            if node_indexes:
                index_props = [
                    prop for index in node_indexes for prop in index.properties
                ]
                presentation.append(f"   Indexes: {', '.join(index_props)}")
            presentation.append("")

        presentation.append("RELATIONSHIPS:")
        for i, rel in enumerate(self.current_graph_model.edges, 1):
            direction = "->" if rel.directionality == "directed" else "<->"
            start_labels = ", ".join(rel.start_node_labels)
            end_labels = ", ".join(rel.end_node_labels)
            presentation.append(
                f"{i}. {start_labels} {direction} [{rel.edge_type}] {direction} {end_labels}"
            )

            prop_names = [
                prop.key if isinstance(prop, GraphProperty) else str(prop)
                for prop in rel.properties
            ]
            if prop_names:
                presentation.append(f"   Properties: {', '.join(prop_names)}")
            presentation.append("")

        return "\n".join(presentation)

    def _get_model_summary(self) -> str:
        """Get concise text summary of current model."""
        if not self.current_graph_model:
            return "No model available"

        summary_parts = ["NODES:"]
        for node in self.current_graph_model.nodes:
            source_name = node.source.name if node.source else "Unknown"
            prop_names = [
                prop.key if isinstance(prop, GraphProperty) else str(prop)
                for prop in node.properties[:3]  # Show first 3 props
            ]
            props_str = ", ".join(prop_names)
            if len(node.properties) > 3:
                props_str += "..."
            summary_parts.append(f"- {node.primary_label} ({source_name}): {props_str}")

        summary_parts.append("\nRELATIONSHIPS:")
        for rel in self.current_graph_model.edges:
            start_labels = ", ".join(rel.start_node_labels)
            end_labels = ", ".join(rel.end_node_labels)
            rel_str = f"- {start_labels} -[{rel.edge_type}]-> {end_labels}"
            summary_parts.append(rel_str)

        return "\n".join(summary_parts)

    def _get_table_columns(self, table_info: Dict[str, Any]) -> List[str]:
        """Extract column names from table info."""
        columns = []
        schema_list = table_info.get("schema", [])
        for col_info in schema_list:
            if isinstance(col_info, dict):
                columns.append(col_info.get("field", ""))
            else:
                columns.append(str(col_info))
        return [col for col in columns if col]

    def _generate_initial_model(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> GraphModel:
        """Generate initial graph model from database structure."""
        logger.info("Generating initial graph model...")

        try:
            # Create simplified model based on database structure
            nodes = []
            relationships = []
            node_indexes = []
            node_constraints = []

            # Create nodes from entity tables
            for table_name, table_info in database_structure["entity_tables"].items():
                # Create node source
                source = NodeSource(
                    type="table",
                    name=table_name,
                    location=f"database.schema.{table_name}",
                    mapping={
                        "labels": [table_name.replace("_", "").title()],
                        "id_field": self._find_primary_key(table_info),
                    },
                )

                # Create properties as GraphProperty objects
                properties = []
                prop_names = self._extract_node_properties_from_table(table_info)
                for prop_name in prop_names:
                    prop_source = PropertySource(field=f"{table_name}.{prop_name}")
                    graph_prop = GraphProperty(key=prop_name, source=prop_source)
                    properties.append(graph_prop)

                node = GraphNode(
                    labels=[table_name.replace("_", "").title()],
                    properties=properties,
                    source=source,
                )
                nodes.append(node)

                # Create indexes
                index_props = self._extract_indexes_from_table(table_info)
                for index_prop in index_props:
                    index_source = IndexSource(
                        origin="migration_requirement",
                        reason="performance_optimization",
                        created_by="migration_agent",
                    )

                    graph_index = GraphIndex(
                        labels=[table_name.replace("_", "").title()],
                        properties=[index_prop],
                        type="label+property",
                        source=index_source,
                    )
                    node_indexes.append(graph_index)

                # Create constraints
                constraint_strs = self._extract_constraints_from_table(table_info)
                for constraint_str in constraint_strs:
                    if "UNIQUE" in constraint_str.upper():
                        prop_name = constraint_str.replace("UNIQUE(", "").replace(
                            ")", ""
                        )
                        constraint_source = ConstraintSource(
                            origin="source_database_constraint",
                            constraint_name=f"{table_name}_{prop_name}_unique",
                            migrated_from=f"database.schema.{table_name}",
                        )

                        graph_constraint = GraphConstraint(
                            type="unique",
                            labels=[table_name.replace("_", "").title()],
                            properties=[prop_name],
                            source=constraint_source,
                        )
                        node_constraints.append(graph_constraint)

            # Create relationships
            for rel in database_structure.get("relationships", []):
                # Create relationship source
                rel_source = RelationshipSource(
                    type="table"
                    if rel.get("type") != "many_to_many"
                    else "junction_table",
                    name=rel.get("constraint_name", ""),
                    location=f"database.schema.{rel['from_table']}",
                    mapping={
                        "start_node": f"{rel['from_table']}.{rel['from_column']}",
                        "end_node": f"{rel['to_table']}.{rel['to_column']}",
                        "edge_type": self._generate_relationship_name(rel),
                    },
                )

                # Find node labels
                from_label = rel["from_table"].replace("_", "").title()
                to_label = rel["to_table"].replace("_", "").title()

                relationship = GraphRelationship(
                    edge_type=self._generate_relationship_name(rel),
                    start_node_labels=[from_label],
                    end_node_labels=[to_label],
                    properties=[],
                    source=rel_source,
                    directionality="directed",
                )
                relationships.append(relationship)

            # Process junction tables (many-to-many relationships)
            join_tables = database_structure.get("join_tables", {})
            for join_table_name, join_table_info in join_tables.items():
                # Extract the foreign keys from the junction table
                foreign_keys = join_table_info.get("foreign_keys", [])
                if len(foreign_keys) >= 2:
                    # Get the first two foreign keys (assuming junction table pattern)
                    fk1 = foreign_keys[0]
                    fk2 = foreign_keys[1]

                    # Create junction table relationship source
                    rel_source = RelationshipSource(
                        type="junction_table",
                        name=join_table_name,
                        location=f"database.schema.{join_table_name}",
                        mapping={
                            "join_table": join_table_name,
                            "from_table": fk1["referenced_table"],
                            "to_table": fk2["referenced_table"],
                            "join_from_column": fk1["column"],
                            "join_to_column": fk2["column"],
                            "from_column": fk1["referenced_column"],
                            "to_column": fk2["referenced_column"],
                        },
                    )

                    # Generate relationship name from junction table
                    def generate_relationship_name(table_name, from_table, to_table):
                        """Generate semantic relationship name from junction table."""
                        # Remove common prefixes/suffixes and convert to relationship
                        table_clean = table_name.lower()
                        from_clean = from_table.lower()
                        to_clean = to_table.lower()

                        # Try to infer semantic meaning from table names
                        if from_clean in table_clean and to_clean in table_clean:
                            # Extract relationship part by removing table names
                            rel_part = table_clean.replace(from_clean, "")
                            rel_part = rel_part.replace(to_clean, "").strip("_")
                            if rel_part:
                                return rel_part.upper()

                        # Fallback: create relationship from table name pattern
                        # Convert table_name to a meaningful relationship
                        parts = table_clean.split("_")
                        if len(parts) >= 2:
                            # Try to find action/relationship words
                            action_words = ["has", "in", "to", "of", "by", "with"]
                            for part in parts:
                                if part in action_words or part.endswith("s"):
                                    return f"{from_clean.upper()}_TO_{to_clean.upper()}"

                        # Final fallback: generic pattern
                        return f"{from_clean.upper()}_TO_{to_clean.upper()}"

                    rel_name = generate_relationship_name(
                        join_table_name,
                        fk1["referenced_table"],
                        fk2["referenced_table"],
                    )

                    # Find node labels
                    from_label = fk1["referenced_table"].replace("_", "").title()
                    to_label = fk2["referenced_table"].replace("_", "").title()

                    relationship = GraphRelationship(
                        edge_type=rel_name,
                        start_node_labels=[from_label],
                        end_node_labels=[to_label],
                        properties=[],
                        source=rel_source,
                        directionality="directed",
                    )
                    relationships.append(relationship)

            return GraphModel(
                nodes=nodes,
                edges=relationships,
                node_indexes=node_indexes,
                node_constraints=node_constraints,
            )

        except Exception as e:
            logger.error(f"Failed to generate initial graph model: {e}")
            return GraphModel(
                nodes=[],
                edges=[],
            )

    def _fix_validation_issues(
        self, graph_model: GraphModel, validation_result: Dict[str, Any]
    ) -> GraphModel:
        """Attempt to fix validation issues automatically."""
        logger.info("Attempting to fix validation issues...")

        for issue in validation_result["issues"]:
            logger.warning(f"Validation issue: {issue}")

        return graph_model

    def _extract_node_properties_from_table(
        self, table_info: Dict[str, Any]
    ) -> List[str]:
        """Extract node properties from table info."""
        properties = []
        if "schema" in table_info:
            for col in table_info["schema"]:
                col_name = col.get("field", "")
                if col_name and (
                    not col_name.endswith("_id") or col.get("key") == "PRI"
                ):
                    properties.append(col_name)
        elif "columns" in table_info:
            for col_name, col_info in table_info["columns"].items():
                if not col_name.endswith("_id") or col_info.get("key") == "PRI":
                    properties.append(col_name)
        return properties

    def _extract_indexes_from_table(self, table_info: Dict[str, Any]) -> List[str]:
        """Extract recommended indexes from table info."""
        indexes = []
        if "schema" in table_info:
            for col in table_info["schema"]:
                if col.get("key") in ["UNI", "MUL"]:
                    indexes.append(col.get("field", ""))
        elif "columns" in table_info:
            for col_name, col_info in table_info["columns"].items():
                if col_info.get("key") in ["UNI", "MUL"]:
                    indexes.append(col_name)
        return [idx for idx in indexes if idx]

    def _extract_constraints_from_table(self, table_info: Dict[str, Any]) -> List[str]:
        """Extract constraints from table info."""
        constraints = []
        if "schema" in table_info:
            for col in table_info["schema"]:
                if col.get("key") == "PRI":
                    constraints.append(f"UNIQUE({col.get('field', '')})")
        elif "columns" in table_info:
            for col_name, col_info in table_info["columns"].items():
                if col_info.get("key") == "PRI":
                    constraints.append(f"UNIQUE({col_name})")
        return constraints

    def _generate_relationship_name(self, rel_data: Dict[str, Any]) -> str:
        """Generate relationship name from relationship data."""
        if rel_data.get("type") == "many_to_many":
            join_table = rel_data.get("join_table", "")
            return join_table.upper().replace("_", "_") if join_table else "CONNECTS"
        else:
            to_table = rel_data["to_table"]
            return f"HAS_{to_table.upper()}"

    def _find_primary_key(self, table_info: Dict[str, Any]) -> str:
        """Find the primary key column for a table."""
        if "schema" in table_info:
            for col in table_info["schema"]:
                if col.get("key") == "PRI":
                    return col["field"]
        elif "columns" in table_info:
            for col_name, col_info in table_info["columns"].items():
                if col_info.get("key") == "PRI":
                    return col_name
        return "id"  # Default assumption
