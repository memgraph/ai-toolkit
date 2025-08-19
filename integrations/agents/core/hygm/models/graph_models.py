"""
Core graph models for Hypothetical Graph Modeling (HyGM).

These models represent the graph structure and provide schema format conversion.
"""

import datetime
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Import from within the same package
try:
    from .sources import (
        PropertySource,
        NodeSource,
        RelationshipSource,
        IndexSource,
        ConstraintSource,
        EnumSource,
    )
except ImportError:
    # Fallback for when imported from different contexts
    from sources import (
        PropertySource,
        NodeSource,
        RelationshipSource,
        IndexSource,
        ConstraintSource,
        EnumSource,
    )


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
        """Create a GraphModel from schema format dictionary."""
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
                    index_name=index_dict["source"].get("index_name"),
                    migrated_from=index_dict["source"].get("migrated_from"),
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
                    index_name=index_dict["source"].get("index_name"),
                    migrated_from=index_dict["source"].get("migrated_from"),
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
                    created_by=enum_dict["source"].get("created_by"),
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
        """Convert to comprehensive schema format dictionary."""
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
            if index.source.index_name:
                schema_index["source"]["index_name"] = index.source.index_name
            if index.source.migrated_from:
                schema_index["source"]["migrated_from"] = index.source.migrated_from

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
            if enum.source.created_by:
                source_dict["created_by"] = enum.source.created_by
            schema_enum["source"] = source_dict

        return schema_enum

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

            # Convert to schema types
            for type_name, count in type_counts.items():
                type_schema = {
                    "type": type_name,
                    "count": count,
                    "examples": examples_by_type[type_name],
                }
                prop_schema["types"].append(type_schema)
        else:
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
            return "Double"
        elif isinstance(value, datetime.datetime):
            return "LocalDateTime"
        elif isinstance(value, datetime.date):
            return "Date"
        elif isinstance(value, datetime.time):
            return "LocalTime"
        elif isinstance(value, str):
            if self._is_datetime_string(value):
                return "LocalDateTime"
            elif self._is_date_string(value):
                return "Date"
            elif self._is_time_string(value):
                return "LocalTime"
            else:
                return "String"
        else:
            return "String"

    def _is_date_string(self, value: str) -> bool:
        """Check if string looks like a date."""
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
            r"^\d{2}/\d{2}/\d{4}$",  # MM/DD/YYYY
            r"^\d{2}-\d{2}-\d{4}$",  # MM-DD-YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)

    def _is_datetime_string(self, value: str) -> bool:
        """Check if string looks like a datetime."""
        datetime_patterns = [
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # YYYY-MM-DD HH:MM:SS
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format
        ]
        return any(re.match(pattern, value) for pattern in datetime_patterns)

    def _is_time_string(self, value: str) -> bool:
        """Check if string looks like a time."""
        time_patterns = [
            r"^\d{2}:\d{2}:\d{2}$",  # HH:MM:SS
            r"^\d{2}:\d{2}$",  # HH:MM
        ]
        return any(re.match(pattern, value) for pattern in time_patterns)
