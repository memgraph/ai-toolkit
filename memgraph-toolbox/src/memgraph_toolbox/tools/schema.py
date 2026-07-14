import json
import re
from typing import Any

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class NodeSchemaTool(BaseTool):
    """
    Tool for describing a node in the graph schema by its labels.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="get_node_schema",
            description=(
                "Describe a node in the graph schema by its labels. Returns the node description, "
                "its properties with types and descriptions, and all relationships where this node "
                "appears as either the start or end node."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "node_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The labels of the node to get the details of.",
                    },
                },
                "required": ["node_labels"],
            },
        )
        self.db = db

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            node_labels = arguments["node_labels"]
            schema = _get_schema(self.db)
            node = _find_node(schema, node_labels)
            if not node:
                return [{"text": f"No node found for the labels: ({_join_labels(node_labels)})"}]

            inbound_edges = _get_inbound_edges(schema, node_labels)
            outbound_edges = _get_outbound_edges(schema, node_labels)
            loopback_edges = _get_loopback_edges(schema, node_labels)
            edge_keys = ["type", "start_node_labels", "end_node_labels"]

            return {
                "node": node,
                "node_indexes": _get_node_indexes(schema, node_labels),
                "node_constraints": _get_node_constraints(schema, node_labels),
                "relationships": {
                    "inbound": [_strip_dict(e, edge_keys) for e in inbound_edges],
                    "outbound": [_strip_dict(e, edge_keys) for e in outbound_edges],
                    "loopback": [_strip_dict(e, edge_keys) for e in loopback_edges],
                },
            }
        except Exception as e:
            return [{"error": f"Failed to describe node: {e!s}"}]


class RelationshipSchemaTool(BaseTool):
    """
    Tool for describing a relationship in the graph schema by its type and connected node labels.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="get_relationship_schema",
            description=(
                "Describe a relationship in the graph schema by its type and connected node labels. "
                "Returns the relationship description, its properties with types and descriptions, "
                "and the start and end nodes it connects."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "relationship_type": {
                        "type": "string",
                        "description": "The type of the relationship to get the details of.",
                    },
                    "start_node_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The labels of the start node of the relationship.",
                    },
                    "end_node_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The labels of the end node of the relationship.",
                    },
                },
                "required": ["relationship_type", "start_node_labels", "end_node_labels"],
            },
        )
        self.db = db

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            edge_type = arguments["relationship_type"]
            start_node_labels = arguments["start_node_labels"]
            end_node_labels = arguments["end_node_labels"]
            schema = _get_schema(self.db)
            edge = _find_edge(schema, edge_type, start_node_labels, end_node_labels)

            if not edge:
                signature = f"({_join_labels(start_node_labels)})-[:{edge_type}]->({_join_labels(end_node_labels)})"
                return [{"text": f"No relationship found: {signature}"}]

            return {
                "relationship": edge,
                "relationship_indexes": _get_edge_indexes(schema, edge_type),
            }
        except Exception as e:
            return [{"error": f"Failed to describe relationship: {e!s}"}]


class EnumSchemaTool(BaseTool):
    """
    Tool for describing an enum in the graph schema by its name.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="get_enum_schema",
            description="Describe an enum in the graph schema by its name. Returns the enum name and its values.",
            input_schema={
                "type": "object",
                "properties": {
                    "enum_name": {
                        "type": "string",
                        "description": "The name of the enum to get the details of.",
                    },
                },
                "required": ["enum_name"],
            },
        )
        self.db = db

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            enum_name = arguments["enum_name"]
            schema = _get_schema(self.db)
            enum = next((e for e in schema.get("enums", []) if e["name"] == enum_name), None)
            if not enum:
                return [{"text": f"No enum found for the name: {enum_name}"}]
            return {
                "enum": enum,
            }
        except Exception as e:
            return [{"error": f"Failed to describe enum: {e!s}"}]


class SearchSchemaTool(BaseTool):
    """
    Tool for searching the graph schema by a regex pattern.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="search_schema",
            description=(
                "Search the entire graph schema (nodes, relationships and enums) by a regex pattern. "
                "Matches against labels, types, descriptions, property keys/descriptions and enum values."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": 'A case-insensitive regex pattern to search for (e.g. "person", "pay.*ment").',
                    },
                },
                "required": ["pattern"],
            },
        )
        self.db = db

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            regex = _build_schema_regex(arguments["pattern"])
            schema = _get_schema(self.db)
            matches: list[dict[str, Any]] = []

            for node in schema.get("nodes", []):
                node_matches = _match_node(node, regex)
                if not node_matches:
                    continue
                matches.append(
                    {
                        "type": "node-match",
                        "labels": node.get("labels", []),
                        "matches": node_matches,
                    }
                )

            for edge in schema.get("edges", []):
                edge_matches = _match_edge(edge, regex)
                if not edge_matches:
                    continue
                matches.append(
                    {
                        "type": "edge-match",
                        "edge_type": edge.get("type", ""),
                        "start_labels": edge.get("start_node_labels", []),
                        "end_labels": edge.get("end_node_labels", []),
                        "matches": edge_matches,
                    }
                )

            for enum in schema.get("enums", []):
                enum_matches = _match_enum(enum, regex)
                if not enum_matches:
                    continue
                matches.append(
                    {
                        "type": "enum-match",
                        "enum": enum,
                        "matches": enum_matches,
                    }
                )

            if not matches:
                return [{"text": f'No schema elements matching pattern "{arguments["pattern"]}" found.'}]
            return matches
        except Exception as e:
            return [{"error": f"Failed to search schema: {e!s}"}]


def _join_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    return f"(:{':'.join(sorted(labels))})"


def _get_schema(db: Memgraph) -> dict[str, Any]:
    """
    Returns schema as a dictionary with the following keys:
    - nodes: list of nodes
    - edges: list of edges
    - node_indexes: list of node indexes
    - edge_indexes: list of edge indexes
    - node_constraints: list of node constraints
    - enums: list of enums
    """
    results = db.query("SHOW SCHEMA INFO")
    if not results:
        raise ValueError("SHOW SCHEMA INFO returned no results")
    if not results[0].get("schema"):
        raise ValueError("SHOW SCHEMA INFO result does not contain a 'schema' key")

    stringified_schema = results[0]["schema"]
    try:
        return json.loads(stringified_schema)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in SHOW SCHEMA INFO result: {stringified_schema!s} - {e!s}") from e


def _build_schema_regex(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def _match_node(node: dict[str, Any], regex: re.Pattern) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    if node.get("labels") and regex.search(_join_labels(node["labels"])):
        reasons.append(
            {
                "reason": "matched node label",
                "matched_value": node["labels"],
            }
        )

    if node.get("description") and regex.search(node["description"]):
        reasons.append(
            {
                "reason": "matched node description",
                "matched_value": node["description"],
            }
        )

    for prop in node.get("properties", []):
        if prop.get("key") and regex.search(prop["key"]):
            reasons.append(
                {
                    "reason": f"matched property key `{prop['key']}`",
                    "matched_value": prop["key"],
                }
            )
        if prop.get("description") and regex.search(prop["description"]):
            reasons.append(
                {
                    "reason": f"matched property key `{prop['key']}` description",
                    "matched_value": prop["description"],
                }
            )
    return reasons


def _match_edge(edge: dict[str, Any], regex: re.Pattern) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    if edge.get("type") and regex.search(edge["type"]):
        reasons.append(
            {
                "reason": "matched relationship type",
                "matched_value": edge["type"],
            }
        )

    if edge.get("start_node_labels") and regex.search(_join_labels(edge["start_node_labels"])):
        reasons.append(
            {
                "reason": "matched start node label",
                "matched_value": edge["start_node_labels"],
            }
        )

    if edge.get("end_node_labels") and regex.search(_join_labels(edge["end_node_labels"])):
        reasons.append(
            {
                "reason": "matched end node label",
                "matched_value": edge["end_node_labels"],
            }
        )

    if edge.get("description") and regex.search(edge["description"]):
        reasons.append(
            {
                "reason": "matched relationship description",
                "matched_value": edge["description"],
            }
        )

    for prop in edge.get("properties", []):
        if prop.get("key") and regex.search(prop["key"]):
            reasons.append(
                {
                    "reason": f"matched property key `{prop['key']}`",
                    "matched_value": prop["key"],
                }
            )
        if prop.get("description") and regex.search(prop["description"]):
            reasons.append(
                {
                    "reason": f"matched property key `{prop['key']}` description",
                    "matched_value": prop["description"],
                }
            )

    return reasons


def _match_enum(enum: dict[str, Any], regex: re.Pattern) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    if enum.get("name") and regex.search(enum["name"]):
        reasons.append(
            {
                "reason": "matched enum name",
                "matched_value": enum,
            }
        )

    for value in enum.get("values", []):
        if regex.search(value):
            reasons.append(
                {
                    "reason": "matched enum value",
                    "matched_value": value,
                }
            )

    return reasons


def _find_node(schema: dict[str, Any], labels: list[str]) -> dict[str, Any] | None:
    target_labels = _join_labels(labels)
    nodes = schema.get("nodes", [])
    return next((n for n in nodes if _join_labels(n["labels"]) == target_labels), None)


def _find_edge(
    schema: dict[str, Any], edge_type: str, start_node_labels: list[str], end_node_labels: list[str]
) -> dict[str, Any] | None:
    start_labels = _join_labels(start_node_labels)
    end_labels = _join_labels(end_node_labels)
    edges = schema.get("edges", [])
    return next(
        (
            e
            for e in edges
            if e["type"] == edge_type
            and _join_labels(e["start_node_labels"]) == start_labels
            and _join_labels(e["end_node_labels"]) == end_labels
        ),
        None,
    )


def _get_inbound_edges(schema: dict[str, Any], node_labels: list[str]) -> list[dict[str, Any]]:
    joined_node_labels = _join_labels(node_labels)
    edges = schema.get("edges", [])
    return [
        e
        for e in edges
        if _join_labels(e["end_node_labels"]) == joined_node_labels and e["start_node_labels"] != joined_node_labels
    ]


def _get_outbound_edges(schema: dict[str, Any], node_labels: list[str]) -> list[dict[str, Any]]:
    joined_node_labels = _join_labels(node_labels)
    edges = schema.get("edges", [])
    return [
        e
        for e in edges
        if _join_labels(e["start_node_labels"]) == joined_node_labels and e["end_node_labels"] != joined_node_labels
    ]


def _get_loopback_edges(schema: dict[str, Any], node_labels: list[str]) -> list[dict[str, Any]]:
    joined_node_labels = _join_labels(node_labels)
    edges = schema.get("edges", [])
    return [
        e
        for e in edges
        if _join_labels(e["start_node_labels"]) == joined_node_labels
        and _join_labels(e["end_node_labels"]) == joined_node_labels
    ]


def _get_node_indexes(schema: dict[str, Any], node_labels: list[str]) -> list[dict[str, Any]]:
    unique_node_labels = set(node_labels)
    indexes = schema.get("node_indexes", [])
    return [i for i in indexes if i.get("labels") and set(i["labels"]).issubset(unique_node_labels)]


def _get_node_constraints(schema: dict[str, Any], node_labels: list[str]) -> list[dict[str, Any]]:
    unique_node_labels = set(node_labels)
    constraints = schema.get("node_constraints", [])
    return [c for c in constraints if c.get("labels") and set(c["labels"]).issubset(unique_node_labels)]


def _get_edge_indexes(schema: dict[str, Any], edge_type: str) -> list[dict[str, Any]]:
    indexes = schema.get("edge_indexes", [])
    return [i for i in indexes if i.get("edge_type") == edge_type]


def _strip_dict(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: d[k] for k in keys if k in d}
