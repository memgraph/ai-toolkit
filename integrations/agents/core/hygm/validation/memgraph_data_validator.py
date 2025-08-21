"""
Memgraph Data Validation Module.

This module provides comprehensive validation functionality for Memgraph 
databases after migration, including schema validation (nodes, relationships, 
indexes, constraints) and data count validation. It compares the expected 
GraphModel specification with the actual Memgraph database state to ensure 
migration success.
"""

import logging
from typing import Dict, List, Any
from ..models.graph_models import GraphModel
from .base import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    ValidationCategory,
)

logger = logging.getLogger(__name__)


class MemgraphDataValidator(BaseValidator):
    """
    Validates Memgraph data and schema against expected GraphModel.

    This class provides comprehensive post-migration validation of Memgraph
    database including schema validation (nodes, relationships, indexes,
    constraints) and data count verification by comparing it with the expected
    GraphModel specification.
    """

    def __init__(self, memgraph_connection):
        """
        Initialize validator with Memgraph connection.

        Args:
            memgraph_connection: Connection to Memgraph database
                (adapter or raw connection)
        """
        super().__init__()
        self.connection = memgraph_connection
        self._cached_data_counts = (
            None  # Cache for data counts to avoid repeated queries
        )

    def validate(self, expected_model: GraphModel, **kwargs) -> ValidationResult:
        """
        Abstract method implementation for BaseValidator.

        This is the main entry point for validation.
        """
        return self.validate_post_migration(expected_model)

    def validate_post_migration(
        self, expected_model: GraphModel, expected_data_counts: Dict[str, int] = None
    ) -> ValidationResult:
        """
        Validate the migrated schema and data against expected model.

        Args:
            expected_model: GraphModel representing expected schema
            expected_data_counts: Optional dict with expected node/relationship counts
                Format: {"nodes": 12345, "relationships": 6789, "selected_tables": ["table1", "table2"]}

        Returns:
            ValidationResult with detailed comparison results
        """
        self.reset()  # Reset state using parent method

        try:
            # Get actual schema from Memgraph
            actual_schema = self._get_actual_schema()
            nodes_count = len(actual_schema.get("nodes", []))
            rels_count = len(actual_schema.get("relationships", []))
            logger.info(
                "Retrieved actual schema: %d nodes, %d relationships",
                nodes_count,
                rels_count,
            )

            # Convert expected model to comparable format
            expected_schema = self._convert_model_to_schema_info(expected_model)
            exp_nodes = len(expected_schema.get("nodes", []))
            exp_rels = len(expected_schema.get("relationships", []))
            logger.info(
                "Expected schema: %d nodes, %d relationships", exp_nodes, exp_rels
            )

            # Perform schema validations
            self._validate_node_labels(expected_schema, actual_schema)
            self._validate_node_properties(expected_schema, actual_schema)
            self._validate_relationships(expected_schema, actual_schema)
            self._validate_indexes(expected_schema, actual_schema)
            self._validate_constraints(expected_schema, actual_schema)

            # Perform data count validation if expected counts provided
            if expected_data_counts:
                self._validate_data_counts(expected_data_counts)

            # Calculate metrics using the base class structure
            self._update_metrics(expected_schema, actual_schema, expected_data_counts)

            # Generate results
            critical_issues = [
                issue
                for issue in self.issues
                if issue.severity == ValidationSeverity.CRITICAL
            ]
            success = not any(critical_issues)
            summary = self._generate_summary()

            return ValidationResult(
                validation_type="memgraph_data_validation",
                success=success,
                summary=summary,
                issues=self.issues,
                metrics=self.metrics,
                details={
                    "expected_schema": expected_schema,
                    "actual_schema": actual_schema,
                    "validation_score": self._calculate_validation_score(),
                    "data_counts": self._get_actual_data_counts()
                    if expected_data_counts
                    else None,
                },
            )

        except Exception as e:
            logger.error("Validation failed with error: %s", str(e))
            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.SCHEMA_MISMATCH,
                f"Validation process failed: {str(e)}",
                recommendation="Check database connection and schema access",
            )

            return ValidationResult(
                validation_type="memgraph_data_validation",
                success=False,
                summary=f"Validation failed: {str(e)}",
                issues=self.issues,
                metrics=self.metrics,
            )

    def _get_actual_schema(self) -> Dict[str, Any]:
        """
        Get actual schema from Memgraph database.

        Returns:
            Structured schema information from Memgraph
        """
        # Handle different connection types
        if hasattr(self.connection, "get_schema_info"):
            # MemgraphAdapter interface
            return self._parse_memgraph_adapter_schema(self.connection)
        else:
            # Raw connection - execute SHOW SCHEMA INFO
            return self._parse_raw_connection_schema(self.connection)

    def _parse_raw_connection_schema(self, connection) -> Dict[str, Any]:
        """
        Parse schema from raw Memgraph connection.

        Args:
            connection: Raw database connection (Memgraph client)

        Returns:
            Structured schema information
        """
        try:
            # Check if it's a Memgraph client with .query() method
            if hasattr(connection, "query"):
                result = connection.query("SHOW SCHEMA INFO;")
                # Convert query result to list to access records
                records = list(result)

                if records and "schema" in records[0]:
                    # Parse the JSON schema from the first record
                    import json

                    schema_json = records[0]["schema"]
                    schema_data = json.loads(schema_json)
                    return self._parse_memgraph_json_schema(schema_data)
                else:
                    logger.warning("SHOW SCHEMA INFO returned unexpected format")
                    return {
                        "nodes": [],
                        "relationships": [],
                        "indexes": [],
                        "constraints": [],
                    }
            else:
                # Fallback for cursor-based connections
                cursor = connection.cursor()
                cursor.execute("SHOW SCHEMA INFO;")
                result = cursor.fetchall()
                return self._parse_schema_info_result(result)
        except Exception as e:
            logger.error(f"Failed to get schema from connection: {e}")
            return {"nodes": [], "relationships": [], "indexes": [], "constraints": []}

    def _parse_schema_info_result(self, schema_info_rows) -> Dict[str, Any]:
        """
        Parse the result of SHOW SCHEMA INFO query.

        Args:
            schema_info_rows: Raw result from SHOW SCHEMA INFO

        Returns:
            Structured schema information
        """
        nodes = {}
        relationships = {}
        indexes = []
        constraints = []

        for row in schema_info_rows:
            if len(row) >= 3:
                element_type = row[0]  # "node" or "relationship"
                element_name = row[1]  # label or relationship type
                properties = row[2] if len(row) > 2 else {}

                if element_type == "node":
                    if element_name not in nodes:
                        # Handle both single labels and label combinations
                        if isinstance(element_name, str):
                            labels = [element_name]
                        else:
                            labels = element_name
                        nodes[element_name] = {"labels": labels, "properties": {}}
                    if isinstance(properties, dict):
                        nodes[element_name]["properties"].update(properties)

                elif element_type == "relationship":
                    if element_name not in relationships:
                        relationships[element_name] = {
                            "type": element_name,
                            "properties": {},
                        }
                    if isinstance(properties, dict):
                        rel_props = relationships[element_name]["properties"]
                        rel_props.update(properties)

        return {
            "nodes": list(nodes.values()),
            "relationships": list(relationships.values()),
            "indexes": indexes,
            "constraints": constraints,
        }

    def _parse_memgraph_adapter_schema(self, adapter) -> Dict[str, Any]:
        """
        Parse schema info from MemgraphAdapter.

        Args:
            adapter: MemgraphAdapter instance

        Returns:
            Structured schema information
        """
        try:
            # Get schema info using adapter methods
            schema_info_rows = adapter.get_schema_info()
            parsed_schema = self._parse_schema_info_result(schema_info_rows)

            # Get additional info
            indexes = adapter.get_indexes()
            constraints = adapter.get_constraints()

            parsed_schema["indexes"] = indexes
            parsed_schema["constraints"] = constraints

            return parsed_schema

        except Exception as e:
            logger.error(f"Failed to parse MemgraphAdapter schema: {e}")
            # Fallback to empty schema
            return {"nodes": [], "relationships": [], "indexes": [], "constraints": []}

    def _parse_memgraph_json_schema(
        self, schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse Memgraph JSON schema format and extract data counts.

        Args:
            schema_data: JSON schema data from SHOW SCHEMA INFO

        Returns:
            Structured schema information with data counts
        """
        nodes = []
        relationships = []
        indexes = []
        constraints = []

        # Track data counts from schema info
        total_nodes = 0
        total_relationships = 0

        # Parse nodes
        for node_data in schema_data.get("nodes", []):
            node_info = {"labels": node_data.get("labels", []), "properties": {}}

            # Calculate node count from property types
            node_count = 0
            for prop in node_data.get("properties", []):
                prop_name = prop.get("key", "")
                prop_types = prop.get("types", [])
                if prop_types:
                    # Get the primary type and its count
                    primary_type = prop_types[0].get("type", "String")
                    # Sum counts across all types for this property (excluding Null)
                    prop_count = sum(
                        type_def.get("count", 0)
                        for type_def in prop_types
                        if type_def.get("type", "") != "Null"
                    )
                    # Use the highest property count as the node count estimate
                    node_count = max(node_count, prop_count)
                    node_info["properties"][prop_name] = primary_type

            # Store the node count in the node info
            node_info["node_count"] = node_count
            total_nodes += node_count
            nodes.append(node_info)

        # Parse relationships
        for edge_data in schema_data.get("edges", []):
            rel_info = {"type": edge_data.get("type", ""), "properties": {}}

            # Calculate relationship count from property types
            rel_count = 0
            properties = edge_data.get("properties", [])

            if properties:
                # If relationship has properties, count from property types
                for prop in properties:
                    prop_name = prop.get("key", "")
                    prop_types = prop.get("types", [])
                    if prop_types:
                        primary_type = prop_types[0].get("type", "String")
                        # Sum counts across all types for this property (excluding Null)
                        prop_count = sum(
                            type_def.get("count", 0)
                            for type_def in prop_types
                            if type_def.get("type", "") != "Null"
                        )
                        # Use the highest property count as the relationship count estimate
                        rel_count = max(rel_count, prop_count)
                        rel_info["properties"][prop_name] = primary_type
            else:
                # If relationship has no properties, we need to count differently
                # For now, we'll mark it as unknown and use a fallback query later
                rel_count = -1  # Mark as needs counting

            # Store the relationship count
            rel_info["relationship_count"] = rel_count
            if rel_count > 0:
                total_relationships += rel_count
            relationships.append(rel_info)

        # Parse indexes (unchanged)
        for index_data in schema_data.get("node_indexes", []):
            index_info = {
                "type": "node",
                "labels": index_data.get("labels", []),
                "properties": index_data.get("properties", []),
                "index_type": index_data.get("type", "label+properties"),
            }
            indexes.append(index_info)

        # Parse constraints (unchanged)
        for constraint_data in schema_data.get("node_constraints", []):
            constraint_info = {
                "type": "node",
                "labels": constraint_data.get("labels", []),
                "properties": constraint_data.get("properties", []),
                "constraint_type": constraint_data.get("type", "unique"),
            }
            constraints.append(constraint_info)

        return {
            "nodes": nodes,
            "relationships": relationships,
            "indexes": indexes,
            "constraints": constraints,
            "data_counts": {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
            },
        }

    def _convert_model_to_schema_info(self, model: GraphModel) -> Dict[str, Any]:
        """
        Convert GraphModel to schema info format for comparison.

        Args:
            model: GraphModel to convert

        Returns:
            Schema info format compatible with Memgraph output
        """
        nodes = []
        for node in model.nodes:
            node_info = {"labels": node.labels, "properties": {}}

            # Convert properties to simple format
            for prop in node.properties:
                if hasattr(prop, "key"):
                    # Determine expected type from GraphProperty types
                    prop_types = prop.types if hasattr(prop, "types") else []
                    primary_type = self._get_primary_type(prop_types)
                    node_info["properties"][prop.key] = primary_type
                else:
                    # Handle string properties
                    node_info["properties"][str(prop)] = "String"

            nodes.append(node_info)

        relationships = []
        for edge in model.edges:
            rel_info = {"type": edge.edge_type, "properties": {}}

            # Convert properties
            for prop in edge.properties:
                if hasattr(prop, "key"):
                    prop_types = prop.types if hasattr(prop, "types") else []
                    primary_type = self._get_primary_type(prop_types)
                    rel_info["properties"][prop.key] = primary_type
                else:
                    rel_info["properties"][str(prop)] = "String"

            relationships.append(rel_info)

        # Convert indexes
        indexes = []
        for index in model.node_indexes:
            index_info = {
                "type": "node",
                "labels": index.labels,
                "properties": index.properties,
                "index_type": index.type,
            }
            indexes.append(index_info)

        for index in model.edge_indexes:
            index_info = {
                "type": "edge",
                "edge_type": index.edge_type,
                "properties": index.properties,
                "index_type": index.type,
            }
            indexes.append(index_info)

        # Convert constraints
        constraints = []
        for constraint in model.node_constraints:
            constraint_info = {
                "type": "node",
                "labels": constraint.labels,
                "properties": constraint.properties,
                "constraint_type": constraint.type,
            }
            constraints.append(constraint_info)

        for constraint in model.edge_constraints:
            constraint_info = {
                "type": "edge",
                "edge_type": constraint.edge_type,
                "properties": constraint.properties,
                "constraint_type": constraint.type,
            }
            constraints.append(constraint_info)

        return {
            "nodes": nodes,
            "relationships": relationships,
            "indexes": indexes,
            "constraints": constraints,
        }

    def _get_primary_type(self, type_list: List[Dict[str, Any]]) -> str:
        """Get the primary type from a list of type definitions."""
        if not type_list:
            return "String"

        # Find the type with highest count, excluding Null
        max_count = 0
        primary_type = "String"

        for type_def in type_list:
            if type_def.get("type", "") != "Null":
                count = type_def.get("count", 0)
                if count > max_count:
                    max_count = count
                    primary_type = type_def.get("type", "String")

        return primary_type

    def _validate_node_labels(self, expected: Dict[str, Any], actual: Dict[str, Any]):
        """Validate that all expected node labels exist in Memgraph."""
        expected_labels = {tuple(node["labels"]) for node in expected["nodes"]}
        actual_labels = {tuple(node["labels"]) for node in actual["nodes"]}

        # Check for missing labels
        missing_labels = expected_labels - actual_labels
        for labels in missing_labels:
            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.SCHEMA_MISMATCH,
                f"Missing node labels in Memgraph: {list(labels)}",
                expected=list(labels),
                actual=None,
                recommendation="Check migration script for node creation issues",
            )

        # Check for unexpected labels
        extra_labels = actual_labels - expected_labels
        for labels in extra_labels:
            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.SCHEMA_MISMATCH,
                f"Unexpected node labels in Memgraph: {list(labels)}",
                expected=None,
                actual=list(labels),
                recommendation="Verify if these labels were intentionally created",
            )

    def _validate_node_properties(
        self, expected: Dict[str, Any], actual: Dict[str, Any]
    ):
        """Validate node properties match expected schema."""
        # Create lookup dictionaries
        expected_nodes = {tuple(node["labels"]): node for node in expected["nodes"]}
        actual_nodes = {tuple(node["labels"]): node for node in actual["nodes"]}

        for labels, expected_node in expected_nodes.items():
            if labels not in actual_nodes:
                continue  # Already handled in label validation

            actual_node = actual_nodes[labels]
            expected_props = expected_node.get("properties", {})
            actual_props = actual_node.get("properties", {})

            # Check for missing properties
            missing_props = set(expected_props.keys()) - set(actual_props.keys())
            for prop in missing_props:
                self.add_issue(
                    ValidationSeverity.CRITICAL,
                    ValidationCategory.SCHEMA_MISMATCH,
                    f"Missing property '{prop}' on node {list(labels)}",
                    expected=prop,
                    actual=None,
                    recommendation="Check property mapping in migration script",
                )

            # TODO: Re-enable property type mismatch validation once type
            # mapping is stabilized
            # Check for type mismatches
            # for prop, expected_type in expected_props.items():
            #     if prop in actual_props:
            #         actual_type = actual_props[prop]
            #         if not self._types_compatible(expected_type, actual_type):
            #             self.add_issue(
            #                 ValidationSeverity.WARNING,
            #                 ValidationCategory.SCHEMA_MISMATCH,
            #                 f"Property '{prop}' type mismatch on node {list(labels)}",
            #                 expected=expected_type,
            #                 actual=actual_type,
            #                 recommendation="Verify data transformation logic",
            #             )

    def _validate_relationships(self, expected: Dict[str, Any], actual: Dict[str, Any]):
        """Validate relationship types and properties."""
        expected_rels = {rel["type"]: rel for rel in expected["relationships"]}
        actual_rels = {rel["type"]: rel for rel in actual["relationships"]}

        # Check for missing relationship types
        missing_rels = set(expected_rels.keys()) - set(actual_rels.keys())
        for rel_type in missing_rels:
            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.SCHEMA_MISMATCH,
                f"Missing relationship type: {rel_type}",
                expected=rel_type,
                actual=None,
                recommendation="Check relationship creation in migration script",
            )

        # Check relationship properties
        for rel_type, expected_rel in expected_rels.items():
            if rel_type not in actual_rels:
                continue

            actual_rel = actual_rels[rel_type]
            expected_props = expected_rel.get("properties", {})
            actual_props = actual_rel.get("properties", {})

            missing_props = set(expected_props.keys()) - set(actual_props.keys())
            for prop in missing_props:
                self.add_issue(
                    ValidationSeverity.WARNING,
                    ValidationCategory.SCHEMA_MISMATCH,
                    f"Missing property '{prop}' on relationship {rel_type}",
                    expected=prop,
                    actual=None,
                    recommendation="Check relationship property mapping",
                )

    def _validate_indexes(self, expected: Dict[str, Any], actual: Dict[str, Any]):
        """Validate that expected indexes exist in Memgraph."""
        expected_indexes = expected.get("indexes", [])
        actual_indexes = actual.get("indexes", [])

        # Create comparable representations
        expected_index_keys = set()
        for idx in expected_indexes:
            if idx.get("type") == "node":
                key = (
                    "node",
                    tuple(idx.get("labels", [])),
                    tuple(idx.get("properties", [])),
                )
            else:
                key = (
                    "edge",
                    idx.get("edge_type", ""),
                    tuple(idx.get("properties", [])),
                )
            expected_index_keys.add(key)

        actual_index_keys = set()
        for idx in actual_indexes:
            if idx.get("type") == "node":
                key = (
                    "node",
                    tuple(idx.get("labels", [])),
                    tuple(idx.get("properties", [])),
                )
            else:
                key = (
                    "edge",
                    idx.get("edge_type", ""),
                    tuple(idx.get("properties", [])),
                )
            actual_index_keys.add(key)

        # Check for missing indexes
        missing_indexes = expected_index_keys - actual_index_keys
        for index_key in missing_indexes:
            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.PERFORMANCE,
                f"Missing index: {index_key}",
                expected=index_key,
                actual=None,
                recommendation="Consider creating missing indexes for performance",
            )

    def _validate_constraints(self, expected: Dict[str, Any], actual: Dict[str, Any]):
        """Validate that expected constraints exist in Memgraph."""
        expected_constraints = expected.get("constraints", [])
        actual_constraints = actual.get("constraints", [])

        # Create comparable representations
        expected_constraint_keys = set()
        for const in expected_constraints:
            if const.get("type") == "node":
                key = (
                    "node",
                    tuple(const.get("labels", [])),
                    tuple(const.get("properties", [])),
                    const.get("constraint_type"),
                )
            else:
                key = (
                    "edge",
                    const.get("edge_type", ""),
                    tuple(const.get("properties", [])),
                    const.get("constraint_type"),
                )
            expected_constraint_keys.add(key)

        actual_constraint_keys = set()
        for const in actual_constraints:
            if const.get("type") == "node":
                key = (
                    "node",
                    tuple(const.get("labels", [])),
                    tuple(const.get("properties", [])),
                    const.get("constraint_type"),
                )
            else:
                key = (
                    "edge",
                    const.get("edge_type", ""),
                    tuple(const.get("properties", [])),
                    const.get("constraint_type"),
                )
            actual_constraint_keys.add(key)

        # Check for missing constraints
        missing_constraints = expected_constraint_keys - actual_constraint_keys
        for constraint_key in missing_constraints:
            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.DATA_INTEGRITY,
                f"Missing constraint: {constraint_key}",
                expected=constraint_key,
                actual=None,
                recommendation="Ensure data integrity by creating missing constraints",
            )

    def _validate_data_counts(self, expected_data_counts: Dict[str, int]):
        """
        Validate actual data counts against expected counts.

        Args:
            expected_data_counts: Expected counts with keys like "nodes", "relationships"
        """
        try:
            actual_counts = self._get_actual_data_counts()

            # Validate node count
            if "nodes" in expected_data_counts:
                expected_nodes = expected_data_counts["nodes"]
                actual_nodes = actual_counts["nodes"]

                if actual_nodes != expected_nodes:
                    severity = (
                        ValidationSeverity.CRITICAL
                        if abs(actual_nodes - expected_nodes) > expected_nodes * 0.1
                        else ValidationSeverity.WARNING
                    )
                    self.add_issue(
                        severity,
                        ValidationCategory.DATA_INTEGRITY,
                        f"Node count mismatch: expected {expected_nodes}, got {actual_nodes}",
                        expected=expected_nodes,
                        actual=actual_nodes,
                        recommendation="Check migration completeness and data source consistency",
                    )
                else:
                    logger.info(f"✅ Node count validation passed: {actual_nodes} nodes")

            # Validate relationship count
            if "relationships" in expected_data_counts:
                expected_rels = expected_data_counts["relationships"]
                actual_rels = actual_counts["relationships"]

                # Relationships can vary more due to optional FKs, so be more lenient
                if actual_rels < expected_rels * 0.5:
                    self.add_issue(
                        ValidationSeverity.WARNING,
                        ValidationCategory.DATA_INTEGRITY,
                        f"Low relationship count: expected ~{expected_rels}, got {actual_rels}",
                        expected=expected_rels,
                        actual=actual_rels,
                        recommendation="Check foreign key constraints and data completeness",
                    )
                else:
                    logger.info(
                        f"✅ Relationship count acceptable: {actual_rels} relationships"
                    )

        except Exception as e:
            logger.error(f"Error validating data counts: {e}")
            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.DATA_INTEGRITY,
                f"Data count validation failed: {str(e)}",
                recommendation="Check database connection and query permissions",
            )

    def _get_actual_data_counts(self) -> Dict[str, int]:
        """
        Get actual node and relationship counts from Memgraph schema info.

        Returns:
            Dictionary with "nodes" and "relationships" counts
        """
        # Return cached result if available
        if self._cached_data_counts is not None:
            return self._cached_data_counts

        try:
            # Use the already-retrieved schema info which contains counts
            actual_schema = self._get_actual_schema()

            # Extract counts from schema data
            data_counts = actual_schema.get("data_counts", {})
            if data_counts:
                nodes = data_counts.get("total_nodes", 0)
                relationships = data_counts.get("total_relationships", 0)

                # Check if we need to count relationships with fallback query
                # This happens when relationships don't have properties
                if relationships == 0 and actual_schema.get("relationships"):
                    # Check if any relationships were marked as needing count
                    needs_counting = any(
                        rel.get("relationship_count", 0) == -1
                        for rel in actual_schema.get("relationships", [])
                    )

                    if needs_counting:
                        # Use fallback query for relationships
                        relationships = self._count_relationships_fallback()
                        logger.info("Used fallback query for relationship count")

                result = {"nodes": nodes, "relationships": relationships}

                # Cache the result
                self._cached_data_counts = result
                logger.info(
                    "Data counts from schema: %d nodes, %d rel", nodes, relationships
                )
                return result

            # Fallback: if schema doesn't have counts, calculate from node lists
            # This happens when using non-JSON schema format
            nodes = len(actual_schema.get("nodes", []))
            relationships = len(actual_schema.get("relationships", []))

            result = {
                "nodes": nodes,  # This will be type count, not data count
                "relationships": relationships,
            }

            # Cache the result
            self._cached_data_counts = result
            logger.warning(
                "Schema info didn't contain data counts, " "using type counts instead"
            )
            logger.info(
                "Schema-based counts: %d node types, %d rel types", nodes, relationships
            )
            return result

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Failed to get data counts from schema: %s", str(e))
            result = {"nodes": 0, "relationships": 0}
            self._cached_data_counts = result
            return result

    def _count_relationships_fallback(self) -> int:
        """
        Fallback method to count relationships using direct Cypher query.
        Used when relationships don't have properties and can't be counted.

        Returns:
            Total number of relationships in the database
        """
        try:
            if hasattr(self.connection, "query"):
                # Direct connection with query method
                query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
                rel_result = self.connection.query(query)
                rel_count = rel_result[0]["rel_count"] if rel_result else 0
            else:
                # Cursor-based connection
                cursor = self.connection.cursor()
                cursor.execute("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = cursor.fetchone()[0]

            logger.debug("Fallback relationship count: %d", rel_count)
            return rel_count

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Fallback relationship counting failed: %s", str(e))
            return 0

    def _types_compatible(self, expected_type: str, actual_type: str) -> bool:
        """Check if actual type is compatible with expected type."""
        # Define type compatibility mappings
        compatible_types = {
            "String": ["String", "TEXT", "VARCHAR"],
            "Integer": ["Integer", "INT", "BIGINT"],
            "Float": ["Float", "DOUBLE", "DECIMAL"],
            "Boolean": ["Boolean", "BOOL"],
            "Date": ["Date", "DATE"],
            "LocalDateTime": ["LocalDateTime", "DATETIME", "TIMESTAMP"],
            "LocalTime": ["LocalTime", "TIME"],
        }

        expected_compatible = compatible_types.get(expected_type, [expected_type])
        return actual_type in expected_compatible

    def _update_metrics(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        expected_data_counts: Dict[str, int] = None,
    ):
        """Update validation metrics using the base ValidationMetrics."""
        # Update basic counts
        self.metrics.tables_total = len(expected.get("nodes", []))
        self.metrics.tables_covered = len(actual.get("nodes", []))

        # Relationships
        self.metrics.relationships_total = len(expected.get("relationships", []))
        self.metrics.relationships_covered = len(actual.get("relationships", []))

        # Indexes
        self.metrics.indexes_total = len(expected.get("indexes", []))
        self.metrics.indexes_covered = len(actual.get("indexes", []))

        # Constraints
        self.metrics.constraints_total = len(expected.get("constraints", []))
        self.metrics.constraints_covered = len(actual.get("constraints", []))

        # Add data count metrics if available
        if expected_data_counts:
            actual_counts = self._get_actual_data_counts()
            # Store additional metrics (can be accessed via metrics object)
            self.metrics.data_nodes_expected = expected_data_counts.get("nodes", 0)
            self.metrics.data_nodes_actual = actual_counts.get("nodes", 0)
            self.metrics.data_relationships_expected = expected_data_counts.get(
                "relationships", 0
            )
            self.metrics.data_relationships_actual = actual_counts.get(
                "relationships", 0
            )

        # Calculate coverage percentage
        self.metrics.calculate_coverage()

    def _calculate_validation_score(self) -> float:
        """Calculate a validation score (0-100)."""
        if not self.issues:
            return 100.0

        # Weight different severities
        critical_weight = 10
        warning_weight = 3
        info_weight = 1

        total_penalty = sum(
            critical_weight
            if issue.severity == ValidationSeverity.CRITICAL
            else warning_weight
            if issue.severity == ValidationSeverity.WARNING
            else info_weight
            for issue in self.issues
        )

        # Calculate score (max penalty of 100)
        max_penalty = 100
        score = max(0, 100 - (total_penalty * 100 / max_penalty))
        return round(score, 2)


def validate_memgraph_data(
    expected_model: GraphModel,
    memgraph_connection,
    expected_data_counts: Dict[str, int] = None,
    detailed_report: bool = True,
) -> ValidationResult:
    """
    Convenience function for post-migration Memgraph data validation.

    Args:
        expected_model: Expected GraphModel/spec.json
        memgraph_connection: Connection to Memgraph database
        expected_data_counts: Optional expected node/relationship counts
        detailed_report: Whether to include detailed issue information

    Returns:
        ValidationResult with comparison results
    """
    validator = MemgraphDataValidator(memgraph_connection)
    result = validator.validate_post_migration(expected_model, expected_data_counts)

    if detailed_report:
        logger.info("Validation Summary: %s", result.summary)
        validation_score = result.details.get("validation_score", 0)
        logger.info("Validation Score: %d/100", validation_score)

        for issue in result.issues:
            log_level = (
                logging.ERROR
                if issue.severity == ValidationSeverity.CRITICAL
                else logging.WARNING
            )
            logger.log(log_level, "%s: %s", issue.category, issue.message)
            if issue.recommendation:
                logger.info("  Recommendation: %s", issue.recommendation)

    return result
