"""
Graph Schema Validator for graph models.

This module provides comprehensive validation of GraphModel objects against
the original database structure to ensure complete coverage and correctness
before migration begins. This is Type 1 validation in the two-tier system.
"""

import logging
from typing import Dict, Any, TYPE_CHECKING
from .base import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    ValidationCategory,
)

if TYPE_CHECKING:
    from ..models.graph_models import GraphModel

logger = logging.getLogger(__name__)


class GraphSchemaValidator(BaseValidator):
    """
    Validates GraphModel against original database structure.

    This validator ensures that the GraphModel properly represents
    all tables, properties, relationships, indexes, and constraints
    from the source database before migration begins.
    """

    def validate(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> ValidationResult:
        """
        Perform comprehensive graph schema validation.

        Args:
            graph_model: GraphModel to validate
            database_structure: Original database structure from data_interface

        Returns:
            ValidationResult with detailed validation results
        """
        self.reset()

        logger.info("Starting graph schema validation...")

        try:
            # Basic validation checks
            if not graph_model:
                self.add_issue(
                    ValidationSeverity.CRITICAL,
                    ValidationCategory.STRUCTURE,
                    "Graph model is not provided",
                    recommendation="Ensure graph model is created",
                )

            if not database_structure:
                self.add_issue(
                    ValidationSeverity.CRITICAL,
                    ValidationCategory.STRUCTURE,
                    "Database structure is not provided",
                    recommendation="Ensure database structure is extracted",
                )

            # If basic checks fail, return early
            critical_issues = [
                issue
                for issue in self.issues
                if issue.severity == ValidationSeverity.CRITICAL
            ]
            if critical_issues:
                return ValidationResult(
                    validation_type="graph_schema",
                    success=False,
                    summary="Validation failed: Missing required inputs",
                    issues=self.issues,
                    metrics=self.metrics,
                )

            # Core validations
            self._validate_table_coverage(graph_model, database_structure)
            self._validate_property_coverage(graph_model, database_structure)
            self._validate_relationship_coverage(graph_model, database_structure)
            self._validate_index_coverage(graph_model, database_structure)
            self._validate_constraint_coverage(graph_model, database_structure)

            # Quality validations
            self._validate_schema_consistency(graph_model)
            self._validate_naming_conventions(graph_model)
            self._validate_performance_considerations(graph_model)

            # Calculate final metrics
            self.metrics.calculate_coverage()

            # Generate summary
            summary = self._generate_summary()
            success = not any(
                issue.severity == ValidationSeverity.CRITICAL for issue in self.issues
            )

            result = ValidationResult(
                validation_type="graph_schema",
                success=success,
                summary=summary,
                issues=self.issues,
                metrics=self.metrics,
                details={
                    "database_structure_summary": self._get_db_summary(
                        database_structure
                    ),
                    "graph_model_summary": self._get_model_summary(graph_model),
                },
            )

            logger.info("Graph schema validation completed: %s", summary)
            return result

        except Exception as e:
            logger.error("Graph schema validation failed: %s", str(e))
            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.STRUCTURE,
                f"Validation process failed: {str(e)}",
                recommendation=("Check graph model and database structure format"),
            )

            return ValidationResult(
                validation_type="graph_schema",
                success=False,
                summary=f"Validation failed: {str(e)}",
                issues=self.issues,
                metrics=self.metrics,
            )

    def _validate_table_coverage(self, graph_model, database_structure):
        """Validate that all entity tables are represented as nodes."""
        logger.debug("Validating table coverage...")

        # Handle both new structured format and legacy format
        if hasattr(database_structure, "entity_tables"):
            # New structured format - work directly with objects
            entity_tables = database_structure.entity_tables
        else:
            # Legacy format fallback
            entity_tables = database_structure.get("entity_tables", {})

        self.metrics.tables_total = len(entity_tables)

        if not entity_tables:
            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.COVERAGE,
                "No entity tables found in database structure",
                recommendation="Verify database structure extraction",
            )
            return

        # Get tables covered by nodes
        covered_tables = set()
        for node in graph_model.nodes:
            if node.source and hasattr(node.source, "name"):
                covered_tables.add(node.source.name)

        self.metrics.tables_covered = len(covered_tables)

        # Check for missing tables
        missing_tables = set(entity_tables.keys()) - covered_tables
        if missing_tables:
            # Get details about missing tables
            missing_table_details = []
            for table_name in sorted(missing_tables):
                if hasattr(database_structure, "entity_tables"):
                    table_info = entity_tables[table_name]
                    column_count = len(table_info.columns)
                    pk_count = len(table_info.primary_keys)
                    fk_count = len(table_info.foreign_keys)
                else:
                    table_info = entity_tables[table_name]
                    column_count = len(table_info.get("schema", []))
                    pk_count = len(table_info.get("primary_keys", []))
                    fk_count = len(table_info.get("foreign_keys", []))

                missing_table_details.append(
                    f"'{table_name}' ({column_count} cols, {pk_count} PKs, "
                    f"{fk_count} FKs)"
                )

            details_str = "; ".join(missing_table_details)

            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.COVERAGE,
                (
                    f"Missing {len(missing_tables)} entity tables in graph "
                    f"model: {details_str}. These tables should be "
                    f"represented as nodes in the graph model."
                ),
                expected=list(entity_tables.keys()),
                actual=list(covered_tables),
                recommendation=(
                    "Create nodes for each missing table. For example:\n"
                    + "\n".join(
                        [
                            f"  - Add '{table}' node with appropriate labels"
                            for table in sorted(missing_tables)
                        ]
                    )
                ),
                details={"missing_tables": list(missing_tables)},
            )

        logger.debug(
            "Table coverage: %d/%d tables covered",
            self.metrics.tables_covered,
            self.metrics.tables_total,
        )

    def _validate_property_coverage(self, graph_model, database_structure):
        """Validate that all table columns are represented as properties."""
        logger.debug("Validating property coverage...")

        # Handle both new structured format and legacy format
        if hasattr(database_structure, "entity_tables"):
            # New structured format - work directly with objects
            entity_tables = database_structure.entity_tables
            foreign_key_columns = self._get_foreign_key_columns(database_structure)

            total_properties = 0
            covered_properties = 0
            missing_by_table = {}

            for table_name, table_info in entity_tables.items():
                # Get column names directly from ColumnInfo objects
                table_columns = {col.name for col in table_info.columns}
                total_properties += len(table_columns)

                # Find corresponding node
                node = self._find_node_for_table(graph_model, table_name)
                if not node:
                    missing_by_table[table_name] = list(table_columns)
                    continue

                # Check property coverage for this node
                node_properties = {prop.key for prop in node.properties}
                missing_props = table_columns - node_properties

                # Separate foreign key columns from regular missing properties
                table_foreign_keys = foreign_key_columns.get(table_name, set())
                missing_foreign_keys = missing_props & table_foreign_keys
                missing_regular_props = missing_props - table_foreign_keys

                # Count coverage: covered + foreign keys that became relationships
                covered_properties += len(table_columns) - len(missing_regular_props)

                # Only report regular properties as missing (not foreign keys)
                if missing_regular_props:
                    missing_by_table[table_name] = list(missing_regular_props)

                # Log foreign keys that became relationships (for debugging)
                if missing_foreign_keys:
                    logger.debug(
                        "Table %s: %d foreign key columns became relationships: %s",
                        table_name,
                        len(missing_foreign_keys),
                        missing_foreign_keys,
                    )
        else:
            # Legacy format fallback
            entity_tables = database_structure.get("entity_tables", {})
            foreign_key_columns = self._get_foreign_key_columns(database_structure)

            total_properties = 0
            covered_properties = 0
            missing_by_table = {}

            for table_name, table_info in entity_tables.items():
                table_columns = {col["field"] for col in table_info.get("schema", [])}
                total_properties += len(table_columns)

                # Find corresponding node
                node = self._find_node_for_table(graph_model, table_name)
                if not node:
                    missing_by_table[table_name] = list(table_columns)
                    continue

                # Check property coverage for this node
                node_properties = {prop.key for prop in node.properties}
                missing_props = table_columns - node_properties

                # Separate foreign key columns from regular missing properties
                table_foreign_keys = foreign_key_columns.get(table_name, set())
                missing_foreign_keys = missing_props & table_foreign_keys
                missing_regular_props = missing_props - table_foreign_keys

                # Count coverage: covered + foreign keys that became relationships
                covered_properties += len(table_columns) - len(missing_regular_props)

                # Only report regular properties as missing (not foreign keys)
                if missing_regular_props:
                    missing_by_table[table_name] = list(missing_regular_props)

                # Log foreign keys that became relationships (for debugging)
                if missing_foreign_keys:
                    logger.debug(
                        "Table %s: %d foreign key columns became relationships: %s",
                        table_name,
                        len(missing_foreign_keys),
                        missing_foreign_keys,
                    )

        self.metrics.properties_total = total_properties
        self.metrics.properties_covered = covered_properties

        if missing_by_table:
            total_missing = sum(len(props) for props in missing_by_table.values())

            # Create detailed message about missing properties
            missing_details = []
            for table_name, missing_props in missing_by_table.items():
                props_str = ", ".join(sorted(missing_props))
                missing_details.append(f"Table '{table_name}': {props_str}")

            details_message = "; ".join(missing_details)

            self.add_issue(
                ValidationSeverity.CRITICAL,
                ValidationCategory.COVERAGE,
                (
                    f"Missing {total_missing} non-foreign-key properties "
                    f"across {len(missing_by_table)} tables. Details: "
                    f"{details_message}. These properties may contain "
                    "important data that should be preserved in "
                    "the graph model."
                ),
                expected=f"{total_properties} properties",
                actual=f"{covered_properties} properties",
                recommendation=(
                    "Add missing properties to corresponding nodes:\n"
                    + "\n".join(
                        [
                            f"  - Add to '{table}' node: {', '.join(props)}"
                            for table, props in missing_by_table.items()
                        ]
                    )
                    + "\nNote: Foreign key columns are correctly modeled "
                    "as relationships, not properties."
                ),
                details={"missing_by_table": missing_by_table},
            )

        logger.debug(
            "Property coverage: %d/%d properties covered (including foreign keys as relationships)",
            covered_properties,
            total_properties,
        )

    def _validate_relationship_coverage(self, graph_model, database_structure):
        """Validate that foreign keys are represented as relationships."""
        logger.debug("Validating relationship coverage...")

        relationships = database_structure.get("relationships", [])
        self.metrics.relationships_total = len(relationships)

        if not relationships:
            self.add_issue(
                ValidationSeverity.INFO,
                ValidationCategory.COVERAGE,
                "No relationships found in database structure",
                recommendation="Verify foreign key extraction is correct",
            )
            return

        # Simple coverage check - count modeled vs database relationships
        modeled_count = len(graph_model.edges)
        self.metrics.relationships_covered = min(modeled_count, len(relationships))

        if modeled_count < len(relationships):
            # Get details about database relationships
            db_relationship_details = []
            for rel in relationships[:5]:  # Show first 5 for brevity
                if isinstance(rel, dict):
                    from_table = rel.get("from_table", "unknown")
                    to_table = rel.get("to_table", "unknown")
                    column = rel.get("column", "unknown")
                    db_relationship_details.append(
                        f"{from_table}.{column} -> {to_table}"
                    )
                else:
                    # Handle object format if needed
                    db_relationship_details.append(str(rel))

            missing_count = len(relationships) - modeled_count
            details_str = "; ".join(db_relationship_details)
            if len(relationships) > 5:
                details_str += f" (and {len(relationships) - 5} more)"

            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.COVERAGE,
                (
                    f"Fewer relationships modeled ({modeled_count}) than in "
                    f"database ({len(relationships)}). Missing "
                    f"{missing_count} potential relationships. Database "
                    f"relationships include: {details_str}. These foreign key "
                    "relationships may represent important "
                    "connections that should be modeled as graph "
                    "relationships."
                ),
                expected=f"{len(relationships)} relationships",
                actual=f"{modeled_count} relationships",
                recommendation=(
                    "Review database foreign keys and consider adding "
                    "missing relationships."
                ),
            )

        logger.debug(
            "Relationship coverage: %d/%d relationships covered",
            self.metrics.relationships_covered,
            len(relationships),
        )

    def _validate_index_coverage(self, graph_model, database_structure):
        """Validate that important indexes are planned."""
        logger.debug("Validating index coverage...")

        # Get planned indexes from model
        planned_indexes = len(graph_model.node_indexes) + len(graph_model.edge_indexes)
        self.metrics.indexes_covered = planned_indexes

        # Handle both new structured format and legacy format
        if hasattr(database_structure, "entity_tables"):
            # New structured format - work directly with objects
            db_indexes_count = sum(
                len(table_info.indexes)
                for table_info in database_structure.entity_tables.values()
            )
        else:
            # Legacy format fallback
            entity_tables = database_structure.get("entity_tables", {})
            db_indexes_count = 0
            for table_info in entity_tables.values():
                db_indexes_count += len(table_info.get("indexes", []))

        self.metrics.indexes_total = db_indexes_count

        if db_indexes_count > 0 and planned_indexes == 0:
            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.PERFORMANCE,
                f"No indexes planned, but {db_indexes_count} exist in source database",
                expected="Indexes planned for performance",
                actual="No indexes planned",
                recommendation="Consider adding indexes for frequently queried properties",
            )

        logger.debug(
            "Index planning: %d indexes planned vs %d in source database",
            planned_indexes,
            db_indexes_count,
        )

    def _validate_constraint_coverage(self, graph_model, database_structure):
        """Validate that database constraints are represented."""
        logger.debug("Validating constraint coverage...")

        # Get planned constraints from model
        planned_constraints = len(graph_model.node_constraints) + len(
            graph_model.edge_constraints
        )
        self.metrics.constraints_covered = planned_constraints

        # Handle both new structured format and legacy format
        if hasattr(database_structure, "entity_tables"):
            # New structured format - work directly with objects
            db_constraints_count = 0
            for table_info in database_structure.entity_tables.values():
                db_constraints_count += len(table_info.primary_keys)
                db_constraints_count += len(table_info.foreign_keys)
        else:
            # Legacy format fallback
            entity_tables = database_structure.get("entity_tables", {})
            db_constraints_count = 0
            for table_info in entity_tables.values():
                db_constraints_count += len(table_info.get("primary_keys", []))
                db_constraints_count += len(table_info.get("foreign_keys", []))

        self.metrics.constraints_total = db_constraints_count

        if db_constraints_count > 0 and planned_constraints == 0:
            self.add_issue(
                ValidationSeverity.WARNING,
                ValidationCategory.CONSISTENCY,
                f"No constraints planned, but {db_constraints_count} "
                "exist in source",
                expected="Constraints planned for data integrity",
                actual="No constraints planned",
                recommendation="Consider adding constraints for data " "integrity",
            )

        logger.debug(
            "Constraint planning: %d constraints planned vs %d in source",
            planned_constraints,
            db_constraints_count,
        )

    def _validate_schema_consistency(self, graph_model):
        """Validate internal consistency of the graph model."""
        logger.debug("Validating schema consistency...")

        # Check for duplicate node labels
        seen_labels = set()
        for node in graph_model.nodes:
            for label in node.labels:
                if label in seen_labels:
                    self.add_issue(
                        ValidationSeverity.WARNING,
                        ValidationCategory.CONSISTENCY,
                        f"Duplicate node label '{label}' found",
                        recommendation="Ensure node labels are unique",
                    )
                seen_labels.add(label)

        # Check for orphaned relationships
        node_labels = set()
        for node in graph_model.nodes:
            node_labels.update(node.labels)

        for edge in graph_model.edges:
            # Check start node labels
            missing_start = set(edge.start_node_labels) - node_labels
            if missing_start:
                self.add_issue(
                    ValidationSeverity.CRITICAL,
                    ValidationCategory.CONSISTENCY,
                    f"Relationship '{edge.edge_type}' references missing "
                    f"start node labels: {missing_start}",
                    recommendation="Ensure all relationship endpoints "
                    "reference existing node labels",
                )

            # Check end node labels
            missing_end = set(edge.end_node_labels) - node_labels
            if missing_end:
                self.add_issue(
                    ValidationSeverity.CRITICAL,
                    ValidationCategory.CONSISTENCY,
                    f"Relationship '{edge.edge_type}' references missing "
                    f"end node labels: {missing_end}",
                    recommendation="Ensure all relationship endpoints "
                    "reference existing node labels",
                )

    def _validate_naming_conventions(self, graph_model):
        """Validate naming conventions and best practices."""
        logger.debug("Validating naming conventions...")

        # Check node label conventions
        for node in graph_model.nodes:
            for label in node.labels:
                if not label[0].isupper():
                    self.add_issue(
                        ValidationSeverity.INFO,
                        ValidationCategory.CONSISTENCY,
                        f"Node label '{label}' should start with uppercase",
                        recommendation="Use PascalCase for node labels",
                    )

        # Check relationship type conventions
        for edge in graph_model.edges:
            if not edge.edge_type.isupper():
                self.add_issue(
                    ValidationSeverity.INFO,
                    ValidationCategory.CONSISTENCY,
                    f"Relationship type '{edge.edge_type}' should be " "uppercase",
                    recommendation="Use UPPER_CASE for relationship types",
                )

    def _validate_performance_considerations(self, graph_model):
        """Validate performance-related aspects."""
        logger.debug("Validating performance considerations...")

        # Check for nodes without indexes on key properties
        for node in graph_model.nodes:
            key_properties = [
                prop.key
                for prop in node.properties
                if "id" in prop.key.lower() or "key" in prop.key.lower()
            ]

            if key_properties:
                # Check if any of these have planned indexes
                has_index = any(
                    set(key_properties) & set(index.properties)
                    for index in graph_model.node_indexes
                    if index.labels
                    and any(label in node.labels for label in index.labels)
                )

                if not has_index:
                    node_label = "/".join(node.labels)
                    key_props = ", ".join(key_properties)
                    self.add_issue(
                        ValidationSeverity.INFO,
                        ValidationCategory.PERFORMANCE,
                        f"Node {node_label} has key properties without " "indexes",
                        recommendation=f"Consider adding indexes for: " f"{key_props}",
                    )

    # Helper methods

    def _get_foreign_key_columns(self, database_structure):
        """
        Extract foreign key columns from database structure.

        Returns a dict mapping table_name -> set of foreign key column names.
        """
        foreign_key_columns = {}

        # Check if we have the new DatabaseStructure model
        if hasattr(database_structure, "entity_tables"):
            # New structured format - work directly with objects
            for table_name, table_info in database_structure.entity_tables.items():
                fk_columns = {fk.column_name for fk in table_info.foreign_keys}
                if fk_columns:
                    foreign_key_columns[table_name] = fk_columns
        else:
            # Legacy format fallback
            entity_tables = database_structure.get("entity_tables", {})
            for table_name, table_info in entity_tables.items():
                foreign_keys = table_info.get("foreign_keys", [])
                fk_columns = set()

                for fk in foreign_keys:
                    if isinstance(fk, dict):
                        # Handle dict format
                        if "column" in fk:
                            fk_columns.add(fk["column"])
                        elif "column_name" in fk:
                            fk_columns.add(fk["column_name"])
                    else:
                        # Handle object format
                        if hasattr(fk, "column_name"):
                            fk_columns.add(fk.column_name)

                if fk_columns:
                    foreign_key_columns[table_name] = fk_columns

        return foreign_key_columns

    def _find_node_for_table(self, graph_model, table_name: str):
        """Find the node that represents a given table."""
        for node in graph_model.nodes:
            if node.source and hasattr(node.source, "name"):
                if node.source.name == table_name:
                    return node
        return None

    def _get_model_summary(self, graph_model) -> Dict[str, Any]:
        """Get a summary of the graph model."""
        return {
            "nodes": len(graph_model.nodes),
            "relationships": len(graph_model.edges),
            "node_indexes": len(graph_model.node_indexes),
            "edge_indexes": len(graph_model.edge_indexes),
            "node_constraints": len(graph_model.node_constraints),
            "edge_constraints": len(graph_model.edge_constraints),
            "node_labels": [
                label for node in graph_model.nodes for label in node.labels
            ],
            "relationship_types": [edge.edge_type for edge in graph_model.edges],
        }

    def _get_db_summary(self, database_structure) -> Dict[str, Any]:
        """Get a summary of the database structure."""
        entity_tables = database_structure.get("entity_tables", {})
        relationships = database_structure.get("relationships", [])

        return {
            "entity_tables": len(entity_tables),
            "relationships": len(relationships),
            "database_type": database_structure.get("database_type", "unknown"),
            "database_name": database_structure.get("database_name", "unknown"),
        }
