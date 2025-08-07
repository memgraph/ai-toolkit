"""
Hypothetical Graph Modeling (HyGM) Module

This module uses LLM to analyze database schemas and provide intelligent
graph modeling suggestions for optimal MySQL to Memgraph migration.
Supports both automatic and interactive modeling modes.
"""

import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ModelingMode(Enum):
    """Modeling modes for HyGM."""

    AUTOMATIC = "automatic"
    INTERACTIVE = "interactive"


@dataclass
class GraphNode:
    """Represents a node in the graph model."""

    name: str
    label: str
    properties: List[str]
    primary_key: str
    indexes: List[str]
    constraints: List[str]
    source_table: str
    modeling_rationale: str


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph model."""

    name: str
    type: str  # "one_to_many", "many_to_many", "one_to_one"
    from_node: str
    to_node: str
    properties: List[str]
    directionality: str  # "directed", "undirected"
    source_info: Dict[str, Any]
    modeling_rationale: str


@dataclass
class GraphModel:
    """Complete graph model for the database."""

    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    modeling_decisions: List[str]
    optimization_suggestions: List[str]
    data_patterns: Dict[str, Any]


class HyGM:
    """
    Uses LLM to create intelligent graph models from relational schemas.

    Supports two modes:
    - AUTOMATIC: Creates graph model without user interaction
    - INTERACTIVE: Interactive mode with user feedback via terminal input
    """

    def __init__(self, llm, mode: ModelingMode = ModelingMode.AUTOMATIC):
        """Initialize with an LLM instance and modeling mode."""
        self.llm = llm
        self.mode = mode
        self.current_graph_model = None
        self.iteration_count = 0
        self.database_structure = None

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

        Args:
            graph_model: The graph model to validate
            database_structure: Original database structure

        Returns:
            Dictionary with validation results and issues
        """
        logger.info("Validating graph model...")

        issues = []
        warnings = []

        # 1. Check all entity tables are represented as nodes
        entity_tables = set(database_structure.get("entity_tables", {}).keys())
        model_source_tables = {node.source_table for node in graph_model.nodes}

        missing_tables = entity_tables - model_source_tables
        if missing_tables:
            issues.append(f"Missing nodes for entity tables: {list(missing_tables)}")

        # 2. Validate node properties exist in source tables
        for node in graph_model.nodes:
            source_table = node.source_table
            if source_table in database_structure.get("entity_tables", {}):
                table_info = database_structure["entity_tables"][source_table]
                available_columns = self._get_table_columns(table_info)

                invalid_props = [
                    prop for prop in node.properties if prop not in available_columns
                ]
                if invalid_props:
                    issues.append(
                        f"Node {node.label} has invalid properties: " f"{invalid_props}"
                    )

        # 3. Check primary key mapping
        for node in graph_model.nodes:
            source_table = node.source_table
            if source_table in database_structure.get("entity_tables", {}):
                table_info = database_structure["entity_tables"][source_table]

                if node.primary_key not in self._get_table_columns(table_info):
                    issues.append(
                        f"Node {node.label} primary key '{node.primary_key}' "
                        f"not found in source table"
                    )

        # 4. Validate relationships
        node_names = {node.name for node in graph_model.nodes}
        for rel in graph_model.relationships:
            if rel.from_node not in node_names:
                issues.append(
                    f"Relationship {rel.name} references unknown from_node: "
                    f"{rel.from_node}"
                )
            if rel.to_node not in node_names:
                issues.append(
                    f"Relationship {rel.name} references unknown to_node: "
                    f"{rel.to_node}"
                )

        # 5. Check for graph modeling best practices
        # Node labels should be meaningful (not just table names)
        for node in graph_model.nodes:
            if node.label.lower() == node.source_table.lower():
                warnings.append(
                    f"Node label '{node.label}' is same as table name, "
                    f"consider more semantic naming"
                )

        # Large tables should have indexes on commonly queried properties
        for node in graph_model.nodes:
            table_info = database_structure.get("entity_tables", {}).get(
                node.source_table, {}
            )
            row_count = table_info.get("row_count", 0)
            if row_count > 10000 and not node.indexes:
                warnings.append(
                    f"Large table {node.source_table} ({row_count} rows) "
                    f"has no indexes defined"
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
        Apply natural language feedback to modify the current graph model.

        Args:
            feedback: Natural language feedback from user
        """
        logger.info("Processing natural language feedback...")

        system_message = SystemMessage(
            content="""
You are a graph modeling expert that processes natural language feedback 
to modify graph models. 

Parse the user's feedback and return specific modifications in JSON format.

Supported operations:
1. Change node label: "change_node_label"
2. Rename property: "rename_property" 
3. Drop property: "drop_property"
4. Add property: "add_property"
5. Change relationship name: "change_relationship_name"
6. Drop relationship: "drop_relationship"
7. Add index: "add_index"
8. Drop index: "drop_index"

Return format:
{
  "operations": [
    {
      "type": "change_node_label",
      "target": "old_label",
      "new_value": "new_label"
    },
    {
      "type": "rename_property",
      "node": "node_label",
      "target": "old_prop_name", 
      "new_value": "new_prop_name"
    }
  ]
}
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
            response = self.llm.invoke([system_message, human_message])
            operations_data = json.loads(response.content)

            self._execute_model_operations(operations_data.get("operations", []))

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            # Could implement fallback or ask for clarification
        except Exception as e:
            logger.error(f"Error processing natural language feedback: {e}")
            # Could implement fallback or ask for clarification

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
        updated_node = None
        for node in self.current_graph_model.nodes:
            if node.label == old_label:
                logger.info(f"Changing node label: {old_label} -> {new_label}")
                node.label = new_label
                updated_node = node
                break

        if updated_node:
            # Update relationships that reference this node
            # Relationships use source_table (lowercase) as identifiers, not labels
            old_node_id = updated_node.source_table.lower()
            new_node_id = new_label.lower()  # Use new label as lowercase identifier

            for rel in self.current_graph_model.relationships:
                if rel.from_node == old_node_id:
                    rel.from_node = new_node_id
                if rel.to_node == old_node_id:
                    rel.to_node = new_node_id

    def _rename_node_property(
        self, node_label: str, old_prop: str, new_prop: str
    ) -> None:
        """Rename a property in a node."""
        for node in self.current_graph_model.nodes:
            if node.label == node_label:
                if old_prop in node.properties:
                    logger.info(
                        f"Renaming property in {node_label}: "
                        f"{old_prop} -> {new_prop}"
                    )
                    idx = node.properties.index(old_prop)
                    node.properties[idx] = new_prop

                    # Update indexes if needed
                    if old_prop in node.indexes:
                        idx = node.indexes.index(old_prop)
                        node.indexes[idx] = new_prop
                break

    def _drop_node_property(self, node_label: str, prop_name: str) -> None:
        """Drop a property from a node."""
        for node in self.current_graph_model.nodes:
            if node.label == node_label:
                if prop_name in node.properties:
                    logger.info(f"Dropping property {prop_name} from {node_label}")
                    node.properties.remove(prop_name)

                    # Remove from indexes if present
                    if prop_name in node.indexes:
                        node.indexes.remove(prop_name)
                break

    def _add_node_property(self, node_label: str, prop_name: str) -> None:
        """Add a property to a node."""
        for node in self.current_graph_model.nodes:
            if node.label == node_label:
                if prop_name not in node.properties:
                    logger.info(f"Adding property {prop_name} to {node_label}")
                    node.properties.append(prop_name)
                break

    def _change_relationship_name(self, old_name: str, new_name: str) -> None:
        """Change a relationship's name."""
        for rel in self.current_graph_model.relationships:
            if rel.name == old_name:
                logger.info(f"Changing relationship name: {old_name} -> {new_name}")
                rel.name = new_name
                break

    def _drop_relationship(self, rel_name: str) -> None:
        """Drop a relationship."""
        self.current_graph_model.relationships = [
            rel
            for rel in self.current_graph_model.relationships
            if rel.name != rel_name
        ]
        logger.info(f"Dropped relationship: {rel_name}")

    def _add_node_index(self, node_label: str, prop_name: str) -> None:
        """Add an index to a node property."""
        for node in self.current_graph_model.nodes:
            if node.label == node_label:
                if prop_name not in node.indexes:
                    logger.info(f"Adding index on {node_label}.{prop_name}")
                    node.indexes.append(prop_name)
                break

    def _drop_node_index(self, node_label: str, prop_name: str) -> None:
        """Drop an index from a node property."""
        for node in self.current_graph_model.nodes:
            if node.label == node_label:
                if prop_name in node.indexes:
                    logger.info(f"Dropping index on {node_label}.{prop_name}")
                    node.indexes.remove(prop_name)
                break

    def _get_model_presentation(self) -> Dict[str, Any]:
        """Get formatted presentation of current model for user review."""
        if not self.current_graph_model:
            return {}

        return {
            "nodes": [
                {
                    "label": node.label,
                    "properties": node.properties,
                    "source_table": node.source_table,
                    "primary_key": node.primary_key,
                    "indexes": node.indexes,
                }
                for node in self.current_graph_model.nodes
            ],
            "relationships": [
                {
                    "name": rel.name,
                    "type": rel.type,
                    "from_node": rel.from_node,
                    "to_node": rel.to_node,
                    "directionality": rel.directionality,
                }
                for rel in self.current_graph_model.relationships
            ],
            "summary": {
                "total_nodes": len(self.current_graph_model.nodes),
                "total_relationships": len(self.current_graph_model.relationships),
            },
        }

    def _get_model_summary(self) -> str:
        """Get concise text summary of current model."""
        if not self.current_graph_model:
            return "No model available"

        summary_parts = ["NODES:"]
        for node in self.current_graph_model.nodes:
            props_str = ", ".join(node.properties[:3])  # Show first 3 props
            if len(node.properties) > 3:
                props_str += "..."
            summary_parts.append(f"- {node.label} ({node.source_table}): {props_str}")

        summary_parts.append("\nRELATIONSHIPS:")
        for rel in self.current_graph_model.relationships:
            rel_str = f"- {rel.from_node} -[{rel.name}]-> {rel.to_node}"
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

        # Use domain context if provided
        context_info = domain_context or "General database migration"

        # Analyze database context
        database_context = self._analyze_database_context(database_structure)

        # Generate comprehensive analysis
        comprehensive_analysis = self._analyze_all_tables_and_relationships(
            database_structure, database_context, context_info
        )

        # Generate final graph model
        graph_model = self._generate_graph_model_from_analysis(
            comprehensive_analysis, database_structure, database_context
        )

        return graph_model

    def _fix_validation_issues(
        self, graph_model: GraphModel, validation_result: Dict[str, Any]
    ) -> GraphModel:
        """Attempt to fix validation issues automatically."""
        logger.info("Attempting to fix validation issues...")

        # For now, just log the issues - could implement automatic fixes
        for issue in validation_result["issues"]:
            logger.warning(f"Validation issue: {issue}")

        # Could implement specific fixes here:
        # - Remove invalid properties
        # - Fix missing primary keys
        # - Validate relationship references

        return graph_model

    # Helper methods for existing functionality that still use old approach
    def _apply_node_changes(self, node_changes: List[Dict[str, Any]]):
        """Apply changes to existing nodes."""
        for change in node_changes:
            node_identifier = change.get("node_identifier")  # name or number

            # Find the node to modify
            target_node = None
            if isinstance(node_identifier, int):
                # By number (1-based)
                if 1 <= node_identifier <= len(self.current_graph_model.nodes):
                    target_node = self.current_graph_model.nodes[node_identifier - 1]
            else:
                # By name
                target_node = next(
                    (
                        n
                        for n in self.current_graph_model.nodes
                        if n.name == node_identifier or n.label == node_identifier
                    ),
                    None,
                )

            if target_node:
                # Apply specific changes
                if "label" in change:
                    target_node.label = change["label"]
                if "properties" in change:
                    target_node.properties = change["properties"]
                if "indexes" in change:
                    target_node.indexes = change["indexes"]
                if "constraints" in change:
                    target_node.constraints = change["constraints"]

                logger.info(f"Modified node {target_node.name}")

    def _apply_relationship_changes(self, relationship_changes: List[Dict[str, Any]]):
        """Apply changes to existing relationships."""
        for change in relationship_changes:
            rel_identifier = change.get("relationship_identifier")

            # Find the relationship to modify
            target_rel = None
            if isinstance(rel_identifier, int):
                # By number (1-based)
                if 1 <= rel_identifier <= len(self.current_graph_model.relationships):
                    target_rel = self.current_graph_model.relationships[
                        rel_identifier - 1
                    ]
            else:
                # By name
                target_rel = next(
                    (
                        r
                        for r in self.current_graph_model.relationships
                        if r.name == rel_identifier
                    ),
                    None,
                )

            if target_rel:
                # Apply specific changes
                if "name" in change:
                    target_rel.name = change["name"]
                if "directionality" in change:
                    target_rel.directionality = change["directionality"]
                if "properties" in change:
                    target_rel.properties = change["properties"]

                logger.info(f"Modified relationship {target_rel.name}")

    def _remove_nodes(self, nodes_to_remove: List[Any]):
        """Remove specified nodes from the model."""
        nodes_removed = []

        for node_identifier in nodes_to_remove:
            # Find nodes to remove
            nodes_to_delete = []

            if isinstance(node_identifier, int):
                # By number (1-based)
                if 1 <= node_identifier <= len(self.current_graph_model.nodes):
                    nodes_to_delete.append(
                        self.current_graph_model.nodes[node_identifier - 1]
                    )
            else:
                # By name
                nodes_to_delete.extend(
                    [
                        n
                        for n in self.current_graph_model.nodes
                        if n.name == node_identifier or n.label == node_identifier
                    ]
                )

            # Remove the nodes
            for node in nodes_to_delete:
                self.current_graph_model.nodes.remove(node)
                nodes_removed.append(node.name)

                # Also remove relationships involving this node
                relationships_to_remove = [
                    r
                    for r in self.current_graph_model.relationships
                    if r.from_node == node.name or r.to_node == node.name
                ]
                for rel in relationships_to_remove:
                    self.current_graph_model.relationships.remove(rel)

        if nodes_removed:
            logger.info(f"Removed nodes: {', '.join(nodes_removed)}")

    def _remove_relationships(self, relationships_to_remove: List[Any]):
        """Remove specified relationships from the model."""
        relationships_removed = []

        for rel_identifier in relationships_to_remove:
            # Find relationships to remove
            rels_to_delete = []

            if isinstance(rel_identifier, int):
                # By number (1-based)
                if 1 <= rel_identifier <= len(self.current_graph_model.relationships):
                    rels_to_delete.append(
                        self.current_graph_model.relationships[rel_identifier - 1]
                    )
            else:
                # By name
                rels_to_delete.extend(
                    [
                        r
                        for r in self.current_graph_model.relationships
                        if r.name == rel_identifier
                    ]
                )

            # Remove the relationships
            for rel in rels_to_delete:
                self.current_graph_model.relationships.remove(rel)
                relationships_removed.append(rel.name)

        if relationships_removed:
            logger.info(f"Removed relationships: {', '.join(relationships_removed)}")

    def _add_nodes(self, nodes_to_add: List[Dict[str, Any]]):
        """Add new nodes to the model."""
        for node_spec in nodes_to_add:
            new_node = GraphNode(
                name=node_spec.get("name", ""),
                label=node_spec.get("label", ""),
                properties=node_spec.get("properties", []),
                primary_key=node_spec.get("primary_key", "id"),
                indexes=node_spec.get("indexes", []),
                constraints=node_spec.get("constraints", []),
                source_table=node_spec.get("source_table", ""),
                modeling_rationale=node_spec.get("rationale", "User-added node"),
            )
            self.current_graph_model.nodes.append(new_node)
            logger.info(f"Added new node: {new_node.name}")

    def _add_relationships(self, relationships_to_add: List[Dict[str, Any]]):
        """Add new relationships to the model."""
        for rel_spec in relationships_to_add:
            new_rel = GraphRelationship(
                name=rel_spec.get("name", ""),
                type=rel_spec.get("type", "one_to_many"),
                from_node=rel_spec.get("from_node", ""),
                to_node=rel_spec.get("to_node", ""),
                properties=rel_spec.get("properties", []),
                directionality=rel_spec.get("directionality", "directed"),
                source_info=rel_spec.get("source_info", {}),
                modeling_rationale=rel_spec.get("rationale", "User-added relationship"),
            )
            self.current_graph_model.relationships.append(new_rel)
            logger.info(f"Added new relationship: {new_rel.name}")

    def _llm_refine_changes(self, feedback: Dict[str, Any]):
        """Use LLM to validate and refine the changes made."""
        system_message = SystemMessage(
            content="""
You are a graph database expert reviewing user-modified graph models.
Validate the changes and suggest improvements for consistency and optimization.
Check for:
1. Semantic consistency of labels and relationship names
2. Proper graph database modeling patterns
3. Performance implications
4. Missing indexes or constraints that should be added
"""
        )

        # Create summary of current model
        model_summary = self.get_current_model_presentation()

        human_message = HumanMessage(
            content=f"""
Review this modified graph model and suggest any improvements:

FEEDBACK APPLIED: {feedback}

CURRENT MODEL:
Nodes: {len(model_summary['nodes'])}
Relationships: {len(model_summary['relationships'])}

NODE DETAILS:
{self._format_nodes_for_llm_review(model_summary['nodes'])}

RELATIONSHIP DETAILS:
{self._format_relationships_for_llm_review(model_summary['relationships'])}

Provide suggestions for:
1. Label consistency and naming improvements
2. Property optimization recommendations  
3. Index and constraint suggestions
4. Overall model improvements
"""
        )

        try:
            response = self.llm.invoke([system_message, human_message])

            # Add LLM suggestions to optimization suggestions
            self.current_graph_model.optimization_suggestions.append(
                f"LLM Review (Iteration {self.iteration_count + 1}): {response.content[:200]}..."
            )

        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")

    def _format_nodes_for_llm_review(self, nodes: List[Dict[str, Any]]) -> str:
        """Format nodes for LLM review."""
        formatted = []
        for node in nodes:
            formatted.append(
                f"- {node['label']} (from {node['source_table']}): "
                f"{len(node['properties'])} properties, "
                f"{len(node['indexes'])} indexes"
            )
        return "\n".join(formatted)

    def _format_relationships_for_llm_review(
        self, relationships: List[Dict[str, Any]]
    ) -> str:
        """Format relationships for LLM review."""
        formatted = []
        for rel in relationships:
            formatted.append(
                f"- {rel['name']}: {rel['from_node']} -> {rel['to_node']} "
                f"({rel['type']}, {rel['directionality']})"
            )
        return "\n".join(formatted)

    def _analyze_database_context(
        self, database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the overall database to understand the domain and context."""

        # Prepare database overview for LLM
        tables_overview = []

        # Use entity_tables if available, otherwise fall back to tables
        tables_to_analyze = database_structure.get(
            "entity_tables", database_structure.get("tables", {})
        )

        for table_name, table_info in tables_to_analyze.items():
            if "schema" in table_info:
                column_names = [col["field"] for col in table_info["schema"]]
            elif "columns" in table_info:
                column_names = list(table_info["columns"].keys())
            else:
                column_names = []

            fk_count = len(table_info.get("foreign_keys", []))
            row_count = table_info.get("row_count", 0)

            tables_overview.append(
                {
                    "name": table_name,
                    "columns": column_names,
                    "foreign_keys": fk_count,
                    "row_count": row_count,
                    "type": table_info.get("type", "entity"),
                }
            )

        system_message = SystemMessage(
            content="""
        You are an expert database architect and graph modeling specialist. 
        Analyze the provided database schema to understand the business domain, 
        data patterns, and optimal graph modeling approach.
        
        Focus on:
        1. Identifying the business domain (e.g., e-commerce, social media, CRM, etc.)
        2. Finding natural graph patterns and hierarchies
        3. Identifying central entities that should be highly connected nodes
        4. Recognizing lookup/reference tables vs core business entities
        5. Understanding data flow and relationships patterns
        """
        )

        human_message = HumanMessage(
            content=f"""
        Analyze this database schema:

        Tables Overview:
        {self._format_tables_for_llm(tables_overview)}

        Relationships:
        {len(database_structure.get('relationships', []))} relationships found

        Please provide:
        1. Business domain identification
        2. Core entities that should be central nodes
        3. Natural graph patterns you observe
        4. Recommended graph modeling approach
        5. Key insights for optimization

        Respond in JSON format with these keys:
        - domain: business domain description
        - core_entities: list of main entity table names
        - graph_patterns: list of observed patterns
        - modeling_approach: recommended approach
        - insights: key insights for graph optimization
        """
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            # Parse LLM response (would need proper JSON parsing in production)
            return {
                "llm_analysis": response.content,
                "tables_count": len(tables_to_analyze),
                "entity_tables_count": len(database_structure.get("entity_tables", {})),
                "relationships_count": len(database_structure.get("relationships", [])),
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Database context analysis failed: %s", e)
            return {
                "llm_analysis": "Context analysis unavailable",
                "tables_count": len(tables_to_analyze),
                "entity_tables_count": len(database_structure.get("entity_tables", {})),
                "relationships_count": len(database_structure.get("relationships", [])),
            }

    def _analyze_table_for_graph_modeling(
        self,
        table_name: str,
        table_info: Dict[str, Any],
        database_structure: Dict[str, Any],
        database_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze a specific table for optimal graph node modeling."""

        # Prepare table analysis context
        schema_details = []

        # Handle both schema formats: entity_tables uses "columns", tables uses "schema"
        if "schema" in table_info:
            # Format from database_analyzer (tables)
            for col in table_info["schema"]:
                schema_details.append(
                    {
                        "name": col["field"],
                        "type": col["type"],
                        "nullable": col["null"] == "YES",
                        "key": col.get("key", ""),
                        "default": col.get("default", ""),
                    }
                )
        elif "columns" in table_info:
            # Format from entity_tables
            for col_name, col_info in table_info["columns"].items():
                schema_details.append(
                    {
                        "name": col_name,
                        "type": col_info["type"],
                        "nullable": col_info.get("nullable", False),
                        "key": col_info.get("key", ""),
                        "default": col_info.get("default", ""),
                    }
                )
        else:
            logger.warning("No schema information found for table %s", table_name)

        foreign_keys = table_info.get("foreign_keys", [])
        row_count = table_info.get("row_count", 0)

        # Find related tables
        related_tables = []
        for rel in database_structure.get("relationships", []):
            if rel["from_table"] == table_name or rel["to_table"] == table_name:
                related_tables.append(rel)

        system_message = SystemMessage(
            content="""
        You are a graph database modeling expert. Analyze this table for optimal 
        graph node representation considering graph database best practices.
        
        Consider:
        1. Which columns should be node properties vs separate nodes
        2. What should be the node label (avoid generic names)
        3. Which properties are good for indexing
        4. What constraints should be applied
        5. How this node fits in the overall graph structure
        """
        )

        human_message = HumanMessage(
            content=f"""
        Analyze table '{table_name}' for graph modeling:

        Database Context: {database_context.get('llm_analysis', 'N/A')}

        Table Details:
        - Row count: {row_count}
        - Columns: {len(schema_details)}
        - Foreign keys: {len(foreign_keys)}

        Schema:
        {self._format_schema_for_llm(schema_details)}

        Foreign Keys:
        {self._format_foreign_keys_for_llm(foreign_keys)}

        Related Relationships:
        {len(related_tables)} relationships involve this table

        Provide graph modeling recommendations in JSON format:
        - node_label: suggested node label (semantic, not just table name)
        - properties: list of columns that should be node properties
        - exclude_properties: columns that shouldn't be properties (with reasons)
        - indexes: recommended property indexes
        - constraints: recommended constraints
        - modeling_rationale: explanation of modeling decisions
        - graph_role: role of this node in the graph (central, lookup, bridge, etc.)
        """
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            return {
                "table_name": table_name,
                "table_info": table_info,
                "llm_analysis": response.content,
                "schema_details": schema_details,
                "foreign_keys": foreign_keys,
                "related_tables": related_tables,
                "row_count": row_count,
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Table analysis failed for %s: %s", table_name, e)
            return {
                "table_name": table_name,
                "table_info": table_info,
                "llm_analysis": "Analysis unavailable",
                "schema_details": schema_details,
                "foreign_keys": foreign_keys,
                "related_tables": related_tables,
                "row_count": row_count,
            }

    def _analyze_relationships_for_graph_modeling(
        self,
        database_structure: Dict[str, Any],
        node_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Analyze relationships for optimal graph edge modeling."""

        relationships = database_structure.get("relationships", [])
        relationship_analyses = []

        # Group relationships by type for batch analysis
        one_to_many_rels = [r for r in relationships if r["type"] == "one_to_many"]
        many_to_many_rels = [r for r in relationships if r["type"] == "many_to_many"]

        # Analyze one-to-many relationships
        if one_to_many_rels:
            analysis = self._analyze_relationship_batch(
                one_to_many_rels, "one_to_many", node_analyses, database_context
            )
            relationship_analyses.extend(analysis)

        # Analyze many-to-many relationships
        if many_to_many_rels:
            analysis = self._analyze_relationship_batch(
                many_to_many_rels, "many_to_many", node_analyses, database_context
            )
            relationship_analyses.extend(analysis)

        return relationship_analyses

    def _analyze_relationship_batch(
        self,
        relationships: List[Dict[str, Any]],
        relationship_type: str,
        node_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of relationships of the same type."""

        system_message = SystemMessage(
            content=f"""
        You are a graph database expert analyzing {relationship_type} relationships 
        for optimal graph edge modeling.
        
        Consider:
        1. Meaningful relationship names (not just table names)
        2. Directionality and semantic meaning
        3. Whether relationship properties should be modeled
        4. Performance implications for graph traversals
        5. Graph query patterns that will be common
        """
        )

        # Prepare relationship context
        rel_context = []
        for rel in relationships:
            from_table = rel["from_table"]
            to_table = rel["to_table"]

            # Find node analysis for context
            from_analysis = next(
                (n for n in node_analyses if n["table_name"] == from_table), None
            )
            to_analysis = next(
                (n for n in node_analyses if n["table_name"] == to_table), None
            )

            rel_context.append(
                {
                    "relationship": rel,
                    "from_table_context": from_analysis.get("llm_analysis", "N/A")
                    if from_analysis
                    else "N/A",
                    "to_table_context": to_analysis.get("llm_analysis", "N/A")
                    if to_analysis
                    else "N/A",
                }
            )

        human_message = HumanMessage(
            content=f"""
        Analyze these {relationship_type} relationships for graph modeling:

        Database Context: {database_context.get('llm_analysis', 'N/A')}

        Relationships to analyze:
        {self._format_relationships_for_llm(rel_context)}

        For each relationship, provide recommendations in JSON format:
        - relationship_name: semantic relationship name
        - directionality: directed/undirected with rationale
        - properties: any relationship properties to model
        - performance_notes: optimization considerations
        - modeling_rationale: explanation of decisions
        """
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            return [
                {
                    "relationship_type": relationship_type,
                    "relationships": relationships,
                    "llm_analysis": response.content,
                    "context": rel_context,
                }
            ]
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Relationship analysis failed for %s: %s", relationship_type, e
            )
            return [
                {
                    "relationship_type": relationship_type,
                    "relationships": relationships,
                    "llm_analysis": "Analysis unavailable",
                    "context": rel_context,
                }
            ]

    def _generate_comprehensive_graph_model(
        self,
        node_analyses: List[Dict[str, Any]],
        relationship_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> GraphModel:
        """Generate the final comprehensive graph model."""

        # Generate nodes from analyses
        nodes = []
        for analysis in node_analyses:
            node = GraphNode(
                name=analysis["table_name"],
                label=self._extract_node_label(analysis),
                properties=self._extract_node_properties(analysis),
                primary_key=self._extract_primary_key(analysis),
                indexes=self._extract_indexes(analysis),
                constraints=self._extract_constraints(analysis),
                source_table=analysis["table_name"],
                modeling_rationale=analysis.get("llm_analysis", ""),
            )
            nodes.append(node)

        # Generate relationships from analyses
        relationships = []
        for analysis in relationship_analyses:
            for rel_data in analysis["relationships"]:
                relationship = GraphRelationship(
                    name=self._extract_relationship_name(rel_data, analysis),
                    type=rel_data["type"],
                    from_node=rel_data["from_table"],
                    to_node=rel_data["to_table"],
                    properties=self._extract_relationship_properties(
                        rel_data, analysis
                    ),
                    directionality="directed",  # Default, could be extracted from LLM
                    source_info=rel_data,
                    modeling_rationale=analysis.get("llm_analysis", ""),
                )
                relationships.append(relationship)

        # Generate overall modeling decisions and suggestions
        modeling_decisions = self._extract_modeling_decisions(
            node_analyses, relationship_analyses, database_context
        )

        optimization_suggestions = self._extract_optimization_suggestions(
            node_analyses, relationship_analyses, database_context
        )

        return GraphModel(
            nodes=nodes,
            relationships=relationships,
            modeling_decisions=modeling_decisions,
            optimization_suggestions=optimization_suggestions,
            data_patterns=database_context,
        )

    # Helper methods for formatting data for LLM
    def _format_tables_for_llm(self, tables_overview: List[Dict]) -> str:
        """Format tables overview for LLM consumption."""
        formatted = []
        for table in tables_overview:
            formatted.append(
                f"- {table['name']}: {len(table['columns'])} columns, "
                f"{table['foreign_keys']} FKs, {table['row_count']} rows, "
                f"type: {table['type']}"
            )
        return "\n".join(formatted)

    def _format_schema_for_llm(self, schema_details: List[Dict]) -> str:
        """Format table schema for LLM consumption."""
        formatted = []
        for col in schema_details:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            key_info = f" ({col['key']})" if col["key"] else ""
            formatted.append(f"- {col['name']}: {col['type']} {nullable}{key_info}")
        return "\n".join(formatted)

    def _format_foreign_keys_for_llm(self, foreign_keys: List[Dict]) -> str:
        """Format foreign keys for LLM consumption."""
        if not foreign_keys:
            return "None"

        formatted = []
        for fk in foreign_keys:
            formatted.append(
                f"- {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}"
            )
        return "\n".join(formatted)

    def _format_relationships_for_llm(self, rel_context: List[Dict]) -> str:
        """Format relationships context for LLM consumption."""
        formatted = []
        for ctx in rel_context:
            rel = ctx["relationship"]
            if rel["type"] == "many_to_many":
                formatted.append(
                    f"- {rel['from_table']} <--> {rel['to_table']} "
                    f"(via {rel['join_table']})"
                )
            else:
                formatted.append(f"- {rel['from_table']} -> {rel['to_table']}")
        return "\n".join(formatted)

    # Helper methods for extracting information from LLM responses
    def _extract_node_label(self, analysis: Dict[str, Any]) -> str:
        """Extract node label from analysis (with fallback)."""
        # In a real implementation, this would parse the JSON response
        # For now, return a sensible default
        table_name = analysis["table_name"]
        return table_name.replace("_", "").title()

    def _extract_node_properties(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract node properties from analysis."""
        # Fallback: exclude obvious foreign keys and system columns
        properties = []
        for col in analysis["schema_details"]:
            if (
                not col["name"].endswith("_id")
                or col["key"] == "PRI"
                and col["name"] not in ["created_at", "updated_at", "deleted_at"]
            ):
                properties.append(col["name"])
        return properties

    def _extract_primary_key(self, analysis: Dict[str, Any]) -> str:
        """Extract primary key from analysis."""
        for col in analysis["schema_details"]:
            if col["key"] == "PRI":
                return col["name"]
        return "id"  # fallback

    def _extract_indexes(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract recommended indexes from analysis."""
        indexes = []
        for col in analysis["schema_details"]:
            if col["key"] in ["UNI", "MUL"] or col["name"] in [
                "email",
                "username",
                "name",
            ]:
                indexes.append(col["name"])
        return indexes

    def _extract_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract recommended constraints from analysis."""
        constraints = []
        for col in analysis["schema_details"]:
            if col["key"] == "PRI":
                constraints.append(f"UNIQUE({col['name']})")
        return constraints

    def _extract_relationship_name(
        self, rel_data: Dict[str, Any], analysis: Dict[str, Any]
    ) -> str:
        # pylint: disable=unused-argument
        """Extract relationship name from analysis."""
        # Simple fallback logic
        if rel_data["type"] == "many_to_many":
            join_table = rel_data.get("join_table", "")
            return join_table.upper().replace("_", "_")
        else:
            to_table = rel_data["to_table"]
            return f"HAS_{to_table.upper()}"

    def _extract_relationship_properties(
        self, rel_data: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[str]:
        # pylint: disable=unused-argument
        """Extract relationship properties from analysis."""
        if rel_data["type"] == "many_to_many":
            return rel_data.get("additional_properties", [])
        return []

    def _extract_modeling_decisions(
        self,
        node_analyses: List[Dict[str, Any]],
        relationship_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[str]:
        # pylint: disable=unused-argument
        """Extract key modeling decisions from all analyses."""
        decisions = [
            f"Identified {len(node_analyses)} entity nodes for the graph",
            (
                "Configured "
                f"{sum(len(ra['relationships']) for ra in relationship_analyses)}"
                " relationships"
            ),
            "Applied semantic labeling based on business domain analysis",
            "Optimized property selection for graph traversal performance",
        ]
        return decisions

    def _extract_optimization_suggestions(
        self,
        node_analyses: List[Dict[str, Any]],
        relationship_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[str]:
        # pylint: disable=unused-argument
        """Extract optimization suggestions from analyses."""
        suggestions = [
            (
                "Consider adding graph-specific indexes for frequently "
                "queried properties"
            ),
            "Monitor relationship cardinality for performance optimization",
            "Implement caching for high-degree nodes",
            "Consider partitioning strategies for large datasets",
        ]
        return suggestions

    def _analyze_all_tables_and_relationships(
        self,
        database_structure: Dict[str, Any],
        database_context: Dict[str, Any],
        domain_context: str,
    ) -> Dict[str, Any]:
        """
        Analyze all tables and relationships in a single comprehensive LLM call.
        Includes sample data from each table for better understanding.
        """
        logger.info("Preparing comprehensive database analysis for single LLM call...")

        # Prepare all table information with sample data
        tables_info = []
        for table_name, table_info in database_structure["entity_tables"].items():
            table_analysis = self._prepare_table_info_with_sample_data(
                table_name, table_info, database_structure
            )
            tables_info.append(table_analysis)

        # Prepare relationship information
        relationships_info = database_structure.get("relationships", [])

        system_message = SystemMessage(
            content="""
You are an expert graph database architect specializing in translating 
relational database schemas to optimal graph models. You will analyze 
a complete database schema with sample data and create a comprehensive 
graph model in a single analysis.

Consider:
1. Semantic node labels that reflect business entities
2. Property selection based on access patterns from sample data
3. Relationship types that reflect business logic
4. Performance optimizations for graph traversals
5. Data patterns visible in the sample data
6. Index and constraint recommendations

Provide a comprehensive JSON response with the complete graph model.
"""
        )

        human_message = HumanMessage(
            content=f"""
Analyze this complete database schema for optimal graph modeling:

DOMAIN CONTEXT: {domain_context}

DATABASE OVERVIEW: {database_context.get('llm_analysis', 'N/A')}

TABLES WITH SAMPLE DATA:
{self._format_tables_with_sample_data(tables_info)}

RELATIONSHIPS:
{self._format_relationships_for_comprehensive_analysis(relationships_info)}

Create a complete graph model with the following JSON structure:
{{
    "nodes": [
        {{
            "name": "semantic_node_name",
            "label": "NodeLabel", 
            "properties": ["prop1", "prop2"],
            "primary_key": "id_field",
            "indexes": ["indexed_prop"],
            "constraints": ["unique_prop"],
            "source_table": "table_name",
            "modeling_rationale": "explanation"
        }}
    ],
    "relationships": [
        {{
            "name": "semantic_relationship_name",
            "type": "RELATIONSHIP_TYPE",
            "from_node": "source_node",
            "to_node": "target_node", 
            "properties": ["rel_prop"],
            "directionality": "directed|undirected",
            "modeling_rationale": "explanation"
        }}
    ],
    "modeling_decisions": ["decision1", "decision2"],
    "optimization_suggestions": ["suggestion1", "suggestion2"],
    "data_patterns": {{
        "identified_patterns": ["pattern1", "pattern2"],
        "recommendations": ["rec1", "rec2"]
    }}
}}
"""
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            return {
                "llm_analysis": response.content,
                "tables_info": tables_info,
                "relationships_info": relationships_info,
                "domain_context": domain_context,
            }
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "llm_analysis": "Comprehensive analysis unavailable",
                "tables_info": tables_info,
                "relationships_info": relationships_info,
                "domain_context": domain_context,
            }

    def _prepare_table_info_with_sample_data(
        self,
        table_name: str,
        table_info: Dict[str, Any],
        database_structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare table information including sample data for analysis."""

        # Extract schema information
        schema_details = []
        if "schema" in table_info:
            for col in table_info["schema"]:
                schema_details.append(
                    {
                        "name": col["field"],
                        "type": col["type"],
                        "nullable": col["null"] == "YES",
                        "key": col.get("key", ""),
                        "default": col.get("default", ""),
                    }
                )
        elif "columns" in table_info:
            for col_name, col_info in table_info["columns"].items():
                schema_details.append(
                    {
                        "name": col_name,
                        "type": col_info["type"],
                        "nullable": col_info.get("nullable", False),
                        "key": col_info.get("key", ""),
                        "default": col_info.get("default", ""),
                    }
                )

        # Get sample data from database_structure
        sample_data = database_structure.get("sample_data", {}).get(table_name, [])
        if not sample_data:
            sample_data = {"note": "No sample data available"}

        return {
            "table_name": table_name,
            "schema": schema_details,
            "foreign_keys": table_info.get("foreign_keys", []),
            "row_count": table_info.get("row_count", 0),
            "sample_data": sample_data,
            "table_type": table_info.get("type", "entity"),
        }

    def _format_tables_with_sample_data(self, tables_info: List[Dict[str, Any]]) -> str:
        """Format tables with sample data for LLM consumption."""
        formatted = []
        for table in tables_info:
            table_text = f"""
TABLE: {table['table_name']}
- Row count: {table['row_count']}
- Schema:"""
            for col in table["schema"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                key_info = f" ({col['key']})" if col["key"] else ""
                table_text += f"\n  * {col['name']}: {col['type']} {nullable}{key_info}"

            if table["foreign_keys"]:
                table_text += "\n- Foreign Keys:"
                for fk in table["foreign_keys"]:
                    table_text += f"\n  * {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}"

            # Format sample data nicely
            sample_data = table["sample_data"]
            if isinstance(sample_data, list) and sample_data:
                table_text += "\n- Sample Data (first few rows):"
                for i, row in enumerate(sample_data[:3], 1):  # Limit to 3 rows
                    table_text += (
                        f"\n  Row {i}: {dict(row) if hasattr(row, 'keys') else row}"
                    )
            else:
                table_text += f"\n- Sample Data: {sample_data}"

            formatted.append(table_text)

        return "\n".join(formatted)

    def _format_relationships_for_comprehensive_analysis(
        self, relationships: List[Dict[str, Any]]
    ) -> str:
        """Format relationships for comprehensive analysis."""
        formatted = []
        for rel in relationships:
            rel_type = rel.get("type", "unknown")
            if rel_type == "many_to_many":
                formatted.append(
                    f"- {rel['from_table']} <--[{rel['join_table']}]--> {rel['to_table']} "
                    f"(many-to-many via {rel['join_table']})"
                )
            else:
                formatted.append(
                    f"- {rel['from_table']}.{rel['from_column']} -> "
                    f"{rel['to_table']}.{rel['to_column']} ({rel_type})"
                )
        return "\n".join(formatted)

    def _generate_graph_model_from_analysis(
        self,
        comprehensive_analysis: Dict[str, Any],
        database_structure: Dict[str, Any],
        database_context: Dict[str, Any],
    ) -> GraphModel:
        """Generate GraphModel from comprehensive analysis results."""

        try:
            # Parse the LLM response - this is a simplified implementation
            # In practice, you'd want more robust JSON parsing

            # For now, create a simplified model based on the database structure
            # This should be enhanced to parse the actual LLM JSON response
            nodes = []
            relationships = []

            # Create nodes from entity tables
            for table_name, table_info in database_structure["entity_tables"].items():
                schema_cols = []
                if "schema" in table_info:
                    schema_cols = [col["field"] for col in table_info["schema"]]
                elif "columns" in table_info:
                    schema_cols = list(table_info["columns"].keys())

                node = GraphNode(
                    name=table_name,
                    label=table_name.title(),
                    properties=schema_cols,
                    primary_key=self._find_primary_key(table_info),
                    indexes=[],
                    constraints=[],
                    source_table=table_name,
                    modeling_rationale="Generated from comprehensive analysis",
                )
                nodes.append(node)

            # Create relationships
            for rel in database_structure.get("relationships", []):
                relationship = GraphRelationship(
                    name=f"{rel['from_table']}_to_{rel['to_table']}",
                    type=rel["type"],
                    from_node=rel["from_table"],
                    to_node=rel["to_table"],
                    properties=[],
                    directionality="directed",
                    source_info=rel,
                    modeling_rationale="Generated from comprehensive analysis",
                )
                relationships.append(relationship)

            modeling_decisions = [
                "Used single LLM call for comprehensive analysis",
                "Included sample data for better context",
                f"Analyzed {len(nodes)} entities and {len(relationships)} relationships",
            ]

            optimization_suggestions = [
                "Consider adding indexes for frequently queried properties",
                "Monitor performance for high-cardinality relationships",
                "Implement data validation based on sample patterns",
            ]

            return GraphModel(
                nodes=nodes,
                relationships=relationships,
                modeling_decisions=modeling_decisions,
                optimization_suggestions=optimization_suggestions,
                data_patterns=comprehensive_analysis.get("data_patterns", {}),
            )

        except Exception as e:
            logger.error(f"Failed to generate graph model from analysis: {e}")
            # Return a basic model as fallback
            return GraphModel(
                nodes=[],
                relationships=[],
                modeling_decisions=["Analysis failed - using fallback model"],
                optimization_suggestions=[],
                data_patterns={},
            )

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
