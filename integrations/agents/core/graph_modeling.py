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
    modeling_rationale: str = Field(description="Explanation for modeling decisions")

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
    modeling_rationale: str = Field(description="Explanation for this relationship")

    class Config:
        """Pydantic config for OpenAI structured output compatibility."""

        extra = "forbid"


class LLMGraphModel(BaseModel):
    """Complete LLM-generated graph model."""

    nodes: List[LLMGraphNode] = Field(description="All nodes in the graph model")
    relationships: List[LLMGraphRelationship] = Field(
        description="All relationships in the graph model"
    )
    modeling_decisions: List[str] = Field(
        description="High-level modeling decisions and rationale"
    )
    optimization_suggestions: List[str] = Field(
        description="Suggestions for query optimization and indexing"
    )
    domain_insights: List[str] = Field(
        description="Domain-specific insights discovered during modeling", default=[]
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
            node = GraphNode(
                name=llm_node.name,
                label=llm_node.label,
                properties=llm_node.properties,
                primary_key=llm_node.primary_key,
                indexes=llm_node.indexes,
                constraints=llm_node.constraints,
                source_table=llm_node.source_table,
                modeling_rationale=llm_node.modeling_rationale,
            )
            nodes.append(node)

        # Convert relationships
        relationships = []
        for llm_rel in llm_model.relationships:
            # Find source info from database structure
            source_info = self._find_relationship_source_info(
                llm_rel, database_structure
            )

            relationship = GraphRelationship(
                name=llm_rel.name,
                type=llm_rel.type,
                from_node=llm_rel.from_node,
                to_node=llm_rel.to_node,
                properties=llm_rel.properties,
                directionality=llm_rel.directionality,
                source_info=source_info,
                modeling_rationale=llm_rel.modeling_rationale,
            )
            relationships.append(relationship)

        return GraphModel(
            nodes=nodes,
            relationships=relationships,
            modeling_decisions=llm_model.modeling_decisions,
            optimization_suggestions=llm_model.optimization_suggestions,
            data_patterns={"domain_insights": llm_model.domain_insights},
        )

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

        # Debug: Log when relationship is not found
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
                        f"Node {node.label} has invalid properties: {invalid_props}"
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
        node_labels = {node.label.lower() for node in graph_model.nodes}
        for rel in graph_model.relationships:
            if rel.from_node not in node_labels:
                # Try to find by source table
                found = False
                for node in graph_model.nodes:
                    if node.source_table.lower() == rel.from_node:
                        found = True
                        break
                if not found:
                    issues.append(
                        f"Relationship {rel.name} references unknown from_node: "
                        f"{rel.from_node}"
                    )

            if rel.to_node not in node_labels:
                # Try to find by source table
                found = False
                for node in graph_model.nodes:
                    if node.source_table.lower() == rel.to_node:
                        found = True
                        break
                if not found:
                    issues.append(
                        f"Relationship {rel.name} references unknown to_node: "
                        f"{rel.to_node}"
                    )

        # 5. Check for graph modeling best practices
        for node in graph_model.nodes:
            if node.label.lower() == node.source_table.lower():
                warnings.append(
                    f"Node label '{node.label}' is same as table name, "
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
                        f"Renaming property in {node_label}: {old_prop} -> {new_prop}"
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

    def _get_model_presentation(self) -> str:
        """Get formatted presentation of current model for user review."""
        if not self.current_graph_model:
            return "No model available"

        presentation = []
        presentation.append("NODES:")
        for i, node in enumerate(self.current_graph_model.nodes, 1):
            presentation.append(f"{i}. {node.label} (from: {node.source_table})")
            presentation.append(f"   Properties: {', '.join(node.properties)}")
            presentation.append(f"   Primary Key: {node.primary_key}")
            if node.indexes:
                presentation.append(f"   Indexes: {', '.join(node.indexes)}")
            presentation.append("")

        presentation.append("RELATIONSHIPS:")
        for i, rel in enumerate(self.current_graph_model.relationships, 1):
            direction = "->" if rel.directionality == "directed" else "<->"
            presentation.append(
                f"{i}. {rel.from_node} {direction} [{rel.name}] {direction} {rel.to_node}"
            )
            presentation.append(f"   Type: {rel.type}")
            if rel.properties:
                presentation.append(f"   Properties: {', '.join(rel.properties)}")
            presentation.append("")

        return "\n".join(presentation)

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

        try:
            # Create simplified model based on database structure
            nodes = []
            relationships = []

            # Create nodes from entity tables
            for table_name, table_info in database_structure["entity_tables"].items():
                node = GraphNode(
                    name=table_name,
                    label=table_name.replace("_", "").title(),
                    properties=self._extract_node_properties_from_table(table_info),
                    primary_key=self._find_primary_key(table_info),
                    indexes=self._extract_indexes_from_table(table_info),
                    constraints=self._extract_constraints_from_table(table_info),
                    source_table=table_name,
                    modeling_rationale="Generated from schema analysis",
                )
                nodes.append(node)

            # Create relationships
            for rel in database_structure.get("relationships", []):
                relationship = GraphRelationship(
                    name=self._generate_relationship_name(rel),
                    type=rel.get("type", "one_to_many"),
                    from_node=rel["from_table"].lower(),
                    to_node=rel["to_table"].lower(),
                    properties=[],
                    directionality="directed",
                    source_info=rel,
                    modeling_rationale="Generated from schema analysis",
                )
                relationships.append(relationship)

            modeling_decisions = [
                f"Analyzed {len(nodes)} entities and {len(relationships)} relationships",
                "Applied semantic labeling based on table names",
                "Configured relationships based on foreign key analysis",
            ]

            optimization_suggestions = [
                "Consider adding indexes for frequently queried properties",
                "Review node labels for better semantic meaning",
                "Validate relationship directions based on business logic",
            ]

            return GraphModel(
                nodes=nodes,
                relationships=relationships,
                modeling_decisions=modeling_decisions,
                optimization_suggestions=optimization_suggestions,
                data_patterns={},
            )

        except Exception as e:
            logger.error(f"Failed to generate initial graph model: {e}")
            return GraphModel(
                nodes=[],
                relationships=[],
                modeling_decisions=["Model generation failed"],
                optimization_suggestions=[],
                data_patterns={},
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
