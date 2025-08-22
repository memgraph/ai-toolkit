"""
Main HyGM (Hypothetical Graph Modeling) class.

This is the primary interface for the modular HyGM system.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models.graph_models import GraphModel
    from .models.operations import ModelModifications

try:
    from .strategies import (
        BaseModelingStrategy,
        DeterministicStrategy,
        LLMStrategy,
    )
    from .validation import GraphSchemaValidator
except ImportError:
    from core.hygm.strategies import (
        BaseModelingStrategy,
        DeterministicStrategy,
        LLMStrategy,
    )
    from core.hygm.validation import GraphSchemaValidator

logger = logging.getLogger(__name__)


class ModelingMode(Enum):
    """Modeling modes for HyGM."""

    AUTOMATIC = "automatic"
    INTERACTIVE = "interactive"


class GraphModelingStrategy(Enum):
    """Graph modeling strategies available."""

    DETERMINISTIC = "deterministic"  # Rule-based graph creation
    LLM_POWERED = "llm_powered"  # LLM generates the graph model


class HyGM:
    """
    Main Hypothetical Graph Modeling class.

    Uses different strategies to create intelligent graph models from
    relational schemas. Supports both automatic and interactive modes.
    """

    def __init__(
        self,
        llm=None,
        mode: ModelingMode = ModelingMode.AUTOMATIC,
        strategy: GraphModelingStrategy = GraphModelingStrategy.DETERMINISTIC,
    ):
        """
        Initialize HyGM with modeling configuration.

        Args:
            llm: LLM instance for AI-powered modeling (optional)
            mode: AUTOMATIC or INTERACTIVE modeling mode
            strategy: DETERMINISTIC or LLM_POWERED strategy
        """
        self.llm = llm
        self.mode = mode
        self.strategy = strategy
        self.current_graph_model = None
        self.iteration_count = 0
        self.database_structure = None
        self._strategy_cache = {}

    def create_graph_model(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        strategy: Optional[GraphModelingStrategy] = None,
    ) -> "GraphModel":
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
        self.database_structure = database_structure

        logger.info("Creating graph model using %s strategy...", used_strategy.value)

        # Check if interactive mode is enabled
        if self.mode == ModelingMode.INTERACTIVE:
            return self._interactive_modeling(
                database_structure, domain_context, used_strategy
            )

        # For automatic mode, use the specified strategy
        strategy_instance = self._get_strategy_instance(used_strategy)
        graph_model = strategy_instance.create_model(database_structure, domain_context)

        # Store the created model as current
        self.current_graph_model = graph_model
        return graph_model

    def _get_strategy_instance(
        self, strategy: GraphModelingStrategy
    ) -> BaseModelingStrategy:
        """Get or create a strategy instance."""
        if strategy not in self._strategy_cache:
            if strategy == GraphModelingStrategy.LLM_POWERED:
                self._strategy_cache[strategy] = LLMStrategy(
                    llm_client=self.llm, model_name="gpt-4", temperature=0.1
                )
            elif strategy == GraphModelingStrategy.DETERMINISTIC:
                self._strategy_cache[strategy] = DeterministicStrategy()
            else:
                msg = f"Unknown strategy: {strategy}"
                raise ValueError(msg)

        return self._strategy_cache[strategy]

    def _interactive_modeling(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        strategy: GraphModelingStrategy = GraphModelingStrategy.DETERMINISTIC,
    ) -> "GraphModel":
        """
        Interactive modeling process with user feedback.

        This method provides an interactive terminal interface for users
        to iteratively refine the graph model.
        """
        logger.info("Starting interactive modeling session...")

        # Create initial model using the specified strategy
        strategy_instance = self._get_strategy_instance(strategy)
        current_model = strategy_instance.create_model(
            database_structure, domain_context
        )
        self.current_graph_model = current_model
        self.iteration_count = 1

        # Interactive refinement loop
        while True:
            self._display_current_model(current_model)

            user_choice = self._get_user_choice()

            if user_choice == "accept":
                logger.info("Model accepted by user")
                break
            elif user_choice == "modify":
                logger.info("Starting interactive model modification...")
                current_model = self._modify_model_interactively(
                    current_model, strategy
                )
                self.iteration_count += 1
            elif user_choice == "regenerate":
                logger.info("Regenerating model with same strategy...")
                # Regenerate with the same strategy
                current_model = strategy_instance.create_model(
                    database_structure, domain_context
                )
                self.iteration_count += 1
            elif user_choice == "switch_strategy":
                logger.info("Switching modeling strategy...")
                # Switch between strategies
                new_strategy = self._switch_strategy(strategy)
                if new_strategy != strategy:
                    strategy_instance = self._get_strategy_instance(new_strategy)
                    current_model = strategy_instance.create_model(
                        database_structure, domain_context
                    )
                    strategy = new_strategy
                    self.iteration_count += 1

        self.current_graph_model = current_model
        return current_model

    def _display_current_model(self, model: "GraphModel") -> None:
        """Display the current graph model to the user."""
        print("\n" + "=" * 60)
        print(f"GRAPH MODEL - ITERATION {self.iteration_count}")
        print("=" * 60)

        print(f"\nNODES ({len(model.nodes)}):")
        for i, node in enumerate(model.nodes, 1):
            print(f"  {i}. {' | '.join(node.labels)}")
            print(f"     Properties: {[p.key for p in node.properties]}")

        print(f"\nRELATIONSHIPS ({len(model.edges)}):")
        for i, edge in enumerate(model.edges, 1):
            start_labels = " | ".join(edge.start_node_labels)
            end_labels = " | ".join(edge.end_node_labels)
            edge_display = f"({start_labels})-[:{edge.edge_type}]->({end_labels})"
            print(f"  {i}. {edge_display}")

        print(f"\nINDEXES ({len(model.node_indexes)}):")
        for i, index in enumerate(model.node_indexes, 1):
            labels = " | ".join(index.labels or [])
            props = ", ".join(index.properties)
            print(f"  {i}. {labels}.{props}")

        print(f"\nCONSTRAINTS ({len(model.node_constraints)}):")
        for i, constraint in enumerate(model.node_constraints, 1):
            labels = " | ".join(constraint.labels or [])
            props = ", ".join(constraint.properties)
            print(f"  {i}. {constraint.type.upper()}: {labels}.{props}")

    def _get_user_choice(self) -> str:
        """Get user choice for next action."""
        print("\nWhat would you like to do?")
        print("1. Accept this model")
        print("2. Modify the model")
        print("3. Regenerate model (same strategy)")
        print("4. Switch modeling strategy")

        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                if choice == "1":
                    return "accept"
                elif choice == "2":
                    return "modify"
                elif choice == "3":
                    return "regenerate"
                elif choice == "4":
                    return "switch_strategy"
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive modeling...")
                return "accept"

    def _switch_strategy(
        self, current_strategy: GraphModelingStrategy
    ) -> GraphModelingStrategy:
        """Allow user to switch between modeling strategies."""
        print("\nAvailable strategies:")
        print("1. Deterministic (rule-based)")
        print("2. LLM-powered (AI-generated)")

        if current_strategy == GraphModelingStrategy.DETERMINISTIC:
            current_name = "Deterministic"
        else:
            current_name = "LLM-powered"
        print(f"\nCurrent strategy: {current_name}")

        while True:
            try:
                choice = input("Choose new strategy (1-2): ").strip()
                if choice == "1":
                    return GraphModelingStrategy.DETERMINISTIC
                elif choice == "2":
                    if self.llm is None:
                        print("LLM not available. Please choose option 1.")
                        continue
                    return GraphModelingStrategy.LLM_POWERED
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except (EOFError, KeyboardInterrupt):
                print(f"\nKeeping current strategy: {current_name}")
                return current_strategy

    def _modify_model_interactively(
        self, model: "GraphModel", strategy: GraphModelingStrategy
    ) -> "GraphModel":
        """Allow user to modify the model using natural language commands."""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODEL MODIFICATION")
        print("=" * 60)
        print("\nDescribe the changes you'd like to make to the graph model.")
        print("You can use natural language like:")
        print("  - 'Rename the Actor label to Person'")
        print("  - 'Add a birth_date property to Actor nodes'")
        print("  - 'Remove the last_update property from all nodes'")
        print("  - 'Change ACTED_IN relationship to PERFORMED_IN'")
        print("  - 'Add an index on Customer email property'")
        print("\nType 'done' when finished, or 'cancel' to return unchanged.")

        while True:
            try:
                print("\n" + "-" * 60)
                user_input = input("Describe your change: ").strip()

                if user_input.lower() == "done":
                    print("Applying changes to model...")
                    break
                elif user_input.lower() == "cancel":
                    print("Cancelling changes...")
                    return model
                elif not user_input:
                    print("Please describe the change you'd like to make.")
                    continue

                # Use LLM to parse natural language into operations
                if self.llm:
                    operations = self._parse_natural_language_to_operations(
                        user_input, model
                    )
                    if operations:
                        print(f"✅ Understood: {operations.reasoning}")
                        # Apply operations to model
                        model = self._apply_operations_to_model(model, operations)
                        print("Changes applied!")

                        # Show the updated model after changes
                        print("\nApplying changes to model...")
                        self.iteration_count += 1
                        self._display_current_model(model)

                        # Perform validation after applying changes
                        model = self._perform_post_operation_validation(
                            model, strategy, operations
                        )
                    else:
                        print(
                            "❌ I didn't understand that command. " "Please try again."
                        )
                else:
                    print("❌ LLM not available for natural language processing.")
                    print("Please use the basic modification menu instead.")
                    break

            except (EOFError, KeyboardInterrupt):
                print("\nCancelling changes...")
                return model

        return model

    def _parse_natural_language_to_operations(
        self, user_input: str, model: "GraphModel"
    ) -> Optional["ModelModifications"]:
        """Parse natural language input into structured operations."""
        if not self.llm:
            return None

        # Get current model structure for context
        model_context = self._get_model_context_for_llm(model)

        system_prompt = (
            "You are an expert at translating natural language instructions "
            "into structured graph model operations.\n\n"
            f"Current graph model structure:\n{model_context}\n\n"
            "Available operations:\n"
            "- change_node_label: Change a node's label\n"
            "- rename_property: Rename a property on a node\n"
            "- drop_property: Remove a property from a node\n"
            "- add_property: Add a new property to a node\n"
            "- change_relationship_name: Change a relationship name\n"
            "- drop_relationship: Remove a relationship\n"
            "- add_index: Add an index on a property\n"
            "- drop_index: Remove an index\n\n"
            "Parse the user's request into appropriate operations. "
            "Return a ModelModifications object with the operations and "
            "reasoning."
        )

        try:
            from langchain_core.output_parsers import PydanticOutputParser

            # Import at runtime to avoid circular imports
            try:
                from .models.operations import ModelModifications
            except ImportError:
                from core.hygm.models.operations import ModelModifications

            parser = PydanticOutputParser(pydantic_object=ModelModifications)

            prompt = f"""
            User request: {user_input}
            
            {parser.get_format_instructions()}
            """

            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            # Parse the response
            operations = parser.parse(response.content)
            return operations

        except (ImportError, ValueError, AttributeError) as e:
            logger.error("Error parsing natural language: %s", e)
            return None
        except Exception as e:  # noqa: BLE001 - Catch-all for LLM errors
            logger.error("Unexpected error in natural language parsing: %s", e)
            return None

    def _get_model_context_for_llm(self, model: "GraphModel") -> str:
        """Get a text description of the current model for LLM context."""
        context_parts = []

        # Nodes
        context_parts.append("NODES:")
        for node in model.nodes:
            props = [p.key for p in node.properties]
            context_parts.append(f"  - {node.primary_label}: {props}")

        # Relationships
        context_parts.append("\nRELATIONSHIPS:")
        for edge in model.edges:
            start = " | ".join(edge.start_node_labels)
            end = " | ".join(edge.end_node_labels)
            context_parts.append(f"  - ({start})-[:{edge.edge_type}]->({end})")

        # Indexes
        if model.node_indexes:
            context_parts.append("\nINDEXES:")
            for index in model.node_indexes:
                labels = " | ".join(index.labels or [])
                props = ", ".join(index.properties)
                context_parts.append(f"  - {labels}.{props}")

        return "\n".join(context_parts)

    def _apply_operations_to_model(
        self, model: "GraphModel", operations: "ModelModifications"
    ) -> "GraphModel":
        """Apply structured operations to the graph model."""
        # Create a deep copy of the model to modify
        import copy

        modified_model = copy.deepcopy(model)

        print(f"\nApplying {len(operations.operations)} operations:")

        for op in operations.operations:
            if isinstance(op, type(op)) and hasattr(op, "operation_type"):
                if op.operation_type == "change_node_label":
                    print(f"  - Change node label: {op.old_label} → {op.new_label}")
                    modified_model = self._apply_change_node_label(
                        modified_model, op.old_label, op.new_label
                    )
                elif op.operation_type == "rename_property":
                    print(
                        f"  - Rename property on {op.node_label}: "
                        f"{op.old_property} → {op.new_property}"
                    )
                    modified_model = self._apply_rename_property(
                        modified_model, op.node_label, op.old_property, op.new_property
                    )
                elif op.operation_type == "drop_property":
                    print(f"  - Drop property: {op.node_label}.{op.property_name}")
                    modified_model = self._apply_drop_property(
                        modified_model, op.node_label, op.property_name
                    )
                elif op.operation_type == "add_property":
                    print(f"  - Add property: {op.node_label}.{op.property_name}")
                    modified_model = self._apply_add_property(
                        modified_model, op.node_label, op.property_name
                    )
                elif op.operation_type == "change_relationship_name":
                    print(f"  - Change relationship: {op.old_name} → {op.new_name}")
                    modified_model = self._apply_change_relationship_name(
                        modified_model, op.old_name, op.new_name
                    )
                elif op.operation_type == "drop_relationship":
                    print(f"  - Drop relationship: {op.relationship_name}")
                    modified_model = self._apply_drop_relationship(
                        modified_model, op.relationship_name
                    )
                elif op.operation_type == "add_index":
                    print(f"  - Add index: {op.node_label}.{op.property_name}")
                    modified_model = self._apply_add_index(
                        modified_model, op.node_label, op.property_name
                    )
                elif op.operation_type == "drop_index":
                    print(f"  - Drop index: {op.node_label}.{op.property_name}")
                    modified_model = self._apply_drop_index(
                        modified_model, op.node_label, op.property_name
                    )

        return modified_model

    def _apply_change_node_label(
        self, model: "GraphModel", old_label: str, new_label: str
    ) -> "GraphModel":
        """Apply change node label operation."""
        for node in model.nodes:
            if old_label in node.labels:
                # Update labels
                node.labels = [
                    new_label if label == old_label else label for label in node.labels
                ]
                # Update source mapping if it exists
                if node.source and "labels" in node.source.mapping:
                    node.source.mapping["labels"] = [
                        new_label if label == old_label else label
                        for label in node.source.mapping["labels"]
                    ]

        # Update relationships that reference this label
        for edge in model.edges:
            edge.start_node_labels = [
                new_label if label == old_label else label
                for label in edge.start_node_labels
            ]
            edge.end_node_labels = [
                new_label if label == old_label else label
                for label in edge.end_node_labels
            ]

        # Update indexes
        for index in model.node_indexes:
            if index.labels:
                index.labels = [
                    new_label if label == old_label else label for label in index.labels
                ]

        # Update constraints
        for constraint in model.node_constraints:
            if constraint.labels:
                constraint.labels = [
                    new_label if label == old_label else label
                    for label in constraint.labels
                ]

        return model

    def _apply_rename_property(
        self, model: "GraphModel", node_label: str, old_property: str, new_property: str
    ) -> "GraphModel":
        """Apply rename property operation."""
        for node in model.nodes:
            if node_label in node.labels:
                for prop in node.properties:
                    if prop.key == old_property:
                        prop.key = new_property
                        # Update source field reference
                        if prop.source:
                            old_field = prop.source.field
                            table_name = old_field.split(".")[0]
                            prop.source.field = f"{table_name}.{new_property}"
        return model

    def _apply_drop_property(
        self, model: "GraphModel", node_label: str, property_name: str
    ) -> "GraphModel":
        """Apply drop property operation."""
        for node in model.nodes:
            if node_label in node.labels:
                node.properties = [
                    prop for prop in node.properties if prop.key != property_name
                ]
        return model

    def _apply_add_property(
        self, model: "GraphModel", node_label: str, property_name: str
    ) -> "GraphModel":
        """Apply add property operation."""
        from .models.graph_models import GraphProperty
        from .models.sources import PropertySource

        for node in model.nodes:
            if node_label in node.labels:
                # Check if property already exists
                existing_props = [prop.key for prop in node.properties]
                if property_name not in existing_props:
                    # Create new property with basic source tracking
                    table_name = node.source.name if node.source else "unknown"
                    prop_source = PropertySource(field=f"{table_name}.{property_name}")
                    new_prop = GraphProperty(key=property_name, source=prop_source)
                    node.properties.append(new_prop)
        return model

    def _apply_change_relationship_name(
        self, model: "GraphModel", old_name: str, new_name: str
    ) -> "GraphModel":
        """Apply change relationship name operation."""
        for edge in model.edges:
            if edge.edge_type == old_name:
                edge.edge_type = new_name
                # Update source mapping if it exists
                if edge.source and "edge_type" in edge.source.mapping:
                    edge.source.mapping["edge_type"] = new_name
        return model

    def _apply_drop_relationship(
        self, model: "GraphModel", relationship_name: str
    ) -> "GraphModel":
        """Apply drop relationship operation."""
        model.edges = [
            edge for edge in model.edges if edge.edge_type != relationship_name
        ]
        return model

    def _apply_add_index(
        self, model: "GraphModel", node_label: str, property_name: str
    ) -> "GraphModel":
        """Apply add index operation."""
        from .models.graph_models import GraphIndex
        from .models.sources import IndexSource

        # Check if index already exists
        for index in model.node_indexes:
            if (
                index.labels
                and node_label in index.labels
                and property_name in index.properties
            ):
                return model  # Index already exists

        # Create new index
        index_source = IndexSource(
            origin="user_request",
            reason="performance_optimization",
            created_by="interactive_modification",
        )
        new_index = GraphIndex(
            labels=[node_label],
            properties=[property_name],
            type="label+property",
            source=index_source,
        )
        model.node_indexes.append(new_index)
        return model

    def _apply_drop_index(
        self, model: "GraphModel", node_label: str, property_name: str
    ) -> "GraphModel":
        """Apply drop index operation."""
        model.node_indexes = [
            index
            for index in model.node_indexes
            if not (
                index.labels
                and node_label in index.labels
                and property_name in index.properties
            )
        ]
        return model

    def validate_graph_model(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of graph model against database structure.

        This method uses the new modular validation system to perform
        graph schema validation of the graph model.

        Args:
            graph_model: The graph model to validate
            database_structure: Original database structure

        Returns:
            Dict with detailed validation results including:
            - is_valid: Boolean indicating if model passes all critical checks
            - issues: List of critical problems that must be fixed
            - warnings: List of recommendations and potential improvements
            - summary: High-level summary of validation results
            - metrics: Quantitative validation metrics
        """
        logger.info("Performing comprehensive graph model validation...")

        # Use the GraphSchemaValidator
        validator = GraphSchemaValidator()
        result = validator.validate(graph_model, database_structure)

        # Convert to the expected format for backward compatibility
        return {
            "is_valid": result.success,
            "issues": [issue.message for issue in result.critical_issues],
            "warnings": [issue.message for issue in result.warnings],
            "suggestions": [issue.message for issue in result.info_issues],
            "summary": result.summary,
            "metrics": {
                "coverage_percentage": result.metrics.coverage_percentage,
                "tables_covered": result.metrics.tables_covered,
                "tables_total": result.metrics.tables_total,
                "properties_covered": result.metrics.properties_covered,
                "properties_total": result.metrics.properties_total,
                "relationships_covered": result.metrics.relationships_covered,
                "relationships_total": result.metrics.relationships_total,
            },
            # Include full result for advanced usage
            "validation_result": result,
        }

    def _perform_post_operation_validation(
        self,
        model: "GraphModel",
        strategy: GraphModelingStrategy,
        operations: "ModelModifications",
    ) -> "GraphModel":
        """
        Perform Graph Schema Validation after applying operations.

        This method validates that the modified graph model still properly
        represents the original database structure, accounting for the
        applied changes. For LLM strategy, it includes automatic model
        improvement loop.

        Args:
            model: Modified graph model to validate
            strategy: Current modeling strategy (affects validation response)
            operations: Operations that were just applied

        Returns:
            The final model (potentially improved by LLM)
        """
        print("\n" + "=" * 60)
        print("GRAPH SCHEMA VALIDATION")
        print("=" * 60)

        if not self.database_structure:
            print("❌ Cannot validate: Original database structure not available")
            return model

        current_model = model
        max_improvement_iterations = 3
        improvement_count = 0

        while improvement_count < max_improvement_iterations:
            # Perform validation using the GraphSchemaValidator
            validator = GraphSchemaValidator()
            validation_result = validator.validate(
                current_model, self.database_structure
            )

            # Print validation summary
            self._display_validation_results(validation_result, strategy)

            # Handle validation results based on strategy
            if strategy == GraphModelingStrategy.DETERMINISTIC:
                self._handle_deterministic_validation(validation_result)
                break  # No automatic improvement for deterministic
            elif strategy == GraphModelingStrategy.LLM_POWERED:
                if validation_result.success:
                    print("\n✅ All validation checks passed!")
                    break
                else:
                    # Try LLM improvement
                    improved_model = self._handle_llm_validation(
                        validation_result, current_model, operations
                    )

                    if improved_model != current_model:
                        # Model was improved, validate again
                        current_model = improved_model
                        improvement_count += 1
                        self.iteration_count += 1

                        print(f"\n🔄 ITERATION {self.iteration_count} - IMPROVED MODEL")
                        self._display_current_model(current_model)

                        if improvement_count < max_improvement_iterations:
                            print(
                                f"\n🔍 Re-validating improved model (iteration {improvement_count + 1}/{max_improvement_iterations})..."
                            )
                            continue
                        else:
                            print(
                                f"\n⚠️ Reached maximum improvement iterations ({max_improvement_iterations})"
                            )
                            break
                    else:
                        # No improvement was made or accepted
                        break

        return current_model

    def _display_validation_results(
        self, validation_result, strategy: GraphModelingStrategy
    ) -> None:
        """Display validation results in a user-friendly format."""
        status = "✅ PASSED" if validation_result.success else "❌ FAILED"
        print(f"\nValidation Status: {status}")
        print(f"Strategy: {strategy.value.upper()}")
        print(f"Summary: {validation_result.summary}")

        # Display metrics
        metrics = validation_result.metrics
        print("\nCoverage Metrics:")
        print(f"  Tables: {metrics.tables_covered}/{metrics.tables_total}")
        print(
            f"  Properties: {metrics.properties_covered}/" f"{metrics.properties_total}"
        )
        print(
            f"  Relationships: {metrics.relationships_covered}/"
            f"{metrics.relationships_total}"
        )
        print(f"  Overall Coverage: {metrics.coverage_percentage:.1f}%")

        # Display critical issues
        critical_issues = validation_result.critical_issues
        if critical_issues:
            print(f"\n🚨 Critical Issues ({len(critical_issues)}):")
            for i, issue in enumerate(critical_issues[:5], 1):  # Show max 5
                print(f"  {i}. {issue.message}")
            if len(critical_issues) > 5:
                print(f"  ... and {len(critical_issues) - 5} more issues")

        # Display warnings
        warnings = validation_result.warnings
        if warnings:
            print(f"\n⚠️ Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings[:3], 1):  # Show max 3
                print(f"  {i}. {warning.message}")
            if len(warnings) > 3:
                print(f"  ... and {len(warnings) - 3} more warnings")

    def _handle_deterministic_validation(self, validation_result) -> None:
        """Handle validation results for deterministic strategy."""
        if not validation_result.success:
            print("\n🔧 DETERMINISTIC STRATEGY GUIDANCE:")
            print("The changes you made have introduced validation issues.")
            print("Consider the following:")

            critical_issues = validation_result.critical_issues
            if critical_issues:
                print("\n📋 Required Fixes:")
                for i, issue in enumerate(critical_issues[:3], 1):
                    has_rec = hasattr(issue, "recommendation") and issue.recommendation
                    if has_rec:
                        print(f"  {i}. {issue.recommendation}")
                    else:
                        print(f"  {i}. {issue.message}")

            print("\nYou can:")
            print("  - Make additional changes to fix these issues")
            print("  - Type 'done' to accept the model as-is")
            print("  - Type 'cancel' to revert your changes")
        else:
            print("\n✅ All validation checks passed!")
            print("Your changes maintain proper database coverage.")

    def _handle_llm_validation(
        self,
        validation_result,
        model: "GraphModel",
        operations: "ModelModifications",
    ) -> "GraphModel":
        """Handle validation results for LLM strategy with automatic model regeneration."""
        if not validation_result.success and self.llm:
            print("\n🤖 LLM STRATEGY: AUTOMATIC MODEL IMPROVEMENT")
            print("The LLM will analyze validation issues and regenerate the model...")

            # Prepare context for LLM
            validation_context = self._prepare_validation_context_for_llm(
                validation_result, model, operations
            )

            try:
                # Use LLM to regenerate the improved model
                improved_model = self._regenerate_model_with_llm_fixes(
                    model, validation_context, validation_result
                )

                if improved_model:
                    print("\n✅ LLM has generated an improved model!")

                    # Offer the user to review the improved model
                    if self._should_apply_llm_improvements(improved_model, model):
                        print("� Applying LLM improvements...")
                        return improved_model
                    else:
                        print("❌ User rejected LLM improvements")
                        return model
                else:
                    print("❌ LLM could not generate improved model")
                    return model
            except Exception as e:
                logger.error("Error getting LLM model improvements: %s", e)
                print(f"❌ Error getting LLM improvements: {e}")
                return model
        else:
            print("\n✅ All validation checks passed!")
            return model

    def _prepare_validation_context_for_llm(
        self,
        validation_result,
        model: "GraphModel",
        operations: "ModelModifications",
    ) -> str:
        """Prepare validation context for LLM analysis."""
        context_parts = []

        # Validation summary
        context_parts.append("VALIDATION RESULTS:")
        status = "PASSED" if validation_result.success else "FAILED"
        context_parts.append(f"Status: {status}")
        context_parts.append(f"Summary: {validation_result.summary}")

        # Critical issues
        critical_issues = validation_result.critical_issues
        if critical_issues:
            context_parts.append(f"\nCRITICAL ISSUES ({len(critical_issues)}):")
            for i, issue in enumerate(critical_issues, 1):
                context_parts.append(f"{i}. {issue.message}")
                has_rec = hasattr(issue, "recommendation") and issue.recommendation
                if has_rec:
                    context_parts.append(f"   Recommendation: {issue.recommendation}")

        # Recent operations
        context_parts.append(f"\nRECENT OPERATIONS ({len(operations.operations)}):")
        for i, op in enumerate(operations.operations, 1):
            context_parts.append(f"{i}. {op.operation_type}: {op.description}")

        # Current model state
        context_parts.append("\nCURRENT MODEL STATE:")
        context_parts.append(f"Nodes: {len(model.nodes)}")
        context_parts.append(f"Relationships: {len(model.edges)}")
        context_parts.append(f"Indexes: {len(model.node_indexes)}")

        return "\n".join(context_parts)

    def _regenerate_model_with_llm_fixes(
        self, current_model: "GraphModel", validation_context: str, validation_result
    ) -> Optional["GraphModel"]:
        """Use LLM to regenerate an improved model based on validation issues."""
        if not self.llm or not self.database_structure:
            return None

        # Get the modeling strategy instance for regeneration
        strategy_instance = self._get_strategy_instance(
            GraphModelingStrategy.LLM_POWERED
        )

        # Prepare enhanced context for the LLM
        improvement_context = self._prepare_improvement_context(
            current_model, validation_result, validation_context
        )

        system_prompt = (
            "You are an expert graph modeling assistant. You need to regenerate "
            "an improved graph model that addresses the validation issues "
            "identified in the current model. Focus on:\n\n"
            "1. Adding any missing entities (tables) to achieve full coverage\n"
            "2. Including all required properties from the original database\n"
            "3. Ensuring all relationships are properly modeled\n"
            "4. Maintaining data integrity and proper graph structure\n\n"
            "Generate a complete, improved graph model that resolves the "
            "validation issues while preserving the user's modifications."
        )

        try:
            print("🔄 LLM is analyzing validation issues and regenerating model...")

            # Use the LLM strategy but with enhanced context
            improved_model = strategy_instance.create_model(
                self.database_structure, domain_context=improvement_context
            )

            if improved_model:
                print("✅ LLM generated improved model")
                return improved_model
            else:
                print("❌ LLM failed to generate improved model")
                return None

        except Exception as e:
            logger.error("Error regenerating model with LLM: %s", e)
            return None

    def _prepare_improvement_context(
        self, current_model: "GraphModel", validation_result, validation_context: str
    ) -> str:
        """Prepare comprehensive context for LLM model improvement."""
        context_parts = []

        # Previous model structure
        context_parts.append("CURRENT MODEL TO IMPROVE:")
        context_parts.append(self._get_model_context_for_llm(current_model))

        # Validation issues that need fixing
        context_parts.append("\nVALIDATION ISSUES TO RESOLVE:")
        context_parts.append(validation_context)

        # Critical requirements
        context_parts.append("\nCRITICAL REQUIREMENTS:")
        critical_issues = validation_result.critical_issues
        for i, issue in enumerate(critical_issues, 1):
            context_parts.append(f"{i}. {issue.message}")
            if hasattr(issue, "recommendation") and issue.recommendation:
                context_parts.append(f"   → {issue.recommendation}")

        # Coverage targets
        metrics = validation_result.metrics
        context_parts.append("\nCOVERAGE TARGETS:")
        context_parts.append(f"- Target tables: {metrics.tables_total}")
        context_parts.append(f"- Target properties: {metrics.properties_total}")
        context_parts.append(f"- Target relationships: {metrics.relationships_total}")

        return "\n".join(context_parts)

    def _should_apply_llm_improvements(
        self, improved_model: "GraphModel", current_model: "GraphModel"
    ) -> bool:
        """Ask user whether to apply LLM improvements."""
        print("\n" + "=" * 60)
        print("🤖 LLM MODEL IMPROVEMENT REVIEW")
        print("=" * 60)

        # Show comparison
        print(f"\nCURRENT MODEL:")
        print(f"  Nodes: {len(current_model.nodes)}")
        print(f"  Relationships: {len(current_model.edges)}")
        print(f"  Indexes: {len(current_model.node_indexes)}")
        print(f"  Constraints: {len(current_model.node_constraints)}")

        print(f"\nIMPROVED MODEL:")
        print(f"  Nodes: {len(improved_model.nodes)}")
        print(f"  Relationships: {len(improved_model.edges)}")
        print(f"  Indexes: {len(improved_model.node_indexes)}")
        print(f"  Constraints: {len(improved_model.node_constraints)}")

        # Show what's new/different
        self._show_model_differences(current_model, improved_model)

        print("\nWould you like to apply these LLM improvements?")
        print("1. Yes - Apply improvements and continue")
        print("2. No - Keep current model")
        print("3. Review - Show detailed improved model first")

        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice == "1":
                    return True
                elif choice == "2":
                    return False
                elif choice == "3":
                    self._display_current_model(improved_model)
                    print(
                        "\nAfter reviewing, would you like to apply these improvements?"
                    )
                    print("1. Yes - Apply improvements")
                    print("2. No - Keep current model")
                    continue
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except (EOFError, KeyboardInterrupt):
                print("\nKeeping current model...")
                return False

    def _show_model_differences(
        self, current_model: "GraphModel", improved_model: "GraphModel"
    ) -> None:
        """Show differences between current and improved models."""
        # Compare nodes
        current_node_labels = {node.primary_label for node in current_model.nodes}
        improved_node_labels = {node.primary_label for node in improved_model.nodes}

        new_nodes = improved_node_labels - current_node_labels
        removed_nodes = current_node_labels - improved_node_labels

        if new_nodes:
            print(f"\n➕ NEW NODES ({len(new_nodes)}):")
            for label in sorted(new_nodes):
                print(f"  + {label}")

        if removed_nodes:
            print(f"\n➖ REMOVED NODES ({len(removed_nodes)}):")
            for label in sorted(removed_nodes):
                print(f"  - {label}")

        # Compare relationships
        current_relationships = {edge.edge_type for edge in current_model.edges}
        improved_relationships = {edge.edge_type for edge in improved_model.edges}

        new_relationships = improved_relationships - current_relationships
        removed_relationships = current_relationships - improved_relationships

        if new_relationships:
            print(f"\n➕ NEW RELATIONSHIPS ({len(new_relationships)}):")
            for rel_type in sorted(new_relationships):
                print(f"  + {rel_type}")

        if removed_relationships:
            print(f"\n➖ REMOVED RELATIONSHIPS ({len(removed_relationships)}):")
            for rel_type in sorted(removed_relationships):
                print(f"  - {rel_type}")

        # Show property changes for existing nodes
        common_nodes = current_node_labels & improved_node_labels
        if common_nodes:
            property_changes = []
            for label in common_nodes:
                current_node = next(
                    n for n in current_model.nodes if n.primary_label == label
                )
                improved_node = next(
                    n for n in improved_model.nodes if n.primary_label == label
                )

                current_props = {p.key for p in current_node.properties}
                improved_props = {p.key for p in improved_node.properties}

                new_props = improved_props - current_props
                removed_props = current_props - improved_props

                if new_props or removed_props:
                    property_changes.append((label, new_props, removed_props))

            if property_changes:
                print(f"\n🔄 PROPERTY CHANGES:")
                for label, new_props, removed_props in property_changes:
                    if new_props:
                        for prop in sorted(new_props):
                            print(f"  + {label}.{prop}")
                    if removed_props:
                        for prop in sorted(removed_props):
                            print(f"  - {label}.{prop}")

    def _get_llm_validation_fixes(self, validation_context: str) -> Optional[str]:
        """Get LLM suggestions for fixing validation issues."""
        if not self.llm:
            return None

        system_prompt = (
            "You are an expert graph modeling assistant. "
            "Analyze the validation results and suggest specific fixes "
            "to improve the graph model's coverage of the original "
            "database structure. "
            "Focus on critical issues that affect data completeness."
        )

        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Please analyze these validation results and suggest "
                            f"fixes:\n\n"
                            f"{validation_context}\n\n"
                            f"Provide specific, actionable recommendations to fix "
                            f"the critical issues."
                        ),
                    },
                ]
            )
            return response.content
        except Exception as e:
            logger.error("Error calling LLM for validation fixes: %s", e)
            return None
