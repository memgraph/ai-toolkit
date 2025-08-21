"""
Main HyGM (Hypothetical Graph Modeling) class.

This is the primary interface for the modular HyGM system.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, TYPE_CHECKING

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
                current_model = self._modify_model_interactively(current_model)
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

    def _modify_model_interactively(self, model: "GraphModel") -> "GraphModel":
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
        # For now, we'll print what would be applied
        # In a full implementation, this would modify the actual model
        print(f"\nWould apply {len(operations.operations)} operations:")

        for op in operations.operations:
            if op.operation_type == "change_node_label":
                print(
                    f"  - Change node label: {getattr(op, 'old_label')} "
                    f"→ {getattr(op, 'new_label')}"
                )
            elif op.operation_type == "rename_property":
                node = getattr(op, "node_label")
                old_prop = getattr(op, "old_property")
                new_prop = getattr(op, "new_property")
                print(f"  - Rename property on {node}: " f"{old_prop} → {new_prop}")
            elif op.operation_type == "drop_property":
                node = getattr(op, "node_label")
                prop = getattr(op, "property_name")
                print(f"  - Drop property: {node}.{prop}")
            elif op.operation_type == "add_property":
                node = getattr(op, "node_label")
                prop = getattr(op, "property_name")
                print(f"  - Add property: {node}.{prop}")
            elif op.operation_type == "change_relationship_name":
                old_name = getattr(op, "old_name")
                new_name = getattr(op, "new_name")
                print(f"  - Change relationship: {old_name} → {new_name}")
            elif op.operation_type == "drop_relationship":
                rel = getattr(op, "relationship_name")
                print(f"  - Drop relationship: {rel}")
            elif op.operation_type == "add_index":
                node = getattr(op, "node_label")
                prop = getattr(op, "property_name")
                print(f"  - Add index: {node}.{prop}")
            elif op.operation_type == "drop_index":
                node = getattr(op, "node_label")
                prop = getattr(op, "property_name")
                print(f"  - Drop index: {node}.{prop}")

        print(f"\nReasoning: {operations.reasoning}")
        print("\n⚠️  Note: Actual model modification not yet implemented.")
        print("This is a preview of what would be changed.")

        # TODO: Implement actual model modifications
        # This would require updating the GraphModel structure

        return model

    def validate_graph_model(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of graph model against database structure.

        This method uses the new modular validation system to perform
        pre-migration validation of the graph model.

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

        # Use the new pre-migration validator
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
