"""
Main HyGM (Hypothetical Graph Modeling) class.

This is the primary interface for the modular HyGM system.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models.graph_models import GraphModel

try:
    from .strategies import BaseModelingStrategy, DeterministicStrategy, LLMStrategy
except ImportError:
    from core.hygm.strategies import (
        BaseModelingStrategy,
        DeterministicStrategy,
        LLMStrategy,
    )

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
        return strategy_instance.create_model(database_structure, domain_context)

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
                current_model = self._modify_model_interactively(current_model)
                self.iteration_count += 1
            elif user_choice == "regenerate":
                # Regenerate with the same strategy
                current_model = strategy_instance.create_model(
                    database_structure, domain_context
                )
                self.iteration_count += 1
            elif user_choice == "switch_strategy":
                # Switch between strategies
                new_strategy = self._switch_strategy(strategy)
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
            labels = " | ".join(index.labels)
            props = ", ".join(index.properties)
            print(f"  {i}. {labels}.{props}")

        print(f"\nCONSTRAINTS ({len(model.node_constraints)}):")
        for i, constraint in enumerate(model.node_constraints, 1):
            labels = " | ".join(constraint.labels)
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
        """Allow user to modify the model interactively."""
        # This would integrate with the interactive modification system
        # For now, return the model unchanged
        print("Interactive model modification not yet implemented.")
        print("Returning model unchanged...")
        return model

    # Backward compatibility methods
    def model_graph(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> "GraphModel":
        """
        Backward compatibility method.

        This maintains the original API while using the new modular system.
        """
        return self.create_graph_model(
            database_structure, domain_context, GraphModelingStrategy.DETERMINISTIC
        )

    def validate_graph_model(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate graph model against database structure.

        Args:
            graph_model: The graph model to validate
            database_structure: Original database structure

        Returns:
            Dict with validation results
        """
        logger.info("Validating graph model...")

        issues = []
        warnings = []

        # Check all entity tables are represented as nodes
        entity_tables = set(database_structure.get("entity_tables", {}).keys())
        model_source_tables = set()

        for node in graph_model.nodes:
            if node.source and hasattr(node.source, "name"):
                model_source_tables.add(node.source.name)

        missing_tables = entity_tables - model_source_tables
        if missing_tables:
            issues.append(f"Missing nodes for tables: {missing_tables}")

        # Check relationships correspond to actual foreign keys
        db_relationships = database_structure.get("relationships", [])
        if len(graph_model.edges) == 0 and len(db_relationships) > 0:
            warnings.append("No relationships created despite foreign keys existing")

        # Basic counts validation
        if len(graph_model.nodes) == 0:
            issues.append("No nodes created in graph model")

        # Summary
        is_valid = len(issues) == 0
        summary = f"{len(graph_model.nodes)} nodes, {len(graph_model.edges)} edges"

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "summary": summary,
        }
