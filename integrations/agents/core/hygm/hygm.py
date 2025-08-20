"""
Main HyGM (Hypothetical Graph Modeling) class.

This is the primary interface for the modular HyGM system.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, TYPE_CHECKING

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

    def export_schema_format(
        self,
        graph_model: Optional["GraphModel"] = None,
        sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Export graph model to schema format (spec.json compatible).

        Args:
            graph_model: Graph model to export. If None, uses current model
            sample_data: Optional sample data for type detection

        Returns:
            Dict in spec.json format
        """
        model_to_export = graph_model or self.current_graph_model

        if model_to_export is None:
            raise ValueError(
                "No graph model available. Create a model first using "
                "create_graph_model()"
            )

        return model_to_export.to_schema_format(sample_data)

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
        Comprehensive validation of graph model against database structure.

        Args:
            graph_model: The graph model to validate
            database_structure: Original database structure

        Returns:
            Dict with detailed validation results including:
            - is_valid: Boolean indicating if model passes all critical checks
            - issues: List of critical problems that must be fixed
            - warnings: List of recommendations and potential improvements
            - suggestions: List of optimization suggestions
            - summary: High-level summary of validation results
            - metrics: Quantitative validation metrics
        """
        logger.info("Performing comprehensive graph model validation...")

        issues = []
        warnings = []
        suggestions = []
        metrics = {}

        # 1. STRUCTURAL VALIDATION
        entity_tables = set(database_structure.get("entity_tables", {}).keys())
        model_source_tables = set()

        # Basic existence checks
        if len(graph_model.nodes) == 0:
            issues.append("Critical: No nodes created in graph model")

        if not entity_tables:
            warnings.append("No entity tables found in source database")

        # 2. NODE COVERAGE VALIDATION
        for node in graph_model.nodes:
            if node.source and hasattr(node.source, "name"):
                model_source_tables.add(node.source.name)

        missing_tables = entity_tables - model_source_tables
        if missing_tables:
            issues.append(
                f"Missing nodes for entity tables: " f"{sorted(missing_tables)}"
            )

        extra_tables = model_source_tables - entity_tables
        if extra_tables:
            warnings.append(
                f"Nodes created for non-entity tables: " f"{sorted(extra_tables)}"
            )

        # 3. NODE QUALITY VALIDATION
        duplicate_labels = set()
        nodes_without_properties = []
        nodes_with_invalid_labels = []

        seen_labels = set()
        for node in graph_model.nodes:
            # Check for duplicate primary labels
            primary_label = node.primary_label
            if primary_label in seen_labels:
                duplicate_labels.add(primary_label)
            seen_labels.add(primary_label)

            # Check node has properties
            if not node.properties or len(node.properties) == 0:
                nodes_without_properties.append(primary_label)

            # Validate label naming conventions
            if not primary_label or not primary_label.replace("_", "").isalnum():
                nodes_with_invalid_labels.append(primary_label)

        if duplicate_labels:
            issues.append(
                f"Duplicate node labels found: " f"{sorted(duplicate_labels)}"
            )
        if nodes_without_properties:
            warnings.append(
                f"Nodes without properties: " f"{sorted(nodes_without_properties)}"
            )
        if nodes_with_invalid_labels:
            warnings.append(
                f"Nodes with invalid label names: "
                f"{sorted(nodes_with_invalid_labels)}"
            )

        # 4. PROPERTY VALIDATION
        property_issues = self._validate_node_properties(
            graph_model, database_structure
        )
        issues.extend(property_issues.get("issues", []))
        warnings.extend(property_issues.get("warnings", []))
        suggestions.extend(property_issues.get("suggestions", []))

        # 5. RELATIONSHIP VALIDATION
        relationship_validation = self._validate_relationships(
            graph_model, database_structure
        )
        issues.extend(relationship_validation.get("issues", []))
        warnings.extend(relationship_validation.get("warnings", []))
        suggestions.extend(relationship_validation.get("suggestions", []))

        # 6. SCHEMA CONSISTENCY VALIDATION
        schema_validation = self._validate_schema_consistency(
            graph_model, database_structure
        )
        warnings.extend(schema_validation.get("warnings", []))
        suggestions.extend(schema_validation.get("suggestions", []))

        # 7. CALCULATE METRICS
        coverage_pct = (
            len(model_source_tables) / len(entity_tables) * 100 if entity_tables else 0
        )
        avg_props = (
            sum(len(node.properties) for node in graph_model.nodes)
            / len(graph_model.nodes)
            if graph_model.nodes
            else 0
        )
        rel_density = (
            len(graph_model.edges) / len(graph_model.nodes) if graph_model.nodes else 0
        )

        metrics = {
            "node_count": len(graph_model.nodes),
            "edge_count": len(graph_model.edges),
            "coverage_percentage": coverage_pct,
            "avg_properties_per_node": avg_props,
            "relationship_density": rel_density,
            "tables_covered": len(model_source_tables),
            "total_entity_tables": len(entity_tables),
        }

        # Determine validity
        is_valid = len(issues) == 0

        # Generate comprehensive summary
        summary = self._generate_validation_summary(
            metrics, issues, warnings, suggestions
        )

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
            "summary": summary,
            "metrics": metrics,
        }

    def _validate_node_properties(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> Dict[str, list]:
        """Validate node properties exist in source schema."""
        issues = []
        warnings = []
        suggestions = []

        entity_tables = database_structure.get("entity_tables", {})

        for node in graph_model.nodes:
            if not node.source or not hasattr(node.source, "name"):
                continue

            table_name = node.source.name
            table_info = entity_tables.get(table_name, {})
            schema_columns = {col["field"]: col for col in table_info.get("schema", [])}

            # Check each property exists in source schema
            missing_props = []

            for prop in node.properties:
                prop_key = prop.key if hasattr(prop, "key") else str(prop)

                if prop_key not in schema_columns:
                    missing_props.append(prop_key)

            if missing_props:
                issues.append(
                    f"Node {node.primary_label} has properties not in source: "
                    f"{missing_props}"
                )

        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}

    def _validate_relationships(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> Dict[str, list]:
        """Validate relationships against database foreign keys."""
        issues = []
        warnings = []
        suggestions = []

        db_relationships = database_structure.get("relationships", [])

        # Check if relationships exist when they should
        if len(graph_model.edges) == 0 and len(db_relationships) > 0:
            warnings.append(
                f"No relationships created despite {len(db_relationships)} "
                f"foreign keys existing"
            )

        # Validate each relationship
        for edge in graph_model.edges:
            # Check if relationship type follows naming conventions
            if not edge.edge_type.isupper():
                suggestions.append(
                    f"Relationship type '{edge.edge_type}' should be UPPERCASE"
                )

            # Check for self-referencing relationships
            if edge.start_node_labels == edge.end_node_labels:
                warnings.append(
                    f"Self-referencing relationship detected: {edge.edge_type}"
                )

            # Validate directionality
            if edge.directionality not in ["directed", "undirected"]:
                issues.append(
                    f"Invalid directionality '{edge.directionality}' "
                    f"for relationship {edge.edge_type}"
                )

        # Check for missing relationships based on foreign keys
        missing_rels = self._identify_missing_relationships(
            graph_model, db_relationships
        )
        if missing_rels:
            suggestions.extend(missing_rels)

        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}

    def _validate_schema_consistency(
        self, graph_model: "GraphModel", database_structure: Dict[str, Any]
    ) -> Dict[str, list]:
        """Validate overall schema consistency and design patterns."""
        warnings = []
        suggestions = []

        # Check for potential many-to-many patterns
        join_tables = database_structure.get("join_tables", {})
        if join_tables and len(graph_model.edges) < len(join_tables):
            warnings.append(
                f"Potential many-to-many relationships not modeled: "
                f"{len(join_tables)} join tables found"
            )

        # Check for hierarchical relationships
        self_refs = [
            edge
            for edge in graph_model.edges
            if edge.start_node_labels == edge.end_node_labels
        ]
        if self_refs:
            suggestions.append(
                f"Consider tree/hierarchy modeling for self-referencing "
                f"relationships: {[e.edge_type for e in self_refs]}"
            )

        return {"warnings": warnings, "suggestions": suggestions}

    def _generate_validation_summary(
        self, metrics: Dict, issues: list, warnings: list, suggestions: list
    ) -> str:
        """Generate a comprehensive validation summary."""
        coverage = metrics.get("coverage_percentage", 0)
        node_count = metrics.get("node_count", 0)
        edge_count = metrics.get("edge_count", 0)

        status = "VALID" if len(issues) == 0 else "INVALID"

        summary_parts = [
            f"Status: {status}",
            f"Coverage: {coverage:.1f}% " f"({node_count} nodes, {edge_count} edges)",
        ]

        if issues:
            summary_parts.append(f"Issues: {len(issues)} critical")
        if warnings:
            summary_parts.append(f"Warnings: {len(warnings)}")
        if suggestions:
            summary_parts.append(f"Suggestions: {len(suggestions)} " "optimizations")

        return " | ".join(summary_parts)

    def _identify_missing_relationships(
        self, graph_model: "GraphModel", db_relationships: list
    ) -> list:
        """Identify relationships that exist in DB but missing in graph."""
        suggestions = []

        # Get existing relationship patterns from graph model
        existing_patterns = set()
        for edge in graph_model.edges:
            pattern = (
                tuple(sorted(edge.start_node_labels)),
                tuple(sorted(edge.end_node_labels)),
                edge.edge_type,
            )
            existing_patterns.add(pattern)

        # Check database relationships
        for db_rel in db_relationships:
            parent_table = db_rel.get("parent_table", "")
            child_table = db_rel.get("child_table", "")

            if parent_table and child_table:
                # Simple heuristic: check if this table pair has a relationship
                has_relationship = any(
                    (
                        parent_table.lower() in str(edge.start_node_labels).lower()
                        and child_table.lower() in str(edge.end_node_labels).lower()
                    )
                    or (
                        child_table.lower() in str(edge.start_node_labels).lower()
                        and parent_table.lower() in str(edge.end_node_labels).lower()
                    )
                    for edge in graph_model.edges
                )

                if not has_relationship:
                    suggestions.append(
                        f"Consider modeling relationship between "
                        f"{parent_table} and {child_table}"
                    )

        return suggestions
