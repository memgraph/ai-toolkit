"""
Main HyGM (Hypothetical Graph Modeling) class.

This is the primary interface for the modular HyGM system.
"""

import copy
import uuid
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models.graph_models import GraphModel, GraphNode
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

try:
    from .models.user_operations import UserOperationHistory
except ImportError:
    from core.hygm.models.user_operations import UserOperationHistory

logger = logging.getLogger(__name__)


class ModelingMode(Enum):
    """Modeling modes for HyGM."""

    AUTOMATIC = "automatic"
    INCREMENTAL = "incremental"


class GraphModelingStrategy(Enum):
    """Graph modeling strategies available."""

    DETERMINISTIC = "deterministic"  # Rule-based graph creation
    LLM_POWERED = "llm_powered"  # LLM generates the graph model


class HyGM:
    """
    Main Hypothetical Graph Modeling class.

    Uses different strategies to create intelligent graph models from
    relational schemas. Supports automatic generation or incremental
    refinement with user feedback.
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
            mode: AUTOMATIC or INCREMENTAL modeling mode
            strategy: DETERMINISTIC or LLM_POWERED strategy
        """
        self.llm = llm
        self.mode = mode
        self.strategy = strategy
        self.current_graph_model = None
        self.iteration_count = 0
        self.database_structure = None
        self._strategy_cache = {}
        # User operation tracking
        self.user_operation_history: Optional["UserOperationHistory"] = None
        self.session_id = str(uuid.uuid4())

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

        # Check if incremental mode is enabled
        if self.mode == ModelingMode.INCREMENTAL:
            return self._incremental_modeling(
                database_structure, domain_context, used_strategy
            )

        # For automatic mode, use the specified strategy
        strategy_instance = self._get_strategy_instance(used_strategy)
        graph_model = strategy_instance.create_model(database_structure, domain_context)

        # Store the created model as current
        self.current_graph_model = graph_model

        # For LLM strategy in automatic mode, perform validation and retry if needed
        if used_strategy == GraphModelingStrategy.LLM_POWERED:
            graph_model = self._validate_and_improve_automatic_llm_model(
                graph_model, database_structure, domain_context
            )
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

    def _interactive_refinement_loop(
        self,
        model: "GraphModel",
        strategy: GraphModelingStrategy,
    ) -> "GraphModel":
        """
        Interactive refinement loop for the incremental modeling flow.

        Allows users to inspect the aggregated graph model, provide feedback,
        and iteratively refine nodes, relationships, indexes, and constraints
        using the natural-language modification helpers.
        """
        logger.info("Entering interactive refinement loop for incremental modeling")

        self.current_graph_model = model
        if self.iteration_count == 0:
            self.iteration_count = 1

        while True:
            self._display_current_model(model)

            user_choice = self._get_refinement_choice()

            if user_choice == "accept":
                logger.info("Combined model accepted by user")
                break
            if user_choice == "modify":
                logger.info("Modifying combined model interactively...")
                model = self._modify_model_interactively(model, strategy)
                self.current_graph_model = model
                continue
            if user_choice == "validate":
                logger.info("Validating combined model on user request")
                model = self._perform_manual_validation(model, strategy)
                self.current_graph_model = model

        return model

    def _incremental_modeling(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        strategy: GraphModelingStrategy = GraphModelingStrategy.DETERMINISTIC,
    ) -> "GraphModel":
        """
        Incremental modeling process with table-by-table confirmation.

        This method processes each table individually, showing the user
        what node will be created and asking for confirmation before
        proceeding to the next table.
        """
        logger.info("Starting incremental modeling session...")

        # Get the strategy instance
        strategy_instance = self._get_strategy_instance(strategy)

        # Initialize an empty graph model to build incrementally
        from .models.graph_models import GraphModel

        incremental_model = GraphModel(
            nodes=[], edges=[], node_indexes=[], node_constraints=[]
        )

        # Extract tables from database structure
        tables = database_structure.get("tables", {})
        if not tables:
            print("❌ No tables found in database structure")
            return incremental_model

        entity_tables = database_structure.get("entity_tables", {})
        join_tables = database_structure.get("join_tables", {})
        view_tables = database_structure.get("views") or database_structure.get(
            "view_tables", {}
        )
        all_relationships = database_structure.get("relationships", [])
        sample_data = database_structure.get("sample_data", {})
        table_counts = database_structure.get("table_counts", {})
        database_name = database_structure.get("database_name")
        database_type = database_structure.get("database_type")

        self._print_banner("INCREMENTAL MODELING SESSION")
        print(f"\nFound {len(tables)} tables to process")
        print(
            "You will review each table and its proposed node before " "proceeding.\n"
        )

        processed_tables = []
        skipped_tables = []

        # Process each table individually
        for table_name, table_info in tables.items():
            print("=" * 60)
            print(f"PROCESSING TABLE: {table_name}")
            print("=" * 60)

            # Create a temporary database structure with just this table
            relevant_relationships = [
                rel
                for rel in all_relationships
                if table_name
                in {
                    rel.get("from_table"),
                    rel.get("to_table"),
                    rel.get("join_table"),
                }
            ]

            single_table_structure: Dict[str, Any] = {
                "tables": {table_name: table_info},
                "entity_tables": {},
                "join_tables": {},
                "views": {},
                "relationships": relevant_relationships,
            }

            if table_name in entity_tables:
                single_table_structure["entity_tables"][table_name] = entity_tables[
                    table_name
                ]
            if table_name in join_tables:
                single_table_structure["join_tables"][table_name] = join_tables[
                    table_name
                ]
            if table_name in view_tables:
                single_table_structure["views"][table_name] = view_tables[table_name]

            if sample_data:
                single_table_structure["sample_data"] = {
                    table_name: sample_data.get(table_name, [])
                }
            if table_counts:
                single_table_structure["table_counts"] = {
                    table_name: table_counts.get(table_name, 0)
                }
            if database_name:
                single_table_structure["database_name"] = database_name
            if database_type:
                single_table_structure["database_type"] = database_type

            # Generate model for this single table
            try:
                table_model = strategy_instance.create_model(
                    single_table_structure, domain_context
                )

                # Display the proposed node for this table
                proposed_nodes = [
                    node
                    for node in table_model.nodes
                    if any(table_name.lower() in label.lower() for label in node.labels)
                ]

                if not proposed_nodes:
                    proposed_nodes = table_model.nodes

                self._display_table_and_proposed_node(
                    table_name, table_info, proposed_nodes
                )

                # Get user decision
                user_choice = self._get_incremental_choice(table_name)

                if user_choice == "accept":
                    # Add this table's nodes to the incremental model
                    self._merge_table_model_into_incremental(
                        incremental_model, table_model
                    )
                    processed_tables.append(table_name)
                    print(f"✅ Added {table_name} to the graph model")
                    # Refinement is now offered after the full session summary.
                elif user_choice == "skip":
                    skipped_tables.append(table_name)
                    print(f"⏭️  Skipped {table_name}")
                elif user_choice == "modify":
                    # Allow user to modify the proposed node
                    modified_model = self._modify_table_node_interactively(
                        table_model, table_name
                    )
                    self._merge_table_model_into_incremental(
                        incremental_model, modified_model
                    )
                    processed_tables.append(table_name)
                    print(f"✅ Added modified {table_name} to the graph model")
                    # Refinement is now offered after the full session summary.
                elif user_choice == "finish":
                    print("🏁 Finishing incremental modeling session...")
                    break

            except Exception as e:
                logger.error("Error processing table %s: %s", table_name, e)
                print(f"❌ Error processing {table_name}: {e}")
                continue

        # Final summary
        self._print_banner("INCREMENTAL MODELING SUMMARY")
        print(f"✅ Processed tables: {len(processed_tables)}")
        if processed_tables:
            print(f"   {', '.join(processed_tables)}")

        if skipped_tables:
            print(f"⏭️  Skipped tables: {len(skipped_tables)}")
            print(f"   {', '.join(skipped_tables)}")

        print("\n📊 Final model statistics:")
        print(f"   Nodes: {len(incremental_model.nodes)}")
        print(f"   Relationships: {len(incremental_model.edges)}")
        print(f"   Indexes: {len(incremental_model.node_indexes)}")
        print(f"   Constraints: {len(incremental_model.node_constraints)}")

        review_choice = self._get_user_input_choice(
            "\nWould you like to review and refine the combined model?\n"
            "1. Finish with the incremental result\n"
            "2. Enter the interactive refinement loop\n"
            "\nSelect option (1-2) or press Enter to finish: ",
            {"1": "finish", "2": "review", "": "finish"},
            "finish",
        )

        if review_choice == "review":
            logger.info(
                "Switching from incremental flow to interactive refinement",
            )
            incremental_model = self._interactive_refinement_loop(
                incremental_model,
                strategy,
            )

        self.current_graph_model = incremental_model
        return incremental_model

    def _display_table_and_proposed_node(
        self,
        table_name: str,
        table_info: Dict[str, Any],
        proposed_nodes: List["GraphNode"],
    ) -> None:
        """Display table information and proposed graph node."""
        print(f"\n📋 TABLE: {table_name}")

        # Show table columns
        columns = table_info.get("schema") or table_info.get("columns", [])
        if columns:
            if isinstance(columns, dict):
                items = columns.items()
            else:
                items = [
                    (
                        col.get("field") or col.get("name"),
                        {
                            "type": col.get("type") or col.get("data_type"),
                            "null": col.get("null"),
                        },
                    )
                    for col in columns
                ]

            print(f"   Columns ({len(columns)}):")
            for col_name, col_info in items:
                if not col_name:
                    continue
                col_type = col_info.get("type", "unknown")
                nullable_flag = col_info.get("null")
                is_nullable = nullable_flag not in {"NO", False, "false", 0}
                nullable = " (nullable)" if is_nullable else ""
                print(f"     - {col_name}: {col_type}{nullable}")

        # Show primary keys
        primary_keys = table_info.get("primary_keys", [])
        if primary_keys:
            print(f"   Primary Keys: {', '.join(primary_keys)}")

        # Show proposed node(s)
        print("\n🎯 PROPOSED NODE(S):")
        if proposed_nodes:
            for i, node in enumerate(proposed_nodes, 1):
                labels = " | ".join(node.labels)
                properties = [p.key for p in node.properties]
                print(f"   {i}. Node Labels: {labels}")
                print(f"      Properties: {properties}")
        else:
            print("   ❌ No nodes proposed for this table")

    def _get_incremental_choice(self, table_name: str) -> str:
        """Get user choice for incremental modeling."""
        print(f"\nWhat would you like to do with table '{table_name}'?")
        print("1. Accept - Add this node to the graph model")
        print("2. Skip - Skip this table for now")
        print("3. Modify - Modify the proposed node before adding")
        print("4. Finish - Stop incremental modeling and return current model")

        choices = {
            "1": "accept",
            "2": "skip",
            "3": "modify",
            "4": "finish",
        }
        return self._get_user_input_choice(
            f"\nEnter your choice for {table_name} (1-4): ", choices, "accept"
        )

    def _merge_table_model_into_incremental(
        self, incremental_model: "GraphModel", table_model: "GraphModel"
    ) -> None:
        """Merge a single table's model into the incremental model."""
        # Add nodes (avoid duplicates based on labels)
        existing_node_labels = {
            tuple(sorted(node.labels)) for node in incremental_model.nodes
        }

        for node in table_model.nodes:
            node_labels_tuple = tuple(sorted(node.labels))
            if node_labels_tuple not in existing_node_labels:
                incremental_model.nodes.append(node)
                existing_node_labels.add(node_labels_tuple)

        # Add edges (avoid duplicates)
        existing_edges = {
            (
                edge.edge_type,
                tuple(edge.start_node_labels),
                tuple(edge.end_node_labels),
            )
            for edge in incremental_model.edges
        }

        for edge in table_model.edges:
            edge_key = (
                edge.edge_type,
                tuple(edge.start_node_labels),
                tuple(edge.end_node_labels),
            )
            if edge_key not in existing_edges:
                incremental_model.edges.append(edge)
                existing_edges.add(edge_key)

        # Add indexes
        incremental_model.node_indexes.extend(table_model.node_indexes)

        # Add constraints
        incremental_model.node_constraints.extend(table_model.node_constraints)

    def _modify_table_node_interactively(
        self, table_model: "GraphModel", table_name: str
    ) -> "GraphModel":
        """Allow user to modify the proposed node for a table."""
        print(f"\n🔧 MODIFYING NODE FOR TABLE: {table_name}")
        print("You can use natural language to modify the proposed node.")
        print("Examples:")
        print(" - Change the label from 'User' to 'Person'")
        print(" - Remove the 'email' property")
        print(" - Add a 'full_name' property")

        if not self.llm:
            print("❌ LLM not available for natural language modifications.")
            print("Returning original model.")
            return table_model

        while True:
            try:
                user_input = input(
                    f"\nDescribe changes for {table_name} " f"(or 'done' to finish): "
                ).strip()

                if user_input.lower() == "done":
                    break
                elif not user_input:
                    print("Please describe the change you'd like to make.")
                    continue

                # Use existing natural language parsing
                operations = self._parse_natural_language_to_operations(
                    user_input, table_model
                )

                if operations:
                    print(f"✅ Understood: {operations.reasoning}")
                    table_model = self._apply_operations_to_model(
                        table_model, operations
                    )

                    # Show updated model
                    print(f"\n📋 UPDATED NODE FOR {table_name}:")
                    for node in table_model.nodes:
                        labels = " | ".join(node.labels)
                        properties = [p.key for p in node.properties]
                        print(f"   Labels: {labels}")
                        print(f"   Properties: {properties}")
                else:
                    print("❌ I didn't understand that command. " "Please try again.")

            except (EOFError, KeyboardInterrupt):
                print(f"\nFinished modifying {table_name}")
                break

        return table_model

    def _print_banner(self, title: str, width: int = 60) -> None:
        """Print a formatted banner with title."""
        print("\n" + "=" * width)
        print(title)
        print("=" * width)

    def _display_current_model(self, model: "GraphModel") -> None:
        """Display the current graph model to the user."""
        self._print_banner(f"GRAPH MODEL - ITERATION {self.iteration_count}")

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

    def _get_user_input_choice(
        self,
        prompt: str,
        choices: Dict[str, str],
        default_action: str = "accept",
    ) -> str:
        """Get validated user input from multiple choices.

        Args:
            prompt: The prompt to display to user
            choices: Dict mapping choice keys to return values
            default_action: Action to return on EOF/interrupt

        Returns:
            The selected choice value
        """
        while True:
            try:
                choice = input(prompt).strip()
                if choice in choices:
                    return choices[choice]
                else:
                    valid_choices = ", ".join(sorted(choices.keys()))
                    print(f"Invalid choice. Please enter {valid_choices}.")
            except (EOFError, KeyboardInterrupt):
                print(f"\nDefaulting to {default_action}...")
                return default_action

    def _get_user_choice(self) -> str:
        """Get user choice for next action."""
        print("\nWhat would you like to do?")
        print("1. Accept this model")
        print("2. Modify the model")
        print("3. Regenerate model (same strategy)")
        print("4. Switch modeling strategy")

        choices = {
            "1": "accept",
            "2": "modify",
            "3": "regenerate",
            "4": "switch_strategy",
        }
        return self._get_user_input_choice(
            "\nEnter your choice (1-4): ", choices, "accept"
        )

    def _get_refinement_choice(self) -> str:
        """Get user choice while refining the combined graph model."""
        print("\nWhat would you like to do with the combined graph model?")
        print("1. Accept and continue")
        print("2. Modify the model using natural language commands")
        print("3. Run graph schema validation")

        choices = {
            "1": "accept",
            "2": "modify",
            "3": "validate",
        }
        return self._get_user_input_choice(
            "\nEnter your choice (1-3): ", choices, "accept"
        )

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
        while True:
            try:
                print("\n" + "=" * 60)
                print("INTERACTIVE MODEL MODIFICATION")
                print("=" * 60)
                print("\nDescribe the changes you'd like to make to the graph model.")
                print("You can use natural language like:")
                print(" - Change the label...")
                print(" - Delete a relationship...")
                print(" - Add a new property...")
                print(" - Remove a property...")

                print("\nType 'done' when finished, 'cancel' to return unchanged.")
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
                    print("🤖 Processing your request... " "(this may take a moment)")
                    operations = self._parse_natural_language_to_operations(
                        user_input, model
                    )
                    if operations:
                        print(f"✅ Understood: {operations.reasoning}")

                        # Initialize user operation history if not exists
                        if not self.user_operation_history:
                            self.user_operation_history = UserOperationHistory(
                                self.session_id
                            )

                        # Track user operations before applying them
                        for operation in operations.operations:
                            self.user_operation_history.add_operation(operation)

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
            "- add_node: Add a new node type with specified properties\n"
            "- drop_node: Remove a node type (and related relationships)\n"
            "- add_relationship: Add a new relationship between nodes\n"
            "- add_index: Add an index on a property\n"
            "- drop_index: Remove an index\n"
            "- add_constraint: Add a constraint "
            "(unique, existence, data_type)\n"
            "- drop_constraint: Remove a constraint\n\n"
            "IMPORTANT: When user says 'all' (e.g., 'drop all unique constraints'), "
            "you must identify ALL matching items from the current model context "
            "and create operations for each one."
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

        # Constraints
        if model.node_constraints:
            context_parts.append("\nCONSTRAINTS:")
            for constraint in model.node_constraints:
                labels = " | ".join(constraint.labels or [])
                props = ", ".join(constraint.properties)
                constraint_desc = f"{constraint.type.upper()}: {labels}.{props}"
                context_parts.append(f"  - {constraint_desc}")

        return "\n".join(context_parts)

    def _apply_operations_to_model(
        self, model: "GraphModel", operations: "ModelModifications"
    ) -> "GraphModel":
        """Apply structured operations to the graph model."""
        # Create a deep copy of the model to modify
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
                        modified_model,
                        op.node_label,
                        op.old_property,
                        op.new_property,
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
                elif op.operation_type == "add_constraint":
                    constraint_desc = f"{op.constraint_type.upper()}"
                    if op.constraint_type == "data_type" and op.data_type:
                        constraint_desc += f" ({op.data_type})"
                    print(
                        f"  - Add constraint: {constraint_desc} on "
                        f"{op.node_label}.{op.property_name}"
                    )
                    modified_model = self._apply_add_constraint(
                        modified_model,
                        op.node_label,
                        op.property_name,
                        op.constraint_type,
                        op.data_type,
                    )
                elif op.operation_type == "drop_constraint":
                    constraint_desc = f"{op.constraint_type.upper()}"
                    print(
                        f"  - Drop constraint: {constraint_desc} on "
                        f"{op.node_label}.{op.property_name}"
                    )
                    modified_model = self._apply_drop_constraint(
                        modified_model,
                        op.node_label,
                        op.property_name,
                        op.constraint_type,
                    )
                elif op.operation_type == "add_node":
                    print(f"  - Add node: {op.node_label}")
                    modified_model = self._apply_add_node(
                        modified_model,
                        op.node_label,
                        op.properties,
                        op.source_table,
                    )
                elif op.operation_type == "drop_node":
                    print(f"  - Drop node: {op.node_label}")
                    modified_model = self._apply_drop_node(
                        modified_model, op.node_label
                    )
                elif op.operation_type == "add_relationship":
                    print(
                        f"  - Add relationship: ({op.start_node_label})"
                        f"-[:{op.relationship_name}]->({op.end_node_label})"
                    )
                    modified_model = self._apply_add_relationship(
                        modified_model,
                        op.relationship_name,
                        op.start_node_label,
                        op.end_node_label,
                        op.properties,
                    )

        return modified_model

    def _perform_manual_validation(
        self,
        model: "GraphModel",
        strategy: GraphModelingStrategy,
    ) -> "GraphModel":
        """Run validation on demand without additional user operations."""
        try:
            from .models.operations import ModelModifications
        except ImportError:
            from core.hygm.models.operations import ModelModifications

        dummy_operations = ModelModifications(
            operations=[],
            reasoning="Interactive refinement validation request",
        )

        return self._validate_and_improve_model(
            model=model,
            strategy=strategy,
            operations=dummy_operations,
            database_structure=self.database_structure or {},
            mode="interactive",
        )

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
        self,
        model: "GraphModel",
        node_label: str,
        old_property: str,
        new_property: str,
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

    def _apply_add_constraint(
        self,
        model: "GraphModel",
        node_label: str,
        property_name: str,
        constraint_type: str,
        data_type: str = "",
    ) -> "GraphModel":
        """Apply add constraint operation."""
        from .models.graph_models import GraphConstraint
        from .models.sources import ConstraintSource

        # Check if constraint already exists
        for constraint in model.node_constraints:
            if (
                constraint.labels
                and node_label in constraint.labels
                and property_name in constraint.properties
                and constraint.type == constraint_type
            ):
                return model  # Constraint already exists

        # Create new constraint
        constraint_source = ConstraintSource(
            origin="user_request",
            reason="data_integrity",
            created_by="interactive_modification",
        )
        new_constraint = GraphConstraint(
            type=constraint_type,
            labels=[node_label],
            properties=[property_name],
            data_type=data_type if constraint_type == "data_type" else None,
            source=constraint_source,
        )
        model.node_constraints.append(new_constraint)
        return model

    def _apply_drop_constraint(
        self,
        model: "GraphModel",
        node_label: str,
        property_name: str,
        constraint_type: str,
    ) -> "GraphModel":
        """Apply drop constraint operation."""
        model.node_constraints = [
            constraint
            for constraint in model.node_constraints
            if not (
                constraint.labels
                and node_label in constraint.labels
                and property_name in constraint.properties
                and constraint.type == constraint_type
            )
        ]
        return model

    def _apply_add_node(
        self,
        model: "GraphModel",
        node_label: str,
        properties: List[str],
        source_table: str = "",
    ) -> "GraphModel":
        """Apply add node operation."""
        from .models.graph_models import GraphNode, GraphProperty
        from .models.sources import PropertySource, NodeSource

        # Check if node already exists
        for node in model.nodes:
            if node_label in node.labels:
                print(f"  ⚠️  Node {node_label} already exists, skipping")
                return model

        # Create node source
        node_source = NodeSource(
            origin="user_request",
            table=source_table or node_label.lower(),
            created_by="interactive_modification",
        )

        # Create properties
        node_properties = []
        for prop_name in properties:
            prop_source = PropertySource(
                field=f"{source_table or node_label.lower()}.{prop_name}"
            )
            node_properties.append(GraphProperty(key=prop_name, source=prop_source))

        # Create new node
        new_node = GraphNode(
            labels=[node_label],
            properties=node_properties,
            source=node_source,
        )
        model.nodes.append(new_node)
        return model

    def _apply_drop_node(self, model: "GraphModel", node_label: str) -> "GraphModel":
        """Apply drop node operation."""
        # Remove the node
        model.nodes = [node for node in model.nodes if node_label not in node.labels]

        # Remove relationships involving this node
        model.edges = [
            edge
            for edge in model.edges
            if (
                node_label not in edge.start_node_labels
                and node_label not in edge.end_node_labels
            )
        ]

        # Remove indexes for this node
        model.node_indexes = [
            index
            for index in model.node_indexes
            if not (index.labels and node_label in index.labels)
        ]

        # Remove constraints for this node
        model.node_constraints = [
            constraint
            for constraint in model.node_constraints
            if not (constraint.labels and node_label in constraint.labels)
        ]

        return model

    def _apply_add_relationship(
        self,
        model: "GraphModel",
        relationship_name: str,
        start_node_label: str,
        end_node_label: str,
        properties: List[str],
    ) -> "GraphModel":
        """Apply add relationship operation."""
        from .models.graph_models import GraphEdge, GraphProperty
        from .models.sources import PropertySource, EdgeSource

        # Check if relationship already exists
        for edge in model.edges:
            if (
                edge.edge_type == relationship_name
                and start_node_label in edge.start_node_labels
                and end_node_label in edge.end_node_labels
            ):
                print(
                    f"  ⚠️  Relationship {relationship_name} already exists "
                    f"between {start_node_label} and {end_node_label}, "
                    "skipping"
                )
                return model

        # Create edge source
        edge_source = EdgeSource(
            origin="user_request",
            created_by="interactive_modification",
        )

        # Create properties
        edge_properties = []
        for prop_name in properties:
            prop_source = PropertySource(
                field=f"{relationship_name.lower()}.{prop_name}"
            )
            edge_properties.append(GraphProperty(key=prop_name, source=prop_source))

        # Create new relationship
        new_edge = GraphEdge(
            edge_type=relationship_name,
            start_node_labels=[start_node_label],
            end_node_labels=[end_node_label],
            properties=edge_properties,
            source=edge_source,
        )
        model.edges.append(new_edge)
        return model

    def _validate_and_improve_automatic_llm_model(
        self,
        graph_model: "GraphModel",
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
    ) -> "GraphModel":
        """
        Validate and automatically improve LLM-generated model in automatic mode.

        This method performs validation and gives the LLM an opportunity to fix
        itself based on validation feedback without user interaction.

        Args:
            graph_model: Initial LLM-generated graph model
            database_structure: Original database structure
            domain_context: Optional domain context for modeling

        Returns:
            Final graph model (potentially improved)
        """
        # Create dummy operations for automatic mode
        from .models.operations import ModelModifications

        dummy_operations = ModelModifications(
            operations=[],
            reasoning="Automatic validation improvement iteration",
        )

        return self._validate_and_improve_model(
            model=graph_model,
            strategy=GraphModelingStrategy.LLM_POWERED,
            operations=dummy_operations,
            database_structure=database_structure,
            mode="automatic",
        )

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

    def _validate_and_improve_model(
        self,
        model: "GraphModel",
        strategy: GraphModelingStrategy,
        operations: "ModelModifications",
        database_structure: Dict[str, Any],
        mode: str = "interactive",
    ) -> "GraphModel":
        """
        Unified validation and improvement method for both automatic and interactive modes.

        Args:
            model: Graph model to validate
            strategy: Current modeling strategy
            operations: Operations that were applied (or dummy for automatic mode)
            database_structure: Original database structure
            mode: "automatic" or "interactive" - affects UI and logging behavior

        Returns:
            Final model (potentially improved)
        """
        # Mode-specific initialization
        if mode == "automatic":
            logger.info("Validating and improving LLM model in automatic mode...")
            print("🔍 Performing automatic validation and improvement for LLM model...")
        else:  # interactive mode
            print("\n" + "=" * 60)
            print("GRAPH SCHEMA VALIDATION")
            print("=" * 60)

        if not database_structure:
            error_msg = "❌ Cannot validate: Original database structure not available"
            print(error_msg)
            return model

        current_model = model
        max_improvement_iterations = 3
        improvement_count = 0

        while improvement_count < max_improvement_iterations:
            # Perform validation using the GraphSchemaValidator
            validator = GraphSchemaValidator()
            validation_result = validator.validate(current_model, database_structure)

            # Mode-specific result display
            if mode == "automatic":
                # Compact display for automatic mode
                status_emoji = "✅" if validation_result.success else "❌"
                print(
                    f"\n{status_emoji} Validation iteration {improvement_count + 1}: "
                    f"{'PASSED' if validation_result.success else 'FAILED'}"
                )

                if validation_result.metrics:
                    coverage = validation_result.metrics.coverage_percentage
                    print(f"📊 Coverage: {coverage:.1f}%")

                logger.info(
                    "Validation iteration %d: %s (Coverage: %.1f%%)",
                    improvement_count + 1,
                    "PASSED" if validation_result.success else "FAILED",
                    (
                        validation_result.metrics.coverage_percentage
                        if validation_result.metrics
                        else 0
                    ),
                )
            else:
                # Detailed display for interactive mode
                self._display_validation_results(validation_result, strategy)

            # Handle validation results based on strategy
            if strategy == GraphModelingStrategy.DETERMINISTIC:
                if mode == "interactive":
                    self._handle_deterministic_validation(validation_result)
                break  # No automatic improvement for deterministic
            elif strategy == GraphModelingStrategy.LLM_POWERED:
                if validation_result.success:
                    success_msg = "✅ Graph model validation passed successfully!"
                    if mode == "automatic":
                        logger.info(success_msg)
                        print(success_msg)
                    else:
                        print("\n✅ All validation checks passed!")
                    break
                else:
                    # Mode-specific improvement attempt messaging
                    if mode == "automatic":
                        critical_count = len(validation_result.critical_issues)
                        warning_count = len(validation_result.warnings)
                        print(
                            f"🔧 Found {critical_count} critical issues and "
                            f"{warning_count} warnings"
                        )

                        logger.info(
                            "🤖 Attempting automatic LLM improvement (iteration %d/%d)",
                            improvement_count + 1,
                            max_improvement_iterations,
                        )
                        print(
                            f"🤖 Attempting automatic LLM improvement "
                            f"(iteration {improvement_count + 1}/{max_improvement_iterations})..."
                        )

                    # Try LLM improvement
                    improved_model = self._handle_llm_validation(
                        validation_result, current_model, operations, mode
                    )

                    if improved_model != current_model:
                        # Model was improved, continue with the improved version
                        current_model = improved_model
                        improvement_count += 1

                        if mode == "automatic":
                            print("🔄 Model improved! Re-validating...")
                            logger.info(
                                "🔄 Model improved, re-validating (iteration %d)",
                                improvement_count,
                            )
                        else:
                            # Interactive mode - show detailed iteration info
                            self.iteration_count += 1
                            print(
                                f"\n🔄 ITERATION {self.iteration_count} - IMPROVED MODEL"
                            )
                            self._display_current_model(current_model)

                            if improvement_count < max_improvement_iterations:
                                print(
                                    f"\n🔍 Re-validating improved model "
                                    f"(iteration {improvement_count + 1}/{max_improvement_iterations})..."
                                )
                            else:
                                print(
                                    f"\n⚠️ Reached maximum improvement iterations "
                                    f"({max_improvement_iterations})"
                                )
                                break
                        continue
                    else:
                        # No improvement was made, break the loop
                        if mode == "automatic":
                            logger.warning("❌ LLM could not improve the model further")
                            print("❌ LLM could not improve the model further")
                        else:
                            # Interactive mode - no improvement accepted
                            pass
                        break

            # Check if we should continue (only for automatic mode without improvement)
            if mode == "automatic" and not validation_result.success and not self.llm:
                reason = "LLM not available" if not self.llm else "Unknown issue"
                logger.warning("⚠️ Cannot improve model: %s", reason)
                print(f"⚠️ Cannot improve model: {reason}")
                break

        # Final summary for automatic mode
        if mode == "automatic":
            if improvement_count > 0:
                logger.info(
                    "✨ Automatic LLM improvement completed after %d iterations",
                    improvement_count,
                )
                print(
                    f"✨ Automatic LLM improvement completed after "
                    f"{improvement_count} iterations"
                )
            else:
                logger.info("📊 Using original LLM model (no improvements needed)")
                print("📊 Using original LLM model")

        return current_model

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
        return self._validate_and_improve_model(
            model=model,
            strategy=strategy,
            operations=operations,
            database_structure=self.database_structure or {},
            mode="interactive",
        )

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

    def _handle_llm_error(self, operation: str, error: Exception, fallback_result=None):
        """Handle LLM errors with consistent logging and fallback."""
        logger.error("Error %s: %s", operation, error)
        print(f"❌ Error {operation}: {error}")
        return fallback_result

    def _handle_llm_validation(
        self,
        validation_result,
        model: "GraphModel",
        operations: "ModelModifications",
        mode: str = "interactive",
    ) -> "GraphModel":
        """Handle validation results for LLM strategy with regeneration."""
        if not validation_result.success and self.llm:
            print("\n🤖 LLM STRATEGY: AUTOMATIC MODEL IMPROVEMENT")
            print(
                "The LLM will analyze validation issues and regenerate " "the model..."
            )

            # Preserve user operation history before LLM improvement
            saved_user_operations = None
            if self.user_operation_history and mode == "interactive":
                saved_user_operations = self.user_operation_history.copy()
                print("📋 Preserving your operation history...")

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
                    should_apply = self._should_apply_llm_improvements(
                        improved_model, model, mode
                    )
                    if should_apply:
                        print("✅ Applying LLM improvements...")

                        # Restore user operation history after improvements
                        if saved_user_operations and mode == "interactive":
                            self.user_operation_history = saved_user_operations
                            print("📋 Restored your operation history!")

                        return improved_model
                    else:
                        print("❌ User rejected LLM improvements")

                        # Restore history if improvements were rejected
                        if saved_user_operations and mode == "interactive":
                            self.user_operation_history = saved_user_operations
                            print("📋 Restored your operation history!")

                        return model
                else:
                    print("❌ LLM could not generate improved model")

                    # Restore history if no improvement was generated
                    if saved_user_operations and mode == "interactive":
                        self.user_operation_history = saved_user_operations
                        print("📋 Restored your operation history!")

                    return model
            except Exception as e:
                logger.error("Error getting LLM model improvements: %s", e)
                print(f"❌ Error getting LLM improvements: {e}")

                # Restore user operation history if there was an error
                if saved_user_operations and mode == "interactive":
                    self.user_operation_history = saved_user_operations
                    print("📋 Restored your operation history!")

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
        self,
        current_model: "GraphModel",
        validation_context: str,
        validation_result,
    ) -> Optional["GraphModel"]:
        """Use LLM to regenerate an improved model based on validation."""
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

        # Extract user operation context separately to preserve user changes
        user_context = None
        if self.user_operation_history and self.user_operation_history.operations:
            user_context = self.user_operation_history.to_llm_context()

        try:
            print("🔄 LLM is analyzing validation issues and regenerating model...")

            # Use the LLM strategy but with enhanced context
            # Pass user operations as user_operation_context to ensure preservation
            if isinstance(strategy_instance, LLMStrategy):
                # LLM strategy supports user_operation_context
                improved_model = strategy_instance.create_model(
                    self.database_structure,
                    domain_context=improvement_context,
                    user_operation_context=user_context,
                )
            else:
                # Fallback for strategies that don't support user_operation_context
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
        self,
        current_model: "GraphModel",
        validation_result,
        validation_context: str,
    ) -> str:
        """Prepare comprehensive context for LLM model improvement."""
        context_parts = []

        # User operations must be preserved - add this FIRST
        if self.user_operation_history and self.user_operation_history.operations:
            user_context = self.user_operation_history.to_llm_context()
            if user_context:  # Only add if there's actual content
                context_parts.append(user_context)
                context_parts.append("")

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
        self,
        improved_model: "GraphModel",
        current_model: "GraphModel",
        mode: str = "interactive",
    ) -> bool:
        """Ask user to apply LLM improvements or auto-apply in automatic mode."""

        # In automatic mode, always apply improvements
        if mode == "automatic":
            print("🤖 Automatic mode: Applying LLM improvements...")
            return True

        # In interactive mode, ask the user
        print("\n" + "=" * 60)
        print("🤖 LLM MODEL IMPROVEMENT REVIEW")
        print("=" * 60)

        # Show comparison
        print("\nCURRENT MODEL:")
        print(f"  Nodes: {len(current_model.nodes)}")
        print(f"  Relationships: {len(current_model.edges)}")
        print(f"  Indexes: {len(current_model.node_indexes)}")
        print(f"  Constraints: {len(current_model.node_constraints)}")

        print("\nIMPROVED MODEL:")
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
                        "\nAfter reviewing, would you like to apply these "
                        "improvements?"
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
                print("\n🔄 PROPERTY CHANGES:")
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
                            "Please analyze these validation results and "
                            f"suggest fixes:\n\n{validation_context}\n\n"
                            "Provide specific, actionable recommendations "
                            "to fix the critical issues."
                        ),
                    },
                ]
            )
            return response.content
        except Exception as e:
            logger.error("Error calling LLM for validation fixes: %s", e)
            return None
