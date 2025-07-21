#!/usr/bin/env python3
"""
Enhanced MySQL to Memgraph Migration Agent Example

This example demonstrates the new features:
1. Foreign keys converted to relationships (FK columns removed from nodes)
2. Join tables converted to many-to-many relationships
3. Configurable relationship naming strategies
"""

import os
import logging
from main import MySQLToMemgraphAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to demonstrate the enhanced migration agent."""

    # Configuration for different relationship naming strategies
    examples = [
        {
            "name": "Table-Based Naming Strategy",
            "strategy": "table_based",
            "description": "Uses table names for relationship labels (default)",
        },
        {
            "name": "LLM-Based Naming Strategy",
            "strategy": "llm",
            "description": "Uses LLM to generate meaningful relationship names",
        },
    ]

    print("Enhanced MySQL to Memgraph Migration Agent")
    print("=" * 50)
    print()

    # Display available strategies
    print("Available relationship naming strategies:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}: {example['description']}")
    print()

    # Get user choice for relationship naming strategy
    while True:
        try:
            choice = input("Select strategy (1-2) or 'q' to quit: ").strip()
            if choice.lower() == "q":
                print("Goodbye!")
                return

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(examples):
                selected_strategy = examples[choice_idx]
                break
            else:
                print("Invalid choice. Please select 1-2.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

    print(f"\nUsing {selected_strategy['name']}")
    print("-" * 30)

    # Get user choice for table selection mode
    print("\nTable selection mode:")
    print("1. Interactive mode - manually select tables to migrate")
    print("2. Automatic mode - migrate all entity tables")
    print()

    while True:
        try:
            mode_choice = input("Select mode (1-2): ").strip()
            if mode_choice == "1":
                interactive_mode = True
                print("Using interactive table selection mode")
                break
            elif mode_choice == "2":
                interactive_mode = False
                print("Using automatic table selection mode")
                break
            else:
                print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")

    print(f"\nUsing {selected_strategy['name']}")
    print("-" * 30)

    # MySQL configuration (from environment or defaults)
    mysql_config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "sakila"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
    }

    # Memgraph configuration
    memgraph_config = {
        "url": os.getenv("MEMGRAPH_URL", "bolt://localhost:7687"),
        "username": os.getenv("MEMGRAPH_USERNAME", ""),
        "password": os.getenv("MEMGRAPH_PASSWORD", ""),
        "database": os.getenv("MEMGRAPH_DATABASE", "memgraph"),
    }

    try:
        # Create agent with selected strategy and table selection mode
        agent = MySQLToMemgraphAgent(
            relationship_naming_strategy=selected_strategy["strategy"],
            interactive_table_selection=interactive_mode,
        )

        mode_desc = "interactive" if interactive_mode else "automatic"
        print(f"Created migration agent with {selected_strategy['strategy']} strategy")
        print(f"Table selection mode: {mode_desc}")

        # Define migration state
        initial_state = {
            "mysql_config": mysql_config,
            "memgraph_config": memgraph_config,
            "database_structure": {},
            "migration_queries": [],
            "migration_plan": "",
            "current_step": "Initializing",
            "errors": [],
            "completed_tables": [],
            "total_tables": 0,
        }

        print("\nStarting migration workflow...")
        print("This will:")
        print("1. Analyze MySQL schema and detect join tables")
        print("2. Generate migration plan with LLM")
        print("3. Create Cypher queries with enhanced relationship handling")
        print("4. Execute migration to Memgraph")
        print("5. Verify the migration results")

        # Run the migration workflow
        if interactive_mode:
            # For interactive mode, we need to handle the workflow differently
            # since it uses interrupts that require special handling
            print("\nNote: Interactive mode requires manual workflow execution")
            print("Please use demo_langgraph_interrupt.py for full interactive demo")

            # For now, run in non-interactive mode to show the results
            print("Running in non-interactive mode for demonstration...")
            agent_demo = MySQLToMemgraphAgent(
                relationship_naming_strategy=selected_strategy["strategy"],
                interactive_table_selection=False,
            )
            compiled_workflow = agent_demo.workflow.compile()
            result = compiled_workflow.invoke(initial_state)
        else:
            # Non-interactive mode: compile and run normally
            compiled_workflow = agent.workflow.compile()
            result = compiled_workflow.invoke(initial_state)

        # Display results
        print("\n" + "=" * 50)
        print("MIGRATION RESULTS")
        print("=" * 50)

        if result["errors"]:
            print("❌ Errors encountered:")
            for error in result["errors"]:
                print(f"  - {error}")
        else:
            print("✅ Migration completed successfully!")

        print(f"\nCompleted tables: {len(result['completed_tables'])}")
        print(f"Total tables: {result['total_tables']}")

        if result.get("database_structure"):
            structure = result["database_structure"]
            print(f"\nSchema Analysis:")
            print(f"  - Entity tables: {len(structure.get('entity_tables', {}))}")
            print(f"  - Join tables: {len(structure.get('join_tables', {}))}")
            print(f"  - Views (excluded): {len(structure.get('views', {}))}")
            print(f"  - Relationships: {len(structure.get('relationships', []))}")

            # Show views that were excluded
            if structure.get("views"):
                print(f"\n👁️ Excluded view tables:")
                for table_name, table_info in structure["views"].items():
                    row_count = table_info.get("row_count", 0)
                    print(f"  - {table_name}: {row_count} rows (view)")

            # Show join tables that were detected
            if structure.get("join_tables"):
                print(f"\n🔗 Detected join tables:")
                for table_name, table_info in structure["join_tables"].items():
                    fk_count = len(table_info.get("foreign_keys", []))
                    row_count = table_info.get("row_count", 0)
                    print(f"  - {table_name}: {fk_count} FKs, {row_count} rows")

            # Show relationship types
            relationships_by_type = {}
            for rel in structure.get("relationships", []):
                rel_type = rel["type"]
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                relationships_by_type[rel_type].append(rel)

            print(f"\n🔗 Relationship breakdown:")
            for rel_type, rels in relationships_by_type.items():
                print(f"  - {rel_type}: {len(rels)} relationships")

        print(f"\nFinal status: {result['current_step']}")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logging.error(f"Migration error: {e}", exc_info=True)


def demo_relationship_naming():
    """Demonstrate different relationship naming strategies."""
    from cypher_generator import CypherGenerator

    print("\n" + "=" * 50)
    print("RELATIONSHIP NAMING DEMO")
    print("=" * 50)

    # Sample relationship data
    sample_relationships = [
        ("customer", "order"),
        ("film", "actor"),
        ("user", "role"),
        ("product", "category"),
        ("order", "order_item"),
    ]

    strategies = ["table_based"]

    for strategy in strategies:
        print(f"\n{strategy.upper()} STRATEGY:")
        print("-" * 20)

        generator = CypherGenerator(strategy)

        for from_table, to_table in sample_relationships:
            rel_name = generator._generate_relationship_type(from_table, to_table)
            print(f"  {from_table} -> {to_table}: {rel_name}")


if __name__ == "__main__":
    main()

    # Optionally run the naming demo
    run_demo = input("\nWould you like to see the relationship naming demo? (y/n): ")
    if run_demo.lower().startswith("y"):
        demo_relationship_naming()
