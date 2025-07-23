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
            "description": ("Uses table names for relationship labels " "(default)"),
        },
        {
            "name": "LLM-Based Naming Strategy",
            "strategy": "llm",
            "description": ("Uses LLM to generate meaningful relationship " "names"),
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
        "host": os.getenv("MYSQL_HOST", "host.docker.internal"),
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
        print(
            f"Created migration agent with " f"{selected_strategy['strategy']} strategy"
        )
        print(f"Table selection mode: {mode_desc}")

        # Define migration state

        print("\nStarting migration workflow...")
        print("This will:")
        print("1. Analyze MySQL schema and detect join tables")
        print("2. Generate migration plan with LLM")
        print("3. Create indexes and constraints automatically")
        print("4. Generate migration queries using Memgraph migrate module")
        print("5. Execute migration to Memgraph")
        print("6. Verify the migration results")

        # Run the migration workflow
        if interactive_mode:
            # Interactive mode requires manual workflow execution
            print("\nNote: Interactive mode requires manual workflow " "execution")
            print("Please use demo_langgraph_interrupt.py for full " "interactive demo")

            # Run in non-interactive mode for demonstration
            print("Running in non-interactive mode for demonstration...")
            agent_demo = MySQLToMemgraphAgent(
                relationship_naming_strategy=selected_strategy["strategy"],
                interactive_table_selection=False,
            )
            result = agent_demo.migrate(mysql_config, memgraph_config)
        else:
            # Non-interactive mode: run normally
            result = agent.migrate(mysql_config, memgraph_config)

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

        # Show index creation results
        if result.get("created_indexes") is not None:
            index_count = len(result.get("created_indexes", []))
            constraint_count = len(result.get("created_constraints", []))
            print(f"Created indexes: {index_count}")
            print(f"Created constraints: {constraint_count}")

        if result.get("database_structure"):
            structure = result["database_structure"]
            print("\nSchema Analysis:")
            print("  - Entity tables: " f"{len(structure.get('entity_tables', {}))}")
            print(f"  - Join tables: " f"{len(structure.get('join_tables', {}))}")
            print("  - Views (excluded): " f"{len(structure.get('views', {}))}")
            print("  - Relationships: " f"{len(structure.get('relationships', []))}")

            # Show views that were excluded
            if structure.get("views"):
                print("\n👁️ Excluded view tables:")
                for table_name, table_info in structure["views"].items():
                    row_count = table_info.get("row_count", 0)
                    print(f"  - {table_name}: {row_count} rows (view)")

            # Show join tables that were detected
            if structure.get("join_tables"):
                print("\n🔗 Detected join tables:")
                for table_name, table_info in structure["join_tables"].items():
                    fk_count = len(table_info.get("foreign_keys", []))
                    row_count = table_info.get("row_count", 0)
                    print(f"  - {table_name}: {fk_count} FKs, " f"{row_count} rows")

            # Show relationship types
            relationships_by_type = {}
            for rel in structure.get("relationships", []):
                rel_type = rel["type"]
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                relationships_by_type[rel_type].append(rel)

            print("\n🔗 Relationship breakdown:")
            for rel_type, rels in relationships_by_type.items():
                print(f"  - {rel_type}: {len(rels)} relationships")

        print(f"\nFinal status: {result['current_step']}")

    except (ValueError, ConnectionError, RuntimeError) as e:
        print(f"❌ Migration failed: {e}")
        logging.error("Migration error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
