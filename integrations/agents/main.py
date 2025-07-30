#!/usr/bin/env python3
"""
MySQL to Memgraph Migration Agent - Main Entry Point

This is the main entry point for the MySQL to Memgraph migration agent.
Run with: uv run main.py
"""

import logging
import sys
from typing import Dict, Any

from utils import (
    MigrationEnvironmentError,
    DatabaseConnectionError,
    setup_and_validate_environment,
    probe_all_connections,
    print_environment_help,
    print_troubleshooting_help,
)
from sql_migration_agent import MySQLToMemgraphAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Print application banner."""
    print("=" * 60)
    print("🚀 MySQL to Memgraph Migration Agent")
    print("=" * 60)
    print("Intelligent database migration with LLM-powered analysis")
    print()


def get_migration_strategy() -> str:
    """
    Get user choice for relationship naming strategy.

    Returns:
        Selected strategy name
    """
    strategies = [
        {
            "name": "table_based",
            "description": "Use table names for relationship labels (default)",
        },
        {
            "name": "llm",
            "description": "Use LLM to generate meaningful relationship names",
        },
    ]

    print("Relationship naming strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy['name']}: {strategy['description']}")
    print()

    while True:
        try:
            choice = input("Select strategy (1-2) or press Enter for default: ").strip()
            if not choice:
                return strategies[0]["name"]  # Default to table_based

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(strategies):
                return strategies[choice_idx]["name"]
            else:
                print("Invalid choice. Please select 1-2.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_table_selection_mode() -> bool:
    """
    Get user choice for table selection mode.

    Returns:
        True for interactive mode, False for automatic mode
    """
    print("Table selection mode:")
    print("  1. Interactive - manually select tables to migrate")
    print("  2. Automatic - migrate all entity tables")
    print()

    while True:
        try:
            choice = input("Select mode (1-2) or press Enter for automatic: ").strip()
            if not choice:
                return False  # Default to automatic mode

            if choice == "1":
                return True
            elif choice == "2":
                return False
            else:
                print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")


def run_migration(
    mysql_config: Dict[str, str],
    memgraph_config: Dict[str, str],
    strategy: str,
    interactive_mode: bool,
) -> Dict[str, Any]:
    """
    Run the migration with the specified configuration.

    Args:
        mysql_config: MySQL connection configuration
        memgraph_config: Memgraph connection configuration
        strategy: Relationship naming strategy
        interactive_mode: Whether to use interactive table selection

    Returns:
        Migration result dictionary
    """
    print(f"🔧 Creating migration agent with {strategy} strategy")
    mode_text = "interactive" if interactive_mode else "automatic"
    print(f"📋 Table selection mode: {mode_text}")
    print()

    # Create agent with selected configuration
    agent = MySQLToMemgraphAgent(
        relationship_naming_strategy=strategy,
        interactive_table_selection=interactive_mode,
    )

    print("🚀 Starting migration workflow...")
    print("This will:")
    print("  1. 🔍 Analyze MySQL schema and detect join tables")
    print("  2. 🤖 Generate migration plan with LLM")
    print("  3. 📝 Create indexes and constraints automatically")
    print("  4. ⚙️  Generate migration queries using Memgraph migrate module")
    print("  5. 🔄 Execute migration to Memgraph")
    print("  6. ✅ Verify the migration results")
    print()

    # Handle interactive vs automatic mode
    if interactive_mode:
        print("⚠️  Note: Interactive mode requires manual workflow execution")
        print("   This demo will run in automatic mode for demonstration")
        print()

        # For demo purposes, create a non-interactive agent
        demo_agent = MySQLToMemgraphAgent(
            relationship_naming_strategy=strategy,
            interactive_table_selection=False,
        )
        return demo_agent.migrate(mysql_config, memgraph_config)
    else:
        return agent.migrate(mysql_config, memgraph_config)


def print_migration_results(result: Dict[str, Any]) -> None:
    """
    Print formatted migration results.

    Args:
        result: Migration result dictionary
    """
    print("\n" + "=" * 60)
    print("📊 MIGRATION RESULTS")
    print("=" * 60)

    if result.get("success", False):
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration encountered errors")

    # Print error details
    if result.get("errors"):
        print(f"\n🚨 Errors ({len(result['errors'])}):")
        for i, error in enumerate(result["errors"], 1):
            print(f"  {i}. {error}")

    # Print completion stats
    completed = len(result.get("completed_tables", []))
    total = result.get("total_tables", 0)
    print(f"\n📋 Tables processed: {completed}/{total}")

    # Print schema analysis details
    if result.get("database_structure"):
        structure = result["database_structure"]
        print("\n🔍 Schema Analysis:")
        print(f"  📁 Entity tables: {len(structure.get('entity_tables', {}))}")
        print(f"  🔗 Join tables: {len(structure.get('join_tables', {}))}")
        print(f"  👁️  Views (excluded): {len(structure.get('views', {}))}")
        print(f"  🔄 Relationships: {len(structure.get('relationships', []))}")

        # Show index/constraint creation results
        if result.get("created_indexes") is not None:
            index_count = len(result.get("created_indexes", []))
            constraint_count = len(result.get("created_constraints", []))
            print(f"  📇 Created indexes: {index_count}")
            print(f"  🔒 Created constraints: {constraint_count}")

        # Show excluded views
        if structure.get("views"):
            print("\n👁️  Excluded view tables:")
            for table_name, table_info in structure["views"].items():
                row_count = table_info.get("row_count", 0)
                print(f"    - {table_name}: {row_count} rows")

        # Show detected join tables
        if structure.get("join_tables"):
            print("\n🔗 Detected join tables:")
            for table_name, table_info in structure["join_tables"].items():
                fk_count = len(table_info.get("foreign_keys", []))
                row_count = table_info.get("row_count", 0)
                print(f"    - {table_name}: {fk_count} FKs, {row_count} rows")

        # Show relationship breakdown
        relationships_by_type = {}
        for rel in structure.get("relationships", []):
            rel_type = rel["type"]
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel)

        if relationships_by_type:
            print("\n🔄 Relationship breakdown:")
            for rel_type, rels in relationships_by_type.items():
                print(f"    - {rel_type}: {len(rels)} relationships")

    print(f"\n🏁 Final status: {result.get('final_step', 'Unknown')}")
    print("=" * 60)


def main() -> None:
    """Main entry point for the migration agent."""
    print_banner()

    try:
        # Setup and validate environment
        print("🔧 Setting up environment...")
        mysql_config, memgraph_config = setup_and_validate_environment()
        print("✅ Environment validation completed")
        print()

        # Probe database connections
        print("🔌 Testing database connections...")
        probe_all_connections(mysql_config, memgraph_config)
        print("✅ All connections verified")
        print()

        # Get user preferences
        strategy = get_migration_strategy()
        interactive_mode = get_table_selection_mode()

        # Run migration
        result = run_migration(
            mysql_config, memgraph_config, strategy, interactive_mode
        )

        # Display results
        print_migration_results(result)

    except MigrationEnvironmentError as e:
        print("\n❌ Environment Setup Error:")
        print(str(e))
        print_environment_help()
        sys.exit(1)

    except DatabaseConnectionError as e:
        print("\n❌ Database Connection Error:")
        print(str(e))
        print_troubleshooting_help()
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Migration cancelled by user")
        sys.exit(0)

    except Exception as e:  # pylint: disable=broad-except
        print(f"\n❌ Unexpected Error: {e}")
        logger.error("Unexpected error in main: %s", e, exc_info=True)
        print_troubleshooting_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
