#!/usr/bin/env python3
"""
Enhanced SQL to Memgraph Migration Agent Entry Point

This is an enhanced entry point that demonstrates the new modular architecture
with configuration presets and better separation of concerns.

Run with: uv run sql2memgraph_agent_enhanced.py
"""

import logging
import sys
from typing import Dict, Any

from utils import (
    MigrationEnvironmentError,
    DatabaseConnectionError,
    probe_all_connections,
    print_environment_help,
    print_troubleshooting_help,
    MigrationConfig,
    get_available_presets,
    get_preset_config,
    print_config_summary,
)
from main import MySQLToMemgraphAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Print application banner."""
    print("=" * 70)
    print("🚀 Enhanced MySQL to Memgraph Migration Agent")
    print("=" * 70)
    print("Modular architecture with configuration presets and utilities")
    print()


def select_configuration_preset() -> str:
    """
    Allow user to select a configuration preset.

    Returns:
        Selected preset name or 'custom' for manual configuration
    """
    presets = get_available_presets()

    print("Available configuration presets:")
    for i, preset in enumerate(presets, 1):
        preset_config = get_preset_config(preset)
        print(f"  {i}. {preset}")
        mysql_host = preset_config["mysql_host"]
        mysql_port = preset_config["mysql_port"]
        print(f"     MySQL: {mysql_host}:{mysql_port}")
        strategy = preset_config["relationship_naming_strategy"]
        print(f"     Strategy: {strategy}")
        print()

    print(f"  {len(presets) + 1}. custom - Manual configuration")
    print()

    while True:
        try:
            choice = input(
                f"Select preset (1-{len(presets) + 1}) or Enter for custom: "
            ).strip()

            if not choice:
                return "custom"

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(presets):
                return presets[choice_idx]
            elif choice_idx == len(presets):
                return "custom"
            else:
                print(f"Invalid choice. Please select 1-{len(presets) + 1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_migration_strategy() -> str:
    """
    Get user choice for relationship naming strategy.

    Returns:
        Selected strategy name
    """
    strategies = [
        {
            "name": "table_based",
            "description": "Use table names for relationship labels (faster)",
        },
        {
            "name": "llm",
            "description": (
                "Use LLM to generate meaningful relationship " "names (slower)"
            ),
        },
    ]

    print("Relationship naming strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy['name']}: {strategy['description']}")
    print()

    while True:
        try:
            choice = input(
                "Select strategy (1-2) or press Enter for table_based: "
            ).strip()
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


def setup_configuration() -> MigrationConfig:
    """
    Setup configuration using presets or environment variables.

    Returns:
        Configured MigrationConfig instance
    """
    # Select preset
    preset_name = select_configuration_preset()

    # Create base configuration from environment
    config = MigrationConfig.from_environment()

    # Apply preset if selected
    if preset_name != "custom":
        preset_data = get_preset_config(preset_name)
        print(f"📋 Applying '{preset_name}' preset...")

        # Override configuration with preset values
        for key, value in preset_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Get user preferences for migration behavior
    print("\n🎯 Migration preferences:")
    strategy = get_migration_strategy()
    interactive_mode = get_table_selection_mode()

    # Update configuration with user choices
    config.relationship_naming_strategy = strategy
    config.interactive_table_selection = interactive_mode

    return config


def run_migration_with_config(config: MigrationConfig) -> Dict[str, Any]:
    """
    Run the migration with the specified configuration.

    Args:
        config: Migration configuration

    Returns:
        Migration result dictionary
    """
    strategy_name = config.relationship_naming_strategy
    print(f"🔧 Creating migration agent with {strategy_name} strategy")
    mode_text = "interactive" if config.interactive_table_selection else "automatic"
    print(f"📋 Table selection mode: {mode_text}")
    print()

    # Create agent with configuration
    agent = MySQLToMemgraphAgent(
        relationship_naming_strategy=config.relationship_naming_strategy,
        interactive_table_selection=config.interactive_table_selection,
    )

    print("🚀 Starting migration workflow...")
    print("Migration steps:")
    print("  1. 🔍 Analyze MySQL schema and detect join tables")
    print("  2. 🤖 Generate migration plan with LLM")
    print("  3. 📝 Create indexes and constraints automatically")
    print("  4. ⚙️  Generate migration queries using Memgraph migrate module")
    print("  5. 🔄 Execute migration to Memgraph")
    print("  6. ✅ Verify the migration results")
    print()

    # Handle interactive vs automatic mode
    if config.interactive_table_selection:
        print("⚠️  Note: Interactive mode requires manual workflow execution")
        print("   This demo will run in automatic mode for demonstration")
        print()

        # For demo purposes, create a non-interactive agent
        demo_agent = MySQLToMemgraphAgent(
            relationship_naming_strategy=config.relationship_naming_strategy,
            interactive_table_selection=False,
        )
        return demo_agent.migrate(config.to_mysql_config(), config.to_memgraph_config())
    else:
        return agent.migrate(config.to_mysql_config(), config.to_memgraph_config())


def print_migration_results(result: Dict[str, Any]) -> None:
    """
    Print formatted migration results.

    Args:
        result: Migration result dictionary
    """
    print("\n" + "=" * 70)
    print("📊 MIGRATION RESULTS")
    print("=" * 70)

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

    print(f"\n🏁 Final status: {result.get('final_step', 'Unknown')}")
    print("=" * 70)


def main() -> None:
    """Main entry point for the enhanced migration agent."""
    print_banner()

    try:
        # Setup configuration with presets
        print("🔧 Setting up configuration...")
        config = setup_configuration()

        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            print_environment_help()
            sys.exit(1)

        # Show configuration summary
        print_config_summary(config)

        # Setup and validate environment with the configuration
        print("🔌 Testing database connections...")
        mysql_config = config.to_mysql_config()
        memgraph_config = config.to_memgraph_config()
        probe_all_connections(mysql_config, memgraph_config)
        print("✅ All connections verified")
        print()

        # Run migration
        result = run_migration_with_config(config)

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
