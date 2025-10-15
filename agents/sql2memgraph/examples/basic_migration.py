#!/usr/bin/env python3
"""
Basic SQL to Memgraph migration example.

This example demonstrates how to use the migration agent to migrate
a SQL database to Memgraph with minimal configuration.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path to import from the agents package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from core import SQLToMemgraphAgent  # noqa: E402
from utils import setup_and_validate_environment  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_basic_migration():
    """Run a basic migration with automatic graph modeling."""
    print("🚀 Basic SQL to Memgraph Migration Example")
    print("=" * 50)

    try:
        # Setup and validate environment
        source_db_config, memgraph_config = setup_and_validate_environment()

        # Create migration agent in automatic mode (non-interactive)
        agent = SQLToMemgraphAgent(interactive_graph_modeling=False)

        # Run migration
        print("Starting automatic migration...")
        result = agent.migrate(source_db_config, memgraph_config)

        # Display results
        if result["success"]:
            print("✅ Migration completed successfully!")
            print(f"📊 Tables migrated: {result['completed_tables']}")
            print(f"📈 Total tables: {result['total_tables']}")
        else:
            print("❌ Migration failed!")
            print(f"🔍 Errors: {result['errors']}")

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Migration example failed: %s", e)
        print(f"❌ Example failed: {e}")


if __name__ == "__main__":
    run_basic_migration()
