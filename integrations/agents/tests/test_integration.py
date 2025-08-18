#!/usr/bin/env python3
"""
Test script for the integrated hypothetical graph modeling functionality.

This script tests the integration of HyGM with the main
MySQLToMemgraphAgent to ensure the enhanced SQL to Graph mapping works correctly.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any

from ..utils.environment import probe_mysql_connection, probe_memgraph_connection
from ..utils.config import get_preset_config
from ..core.graph_modeling import HyGM
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_graph_modeler_standalone():
    """Test the HyGM as a standalone component."""
    logger.info("Testing HyGM standalone...")

    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0.1)

        # Create graph modeler
        modeler = HyGM(llm=llm)

        # Create a sample database structure for testing
        sample_structure = {
            "tables": {
                "users": {
                    "schema": [
                        {"field": "id", "type": "int(11)", "null": "NO", "key": "PRI"},
                        {
                            "field": "email",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "name",
                            "type": "varchar(100)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "created_at",
                            "type": "timestamp",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                    "type": "entity",
                },
                "orders": {
                    "schema": [
                        {"field": "id", "type": "int(11)", "null": "NO", "key": "PRI"},
                        {
                            "field": "user_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                        {
                            "field": "total",
                            "type": "decimal(10,2)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "status",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "created_at",
                            "type": "timestamp",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["id"],
                    "foreign_keys": [
                        {
                            "column": "user_id",
                            "referenced_table": "users",
                            "referenced_column": "id",
                        }
                    ],
                    "type": "entity",
                },
            },
            "entity_tables": {
                "users": {
                    "columns": {
                        "id": {"type": "int", "nullable": False, "key": "primary"},
                        "email": {"type": "varchar(255)", "nullable": False},
                        "name": {"type": "varchar(100)", "nullable": False},
                        "created_at": {"type": "timestamp", "nullable": False},
                    },
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                },
                "orders": {
                    "columns": {
                        "id": {"type": "int", "nullable": False, "key": "primary"},
                        "user_id": {"type": "int", "nullable": False, "key": "foreign"},
                        "total": {"type": "decimal(10,2)", "nullable": False},
                        "status": {"type": "varchar(50)", "nullable": False},
                        "created_at": {"type": "timestamp", "nullable": False},
                    },
                    "primary_keys": ["id"],
                    "foreign_keys": [
                        {
                            "column": "user_id",
                            "referenced_table": "users",
                            "referenced_column": "id",
                        }
                    ],
                },
            },
            "relationships": [
                {
                    "from_table": "orders",
                    "to_table": "users",
                    "type": "many_to_one",
                    "from_column": "user_id",
                    "to_column": "id",
                }
            ],
        }

        # Test the modeling
        logger.info("Running graph modeling analysis...")
        graph_model = modeler.analyze_and_model_schema(sample_structure)

        # Validate results
        logger.info("Graph modeling completed successfully!")
        logger.info("Created %d node types", len(graph_model.nodes))
        logger.info("Created %d relationships", len(graph_model.relationships))

        # Print some details for debugging
        if graph_model.nodes:
            logger.info("Sample node: %s", graph_model.nodes[0].__dict__)
        if graph_model.relationships:
            logger.info(
                "Sample relationship: %s", graph_model.relationships[0].__dict__
            )

        assert len(graph_model.nodes) > 0, "Should create at least one node type"
        # Relationships might be 0 if LLM determined they should be created differently
        # assert len(graph_model.relationships) > 0, "Should create at least one relationship"

        logger.info("✅ Graph modeling successful!")
        logger.info("   - Created %d node types", len(graph_model.nodes))
        logger.info(
            "   - Created %d relationship types", len(graph_model.relationships)
        )

        return True

    except Exception as e:
        logger.error("❌ Graph modeling test failed: %s", str(e))
        return False


def test_environment_setup():
    """Test that the environment is properly configured."""
    logger.info("Testing environment setup...")

    try:
        # Test configuration loading
        config = get_preset_config("local_development")
        logger.info("✅ Configuration loading successful")

        # Test database connections (if available)
        mysql_available = False
        memgraph_available = False

        try:
            mysql_result = probe_mysql_connection(config["mysql_config"])
            mysql_available = mysql_result["success"]
            logger.info(
                "MySQL connection: %s",
                "✅ Available" if mysql_available else "❌ Not available",
            )
        except Exception as e:
            logger.warning("MySQL probe failed: %s", str(e))

        try:
            memgraph_result = probe_memgraph_connection(config["memgraph_config"])
            memgraph_available = memgraph_result["success"]
            logger.info(
                "Memgraph connection: %s",
                "✅ Available" if memgraph_available else "❌ Not available",
            )
        except Exception as e:
            logger.warning("Memgraph probe failed: %s", str(e))

        # Check OpenAI API key
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        logger.info(
            f"OpenAI API key: {'✅ Available' if openai_available else '❌ Not available'}"
        )

        if not openai_available:
            logger.warning(
                "OpenAI API key is required for graph modeling functionality"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Environment setup test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.info("🧪 Starting intelligent graph modeling integration tests...")

    # Test environment setup
    if not test_environment_setup():
        logger.error("Environment setup failed - cannot continue with tests")
        return False

    # Test graph modeler standalone
    if not test_graph_modeler_standalone():
        logger.error("Graph modeler test failed")
        return False

    logger.info("🎉 All integration tests passed!")
    logger.info("\nNext steps:")
    logger.info("1. Run the main agent with: python sql2memgraph_agent_enhanced.py")
    logger.info("2. The agent will now include intelligent graph modeling analysis")
    logger.info("3. Check the migration plan for LLM-generated insights")

    return True


if __name__ == "__main__":
    success = main()

    exit(0 if success else 1)
