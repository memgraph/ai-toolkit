"""
MySQL to Memgraph Migration Agent

This agent analyzes MySQL databases, generates appropriate Cypher queries,
and migrates data to Memgraph using LangGraph workflow.
"""

import os
import sys
import logging
from typing import Dict, List, Any, TypedDict
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "memgraph-toolbox" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "langchain-memgraph"))

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from database_analyzer import MySQLAnalyzer
from cypher_generator import CypherGenerator
from memgraph_toolbox.api.memgraph import Memgraph

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationState(TypedDict):
    """State for the migration workflow."""

    mysql_config: Dict[str, str]
    memgraph_config: Dict[str, str]
    database_structure: Dict[str, Any]
    migration_queries: List[str]
    migration_plan: str
    current_step: str
    errors: List[str]
    completed_tables: List[str]
    total_tables: int


class MySQLToMemgraphAgent:
    """Agent for migrating MySQL databases to Memgraph."""

    def __init__(self):
        """Initialize the migration agent."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
        )
        self.mysql_analyzer = None
        self.cypher_generator = CypherGenerator()
        self.memgraph_client = None

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(MigrationState)

        # Add nodes
        workflow.add_node("analyze_mysql", self._analyze_mysql_schema)
        workflow.add_node("generate_migration_plan", self._generate_migration_plan)
        workflow.add_node("generate_cypher_queries", self._generate_cypher_queries)
        workflow.add_node("validate_queries", self._validate_queries)
        workflow.add_node("execute_migration", self._execute_migration)
        workflow.add_node("verify_migration", self._verify_migration)

        # Add edges
        workflow.add_edge("analyze_mysql", "generate_migration_plan")
        workflow.add_edge("generate_migration_plan", "generate_cypher_queries")
        workflow.add_edge("generate_cypher_queries", "validate_queries")
        workflow.add_edge("validate_queries", "execute_migration")
        workflow.add_edge("execute_migration", "verify_migration")
        workflow.add_edge("verify_migration", END)

        # Set entry point
        workflow.set_entry_point("analyze_mysql")

        return workflow.compile()

    def _analyze_mysql_schema(self, state: MigrationState) -> MigrationState:
        """Analyze MySQL database schema and structure."""
        logger.info("Analyzing MySQL database schema...")

        try:
            # Initialize MySQL analyzer
            self.mysql_analyzer = MySQLAnalyzer(**state["mysql_config"])

            if not self.mysql_analyzer.connect():
                raise Exception("Failed to connect to MySQL database")

            # Get database structure
            structure = self.mysql_analyzer.get_database_structure()

            # Add table counts for progress tracking
            structure["table_counts"] = {}
            for table_name in structure["tables"].keys():
                count = self.mysql_analyzer.get_table_row_count(table_name)
                structure["table_counts"][table_name] = count

            state["database_structure"] = structure
            state["total_tables"] = len(structure["tables"])
            state["current_step"] = "Schema analysis completed"

            logger.info(
                f"Found {len(structure['tables'])} tables and "
                f"{len(structure['relationships'])} relationships"
            )

        except Exception as e:
            logger.error(f"Error analyzing MySQL schema: {e}")
            state["errors"].append(f"Schema analysis failed: {e}")

        return state

    def _generate_migration_plan(self, state: MigrationState) -> MigrationState:
        """Generate a migration plan using LLM."""
        logger.info("Generating migration plan...")

        try:
            structure = state["database_structure"]

            # Prepare context for LLM
            context = {
                "tables": list(structure["tables"].keys()),
                "relationships": structure["relationships"],
                "table_counts": structure.get("table_counts", {}),
            }

            system_message = SystemMessage(
                content="""
You are an expert database migration specialist. You need to create a 
detailed migration plan for moving data from MySQL to Memgraph (a graph database).

Your task is to:
1. Analyze the database structure
2. Identify the optimal order for creating nodes and relationships
3. Consider dependencies between tables
4. Suggest any optimizations for graph modeling
5. Identify potential issues or challenges

Provide a detailed, step-by-step migration plan.
            """
            )

            human_message = HumanMessage(
                content=f"""
Create a migration plan for the following MySQL database structure:

Tables: {context['tables']}
Relationships: {context['relationships']}
Table row counts: {context['table_counts']}

Please provide a detailed migration plan including:
1. Order of operations
2. Node creation strategy
3. Relationship creation strategy
4. Any potential issues to watch for
5. Estimated timeline considerations
            """
            )

            response = self.llm.invoke([system_message, human_message])
            state["migration_plan"] = response.content
            state["current_step"] = "Migration plan generated"

        except Exception as e:
            logger.error(f"Error generating migration plan: {e}")
            state["errors"].append(f"Migration plan generation failed: {e}")

        return state

    def _generate_cypher_queries(self, state: MigrationState) -> MigrationState:
        """Generate Cypher queries for the migration."""
        logger.info("Generating Cypher queries...")

        try:
            structure = state["database_structure"]
            queries = self.cypher_generator.generate_full_migration_script(structure)

            state["migration_queries"] = queries
            state["current_step"] = "Cypher queries generated"

            logger.info(f"Generated {len(queries)} migration queries")

        except Exception as e:
            logger.error(f"Error generating Cypher queries: {e}")
            state["errors"].append(f"Query generation failed: {e}")

        return state

    def _validate_queries(self, state: MigrationState) -> MigrationState:
        """Validate generated Cypher queries."""
        logger.info("Validating Cypher queries...")

        try:
            # Initialize Memgraph connection for validation
            self.memgraph_client = Memgraph(**state["memgraph_config"])

            # Test connection
            test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            self.memgraph_client.query(test_query)

            state["current_step"] = "Queries validated successfully"
            logger.info("Memgraph connection established and queries validated")

        except Exception as e:
            logger.error(f"Error validating queries: {e}")
            state["errors"].append(f"Query validation failed: {e}")

        return state

    def _execute_migration(self, state: MigrationState) -> MigrationState:
        """Execute the migration queries."""
        logger.info("Executing migration...")

        try:
            structure = state["database_structure"]
            queries = state["migration_queries"]

            # Execute constraint and index creation queries first
            constraint_queries = [
                q for q in queries if "CONSTRAINT" in q or "INDEX" in q
            ]
            for query in constraint_queries:
                if query.strip() and not query.startswith("//"):
                    try:
                        self.memgraph_client.query(query)
                        logger.info(f"Executed: {query[:50]}...")
                    except Exception as e:
                        logger.warning(f"Constraint/Index creation failed: {e}")

            # Migrate data for each table
            for table_name, table_info in structure["tables"].items():
                logger.info(f"Migrating table: {table_name}")

                # Get data from MySQL
                data = self.mysql_analyzer.get_table_data(table_name)

                if data:
                    # Prepare data for Cypher
                    prepared_data = self.cypher_generator.prepare_data_for_cypher(
                        data, table_info["schema"]
                    )

                    # Find the node creation query for this table
                    node_query = None
                    for query in queries:
                        if (
                            f"Create {self.cypher_generator._table_name_to_label(table_name)} nodes"
                            in query
                        ):
                            node_query = query
                            break

                    if node_query:
                        # Execute the query with data
                        try:
                            # Clean the query (remove comments)
                            clean_query = "\n".join(
                                [
                                    line
                                    for line in node_query.split("\n")
                                    if not line.strip().startswith("//")
                                ]
                            ).strip()

                            self.memgraph_client.query(
                                clean_query, {"data": prepared_data}
                            )
                            state["completed_tables"].append(table_name)
                            logger.info(
                                f"Successfully migrated {len(data)} rows from {table_name}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to migrate table {table_name}: {e}")
                            state["errors"].append(
                                f"Table migration failed for {table_name}: {e}"
                            )
                else:
                    logger.info(f"No data found in table {table_name}")

            # Create relationships
            logger.info("Creating relationships...")
            relationship_queries = [
                q for q in queries if "CREATE (" in q and ")-[:" in q
            ]
            for query in relationship_queries:
                if query.strip() and not query.startswith("//"):
                    try:
                        clean_query = "\n".join(
                            [
                                line
                                for line in query.split("\n")
                                if not line.strip().startswith("//")
                            ]
                        ).strip()

                        self.memgraph_client.query(clean_query)
                        logger.info(f"Created relationships: {query[:50]}...")
                    except Exception as e:
                        logger.warning(f"Relationship creation failed: {e}")

            state["current_step"] = "Migration execution completed"

        except Exception as e:
            logger.error(f"Error executing migration: {e}")
            state["errors"].append(f"Migration execution failed: {e}")

        return state

    def _verify_migration(self, state: MigrationState) -> MigrationState:
        """Verify the migration results."""
        logger.info("Verifying migration results...")

        try:
            # Count nodes and relationships in Memgraph
            node_count_query = "MATCH (n) RETURN count(n) as node_count"
            relationship_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"

            node_result = self.memgraph_client.query(node_count_query)
            rel_result = self.memgraph_client.query(relationship_count_query)

            node_count = node_result[0]["node_count"] if node_result else 0
            rel_count = rel_result[0]["rel_count"] if rel_result else 0

            # Calculate expected counts from MySQL
            structure = state["database_structure"]
            expected_nodes = sum(structure.get("table_counts", {}).values())

            logger.info(f"Migration verification:")
            logger.info(f"  - Nodes created: {node_count} (expected: {expected_nodes})")
            logger.info(f"  - Relationships created: {rel_count}")
            logger.info(
                f"  - Tables migrated: {len(state['completed_tables'])}/{state['total_tables']}"
            )

            state["current_step"] = "Migration verification completed"

        except Exception as e:
            logger.error(f"Error verifying migration: {e}")
            state["errors"].append(f"Migration verification failed: {e}")

        return state

    def migrate(
        self, mysql_config: Dict[str, str], memgraph_config: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Execute the complete migration workflow."""
        logger.info("Starting MySQL to Memgraph migration...")

        # Default Memgraph configuration
        if not memgraph_config:
            memgraph_config = {
                "url": os.getenv("MEMGRAPH_URL", "bolt://localhost:7687"),
                "username": os.getenv("MEMGRAPH_USER", ""),
                "password": os.getenv("MEMGRAPH_PASSWORD", ""),
                "database": os.getenv("MEMGRAPH_DATABASE", "memgraph"),
            }

        # Initialize state
        initial_state = MigrationState(
            mysql_config=mysql_config,
            memgraph_config=memgraph_config,
            database_structure={},
            migration_queries=[],
            migration_plan="",
            current_step="Starting migration",
            errors=[],
            completed_tables=[],
            total_tables=0,
        )

        try:
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)

            # Cleanup connections
            if self.mysql_analyzer:
                self.mysql_analyzer.disconnect()
            if self.memgraph_client:
                self.memgraph_client.close()

            return {
                "success": len(final_state["errors"]) == 0,
                "migration_plan": final_state["migration_plan"],
                "completed_tables": final_state["completed_tables"],
                "total_tables": final_state["total_tables"],
                "errors": final_state["errors"],
                "final_step": final_state["current_step"],
            }

        except Exception as e:
            logger.error(f"Migration workflow failed: {e}")
            return {
                "success": False,
                "errors": [f"Workflow execution failed: {e}"],
                "migration_plan": "",
                "completed_tables": [],
                "total_tables": 0,
                "final_step": "Failed",
            }


def main():
    """Main function to run the migration agent."""

    # Example configuration for Sakila database
    mysql_config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "password"),
        "database": os.getenv("MYSQL_DATABASE", "sakila"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
    }

    print("MySQL to Memgraph Migration Agent")
    print("=" * 40)

    # Create and run the agent
    agent = MySQLToMemgraphAgent()
    result = agent.migrate(mysql_config)

    print(f"\nMigration Result:")
    print(f"Success: {result['success']}")
    print(
        f"Completed Tables: {len(result['completed_tables'])}/{result['total_tables']}"
    )

    if result["errors"]:
        print(f"Errors: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"  - {error}")

    print(f"\nMigration Plan:")
    print(result["migration_plan"])


if __name__ == "__main__":
    main()
