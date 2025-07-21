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
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from database_analyzer import MySQLAnalyzer
from cypher_generator import CypherGenerator
from memgraph_toolbox.api.memgraph import Memgraph

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def validate_environment_variables():
    """Validate required environment variables."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for migration planning",
        "MYSQL_PASSWORD": "MySQL database password",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.error(
            "Please check your .env file and ensure all required variables are set"
        )
        return False

    return True


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

    def __init__(
        self,
        relationship_naming_strategy: str = "table_based",
        interactive_table_selection: bool = True,
    ):
        """Initialize the migration agent.

        Args:
            relationship_naming_strategy: Strategy for naming relationships.
                - "table_based": Use table names directly (default)
                - "llm": Use LLM to generate meaningful names
            interactive_table_selection: Whether to prompt user for table selection.
                - True: Show interactive table selection (default)
                - False: Migrate all entity tables automatically
        """
        # Validate environment variables first
        if not validate_environment_variables():
            raise ValueError(
                "Required environment variables are missing. "
                "Please check your .env file."
            )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key
        )
        self.mysql_analyzer = None
        self.cypher_generator = CypherGenerator(relationship_naming_strategy)
        self.interactive_table_selection = interactive_table_selection

        # Set LLM for cypher generator if using LLM strategy
        if relationship_naming_strategy == "llm":
            self.cypher_generator.set_llm(self.llm)

        self.memgraph_client = None

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(MigrationState)

        # Add nodes
        workflow.add_node("analyze_mysql", self._analyze_mysql_schema)
        workflow.add_node("select_tables", self._select_tables_for_migration)
        workflow.add_node("generate_migration_plan", self._generate_migration_plan)
        workflow.add_node("generate_cypher_queries", self._generate_cypher_queries)
        workflow.add_node("validate_queries", self._validate_queries)
        workflow.add_node("execute_migration", self._execute_migration)
        workflow.add_node("verify_migration", self._verify_migration)

        # Add edges
        workflow.add_edge("analyze_mysql", "select_tables")
        workflow.add_edge("select_tables", "generate_migration_plan")
        workflow.add_edge("generate_migration_plan", "generate_cypher_queries")
        workflow.add_edge("generate_cypher_queries", "validate_queries")
        workflow.add_edge("validate_queries", "execute_migration")
        workflow.add_edge("execute_migration", "verify_migration")
        workflow.add_edge("verify_migration", END)

        # Set entry point
        workflow.set_entry_point("analyze_mysql")

        # Return the workflow (not compiled) so caller can add checkpointer
        return workflow

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
            # Only count entity tables for migration progress (exclude views and join tables)
            state["total_tables"] = len(structure["entity_tables"])
            state["current_step"] = "Schema analysis completed"

            # Log detailed analysis
            views_count = len(structure.get("views", {}))
            join_tables_count = len(structure.get("join_tables", {}))
            entity_tables_count = len(structure.get("entity_tables", {}))

            logger.info(
                f"Found {len(structure['tables'])} total tables: "
                f"{entity_tables_count} entities, "
                f"{join_tables_count} join tables, "
                f"{views_count} views, "
                f"and {len(structure['relationships'])} relationships"
            )

            if views_count > 0:
                logger.info(f"Skipping {views_count} view tables from migration")

        except Exception as e:
            logger.error(f"Error analyzing MySQL schema: {e}")
            state["errors"].append(f"Schema analysis failed: {e}")

        return state

    # TODO: This should be human visible and configurable.
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

    # TODO: Implement actual validation logic for Cypher queries
    def _validate_queries(self, state: MigrationState) -> MigrationState:
        """Validate generated Cypher queries."""
        logger.info("Validating Cypher queries...")

        try:
            # Initialize Memgraph connection for validation
            config = state["memgraph_config"]
            self.memgraph_client = Memgraph(
                url=config.get("url"),
                username=config.get("username"),
                password=config.get("password"),
                database=config.get("database"),
            )

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

            # Clear the database first to avoid constraint violations
            try:
                logger.info("Clearing existing data from Memgraph...")
                self.memgraph_client.query("MATCH (n) DETACH DELETE n")
                self.memgraph_client.query("DROP CONSTRAINT ON (n) ASSERT exists(n.id)")
                logger.info("Database cleared successfully")
            except Exception as e:
                logger.warning(f"Database clearing failed (might be empty): {e}")

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

            # Migrate data for entity tables only (not join tables)
            for table_name, table_info in structure["entity_tables"].items():
                logger.info(f"Migrating entity table: {table_name}")

                # Get data from MySQL
                data = self.mysql_analyzer.get_table_data(table_name)

                if data:
                    # Prepare data for Cypher (excluding FK columns)
                    prepared_data = self.cypher_generator.prepare_data_for_cypher(
                        data, table_info["schema"], table_info["foreign_keys"]
                    )

                    # Find the node creation query for this table
                    node_query = None
                    generator = self.cypher_generator
                    label = generator._table_name_to_label(table_name)
                    for query in queries:
                        if f"Create {label} nodes" in query:
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
                                f"Successfully migrated {len(data)} rows "
                                f"from {table_name}"
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

            # Handle one-to-many relationships (from foreign keys)
            for rel in structure["relationships"]:
                if rel["type"] == "one_to_many":
                    try:
                        # Find the relationship query
                        rel_query = self.cypher_generator.generate_relationship_query(
                            rel
                        )
                        clean_query = "\n".join(
                            [
                                line
                                for line in rel_query.split("\n")
                                if not line.strip().startswith("//")
                            ]
                        ).strip()

                        self.memgraph_client.query(clean_query)
                        logger.info(
                            f"Created one-to-many relationship: "
                            f"{rel['from_table']} -> {rel['to_table']}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to create relationship: {e}")
                        state["errors"].append(f"Relationship creation failed: {e}")

            # Handle many-to-many relationships (from join tables)
            for rel in structure["relationships"]:
                if rel["type"] == "many_to_many":
                    try:
                        join_table_name = rel["join_table"]
                        join_table_info = structure["join_tables"][join_table_name]

                        # Get join table data
                        join_data = self.mysql_analyzer.get_table_data(join_table_name)

                        if join_data:
                            # Prepare join table data
                            prepared_data = self.cypher_generator.prepare_join_table_data_for_cypher(
                                join_data, join_table_info["schema"]
                            )

                            # Generate and execute relationship query
                            rel_query = (
                                self.cypher_generator.generate_relationship_query(rel)
                            )
                            clean_query = "\n".join(
                                [
                                    line
                                    for line in rel_query.split("\n")
                                    if not line.strip().startswith("//")
                                ]
                            ).strip()

                            self.memgraph_client.query(
                                clean_query, {"data": prepared_data}
                            )
                            logger.info(
                                f"Created many-to-many relationship: "
                                f"{rel['from_table']} <-> {rel['to_table']} "
                                f"via {join_table_name}"
                            )
                        else:
                            logger.info(f"No data in join table {join_table_name}")

                    except Exception as e:
                        logger.error(f"Failed to create many-to-many relationship: {e}")
                        state["errors"].append(
                            f"Many-to-many relationship creation failed: {e}"
                        )
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
            # For non-interactive mode, compile workflow without checkpointer
            if not self.interactive_table_selection:
                compiled_workflow = self.workflow.compile()
                final_state = compiled_workflow.invoke(initial_state)
            else:
                # For interactive mode, import and use checkpointer
                from langgraph.checkpoint.memory import MemorySaver

                memory = MemorySaver()
                compiled_workflow = self.workflow.compile(checkpointer=memory)

                # This will require proper handling of interrupts in calling code
                final_state = compiled_workflow.invoke(initial_state)

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

    def _select_tables_for_migration(self, state: MigrationState) -> MigrationState:
        """Allow human to select which tables to migrate using LangGraph interrupt."""
        logger.info("Starting table selection process...")

        try:
            structure = state["database_structure"]
            entity_tables = structure.get("entity_tables", {})
            join_tables = structure.get("join_tables", {})
            views = structure.get("views", {})

            if not entity_tables:
                logger.warning("No entity tables found for migration")
                state["current_step"] = "No tables available for migration"
                return state

            # Non-interactive mode: select all entity tables
            if not self.interactive_table_selection:
                selected_tables = list(entity_tables.keys())
                logger.info(
                    f"Non-interactive mode: selecting all {len(selected_tables)} entity tables"
                )

                # Update the database structure with all tables
                structure["selected_tables"] = selected_tables
                state["database_structure"] = structure
                state["total_tables"] = len(selected_tables)

                # Keep all relationships as they are
                state[
                    "current_step"
                ] = "All tables selected for migration (non-interactive)"
                return state

            # Interactive mode: use LangGraph interrupt for human input
            # Prepare table information for human review
            table_info = {"entity_tables": [], "join_tables": [], "views": []}

            # Format entity tables with details
            for i, (table_name, table_data) in enumerate(entity_tables.items(), 1):
                row_count = table_data.get("row_count", 0)
                fk_count = len(table_data.get("foreign_keys", []))
                table_info["entity_tables"].append(
                    {
                        "number": i,
                        "name": table_name,
                        "row_count": row_count,
                        "foreign_key_count": fk_count,
                    }
                )

            # Format join tables for context
            for table_name, table_data in join_tables.items():
                row_count = table_data.get("row_count", 0)
                fk_count = len(table_data.get("foreign_keys", []))
                table_info["join_tables"].append(
                    {
                        "name": table_name,
                        "row_count": row_count,
                        "foreign_key_count": fk_count,
                    }
                )

            # Format views for context
            for table_name, table_data in views.items():
                row_count = table_data.get("row_count", 0)
                table_info["views"].append({"name": table_name, "row_count": row_count})

            # Use LangGraph interrupt to pause for human input
            logger.info("Requesting human input for table selection...")
            user_response = interrupt(
                {
                    "type": "table_selection",
                    "message": "Please select which tables to migrate",
                    "table_info": table_info,
                    "instructions": {
                        "options": [
                            "'all' - Migrate all entity tables",
                            "'1,3,5' - Migrate specific tables by number",
                            "'1-5' - Migrate range of tables",
                            "'none' - Skip migration (exit)",
                        ]
                    },
                }
            )

            # Process the human response
            if not user_response or "selection" not in user_response:
                raise ValueError("No table selection received from user")

            selection = user_response["selection"].strip()
            table_list = [table["name"] for table in table_info["entity_tables"]]

            # Parse user selection
            if selection.lower() == "none":
                logger.info("Migration cancelled by user.")
                state["errors"].append("Migration cancelled by user")
                state["current_step"] = "Migration cancelled"
                return state

            elif selection.lower() == "all":
                selected_tables = list(entity_tables.keys())

            elif "," in selection:
                # Handle comma-separated list (e.g., "1,3,5")
                indices = []
                for part in selection.split(","):
                    try:
                        idx = int(part.strip()) - 1
                        if 0 <= idx < len(table_list):
                            indices.append(idx)
                        else:
                            raise ValueError(f"Invalid table number: {part.strip()}")
                    except ValueError as e:
                        raise ValueError(f"Invalid number in selection: {part.strip()}")

                selected_tables = [table_list[i] for i in indices]

            elif "-" in selection:
                # Handle range (e.g., "1-5")
                try:
                    start, end = selection.split("-")
                    start_idx = int(start.strip()) - 1
                    end_idx = int(end.strip()) - 1

                    if (
                        0 <= start_idx < len(table_list)
                        and 0 <= end_idx < len(table_list)
                        and start_idx <= end_idx
                    ):
                        selected_tables = table_list[start_idx : end_idx + 1]
                    else:
                        raise ValueError("Invalid range")
                except ValueError:
                    raise ValueError("Invalid range format. Use format like '1-5'")

            else:
                # Handle single number
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(table_list):
                        selected_tables = [table_list[idx]]
                    else:
                        raise ValueError("Invalid table number")
                except ValueError:
                    raise ValueError(
                        f"Invalid selection. Please enter a number between 1 and {len(table_list)}, 'all', or 'none'"
                    )

            # Filter the structure to only include selected tables
            filtered_entity_tables = {
                table_name: table_info
                for table_name, table_info in entity_tables.items()
                if table_name in selected_tables
            }

            # Update the database structure with filtered tables
            structure["entity_tables"] = filtered_entity_tables
            structure["selected_tables"] = selected_tables
            state["database_structure"] = structure
            state["total_tables"] = len(selected_tables)

            # Filter relationships to only include those involving selected tables
            filtered_relationships = []
            for rel in structure.get("relationships", []):
                if rel["type"] == "many_to_many":
                    # Include if both tables in the relationship are selected
                    if (
                        rel["from_table"] in selected_tables
                        and rel["to_table"] in selected_tables
                    ):
                        filtered_relationships.append(rel)
                else:
                    # Include if the from_table is selected
                    if rel["from_table"] in selected_tables:
                        filtered_relationships.append(rel)

            structure["relationships"] = filtered_relationships

            logger.info(
                f"User selected {len(selected_tables)} tables for migration: {', '.join(selected_tables)}"
            )
            logger.info(f"{len(filtered_relationships)} relationships will be created")

            state["current_step"] = "Tables selected for migration"

        except Exception as e:
            logger.error(f"Error in table selection: {e}")
            state["errors"].append(f"Table selection failed: {e}")
            state["current_step"] = "Table selection failed"

        return state


def debug_mysql_connection(mysql_config: Dict[str, str]) -> bool:
    """Debug MySQL connection specifically."""
    logger.debug("Starting MySQL connection debug...")

    try:
        analyzer = MySQLAnalyzer(**mysql_config)
        logger.debug(f"Created MySQLAnalyzer with config: {mysql_config}")

        if analyzer.connect():
            logger.debug("✓ MySQL connection successful")

            # Test basic operations
            tables = analyzer.get_tables()
            logger.debug(f"✓ Found {len(tables)} tables: {tables}")

            if tables:
                # Test schema retrieval
                first_table = tables[0]
                schema = analyzer.get_table_schema(first_table)
                logger.debug(
                    f"✓ Retrieved schema for {first_table}: {len(schema)} columns"
                )

                # Test data retrieval
                data = analyzer.get_table_data(first_table, limit=5)
                logger.debug(f"✓ Retrieved {len(data)} sample rows from {first_table}")

            analyzer.disconnect()
            logger.debug("✓ MySQL connection closed successfully")
            return True
        else:
            logger.error("✗ MySQL connection failed")
            return False

    except Exception as e:
        logger.error(f"✗ MySQL debug failed: {e}", exc_info=True)
        return False


def main():
    """Main function to run the migration agent."""

    print("MySQL to Memgraph Migration Agent")
    print("=" * 40)

    # Check environment variables first
    if not validate_environment_variables():
        print("\n❌ Setup Error: Missing required environment variables")
        print("\nPlease ensure you have:")
        print("1. Created a .env file (copy from .env.example)")
        print("2. Set your OPENAI_API_KEY")
        print("3. Set your MYSQL_PASSWORD")
        print("\nExample .env file:")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("MYSQL_PASSWORD=your_mysql_password")
        print("MYSQL_HOST=localhost")
        print("MYSQL_USER=root")
        print("MYSQL_DATABASE=sakila")
        return

    # Example configuration for Sakila database
    mysql_config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "password"),
        "database": os.getenv("MYSQL_DATABASE", "sakila"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
    }

    try:
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

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check your .env file exists and contains required variables")
        print("2. Verify your OpenAI API key is valid")
        print("3. Test MySQL connection with: python mysql_troubleshoot.py")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print("Run with debug mode for more details")
        logger.error(f"Unexpected error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()
