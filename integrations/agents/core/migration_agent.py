"""
SQL Database to Memgraph Migration Agent

This agent analyzes SQL databases, generates appropriate Cypher queries,
and migrates data to Memgraph using LangGraph workflow.
"""

import os
import sys
import logging
from time import sleep
from typing import Dict, List, Any, TypedDict, Optional
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "memgraph-toolbox" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "langchain-memgraph"))
sys.path.append(str(Path(__file__).parent.parent))  # Add agents root to path

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from query_generation.cypher_generator import CypherGenerator
from core.hygm import HyGM, ModelingMode, GraphModelingStrategy
from core.hygm.validation import validate_memgraph_data
from memgraph_toolbox.api.memgraph import Memgraph
from database.factory import DatabaseAnalyzerFactory

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MigrationState(TypedDict):
    """State for the migration workflow."""

    source_db_config: Dict[str, str]
    memgraph_config: Dict[str, str]
    database_structure: Dict[str, Any]
    graph_model: Any  # HyGM GraphModel object
    migration_queries: List[str]
    current_step: str
    errors: List[str]
    completed_tables: List[str]
    total_tables: int
    created_indexes: List[str]
    created_constraints: List[str]
    validation_report: Dict[str, Any]  # Post-migration validation results


class SQLToMemgraphAgent:
    """Agent for migrating SQL databases to Memgraph."""

    def __init__(
        self,
        modeling_mode: ModelingMode = ModelingMode.AUTOMATIC,
        graph_modeling_strategy: GraphModelingStrategy = GraphModelingStrategy.DETERMINISTIC,
    ):
        """Initialize the migration agent.

        Args:
            modeling_mode: Graph modeling mode
                - AUTOMATIC: Generate graph model automatically (default)
                - INTERACTIVE: Allow user to modify and refine graph model
            graph_modeling_strategy: Strategy for graph model creation
                - DETERMINISTIC: Rule-based graph creation (default)
                - LLM_POWERED: LLM generates the graph model
        """

        openai_api_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key
        )
        self.database_analyzer = None
        self.cypher_generator = CypherGenerator()
        self.modeling_mode = modeling_mode
        self.graph_modeling_strategy = graph_modeling_strategy

        self.memgraph_client = None

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _get_db_config_for_migrate(self, db_config: Dict[str, str]) -> str:
        """
        Convert database config for use with migrate module in Memgraph.

        Adjusts localhost/127.0.0.1 to host.docker.internal for Docker.
        """
        migrate_host = db_config["host"]
        if migrate_host == "localhost" or migrate_host == "127.0.0.1":
            migrate_host = "host.docker.internal"

        return f"""{{
            user: '{db_config['user']}',
            password: '{db_config['password']}',
            host: '{migrate_host}',
            database: '{db_config['database']}'
        }}"""

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with improved separation of concerns."""
        workflow = StateGraph(MigrationState)

        # Add nodes - refactored for better modularity
        workflow.add_node(
            "connect_and_analyze_schema", self._connect_and_analyze_schema
        )
        workflow.add_node("create_graph_model", self._create_graph_model)
        workflow.add_node("create_indexes", self._create_indexes)
        workflow.add_node("generate_cypher_queries", self._generate_cypher_queries)
        workflow.add_node("prepare_target_database", self._prepare_target_database)
        workflow.add_node("execute_data_migration", self._execute_data_migration)
        workflow.add_node("validate_post_migration", self._validate_post_migration)

        # Add conditional edges for better error handling
        workflow.add_edge("connect_and_analyze_schema", "create_graph_model")
        workflow.add_edge("create_graph_model", "prepare_target_database")
        workflow.add_edge("prepare_target_database", "create_indexes")
        workflow.add_edge("create_indexes", "generate_cypher_queries")
        workflow.add_edge("generate_cypher_queries", "execute_data_migration")
        workflow.add_edge("execute_data_migration", "validate_post_migration")
        workflow.add_edge("validate_post_migration", END)

        # Set entry point
        workflow.set_entry_point("connect_and_analyze_schema")

        # Return the workflow (not compiled) so caller can add checkpointer
        return workflow

    def _connect_and_analyze_schema(self, state: MigrationState) -> MigrationState:
        """Connect to source database and analyze schema structure."""
        logger.info("Connecting to source database and analyzing schema...")

        try:
            # Initialize database analyzer using factory
            database_analyzer = DatabaseAnalyzerFactory.create_analyzer(
                database_type="mysql", **state["source_db_config"]
            )

            if not database_analyzer.connect():
                raise Exception("Failed to connect to source database")

            # Get standardized database structure
            db_structure = database_analyzer.get_database_structure()

            # Use the built-in HyGM format conversion
            hygm_data = db_structure.to_hygm_format()

            # Store the database structure for next steps
            state["database_structure"] = hygm_data

            # Only count entity tables for migration progress
            state["total_tables"] = len(hygm_data["entity_tables"])

            # Automatically select all entity tables for migration
            entity_tables = hygm_data.get("entity_tables", {})
            if entity_tables:
                selected_tables = list(entity_tables.keys())
                logger.info(
                    f"Automatically selecting all {len(selected_tables)} "
                    f"entity tables for migration"
                )
                hygm_data["selected_tables"] = selected_tables
                state["database_structure"] = hygm_data
            else:
                logger.warning("No entity tables found for migration")
                state["errors"].append("No entity tables available for migration")

            state["current_step"] = "Schema analysis completed"

            # Log detailed analysis
            views_count = len(hygm_data.get("views", {}))
            join_tables_count = len(hygm_data.get("join_tables", {}))
            entity_tables_count = len(hygm_data.get("entity_tables", {}))

            logger.info(
                f"Found {len(hygm_data['tables'])} total tables: "
                f"{entity_tables_count} entities, "
                f"{join_tables_count} join tables, "
                f"{views_count} views, "
                f"and {len(hygm_data['relationships'])} relationships"
            )

            if views_count > 0:
                logger.info(f"Skipping {views_count} view tables from migration")

        except Exception as e:
            logger.error(f"Error analyzing database schema: {e}")
            state["errors"].append(f"Schema analysis failed: {e}")

        return state

    def _create_graph_model(self, state: MigrationState) -> MigrationState:
        """Create graph model using HyGM based on analyzed schema."""
        logger.info("Creating graph model using HyGM...")

        try:
            hygm_data = state["database_structure"]

            # Log the modeling mode being used
            if self.modeling_mode == ModelingMode.INTERACTIVE:
                logger.info("Using interactive graph modeling mode")
            else:
                logger.info("Using automatic graph modeling mode")

            # Create graph modeler with strategy and mode
            graph_modeler = HyGM(
                llm=self.llm,
                mode=self.modeling_mode,
                strategy=self.graph_modeling_strategy,
            )

            # Log the strategy being used
            strategy_name = self.graph_modeling_strategy.value
            logger.info(f"Using {strategy_name} graph modeling strategy")

            # Generate graph model using new unified interface
            graph_model = graph_modeler.create_graph_model(
                hygm_data, domain_context="Database migration to graph database"
            )

            # Store the graph model in state
            state["graph_model"] = graph_model

            logger.info(
                f"Graph model created with {len(graph_model.nodes)} "
                f"node types and {len(graph_model.edges)} "
                f"relationship types"
            )

            state["current_step"] = "Graph model created successfully"

        except Exception as e:
            logger.error(f"Graph modeling failed: {e}")
            # HyGM is required - propagate the error
            return self._handle_step_error(state, "creating graph model", e)

        return state

    def _validate_graph_model(self, state: MigrationState) -> MigrationState:
        """Validate the created graph model."""
        logger.info("Validating graph model...")

        # Validate that we have a graph model
        if not state.get("graph_model"):
            logger.error("No graph model found - this should not happen")
            state["errors"].append("Graph modeling failed to produce a model")
            state["current_step"] = "Graph model validation failed"
            return state

        # Perform validation
        try:
            from core.hygm import HyGM

            # Store graph model for later use in query generation
            self._current_graph_model = state["graph_model"]

            validator = HyGM(llm=self.llm)
            validation_result = validator.validate_graph_model(
                state["graph_model"], state["database_structure"]
            )

            if not validation_result["is_valid"]:
                logger.warning("Graph model has validation issues:")
                for issue in validation_result["issues"]:
                    logger.warning(f"- {issue}")
                state["errors"].extend(validation_result["issues"])

            if validation_result["warnings"]:
                logger.info("Graph model validation warnings:")
                for warning in validation_result["warnings"]:
                    logger.info(f"- {warning}")

            state[
                "current_step"
            ] = f"Graph model validated - {validation_result['summary']}"

        except Exception as e:
            logger.error(f"Error validating graph model: {e}")
            state["errors"].append(f"Graph model validation failed: {e}")
            state["current_step"] = "Graph model validation failed"

        return state

    def _prepare_target_database(self, state: MigrationState) -> MigrationState:
        """Prepare the target Memgraph database for migration."""
        logger.info("Preparing target database for migration...")

        try:
            # Initialize Memgraph connection
            config = state["memgraph_config"]
            self.memgraph_client = Memgraph(
                url=config.get("url"),
                username=config.get("username"),
                password=config.get("password"),
                database=config.get("database"),
            )

            # Test Memgraph connection
            test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            self.memgraph_client.query(test_query)
            logger.info("Memgraph connection established successfully")

            # Clear the database first to avoid constraint violations
            try:
                logger.info("Clearing existing data from Memgraph...")
                self.memgraph_client.query("STORAGE MODE IN_MEMORY_ANALYTICAL;")
                sleep(1)
                self.memgraph_client.query("DROP GRAPH")
                sleep(5)
                self.memgraph_client.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL;")
                sleep(1)
                logger.info("Database cleared successfully")
            except Exception as e:
                logger.warning(f"Database clearing failed (might be empty): {e}")

            state["current_step"] = "Target database prepared successfully"

        except Exception as e:
            logger.error(f"Error preparing target database: {e}")
            state["errors"].append(f"Database preparation failed: {e}")
            state["current_step"] = "Database preparation failed"

        return state

    def _execute_data_migration(self, state: MigrationState) -> MigrationState:
        """Execute the actual data migration queries."""
        logger.info("Executing data migration...")

        try:
            queries = state["migration_queries"]

            # Execute all migration queries sequentially
            successful_queries = 0
            for i, query in enumerate(queries):
                # Skip empty queries, but don't skip queries that contain comments
                query_lines = [line.strip() for line in query.strip().split("\n")]
                non_comment_lines = [
                    line for line in query_lines if line and not line.startswith("//")
                ]

                if non_comment_lines:  # Has actual Cypher code
                    try:
                        logger.info(f"Executing query {i+1}/{len(queries)}...")
                        self.memgraph_client.query(query)
                        successful_queries += 1

                        # Log progress for node creation queries
                        if "CREATE (n:" in query:
                            # Extract table name from query comment or FROM clause
                            table_name = None
                            if "FROM " in query:
                                try:
                                    from_part = query.split("FROM ")[1]
                                    table_name = from_part.split()[0].rstrip(",")
                                except (IndexError, AttributeError):
                                    pass

                            if table_name:
                                logger.info(
                                    f"Successfully migrated data from table: "
                                    f"{table_name}"
                                )
                                # Update completed tables list
                                if table_name not in state["completed_tables"]:
                                    state["completed_tables"].append(table_name)
                        elif "CREATE (" in query and "-[:" in query:
                            logger.info("Successfully created relationships")

                    except Exception as e:
                        logger.error(f"Failed to execute query {i+1}: {e}")
                        logger.error(f"Query: {query[:100]}...")
                        state["errors"].append(f"Query execution failed: {e}")

            logger.info(
                f"Migration completed: {successful_queries}/{len(queries)} "
                f"queries executed successfully"
            )
            state["current_step"] = "Data migration completed"

        except Exception as e:
            logger.error(f"Error executing data migration: {e}")
            state["errors"].append(f"Data migration failed: {e}")

        return state

    def _execute_queries_with_logging(
        self,
        queries: List[str],
        query_type: str,
        memgraph_client: Memgraph,
        success_list: List[str],
        warning_prefix: str = "warning",
    ) -> None:
        """Execute queries with consistent logging and error handling."""
        for query in queries:
            try:
                logger.info(f"Creating {query_type}: %s", query)
                memgraph_client.query(query)
                success_list.append(query)
            except Exception as e:
                # Some queries might already exist, log but continue
                logger.warning(
                    f"{query_type.capitalize()} creation {warning_prefix}: %s", e
                )

    def _handle_step_error(
        self,
        state: MigrationState,
        step_name: str,
        error: Exception,
    ) -> MigrationState:
        """Standardized error handling for workflow steps."""
        error_msg = f"Error {step_name}: {error}"
        failure_msg = f"{step_name.capitalize()} failed: {error}"

        logger.error(error_msg)
        state["errors"].append(failure_msg)
        state["current_step"] = f"{step_name.capitalize()} failed"

        return state

    def _create_indexes(self, state: MigrationState) -> MigrationState:
        """Create indexes and constraints in Memgraph before migration."""
        logger.info("Creating indexes and constraints from HyGM graph model...")

        try:
            # Use the existing Memgraph connection from prepare_target_database
            if not self.memgraph_client:
                raise Exception("No Memgraph connection available")

            # Track created indexes and constraints
            created_indexes = []
            created_constraints = []

            # Get the HyGM graph model (required)
            graph_model = state.get("graph_model")
            if not graph_model or not hasattr(graph_model, "node_indexes"):
                raise Exception("HyGM graph model with indexes is required")

            logger.info("Using HyGM-provided indexes and constraints")

            # Generate index queries from HyGM graph model
            index_queries = self.cypher_generator.generate_index_queries_from_hygm(
                graph_model.node_indexes
            )

            # Generate constraint queries from HyGM graph model
            constraint_queries = (
                self.cypher_generator.generate_constraint_queries_from_hygm(
                    graph_model.node_constraints
                )
            )

            logger.info(
                "HyGM provided %d indexes and %d constraints",
                len(index_queries),
                len(constraint_queries),
            )

            # Execute constraint queries first
            self._execute_queries_with_logging(
                constraint_queries,
                "constraint",
                self.memgraph_client,
                created_constraints,
            )

            # Execute index queries
            self._execute_queries_with_logging(
                index_queries, "index", self.memgraph_client, created_indexes
            )

            # Store results in state
            state["created_indexes"] = created_indexes
            state["created_constraints"] = created_constraints
            state["current_step"] = "HyGM indexes and constraints created"

            logger.info(
                "Created %d constraints and %d indexes from HyGM model",
                len(created_constraints),
                len(created_indexes),
            )

        except Exception as e:
            return self._handle_step_error(state, "creating indexes", e)

        return state

    def _generate_cypher_queries(self, state: MigrationState) -> MigrationState:
        """Generate Cypher queries based on HyGM graph model recommendations."""
        logger.info("Generating Cypher queries based on HyGM graph model...")

        try:
            hygm_data = state["database_structure"]
            source_db_config = state["source_db_config"]

            # Get the HyGM graph model (required)
            graph_model = state.get("graph_model")
            if not graph_model:
                raise Exception("HyGM graph model is required for migration")

            # Store graph model in instance for use by helper methods
            self._current_graph_model = graph_model
            queries = []

            # Create database connection config for migrate module
            db_config_str = self._get_db_config_for_migrate(source_db_config)

            # Generate node creation queries based on HyGM recommendations
            logger.info(
                f"Creating nodes based on {len(graph_model.nodes)} HyGM node definitions"
            )

            for node_def in graph_model.nodes:
                source_table = node_def.source.name if node_def.source else "unknown"
                node_label = node_def.primary_label

                # Extract property names from HyGM GraphProperty objects
                properties = [
                    prop.key if hasattr(prop, "key") else str(prop)
                    for prop in node_def.properties
                ]

                # Use validated properties from HyGM graph model
                if properties:
                    properties_str = ", ".join(properties)
                    node_query = f"""
// Create {node_label} nodes from {source_table} table (HyGM optimized)
CALL migrate.mysql('SELECT {properties_str} FROM {source_table}',
                   {db_config_str})
YIELD row
CREATE (n:{node_label})
SET n += row;"""
                    queries.append(node_query)
                    logger.info(
                        f"Added node creation for {node_label} with "
                        f"{len(properties)} properties"
                    )
                else:
                    logger.warning(
                        f"No properties found for node {node_label} "
                        f"from table {source_table}"
                    )

            # Generate relationship creation queries based on HyGM recommendations
            logger.info(
                f"Creating relationships based on {len(graph_model.edges)} HyGM relationship definitions"
            )

            for rel_def in graph_model.edges:
                rel_query = self._generate_hygm_relationship_query(
                    rel_def, hygm_data, db_config_str
                )
                if rel_query:
                    queries.append(rel_query)
                    logger.info(f"Added relationship creation: {rel_def.edge_type}")

            state["migration_queries"] = queries
            state["current_step"] = "Cypher queries generated using HyGM graph model"

            logger.info(
                f"Generated {len(queries)} migration queries based on HyGM recommendations"
            )

        except Exception as e:
            logger.error(f"Error generating HyGM-based Cypher queries: {e}")
            return self._handle_step_error(state, "generating cypher queries", e)

        return state

    def _generate_hygm_relationship_query(
        self, rel_def, hygm_data: Dict[str, Any], mysql_config_str: str
    ) -> str:
        """Generate relationship query based on HyGM relationship definition."""

        try:
            rel_name = rel_def.edge_type
            from_node = (
                rel_def.start_node_labels[0] if rel_def.start_node_labels else ""
            )
            to_node = rel_def.end_node_labels[0] if rel_def.end_node_labels else ""
            source_info = rel_def.source.mapping if rel_def.source else {}

            # Determine relationship type based on source information
            # If we have start_node and end_node mapping, it's likely one_to_many (foreign key)
            # If we have a junction table type, it's many_to_many
            if rel_def.source and rel_def.source.type == "junction_table":
                rel_type = "many_to_many"
            elif "start_node" in source_info and "end_node" in source_info:
                rel_type = "one_to_many"
            else:
                # Try to infer relationship type from mapping
                if source_info.get("join_table"):
                    rel_type = "many_to_many"
                else:
                    # HyGM should provide relationship type
                    raise Exception(
                        f"HyGM must specify relationship type for {rel_name}"
                    )

            # Find the corresponding node labels from HyGM model
            from_label = from_node
            to_label = to_node

            # Try to get actual labels from the graph model if available
            if hasattr(self, "_current_graph_model") and self._current_graph_model:
                for node in self._current_graph_model.nodes:
                    if node.primary_label.lower() == from_node.lower():
                        from_label = node.primary_label
                    if node.primary_label.lower() == to_node.lower():
                        to_label = node.primary_label

            if not from_label or not to_label:
                logger.warning(
                    "Could not find node labels for relationship %s", rel_name
                )
                return ""

            if rel_type == "one_to_many":
                return self._generate_one_to_many_hygm_query(
                    rel_name,
                    from_label,
                    to_label,
                    source_info,
                    mysql_config_str,
                    hygm_data,
                )
            elif rel_type == "many_to_many":
                return self._generate_many_to_many_hygm_query(
                    rel_name,
                    from_label,
                    to_label,
                    source_info,
                    mysql_config_str,
                    hygm_data,
                )
            else:
                logger.warning("Unsupported relationship type: %s", rel_type)
                return ""

        except Exception as e:
            logger.error(
                "Error generating relationship query for %s: %s", rel_def.edge_type, e
            )
            return ""

    def _generate_one_to_many_hygm_query(
        self,
        rel_name: str,
        from_label: str,
        to_label: str,
        source_info: Dict[str, Any],
        mysql_config_str: str,
        hygm_data: Dict[str, Any],
    ) -> str:
        """Generate one-to-many relationship query using HyGM information."""

        # Extract table and column information from the mapping
        start_node = source_info.get("start_node", "")  # e.g., "address.city_id"
        end_node = source_info.get("end_node", "")  # e.g., "city.city_id"

        if not start_node or not end_node:
            logger.error("Missing relationship information for %s", rel_name)
            raise Exception(
                f"HyGM must provide complete relationship mapping for {rel_name}"
            )

        # Parse the table.column format
        try:
            from_table, fk_column = start_node.split(".", 1)
            to_table, to_column = end_node.split(".", 1)
            logger.info(
                "Parsed relationship mapping for %s: from_table=%s, fk_column=%s, to_table=%s, to_column=%s",
                rel_name,
                from_table,
                fk_column,
                to_table,
                to_column,
            )
        except ValueError:
            logger.error("Invalid mapping format for %s: %s", rel_name, source_info)
            raise Exception(
                f"HyGM must provide valid relationship mapping for {rel_name}"
            )

        # Get primary key from source table
        from_table_info = hygm_data.get("entity_tables", {}).get(from_table, {})
        primary_keys = from_table_info.get("primary_keys", [])

        if not primary_keys:
            logger.error("Could not determine primary key for table %s", from_table)
            raise Exception(
                f"HyGM must provide complete table information with primary keys for {from_table}"
            )

        # Use the first primary key (most common case)
        from_pk = primary_keys[0]

        # Debug the actual query components
        logger.info(
            "Generating query for %s: from_pk=%s, fk_column=%s, from_table=%s, to_column=%s, to_table=%s",
            rel_name,
            from_pk,
            fk_column,
            from_table,
            to_column,
            to_table,
        )

        query = f"""
// Create {rel_name} relationships (HyGM: {from_label} -> {to_label})
CALL migrate.mysql('SELECT {from_pk}, {fk_column} FROM {from_table} WHERE {fk_column} IS NOT NULL', {mysql_config_str})
YIELD row
MATCH (from_node:{from_label} {{{from_pk}: row.{from_pk}}})
MATCH (to_node:{to_label} {{{to_column}: row.{fk_column}}})
CREATE (from_node)-[:{rel_name}]->(to_node);"""

        logger.info("Generated query for %s: %s", rel_name, query[:200] + "...")
        return query

    def _find_table_for_label(self, label: str, hygm_data: Dict[str, Any]) -> str:
        """Find the database table that corresponds to a graph label."""
        # First, try to find the table using the graph model source information
        if hasattr(self, "_current_graph_model") and self._current_graph_model:
            for node in self._current_graph_model.nodes:
                # Check if this node has the label we're looking for
                if label in node.labels and node.source and node.source.name:
                    return node.source.name

        # Use HyGM data for table mapping if no graph model source
        entity_tables = hygm_data.get("entity_tables", {})

        # Direct match
        if label.lower() in entity_tables:
            return label.lower()

        # Try pluralized version
        plural_label = label.lower() + "s"
        if plural_label in entity_tables:
            return plural_label

        # Try without 's' (singularize)
        if label.lower().endswith("s"):
            singular_label = label.lower()[:-1]
            if singular_label in entity_tables:
                return singular_label

        # Try case variations
        for table_name in entity_tables.keys():
            if table_name.lower() == label.lower():
                return table_name

        return ""

    def _generate_many_to_many_hygm_query(
        self,
        rel_name: str,
        from_label: str,
        to_label: str,
        source_info: Dict[str, Any],
        mysql_config_str: str,
        hygm_data: Dict[str, Any],
    ) -> str:
        """Generate many-to-many relationship query using HyGM information."""

        # Extract junction table information from the mapping
        join_table = source_info.get("join_table")
        from_table = source_info.get("from_table")
        to_table = source_info.get("to_table")
        from_fk = source_info.get("join_from_column")  # FK in junction table
        to_fk = source_info.get("join_to_column")  # FK in junction table
        from_pk = source_info.get("from_column")  # PK in source table
        to_pk = source_info.get("to_column")  # PK in target table

        if not all([join_table, from_table, to_table, from_fk, to_fk, from_pk, to_pk]):
            logger.error(
                "Missing many-to-many relationship information for %s", rel_name
            )
            raise Exception(
                f"HyGM must provide complete many-to-many mapping for {rel_name}"
            )

        query = f"""
// Create {rel_name} relationships via {join_table}
// (HyGM: {from_label} <-> {to_label})
CALL migrate.mysql('SELECT {from_fk}, {to_fk} FROM {join_table}', {mysql_config_str})
YIELD row
MATCH (from:{from_label} {{{from_pk}: row.{from_fk}}})
MATCH (to:{to_label} {{{to_pk}: row.{to_fk}}})
CREATE (from)-[:{rel_name}]->(to);"""
        return query

    def _validate_post_migration(self, state: MigrationState) -> MigrationState:
        """Validate post-migration results using HyGM schema comparison."""
        logger.info("Running post-migration validation...")

        try:
            # Check if we have a graph model to validate against
            if not state.get("graph_model"):
                logger.warning("No graph model available for validation")
                state["validation_report"] = {
                    "success": False,
                    "reason": "No graph model available",
                }
                state["current_step"] = "Post-migration validation skipped"
                return state

            # Reuse existing Memgraph connection from previous steps
            if not self.memgraph_client:
                logger.error("No Memgraph connection available for validation")
                state["validation_report"] = {
                    "success": False,
                    "reason": "No Memgraph connection available",
                }
                state["current_step"] = "Post-migration validation failed"
                return state

            # Get the graph model from state
            graph_model = state.get("graph_model")

            # Calculate expected data counts from MySQL for validation
            structure = state["database_structure"]
            expected_nodes = 0
            selected_tables = structure.get("selected_tables", [])
            table_counts = structure.get("table_counts", {})

            for table_name in selected_tables:
                if table_name in table_counts:
                    expected_nodes += table_counts[table_name]

            # Create expected data counts for the validator
            expected_data_counts = {
                "nodes": expected_nodes,
                "selected_tables": selected_tables,
            }

            # Run post-migration validation using existing connection with data counts
            logger.info("Executing post-migration validation...")
            validation_result = validate_memgraph_data(
                expected_model=graph_model,
                memgraph_connection=self.memgraph_client,
                expected_data_counts=expected_data_counts,
                detailed_report=True,
            )

            # Store validation results in state
            state["validation_report"] = {
                "success": validation_result.success,
                "summary": validation_result.summary,
                "validation_score": validation_result.details.get(
                    "validation_score", 0
                ),
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "category": issue.category,
                        "message": issue.message,
                        "expected": issue.expected,
                        "actual": issue.actual,
                        "recommendation": issue.recommendation,
                    }
                    for issue in validation_result.issues
                ],
                "metrics": validation_result.metrics,
            }

            # Log validation summary
            if validation_result.success:
                logger.info("✅ Post-migration validation PASSED")
                score = int(validation_result.details.get("validation_score", 0))
                logger.info(f"Validation score: {score}/100")
            else:
                logger.warning("⚠️ Post-migration validation found issues")
                score = int(validation_result.details.get("validation_score", 0))
                logger.warning(f"Validation score: {score}/100")

                # Log critical issues
                critical_issues = [
                    issue
                    for issue in validation_result.issues
                    if issue.severity.value == "CRITICAL"
                ]
                if critical_issues:
                    count = len(critical_issues)
                    logger.error(f"Found {count} critical validation issues:")
                    # Show first 3 critical issues
                    for issue in critical_issues[:3]:
                        logger.error(f"  - {issue.message}")

            state["current_step"] = "Post-migration validation completed"

        except Exception as e:
            logger.error(f"Error during post-migration validation: {e}")
            state["errors"].append(f"Post-migration validation failed: {e}")
            state["validation_report"] = {
                "validation_performed": False,
                "reason": f"Validation error: {e}",
            }
            state["current_step"] = "Post-migration validation failed"

        return state

    def migrate(
        self,
        source_db_config: Dict[str, str],
        memgraph_config: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute the complete migration workflow."""
        logger.info("Starting SQL database to Memgraph migration...")

        # Initialize state
        initial_state = MigrationState(
            source_db_config=source_db_config,
            memgraph_config=memgraph_config,
            database_structure={},
            graph_model=None,
            migration_queries=[],
            current_step="Starting migration",
            errors=[],
            completed_tables=[],
            total_tables=0,
            created_indexes=[],
            created_constraints=[],
            validation_report={},
        )

        try:
            # For automatic graph modeling mode, compile workflow without checkpointer
            if self.modeling_mode == ModelingMode.AUTOMATIC:
                compiled_workflow = self.workflow.compile()
                final_state = compiled_workflow.invoke(initial_state)
            else:
                # For interactive graph modeling mode, import and use checkpointer
                from langgraph.checkpoint.memory import MemorySaver

                memory = MemorySaver()
                compiled_workflow = self.workflow.compile(checkpointer=memory)

                # Provide required configuration for checkpointer
                config = {"configurable": {"thread_id": "migration_thread_1"}}
                final_state = compiled_workflow.invoke(initial_state, config=config)

            # Cleanup connections
            if self.database_analyzer:
                self.database_analyzer.disconnect()
            if self.memgraph_client:
                self.memgraph_client.close()

            return {
                "success": len(final_state["errors"]) == 0,
                "completed_tables": final_state["completed_tables"],
                "total_tables": final_state["total_tables"],
                "errors": final_state["errors"],
                "final_step": final_state["current_step"],
                "validation_report": final_state.get("validation_report", {}),
            }

        except Exception as e:
            logger.error(f"Migration workflow failed: {e}")
            return {
                "success": False,
                "errors": [f"Workflow execution failed: {e}"],
                "completed_tables": [],
                "total_tables": 0,
                "final_step": "Failed",
                "validation_report": {},
            }
