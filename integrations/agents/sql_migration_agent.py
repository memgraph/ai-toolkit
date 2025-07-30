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
from dotenv import load_dotenv

from sql_database_analyzer import MySQLAnalyzer
from cypher_generator import CypherGenerator
from hygm import HyGM, GraphModel
from memgraph_toolbox.api.memgraph import Memgraph

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MigrationState(TypedDict):
    """State for the migration workflow."""

    mysql_config: Dict[str, str]
    memgraph_config: Dict[str, str]
    database_structure: Dict[str, Any]
    migration_queries: List[str]
    current_step: str
    errors: List[str]
    completed_tables: List[str]
    total_tables: int
    created_indexes: List[str]
    created_constraints: List[str]


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
            interactive_table_selection: Whether to prompt user for table
                selection.
                - True: Show interactive table selection (default)
                - False: Migrate all entity tables automatically
        """
        # Environment validation is now handled by utils.environment module
        # This makes the agent more modular and reusable

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

    def _get_mysql_config_for_migrate(self, mysql_config: Dict[str, str]) -> str:
        """
        Convert MySQL config for use with migrate module in Memgraph container.

        Adjusts localhost/127.0.0.1 to host.docker.internal for Docker networking.
        """
        migrate_host = mysql_config["host"]
        if migrate_host == "localhost" or migrate_host == "127.0.0.1":
            migrate_host = "host.docker.internal"

        return f"""{{
            user: '{mysql_config['user']}',
            password: '{mysql_config['password']}',
            host: '{migrate_host}',
            database: '{mysql_config['database']}'
        }}"""

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(MigrationState)

        # Add nodes
        workflow.add_node("analyze_mysql", self._analyze_mysql_schema)
        workflow.add_node("select_tables", self._select_tables_for_migration)
        workflow.add_node(
            "interactive_graph_modeling", self._interactive_graph_modeling
        )
        workflow.add_node("create_indexes", self._create_indexes)
        workflow.add_node("generate_cypher_queries", self._generate_cypher_queries)
        workflow.add_node("validate_queries", self._validate_queries)
        workflow.add_node("execute_migration", self._execute_migration)
        workflow.add_node("verify_migration", self._verify_migration)

        # Add edges
        workflow.add_edge("analyze_mysql", "select_tables")
        workflow.add_edge("select_tables", "interactive_graph_modeling")
        workflow.add_edge("interactive_graph_modeling", "create_indexes")
        workflow.add_edge("create_indexes", "generate_cypher_queries")
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

            # Add table counts for progress tracking and collect sample data
            structure["table_counts"] = {}
            structure["sample_data"] = {}
            for table_name in structure["tables"].keys():
                count = self.mysql_analyzer.get_table_row_count(table_name)
                structure["table_counts"][table_name] = count

                # Get sample data for better graph modeling context
                try:
                    sample_data = self.mysql_analyzer.get_table_data(
                        table_name, limit=3
                    )
                    structure["sample_data"][table_name] = sample_data
                    logger.debug(
                        f"Collected {len(sample_data)} sample rows from {table_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not fetch sample data from {table_name}: {e}"
                    )
                    structure["sample_data"][table_name] = []

            # Enhance with intelligent graph modeling
            logger.info(
                "Applying intelligent graph modeling analysis "
                "with single LLM call..."
            )
            try:
                graph_modeler = HyGM(llm=self.llm)

                # Use interactive if enabled, otherwise standard approach
                if self.interactive_table_selection:
                    logger.info("Using interactive graph modeling approach")
                    graph_model = graph_modeler.analyze_and_model_schema_interactive(
                        structure,
                        domain_context=("Database migration with user feedback"),
                    )
                else:
                    logger.info("Using standard graph modeling approach")
                    graph_model = graph_modeler.analyze_and_model_schema(structure)

                # Add graph modeling results to structure
                structure["graph_model"] = {
                    "nodes": [node.__dict__ for node in graph_model.nodes],
                    "relationships": [
                        rel.__dict__ for rel in graph_model.relationships
                    ],
                    "modeling_decisions": graph_model.modeling_decisions,
                    "optimization_suggestions": (graph_model.optimization_suggestions),
                }

                logger.info(
                    f"Graph model created with {len(graph_model.nodes)} "
                    f"node types and {len(graph_model.relationships)} "
                    f"relationship types"
                )

            except Exception as e:
                logger.warning(f"Graph modeling enhancement failed: {e}")
                # Continue without graph modeling enhancement

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

    def _interactive_graph_modeling(self, state: MigrationState) -> MigrationState:
        """Interactive graph modeling with user feedback loop."""
        logger.info("Starting interactive graph modeling process...")

        try:
            structure = state["database_structure"]

            # Skip interactive modeling if not in interactive mode
            if not self.interactive_table_selection:
                logger.info("Non-interactive mode: skipping graph modeling feedback")
                state["current_step"] = "Graph modeling completed (non-interactive)"
                return state

            # Check if we have an existing graph model
            if "graph_model" not in structure:
                logger.warning("No graph model found for interactive refinement")
                state["current_step"] = "Graph modeling skipped - no initial model"
                return state

            # Initialize interactive HyGM
            graph_modeler = HyGM(llm=self.llm)

            # Load the existing graph model into HyGM for interaction
            existing_model = structure["graph_model"]
            graph_modeler.current_graph_model = self._convert_dict_to_graph_model(
                existing_model
            )

            # Present initial model to user
            while True:
                model_presentation = graph_modeler.get_current_model_presentation()

                # Use LangGraph interrupt for user interaction
                user_response = interrupt(
                    {
                        "type": "graph_model_review",
                        "message": "Review and modify the graph model",
                        "model_presentation": model_presentation,
                        "instructions": {
                            "actions": [
                                "'continue' - Proceed with current model",
                                "'modify' - Make changes to the model",
                                "'reset' - Start over with original analysis",
                            ],
                            "modification_examples": [
                                "Change node labels, properties, or constraints",
                                "Modify relationship names or directionality",
                                "Remove unwanted nodes or relationships",
                                "Add new nodes or relationships",
                            ],
                        },
                    }
                )

                if not user_response:
                    logger.info(
                        "No user response received, proceeding with current model"
                    )
                    break

                action = user_response.get("action", "continue").lower()

                if action == "continue":
                    logger.info("User approved the current graph model")
                    break
                elif action == "reset":
                    # Reset to original model
                    graph_modeler.current_graph_model = (
                        self._convert_dict_to_graph_model(existing_model)
                    )
                    graph_modeler.iteration_count = 0
                    logger.info("Reset graph model to original state")
                    continue
                elif action == "modify":
                    # Apply user modifications
                    feedback = user_response.get("feedback", {})
                    if feedback:
                        result = graph_modeler.apply_user_feedback(feedback)
                        if result.get("success"):
                            logger.info(
                                f"Applied user feedback successfully: {result['message']}"
                            )
                        else:
                            logger.error(
                                f"Failed to apply feedback: {result.get('error', 'Unknown error')}"
                            )
                    continue
                else:
                    logger.warning(
                        f"Unknown action: {action}, proceeding with current model"
                    )
                    break

            # Update the state with the refined graph model
            final_model = graph_modeler.current_graph_model
            structure["graph_model"] = {
                "nodes": [node.__dict__ for node in final_model.nodes],
                "relationships": [rel.__dict__ for rel in final_model.relationships],
                "modeling_decisions": final_model.modeling_decisions,
                "optimization_suggestions": final_model.optimization_suggestions,
                "iterations": graph_modeler.iteration_count,
            }

            state["database_structure"] = structure
            state[
                "current_step"
            ] = f"Interactive graph modeling completed ({graph_modeler.iteration_count} iterations)"

            logger.info(
                f"Completed interactive graph modeling after {graph_modeler.iteration_count} iterations"
            )

        except Exception as e:
            logger.error(f"Error in interactive graph modeling: {e}")
            state["errors"].append(f"Interactive graph modeling failed: {e}")
            state["current_step"] = "Interactive graph modeling failed"

        return state

    def _convert_dict_to_graph_model(self, model_dict: Dict[str, Any]) -> GraphModel:
        """Convert dictionary representation back to GraphModel object."""
        from hygm import GraphModel, GraphNode, GraphRelationship

        # Convert nodes
        nodes = []
        for node_dict in model_dict.get("nodes", []):
            node = GraphNode(
                name=node_dict.get("name", ""),
                label=node_dict.get("label", ""),
                properties=node_dict.get("properties", []),
                primary_key=node_dict.get("primary_key", "id"),
                indexes=node_dict.get("indexes", []),
                constraints=node_dict.get("constraints", []),
                source_table=node_dict.get("source_table", ""),
                modeling_rationale=node_dict.get("modeling_rationale", ""),
            )
            nodes.append(node)

        # Convert relationships
        relationships = []
        for rel_dict in model_dict.get("relationships", []):
            rel = GraphRelationship(
                name=rel_dict.get("name", ""),
                type=rel_dict.get("type", ""),
                from_node=rel_dict.get("from_node", ""),
                to_node=rel_dict.get("to_node", ""),
                properties=rel_dict.get("properties", []),
                directionality=rel_dict.get("directionality", "directed"),
                source_info=rel_dict.get("source_info", {}),
                modeling_rationale=rel_dict.get("modeling_rationale", ""),
            )
            relationships.append(rel)

        return GraphModel(
            nodes=nodes,
            relationships=relationships,
            modeling_decisions=model_dict.get("modeling_decisions", []),
            optimization_suggestions=model_dict.get("optimization_suggestions", []),
            data_patterns=model_dict.get("data_patterns", {}),
        )

    # TODO: This should be human visible and configurable.
    def _create_indexes(self, state: MigrationState) -> MigrationState:
        """Create indexes and constraints in Memgraph before migration."""
        logger.info("Creating indexes and constraints in Memgraph...")

        try:
            memgraph = Memgraph(**state["memgraph_config"])
            structure = state["database_structure"]

            # Track created indexes
            created_indexes = []
            created_constraints = []

            # Create indexes and constraints for each entity table
            for table_name, table_info in structure["entity_tables"].items():
                schema = table_info["schema"]

                # Generate index queries
                index_queries = self.cypher_generator.generate_index_queries(
                    table_name, schema
                )

                # Generate constraint queries
                constraint_queries = self.cypher_generator.generate_constraint_queries(
                    table_name, schema
                )

                # Execute constraint queries first
                for query in constraint_queries:
                    try:
                        logger.info(f"Creating constraint: {query}")
                        memgraph.query(query)
                        created_constraints.append(query)
                    except Exception as e:
                        # Some constraints might already exist, continue
                        logger.warning("Constraint creation warning: %s", e)

                # Execute index queries
                for query in index_queries:
                    try:
                        logger.info(f"Creating index: {query}")
                        memgraph.query(query)
                        created_indexes.append(query)
                    except Exception as e:
                        # Some indexes might already exist, log but continue
                        logger.warning("Index creation warning: %s", e)

            # Store results in state
            state["created_indexes"] = created_indexes
            state["created_constraints"] = created_constraints
            state["current_step"] = "Indexes and constraints created"

            logger.info(
                "Created %d constraints and %d indexes",
                len(created_constraints),
                len(created_indexes),
            )

        except Exception as e:
            logger.error("Error creating indexes: %s", e)
            state["errors"].append(f"Index creation failed: {e}")

        return state

    def _generate_cypher_queries(self, state: MigrationState) -> MigrationState:
        """Generate Cypher queries based on HyGM graph model recommendations."""
        logger.info("Generating Cypher queries based on HyGM graph model...")

        try:
            structure = state["database_structure"]
            mysql_config = state["mysql_config"]

            # Check if we have HyGM graph model
            if "graph_model" not in structure:
                logger.warning(
                    "No HyGM graph model found, falling back to basic migration"
                )
                return self._generate_cypher_queries_fallback(state)

            graph_model = structure["graph_model"]
            queries = []

            # Create MySQL connection config for migrate module
            mysql_config_str = self._get_mysql_config_for_migrate(mysql_config)

            # Generate node creation queries based on HyGM recommendations
            logger.info(
                f"Creating nodes based on {len(graph_model['nodes'])} HyGM node definitions"
            )

            for node_def in graph_model["nodes"]:
                source_table = node_def["source_table"]
                node_label = node_def["label"]
                properties = node_def["properties"]

                # Get table info for column validation
                table_info = structure.get("entity_tables", {}).get(source_table, {})

                # Validate properties exist in source table
                valid_properties = self._validate_node_properties(
                    properties, table_info
                )

                if valid_properties:
                    properties_str = ", ".join(valid_properties)
                    node_query = f"""
// Create {node_label} nodes from {source_table} table (HyGM optimized)
// Rationale: {node_def.get('modeling_rationale', 'N/A')}
CALL migrate.mysql('SELECT {properties_str} FROM {source_table}', {mysql_config_str})
YIELD row
CREATE (n:{node_label})
SET n += row;"""
                    queries.append(node_query)
                    logger.info(
                        f"Added node creation for {node_label} with {len(valid_properties)} properties"
                    )
                else:
                    logger.warning(
                        f"No valid properties found for node {node_label} from table {source_table}"
                    )

            # Generate relationship creation queries based on HyGM recommendations
            logger.info(
                f"Creating relationships based on {len(graph_model['relationships'])} HyGM relationship definitions"
            )

            for rel_def in graph_model["relationships"]:
                rel_query = self._generate_hygm_relationship_query(
                    rel_def, structure, mysql_config_str
                )
                if rel_query:
                    queries.append(rel_query)
                    logger.info(f"Added relationship creation: {rel_def['name']}")

            state["migration_queries"] = queries
            state["current_step"] = "Cypher queries generated using HyGM graph model"

            logger.info(
                f"Generated {len(queries)} migration queries based on HyGM recommendations"
            )

        except Exception as e:
            logger.error(f"Error generating HyGM-based Cypher queries: {e}")
            state["errors"].append(f"HyGM Cypher generation failed: {e}")
            # Fallback to basic migration
            return self._generate_cypher_queries_fallback(state)

        return state

    def _validate_node_properties(
        self, properties: List[str], table_info: Dict[str, Any]
    ) -> List[str]:
        """Validate that properties exist in the source table schema."""
        if not table_info or not properties:
            return []

        # Get available columns from table schema
        available_columns = set()
        schema_list = table_info.get("schema", [])
        for col_info in schema_list:
            col_name = col_info.get("field")
            if col_name:
                available_columns.add(col_name)

        # Return only properties that exist in the table
        valid_properties = [prop for prop in properties if prop in available_columns]

        if len(valid_properties) != len(properties):
            missing = set(properties) - set(valid_properties)
            logger.warning(f"Some properties not found in table schema: {missing}")

        return valid_properties

    def _generate_hygm_relationship_query(
        self, rel_def: Dict[str, Any], structure: Dict[str, Any], mysql_config_str: str
    ) -> str:
        """Generate relationship query based on HyGM relationship definition."""

        try:
            rel_name = rel_def["name"]
            rel_type = rel_def["type"]
            from_node = rel_def["from_node"]
            to_node = rel_def["to_node"]
            source_info = rel_def.get("source_info", {})

            # Find the corresponding node labels from HyGM
            from_label = self._find_hygm_node_label(from_node, structure)
            to_label = self._find_hygm_node_label(to_node, structure)

            if not from_label or not to_label:
                logger.warning(
                    f"Could not find node labels for relationship {rel_name}"
                )
                return ""

            if rel_type == "one_to_many":
                return self._generate_one_to_many_hygm_query(
                    rel_name,
                    from_label,
                    to_label,
                    source_info,
                    mysql_config_str,
                    structure,
                )
            elif rel_type == "many_to_many":
                return self._generate_many_to_many_hygm_query(
                    rel_name,
                    from_label,
                    to_label,
                    source_info,
                    mysql_config_str,
                    structure,
                )
            else:
                logger.warning(f"Unsupported relationship type: {rel_type}")
                return ""

        except Exception as e:
            logger.error(
                f"Error generating relationship query for {rel_def.get('name', 'unknown')}: {e}"
            )
            return ""

    def _find_hygm_node_label(self, node_name: str, structure: Dict[str, Any]) -> str:
        """Find the HyGM node label for a given node name."""
        graph_model = structure.get("graph_model", {})
        nodes = graph_model.get("nodes", [])

        for node in nodes:
            if node.get("name") == node_name or node.get("source_table") == node_name:
                return node.get("label", "")

        # Fallback to original table name transformation
        return self.cypher_generator._table_name_to_label(node_name)

    def _generate_one_to_many_hygm_query(
        self,
        rel_name: str,
        from_label: str,
        to_label: str,
        source_info: Dict[str, Any],
        mysql_config_str: str,
        structure: Dict[str, Any],
    ) -> str:
        """Generate one-to-many relationship query using HyGM information."""

        from_table = source_info.get("from_table")
        to_table = source_info.get("to_table")
        fk_column = source_info.get("from_column")
        to_column = source_info.get("to_column")

        if not all([from_table, to_table, fk_column, to_column]):
            logger.warning(f"Missing relationship information for {rel_name}")
            return ""

        # Get primary key from source table
        from_table_info = structure.get("entity_tables", {}).get(from_table, {})
        from_pk = self._get_primary_key(from_table_info)

        if not from_pk:
            logger.warning(f"Could not determine primary key for table {from_table}")
            return ""

        return f"""
// Create {rel_name} relationships (HyGM: {from_label} -> {to_label})
CALL migrate.mysql('SELECT {from_pk}, {fk_column} FROM {from_table} WHERE {fk_column} IS NOT NULL', {mysql_config_str})
YIELD row
MATCH (from_node:{from_label} {{{from_pk}: row.{from_pk}}})
MATCH (to_node:{to_label} {{{to_column}: row.{fk_column}}})
CREATE (from_node)-[:{rel_name}]->(to_node);"""

    def _generate_many_to_many_hygm_query(
        self,
        rel_name: str,
        from_label: str,
        to_label: str,
        source_info: Dict[str, Any],
        mysql_config_str: str,
        structure: Dict[str, Any],
    ) -> str:
        """Generate many-to-many relationship query using HyGM information."""

        join_table = source_info.get("join_table")
        from_table = source_info.get("from_table")
        to_table = source_info.get("to_table")
        from_fk = source_info.get("join_from_column")
        to_fk = source_info.get("join_to_column")
        from_pk = source_info.get("from_column")
        to_pk = source_info.get("to_column")

        if not all([join_table, from_table, to_table, from_fk, to_fk, from_pk, to_pk]):
            logger.warning(
                f"Missing many-to-many relationship information for {rel_name}"
            )
            return ""

        return f"""
// Create {rel_name} relationships via {join_table} (HyGM: {from_label} <-> {to_label})
CALL migrate.mysql('SELECT {from_fk}, {to_fk} FROM {join_table}', {mysql_config_str})
YIELD row
MATCH (from:{from_label} {{{from_pk}: row.{from_fk}}})
MATCH (to:{to_label} {{{to_pk}: row.{to_fk}}})
CREATE (from)-[:{rel_name}]->(to);"""

    def _get_primary_key(self, table_info: Dict[str, Any]) -> str:
        """Get the primary key column name from table info."""
        schema_list = table_info.get("schema", [])
        for col_info in schema_list:
            if col_info.get("key") == "PRI":
                return col_info.get("field", "")

        # Fallback: assume first column is primary key
        if schema_list:
            return schema_list[0].get("field", "id")

        return "id"  # Default assumption

    def _generate_cypher_queries_fallback(
        self, state: MigrationState
    ) -> MigrationState:
        """Fallback method for generating Cypher queries without HyGM model."""
        logger.info("Using fallback migration query generation...")

        try:
            structure = state["database_structure"]
            mysql_config = state["mysql_config"]

            # Generate migration queries using migrate.mysql() procedure
            queries = []

            # Create MySQL connection config for migrate module
            mysql_config_str = self._get_mysql_config_for_migrate(mysql_config)

            # Generate node creation queries for each entity table
            entity_tables = structure.get("entity_tables", {})

            for table_name, table_info in entity_tables.items():
                label = self.cypher_generator._table_name_to_label(table_name)

                # Get columns excluding foreign keys for node properties
                node_columns = []
                fk_columns = {fk["column"] for fk in table_info.get("foreign_keys", [])}

                # Schema is a list of column dictionaries, not a dict
                schema_list = table_info.get("schema", [])
                for col_info in schema_list:
                    col_name = col_info.get("field")
                    if col_name and col_name not in fk_columns:
                        node_columns.append(col_name)

                if node_columns:
                    columns_str = ", ".join(node_columns)
                    node_query = f"""
// Create {label} nodes from {table_name} table (fallback)
CALL migrate.mysql('SELECT {columns_str} FROM {table_name}', {mysql_config_str})
YIELD row
CREATE (n:{label})
SET n += row;"""
                    queries.append(node_query)

            # Generate relationship creation queries
            for rel in structure["relationships"]:
                if rel["type"] == "one_to_many":
                    from_table = rel["from_table"]  # Table with FK
                    to_table = rel["to_table"]  # Referenced table
                    from_label = self.cypher_generator._table_name_to_label(from_table)
                    to_label = self.cypher_generator._table_name_to_label(to_table)
                    rel_name = self.cypher_generator._generate_relationship_type(
                        from_table, to_table
                    )

                    # FK column and what it references
                    fk_column = rel["from_column"]  # FK column name
                    to_column = rel["to_column"]  # Referenced column name

                    # Get the PK of the from_table (assume first column is PK)
                    from_table_info = structure["entity_tables"][from_table]
                    from_pk = from_table_info["schema"][0]["field"]

                    rel_query = f"""
// Create {rel_name} relationships between {from_label} and {to_label} (fallback)
CALL migrate.mysql('SELECT {from_pk}, {fk_column} FROM {from_table} WHERE {fk_column} IS NOT NULL', {mysql_config_str})
YIELD row
MATCH (from_node:{from_label} {{{from_pk}: row.{from_pk}}})
MATCH (to_node:{to_label} {{{to_column}: row.{fk_column}}})
CREATE (from_node)-[:{rel_name}]->(to_node);"""
                    queries.append(rel_query)

                elif rel["type"] == "many_to_many":
                    join_table = rel["join_table"]
                    from_table = rel["from_table"]
                    to_table = rel["to_table"]
                    from_label = self.cypher_generator._table_name_to_label(from_table)
                    to_label = self.cypher_generator._table_name_to_label(to_table)
                    rel_name = self.cypher_generator._generate_relationship_type(
                        from_table, to_table, join_table
                    )

                    from_fk = rel["join_from_column"]  # FK column in join table
                    to_fk = rel["join_to_column"]  # FK column in join table
                    from_pk = rel["from_column"]  # PK column in from_table
                    to_pk = rel["to_column"]  # PK column in to_table

                    rel_query = f"""
// Create {rel_name} relationships via {join_table} table (fallback)
CALL migrate.mysql('SELECT {from_fk}, {to_fk} FROM {join_table}', {mysql_config_str})
YIELD row
MATCH (from:{from_label} {{{from_pk}: row.{from_fk}}})
MATCH (to:{to_label} {{{to_pk}: row.{to_fk}}})
CREATE (from)-[:{rel_name}]->(to);"""
                    queries.append(rel_query)

            state["migration_queries"] = queries
            state["current_step"] = "Cypher queries generated (fallback mode)"

            logger.info(
                f"Generated {len(queries)} migration queries using fallback method"
            )

        except Exception as e:
            logger.error(f"Error generating fallback Cypher queries: {e}")
            state["errors"].append(f"Fallback Cypher generation failed: {e}")

        return state

    def _validate_queries(self, state: MigrationState) -> MigrationState:
        """Validate generated Cypher queries and test Memgraph connection."""
        logger.info("Validating queries and testing connections...")

        try:
            # Initialize Memgraph connection for validation
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

            # Test migrate.mysql connection by querying a small dataset
            mysql_config = state["mysql_config"]
            mysql_config_str = self._get_mysql_config_for_migrate(mysql_config)

            test_mysql_query = f"""
            CALL migrate.mysql('SELECT 1 as test_column LIMIT 1', {mysql_config_str})
            YIELD row
            RETURN row.test_column as test_result
            """

            self.memgraph_client.query(test_mysql_query)
            logger.info("MySQL connection through migrate module validated")

            state["current_step"] = "Queries and connections validated successfully"

        except Exception as e:
            logger.error(f"Error validating queries: {e}")
            state["errors"].append(f"Query validation failed: {e}")

        return state

    def _execute_migration(self, state: MigrationState) -> MigrationState:
        """Execute the migration using Memgraph migrate module."""
        logger.info("Executing migration using migrate module...")

        try:
            queries = state["migration_queries"]

            # Clear the database first to avoid constraint violations
            try:
                logger.info("Clearing existing data from Memgraph...")
                self.memgraph_client.query("MATCH (n) DETACH DELETE n")
                # Skip constraint dropping as it's not critical and has syntax issues
                logger.info("Database cleared successfully")
            except Exception as e:
                logger.warning(f"Database clearing failed (might be empty): {e}")

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
                            table_match = query.split("FROM ")[1].split()[0]
                            logger.info(
                                f"Successfully migrated data from table: {table_match}"
                            )
                        elif "CREATE (" in query and "-[:" in query:
                            logger.info("Successfully created relationships")

                    except Exception as e:
                        logger.error(f"Failed to execute query {i+1}: {e}")
                        logger.error(f"Query: {query[:100]}...")
                        state["errors"].append(f"Query execution failed: {e}")

            logger.info(
                f"Migration completed: {successful_queries}/{len(queries)} queries executed successfully"
            )
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
            current_step="Starting migration",
            errors=[],
            completed_tables=[],
            total_tables=0,
            created_indexes=[],
            created_constraints=[],
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
