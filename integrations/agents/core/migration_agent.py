"""
SQL Database to Memgraph Migration Agent

This agent analyzes SQL databases, generates appropriate Cypher queries,
and migrates data to Memgraph using LangGraph workflow.
"""

import os
import sys
import logging
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
from core.hygm import HyGM, GraphModel, ModelingMode, GraphModelingStrategy
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
        """Build the LangGraph workflow."""
        workflow = StateGraph(MigrationState)

        # Add nodes
        workflow.add_node("analyze_database", self._analyze_database_schema)
        workflow.add_node(
            "interactive_graph_modeling", self._interactive_graph_modeling
        )
        workflow.add_node("create_indexes", self._create_indexes)
        workflow.add_node("generate_cypher_queries", self._generate_cypher_queries)
        workflow.add_node("validate_queries", self._validate_queries)
        workflow.add_node("execute_migration", self._execute_migration)
        workflow.add_node("verify_migration", self._verify_migration)
        workflow.add_node("validate_post_migration", self._validate_post_migration)

        # Add edges - direct flow without table selection
        workflow.add_edge("analyze_database", "interactive_graph_modeling")
        workflow.add_edge("interactive_graph_modeling", "create_indexes")
        workflow.add_edge("create_indexes", "generate_cypher_queries")
        workflow.add_edge("generate_cypher_queries", "validate_queries")
        workflow.add_edge("validate_queries", "execute_migration")
        workflow.add_edge("execute_migration", "verify_migration")
        workflow.add_edge("verify_migration", "validate_post_migration")
        workflow.add_edge("validate_post_migration", END)

        # Set entry point
        workflow.set_entry_point("analyze_database")

        # Return the workflow (not compiled) so caller can add checkpointer
        return workflow

    def _analyze_database_schema(self, state: MigrationState) -> MigrationState:
        """Analyze source database schema and structure."""
        logger.info("Analyzing source database schema...")

        try:
            # Initialize database analyzer using factory
            database_analyzer = DatabaseAnalyzerFactory.create_analyzer(
                database_type="mysql", **state["source_db_config"]
            )

            if not database_analyzer.connect():
                raise Exception("Failed to connect to source database")

            # Get standardized database structure
            db_structure = database_analyzer.get_database_structure()

            # Store database structure for later use (e.g., primary key lookup)
            self._database_structure = db_structure

            # Use the built-in HyGM format conversion
            hygm_data = db_structure.to_hygm_format()

            # Enhance with intelligent graph modeling
            logger.info("Starting graph modeling analysis...")
            try:
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

                # Store the graph model in state for use by migration agent
                state["database_structure"] = hygm_data
                state["graph_model"] = graph_model

                logger.info(
                    f"Graph model created with {len(graph_model.nodes)} "
                    f"node types and {len(graph_model.edges)} "
                    f"relationship types"
                )

            except Exception as e:
                logger.warning(f"Graph modeling enhancement failed: {e}")
                # Continue without graph modeling enhancement
                state["database_structure"] = hygm_data
                state["graph_model"] = None

            # Only count entity tables for migration progress (exclude views and join tables)
            state["total_tables"] = len(hygm_data["entity_tables"])

            # Automatically select all entity tables for migration
            entity_tables = hygm_data.get("entity_tables", {})
            if entity_tables:
                selected_tables = list(entity_tables.keys())
                logger.info(
                    f"Automatically selecting all {len(selected_tables)} entity tables for migration"
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

    def _interactive_graph_modeling(self, state: MigrationState) -> MigrationState:
        """
        Graph modeling step - now handled by HyGM class internally.
        This method validates the completed graph model.
        """
        logger.info("Validating completed graph model...")

        # Validate that we have a graph model
        if not state.get("graph_model"):
            logger.error("No graph model found - this should not happen")
            state["errors"].append("Graph modeling failed to produce a model")
            state["current_step"] = "Graph modeling failed"
            return state

        # Perform final validation
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

    def _convert_dict_to_graph_model(self, model_dict: Dict[str, Any]) -> GraphModel:
        """Convert dictionary representation back to GraphModel object."""
        from .hygm import GraphModel, GraphNode, GraphRelationship

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
            )
            relationships.append(rel)

        return GraphModel(
            nodes=nodes,
            relationships=relationships,
        )

    # TODO: This should be human visible and configurable.
    def _create_indexes(self, state: MigrationState) -> MigrationState:
        """Create indexes and constraints in Memgraph before migration."""
        logger.info("Creating indexes and constraints from HyGM graph model...")

        try:
            memgraph = Memgraph(**state["memgraph_config"])

            # Track created indexes and constraints
            created_indexes = []
            created_constraints = []

            # Check if we have a HyGM graph model with indexes and constraints
            graph_model = state.get("graph_model")
            if graph_model and hasattr(graph_model, "node_indexes"):
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
            else:
                # Fallback to legacy method if no HyGM model available
                logger.warning("No HyGM graph model found, using fallback method")
                return self._create_indexes_fallback(state)

            # Execute constraint queries first
            for query in constraint_queries:
                try:
                    logger.info("Creating constraint: %s", query)
                    memgraph.query(query)
                    created_constraints.append(query)
                except Exception as e:
                    # Some constraints might already exist, continue
                    logger.warning("Constraint creation warning: %s", e)

            # Execute index queries
            for query in index_queries:
                try:
                    logger.info("Creating index: %s", query)
                    memgraph.query(query)
                    created_indexes.append(query)
                except Exception as e:
                    # Some indexes might already exist, log but continue
                    logger.warning("Index creation warning: %s", e)

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
            logger.error("Error creating indexes: %s", e)
            state["errors"].append(f"Index creation failed: {e}")

        return state

    def _create_indexes_fallback(self, state: MigrationState) -> MigrationState:
        """Fallback method for creating indexes when no HyGM model available."""
        logger.info("Creating indexes and constraints using fallback method...")

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
                        logger.info("Creating constraint: %s", query)
                        memgraph.query(query)
                        created_constraints.append(query)
                    except Exception as e:
                        # Some constraints might already exist, continue
                        logger.warning("Constraint creation warning: %s", e)

                # Execute index queries
                for query in index_queries:
                    try:
                        logger.info("Creating index: %s", query)
                        memgraph.query(query)
                        created_indexes.append(query)
                    except Exception as e:
                        # Some indexes might already exist, log but continue
                        logger.warning("Index creation warning: %s", e)

            # Store results in state
            state["created_indexes"] = created_indexes
            state["created_constraints"] = created_constraints
            state["current_step"] = "Fallback indexes and constraints created"

            logger.info(
                "Created %d constraints and %d indexes using fallback method",
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
            hygm_data = state["database_structure"]
            source_db_config = state["source_db_config"]

            # Check if we have HyGM graph model
            if not state.get("graph_model"):
                logger.warning(
                    "No HyGM graph model found, falling back to basic migration"
                )
                return self._generate_cypher_queries_fallback(state)

            graph_model = state["graph_model"]
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

                # Extract property names from GraphProperty objects
                properties = []
                for prop in node_def.properties:
                    if isinstance(prop, type(node_def.properties[0])) and hasattr(
                        prop, "key"
                    ):
                        properties.append(prop.key)
                    else:
                        properties.append(str(prop))

                # Get table info for column validation
                table_info = hygm_data.get("entity_tables", {}).get(source_table, {})

                # Validate properties exist in source table
                valid_properties = self._validate_node_properties(
                    properties, table_info
                )

                if valid_properties:
                    properties_str = ", ".join(valid_properties)
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
                        f"{len(valid_properties)} properties"
                    )
                else:
                    logger.warning(
                        f"No valid properties found for node {node_label} from table {source_table}"
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
                rel_type = "many_to_many"  # Default fallback

            # Find the corresponding node labels from HyGM
            from_label = self._find_hygm_node_label(from_node, hygm_data)
            to_label = self._find_hygm_node_label(to_node, hygm_data)

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
                logger.warning(f"Unsupported relationship type: {rel_type}")
                return ""

        except Exception as e:
            logger.error(
                f"Error generating relationship query for {rel_def.edge_type}: {e}"
            )
            return ""

    def _find_hygm_node_label(self, node_name: str, hygm_data: Dict[str, Any]) -> str:
        """Find the HyGM node label for a given node name."""
        # Access the graph model from the current state to get actual node labels
        if hasattr(self, "_current_graph_model") and self._current_graph_model:
            for node in self._current_graph_model.nodes:
                # First try to match by primary label (for LLM-generated models)
                if node.primary_label.lower() == node_name.lower():
                    return node.primary_label
                # Then try to match by source table (for deterministic models)
                source_table = node.source.name if node.source else ""
                if source_table.lower() == node_name.lower():
                    return node.primary_label
                # Also check all labels
                for label in node.labels:
                    if label.lower() == node_name.lower():
                        return label

        # Fallback to table name transformation if graph model not available
        return self.cypher_generator._table_name_to_label(node_name)

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
            logger.warning(f"Missing relationship information for {rel_name}")
            return ""

        # Parse the table.column format
        try:
            from_table, fk_column = start_node.split(".", 1)
            to_table, to_column = end_node.split(".", 1)
        except ValueError:
            logger.warning(f"Invalid mapping format for {rel_name}: {source_info}")
            return ""

        # Get primary key from source table
        from_table_info = hygm_data.get("entity_tables", {}).get(from_table, {})
        primary_keys = from_table_info.get("primary_keys", [])

        if not primary_keys:
            logger.warning(f"Could not determine primary key for table {from_table}")
            return ""

        # Use the first primary key (most common case)
        from_pk = primary_keys[0]

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

    def _generate_cypher_queries_fallback(
        self, state: MigrationState
    ) -> MigrationState:
        """Fallback method for generating Cypher queries without HyGM model."""
        logger.info("Using fallback migration query generation...")

        try:
            structure = state["database_structure"]
            source_db_config = state["source_db_config"]

            # Generate migration queries using migrate.mysql() procedure
            queries = []

            # Create MySQL connection config for migrate module
            mysql_config_str = self._get_db_config_for_migrate(source_db_config)

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
                    rel_name = self.cypher_generator.generate_relationship_type(
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
                    rel_name = self.cypher_generator.generate_relationship_type(
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
            source_db_config = state["source_db_config"]
            mysql_config_str = self._get_db_config_for_migrate(source_db_config)

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
                                    f"Successfully migrated data from table: {table_name}"
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

            # Run post-migration validation using existing connection
            logger.info("Executing post-migration validation...")
            validation_result = validate_memgraph_data(
                expected_model=graph_model,
                memgraph_connection=self.memgraph_client,
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
