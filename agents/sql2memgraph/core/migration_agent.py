# flake8: noqa
"""
SQL Database to Memgraph Migration Agent

This agent analyzes SQL databases, generates appropriate Cypher queries,
and migrates data to Memgraph using LangGraph workflow.
"""

import hashlib
import json
import logging
import sys
from typing import Dict, List, Any, TypedDict, Optional, cast
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "memgraph-toolbox" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "langchain-memgraph"))
sys.path.append(str(Path(__file__).parent.parent))  # Add agents root to path

from langgraph.graph import StateGraph, END  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.runnables.config import RunnableConfig  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from query_generation.cypher_generator import CypherGenerator  # noqa: E402
from core.hygm import HyGM, ModelingMode, GraphModelingStrategy  # noqa: E402
from core.hygm.validation import validate_memgraph_data  # noqa: E402
from memgraph_toolbox.api.memgraph import Memgraph  # noqa: E402
from database.factory import DatabaseAnalyzerFactory  # noqa: E402
from core.utils.meta_graph import (  # noqa: E402
    node_key as meta_node_key,
    relationship_key as meta_relationship_key,
    summarize_node as meta_summarize_node,
    summarize_relationship as meta_summarize_relationship,
    summarize_nodes as meta_summarize_nodes,
    summarize_relationships as meta_summarize_relationships,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MigrationState(TypedDict):
    """State for the migration workflow."""

    source_db_config: Dict[str, str]
    memgraph_config: Optional[Dict[str, str]]
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
    existing_meta_graph: Optional[Dict[str, Any]]


class SQLToMemgraphAgent:
    """Agent for migrating SQL databases to Memgraph."""

    def __init__(
        self,
        modeling_mode: ModelingMode = ModelingMode.AUTOMATIC,
        graph_modeling_strategy: GraphModelingStrategy = (
            GraphModelingStrategy.DETERMINISTIC
        ),
        meta_graph_policy: str = "auto",
    ):
        """Initialize the migration agent.

        Args:
            modeling_mode: Graph modeling mode
                - AUTOMATIC: Generate graph model automatically (default)
                - INCREMENTAL: Review tables and refine the model interactively
            graph_modeling_strategy: Strategy for graph model creation
                - DETERMINISTIC: Rule-based graph creation (default)
                - LLM_POWERED: LLM generates the graph model
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.database_analyzer = None
        self.cypher_generator = CypherGenerator()
        self.modeling_mode = modeling_mode
        self.graph_modeling_strategy = graph_modeling_strategy
        policy = (meta_graph_policy or "auto").lower()
        if policy not in {"auto", "skip", "reset"}:
            logger.warning(
                "Unknown meta graph policy '%s'; defaulting to auto",
                meta_graph_policy,
            )
            policy = "auto"
        self.meta_graph_policy = policy

        self.memgraph_client: Optional[Memgraph] = None
        self._existing_meta_graph: Optional[Dict[str, Any]] = None
        self._current_graph_model: Optional[Any] = None
        self._ingestion_plan: Dict[str, Any] = {}
        self._source_signature: Dict[str, str] = {}

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
            user: '{db_config["user"]}',
            password: '{db_config["password"]}',
            host: '{migrate_host}',
            database: '{db_config["database"]}'
        }}"""

    def _compute_source_signature(
        self,
        state: MigrationState,
    ) -> Dict[str, str]:
        """Create a deterministic signature for the source database."""
        config = state.get("source_db_config", {})
        structure = state.get("database_structure", {})
        host = config.get("host", "")
        database = config.get("database", "")
        db_type = (
            structure.get("database_type") or config.get("database_type") or "mysql"
        )
        signature = {
            "host": host,
            "database": database,
            "type": db_type,
        }
        self._source_signature = signature
        return signature

    def _node_key(self, node: Any) -> str:
        """Generate a stable key for a graph node definition."""
        return meta_node_key(node)

    def _relationship_key(self, rel: Any) -> str:
        """Generate a stable key for a graph relationship."""
        return meta_relationship_key(rel)

    def _summarize_node(self, node: Any) -> Dict[str, Any]:
        """Create a JSON-serializable summary for a node definition."""
        return meta_summarize_node(node)

    def _summarize_relationship(self, rel: Any) -> Dict[str, Any]:
        """Create a JSON-serializable summary for a relationship."""
        return meta_summarize_relationship(rel)

    def _graph_model_schema(self, model: Any) -> Dict[str, Any]:
        """Convert a graph model to schema format if possible."""
        if hasattr(model, "to_schema_format"):
            return model.to_schema_format()
        return {}

    def _graph_model_hash(self, schema: Dict[str, Any]) -> str:
        """Compute a stable hash for a schema dictionary."""
        schema_json = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

    def _build_node_summaries(self, model: Any) -> Dict[str, Any]:
        """Build summaries for all nodes in a model."""
        return meta_summarize_nodes(getattr(model, "nodes", []))

    def _build_relationship_summaries(self, model: Any) -> Dict[str, Any]:
        """Build summaries for all relationships in a model."""
        return meta_summarize_relationships(getattr(model, "edges", []))

    def _load_existing_meta_graph(self, state: MigrationState) -> None:
        """Read stored migration metadata from Memgraph if available."""
        if not self.memgraph_client:
            return

        signature = self._compute_source_signature(state)
        query = (
            "MATCH (meta:MigrationAgent {source_host: $host, "
            "source_database: $database, source_type: $type}) "
            "RETURN meta LIMIT 1"
        )
        result = self.memgraph_client.query(
            query,
            {
                "host": signature["host"],
                "database": signature["database"],
                "type": signature["type"],
            },
        )

        if result:
            meta = result[0].get("meta", {})
            node_data = meta.get("node_summaries") or "{}"
            rel_data = meta.get("relationship_summaries") or "{}"
            table_counts = meta.get("table_counts") or "{}"
            self._existing_meta_graph = {
                "model_hash": meta.get("model_hash"),
                "node_summaries": json.loads(node_data),
                "relationship_summaries": json.loads(rel_data),
                "table_counts": json.loads(table_counts),
            }
            state["existing_meta_graph"] = self._existing_meta_graph
            logger.info(
                "Loaded existing migration metadata for %s/%s",
                signature["host"],
                signature["database"],
            )
        else:
            self._existing_meta_graph = None
            state["existing_meta_graph"] = None
            logger.info(
                "No existing migration metadata found for %s/%s",
                signature["host"],
                signature["database"],
            )

    def _calculate_ingestion_plan(
        self,
        graph_model: Any,
        structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine which nodes and relationships need migration."""
        plan = {
            "nodes": set(),
            "relationships": set(),
            "node_reasons": {},
            "relationship_reasons": {},
        }

        table_counts = structure.get("table_counts", {}) or {}
        existing = self._existing_meta_graph or {}
        existing_nodes = existing.get("node_summaries", {}) or {}
        existing_rels = existing.get("relationship_summaries", {}) or {}
        existing_counts = existing.get("table_counts", {}) or {}

        node_keys_by_source: Dict[str, str] = {}
        label_keys: Dict[str, str] = {}

        for node in getattr(graph_model, "nodes", []):
            key = self._node_key(node)
            summary = self._summarize_node(node)
            source_name = summary.get("source")
            if source_name:
                node_keys_by_source[source_name] = key
            label_key = "|".join(sorted(summary.get("labels", [])))
            label_keys[label_key] = key

            reasons: List[str] = []
            stored = existing_nodes.get(key)
            if not stored:
                reasons.append("new node definition")
            else:
                if summary["properties"] != stored.get("properties", []):
                    reasons.append("properties changed")
                if summary["id_field"] != stored.get("id_field"):
                    reasons.append("identifier changed")

            table_name = summary.get("source")
            if table_name:
                new_count = table_counts.get(table_name)
                old_count = existing_counts.get(table_name)
                if new_count is not None:
                    if old_count is None:
                        reasons.append("table count unavailable previously")
                    elif new_count != old_count:
                        if new_count > old_count:
                            reasons.append("source data increased")
                        else:
                            reasons.append("source data changed")

            if reasons or not existing_nodes:
                plan["nodes"].add(key)
                plan["node_reasons"][key] = reasons or ["initial migration"]

        for rel in getattr(graph_model, "edges", []):
            key = self._relationship_key(rel)
            summary = self._summarize_relationship(rel)
            reasons: List[str] = []
            stored = existing_rels.get(key)
            if not stored:
                reasons.append("new relationship definition")
            else:
                if summary["mapping"] != stored.get("mapping", {}):
                    reasons.append("mapping changed")
                if summary["start"] != stored.get("start", []):
                    reasons.append("start labels changed")
                if summary["end"] != stored.get("end", []):
                    reasons.append("end labels changed")

            start_key = None
            end_key = None
            start_table = summary.get("start_table")
            end_table = summary.get("end_table")
            if start_table and start_table in node_keys_by_source:
                start_key = node_keys_by_source[start_table]
            if end_table and end_table in node_keys_by_source:
                end_key = node_keys_by_source[end_table]
            if not start_key:
                label = "|".join(summary.get("start", []))
                start_key = label_keys.get(label)
            if not end_key:
                label = "|".join(summary.get("end", []))
                end_key = label_keys.get(label)

            dependent_update = False
            if start_key and start_key in plan["nodes"]:
                dependent_update = True
            if end_key and end_key in plan["nodes"]:
                dependent_update = True
            if dependent_update and "dependent node update" not in reasons:
                reasons.append("dependent node update")

            if reasons or not existing_rels:
                plan["relationships"].add(key)
                plan["relationship_reasons"][key] = reasons or ["initial migration"]

        self._ingestion_plan = plan
        return plan

    def _store_meta_graph(self, state: MigrationState) -> None:
        """Persist the current graph model metadata to Memgraph."""
        if not self.memgraph_client:
            return

        graph_model = state.get("graph_model")
        if not graph_model:
            return

        structure = state.get("database_structure", {})
        schema = self._graph_model_schema(graph_model)
        node_summaries = self._build_node_summaries(graph_model)
        rel_summaries = self._build_relationship_summaries(graph_model)
        table_counts = structure.get("table_counts", {}) or {}

        model_hash = self._graph_model_hash(schema)
        signature = self._source_signature or self._compute_source_signature(state)

        query = (
            "MERGE (meta:MigrationAgent {source_host: $host, "
            "source_database: $database, source_type: $type}) "
            "SET meta.last_migrated_at = datetime(), "
            "meta.model_hash = $model_hash, "
            "meta.schema = $schema, "
            "meta.node_summaries = $node_summaries, "
            "meta.relationship_summaries = $relationship_summaries, "
            "meta.table_counts = $table_counts"
        )

        self.memgraph_client.query(
            query,
            {
                "host": signature.get("host", ""),
                "database": signature.get("database", ""),
                "type": signature.get("type", ""),
                "model_hash": model_hash,
                "schema": json.dumps(schema, sort_keys=True),
                "node_summaries": json.dumps(node_summaries, sort_keys=True),
                "relationship_summaries": json.dumps(
                    rel_summaries,
                    sort_keys=True,
                ),
                "table_counts": json.dumps(table_counts, sort_keys=True),
            },
        )

        logger.info(
            "Stored migration metadata for %s/%s",
            signature.get("host", ""),
            signature.get("database", ""),
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with clear separation of concerns."""
        workflow = StateGraph(MigrationState)

        # Add nodes - refactored for better modularity
        workflow.add_node(
            "connect_and_analyze_schema",
            self._connect_and_analyze_schema,
        )
        workflow.add_node(
            "create_graph_model",
            self._create_graph_model,
        )
        workflow.add_node(
            "create_indexes",
            self._create_indexes,
        )
        workflow.add_node(
            "generate_cypher_queries",
            self._generate_cypher_queries,
        )
        workflow.add_node(
            "prepare_target_database",
            self._prepare_target_database,
        )
        workflow.add_node(
            "execute_data_migration",
            self._execute_data_migration,
        )
        workflow.add_node(
            "validate_post_migration",
            self._validate_post_migration,
        )

        # Add conditional edges for better error handling
        workflow.add_edge(
            "connect_and_analyze_schema",
            "prepare_target_database",
        )
        workflow.add_edge(
            "prepare_target_database",
            "create_graph_model",
        )
        workflow.add_edge("create_graph_model", "create_indexes")
        workflow.add_edge("create_indexes", "generate_cypher_queries")
        workflow.add_edge("generate_cypher_queries", "execute_data_migration")
        workflow.add_edge("execute_data_migration", "validate_post_migration")
        workflow.add_edge("validate_post_migration", END)

        # Set entry point
        workflow.set_entry_point("connect_and_analyze_schema")

        # Return the workflow (not compiled) so caller can add checkpointer
        return workflow

    def _connect_and_analyze_schema(
        self,
        state: MigrationState,
    ) -> MigrationState:
        """Connect to source database and prepare info for HyGM."""
        logger.info("Preparing database connection for HyGM analysis...")

        try:
            # Initialize database analyzer to test connection
            database_analyzer = DatabaseAnalyzerFactory.create_analyzer(
                database_type="mysql", **state["source_db_config"]
            )

            if not database_analyzer.connect():
                raise Exception("Failed to connect to source database")

            # Get basic database structure for HyGM
            db_structure = database_analyzer.get_database_structure()
            hygm_data = db_structure.to_hygm_format()

            # Store the database structure for HyGM
            state["database_structure"] = hygm_data
            state["total_tables"] = len(hygm_data.get("entity_tables", {}))
            state["current_step"] = "Database connection established"

            logger.info("Database structure prepared for HyGM analysis")

        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            state["errors"].append(f"Database connection failed: {e}")

        return state

    def _create_graph_model(self, state: MigrationState) -> MigrationState:
        """Create graph model using HyGM based on analyzed schema."""
        logger.info("Creating graph model using HyGM...")

        try:
            hygm_data = state["database_structure"]

            # Log the modeling mode being used
            if self.modeling_mode == ModelingMode.INCREMENTAL:
                logger.info(
                    "Using incremental graph modeling mode with an "
                    "end-of-session interactive refinement option"
                )
            else:
                logger.info("Using automatic graph modeling mode")

            # Create graph modeler with strategy and mode
            graph_modeler = HyGM(
                llm=self.llm,
                mode=self.modeling_mode,
                strategy=self.graph_modeling_strategy,
                existing_meta_graph=state.get("existing_meta_graph"),
            )

            # Log the strategy being used
            strategy_name = self.graph_modeling_strategy.value
            logger.info(f"Using {strategy_name} graph modeling strategy")

            # Generate graph model using new unified interface
            graph_model = graph_modeler.create_graph_model(
                hygm_data,
                domain_context="Database migration to graph database",
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

    def _prepare_target_database(
        self,
        state: MigrationState,
    ) -> MigrationState:
        """Prepare the target Memgraph database for migration."""
        logger.info("Preparing target database for migration...")

        try:
            # Initialize Memgraph connection
            config_value = state.get("memgraph_config")
            if not config_value:
                raise Exception("Memgraph configuration is required")
            config = cast(Dict[str, str], config_value)

            url = config.get("url")
            if not url:
                raise Exception("Memgraph configuration must include 'url'")

            username = config.get("username") or ""
            password = config.get("password") or ""
            database = config.get("database") or "memgraph"

            self.memgraph_client = Memgraph(
                url=url,
                username=username,
                password=password,
                database=database,
            )

            # Test Memgraph connection
            test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            self.memgraph_client.query(test_query)
            logger.info("Memgraph connection established successfully")

            # Load existing meta graph to plan incremental ingestion
            policy = getattr(self, "meta_graph_policy", "auto")
            if policy == "skip":
                logger.info("Meta graph loading skipped by configuration")
                self._existing_meta_graph = None
                state["existing_meta_graph"] = None
            else:
                self._load_existing_meta_graph(state)
                if self._existing_meta_graph:
                    if policy == "reset":
                        logger.info(
                            "Existing migration metadata ignored due to reset policy",
                        )
                        self._existing_meta_graph = None
                        state["existing_meta_graph"] = None
                    else:
                        logger.info(
                            "Existing migration metadata detected; data will be merged",
                        )
                else:
                    logger.info(
                        "No migration metadata found; treating this as an initial run",
                    )

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
            memgraph_client = self.memgraph_client
            if not memgraph_client:
                raise Exception("Memgraph client is not initialized")

            queries = state["migration_queries"]

            # Execute all migration queries sequentially
            successful_queries = 0
            for i, query in enumerate(queries):
                # Skip empty queries but keep comment-only blocks for context
                query_lines = [line.strip() for line in query.strip().split("\n")]
                non_comment_lines = [
                    line for line in query_lines if line and not line.startswith("//")
                ]

                if non_comment_lines:  # Has actual Cypher code
                    try:
                        logger.info(
                            "Executing query %d/%d...",
                            i + 1,
                            len(queries),
                        )
                        memgraph_client.query(query)
                        successful_queries += 1

                        # Log progress for node creation queries
                        if "CREATE (n:" in query:
                            # Extract table name from comment or FROM clause
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
                        logger.error(f"Failed to execute query {i + 1}: {e}")
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
                logger.info("Creating %s: %s", query_type, query)
                memgraph_client.query(query)
                success_list.append(query)
            except Exception as e:
                # Some queries might already exist, log but continue
                logger.warning(
                    f"{query_type.capitalize()} creation {warning_prefix}: %s",
                    e,
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
        logger.info("Creating HyGM indexes and constraints...")

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

    def _generate_cypher_queries(
        self,
        state: MigrationState,
    ) -> MigrationState:
        """Generate merge-based Cypher queries using the ingestion plan."""
        logger.info("Generating Cypher queries based on HyGM graph model...")

        try:
            source_db_config = state["source_db_config"]
            graph_model = state.get("graph_model")
            if not graph_model:
                raise Exception("HyGM graph model is required for migration")

            self._current_graph_model = graph_model

            structure = state.get("database_structure", {})
            plan = self._calculate_ingestion_plan(graph_model, structure)
            nodes_to_migrate = plan["nodes"]
            relationships_to_migrate = plan["relationships"]

            if not nodes_to_migrate and not relationships_to_migrate:
                logger.info(
                    "Schema and table counts already match stored metadata; "
                    "no migration queries generated"
                )
                state["migration_queries"] = []
                state["current_step"] = "No new data to migrate"
                return state

            for node_key in sorted(nodes_to_migrate):
                reasons = plan["node_reasons"].get(node_key, [])
                reason_text = ", ".join(reasons) if reasons else "initial migration"
                logger.info("Node plan %s → %s", node_key, reason_text)

            for rel_key in sorted(relationships_to_migrate):
                reasons = plan["relationship_reasons"].get(rel_key, [])
                reason_text = ", ".join(reasons) if reasons else "initial migration"
                logger.info("Relationship plan %s → %s", rel_key, reason_text)

            queries: List[str] = []
            db_config_str = self._get_db_config_for_migrate(source_db_config)

            for node_def in graph_model.nodes:
                node_key = self._node_key(node_def)
                if nodes_to_migrate and node_key not in nodes_to_migrate:
                    continue

                source = getattr(node_def, "source", None)
                source_table = getattr(source, "name", None) or "unknown"
                node_label = node_def.primary_label

                properties = [
                    prop.key if hasattr(prop, "key") else str(prop)
                    for prop in getattr(node_def, "properties", [])
                ]

                node_mapping = getattr(source, "mapping", {}) if source else {}
                id_field = node_mapping.get("id_field")
                if not id_field and properties:
                    id_field = properties[0]

                if id_field and id_field not in properties:
                    properties.append(id_field)

                if not id_field:
                    logger.warning(
                        "Skipping node %s: identifier field missing",
                        node_label,
                    )
                    continue

                if not properties:
                    logger.warning(
                        "No properties found for node %s from table %s",
                        node_label,
                        source_table,
                    )
                    continue

                properties_str = ", ".join(properties)
                node_query = f"""
// Merge {node_label} nodes from {source_table} table (HyGM optimized)
CALL migrate.mysql(
    'SELECT {properties_str} FROM {source_table}',
    {db_config_str}
)
YIELD row
MERGE (n:{node_label} {{{id_field}: row.{id_field}}})
SET n += row;"""
                queries.append(node_query)
                logger.info("Prepared merge query for %s", node_label)

            logger.info(
                "Preparing relationship queries for %d definitions",
                len(graph_model.edges),
            )

            for rel_def in graph_model.edges:
                rel_key = self._relationship_key(rel_def)
                if relationships_to_migrate and rel_key not in relationships_to_migrate:
                    continue

                rel_query = self._generate_hygm_relationship_query(
                    rel_def, db_config_str
                )
                if rel_query:
                    queries.append(rel_query)
                    logger.info(
                        "Prepared merge query for relationship %s",
                        rel_def.edge_type,
                    )

            state["migration_queries"] = queries
            state["current_step"] = "Migration queries prepared"

            logger.info("Generated %d migration queries", len(queries))

        except Exception as e:
            logger.error(f"Error generating HyGM-based Cypher queries: {e}")
            return self._handle_step_error(
                state,
                "generating cypher queries",
                e,
            )

        return state

    def _generate_hygm_relationship_query(
        self,
        rel_def,
        mysql_config_str: str,
    ) -> str:
        """Create relationship query from HyGM definition."""

        try:
            if not rel_def.source or not rel_def.source.mapping:
                logger.warning(
                    f"No source mapping for relationship {rel_def.edge_type}"
                )
                return ""

            rel_name = rel_def.edge_type
            source_info = rel_def.source.mapping

            # Determine relationship type from HyGM source
            if rel_def.source.type == "many_to_many":
                return self._generate_many_to_many_hygm_query(
                    rel_name, rel_def, source_info, mysql_config_str
                )
            elif rel_def.source.type in ["table", "foreign_key"]:
                return self._generate_one_to_many_hygm_query(
                    rel_name, rel_def, source_info, mysql_config_str
                )
            else:
                logger.warning(
                    "Unsupported relationship type: %s",
                    rel_def.source.type,
                )
                return ""

        except Exception as e:
            logger.error(
                "Error generating relationship query for %s: %s",
                rel_def.edge_type,
                e,
            )
            return ""

    def _generate_one_to_many_hygm_query(
        self,
        rel_name: str,
        rel_def,
        source_info: Dict[str, Any],
        mysql_config_str: str,
    ) -> str:
        """Generate one-to-many relationship query from HyGM mapping."""

        start_node = source_info.get("start_node", "")
        end_node = source_info.get("end_node", "")
        from_pk = source_info.get("from_pk")

        if not start_node or not end_node:
            logger.error("Missing relationship information for %s", rel_name)
            raise Exception(
                "HyGM must provide complete relationship mapping for " f"{rel_name}"
            )

        try:
            from_table, fk_column = start_node.split(".", 1)
            to_table, to_column = end_node.split(".", 1)
        except ValueError:
            logger.error(
                "Invalid mapping format for %s: %s",
                rel_name,
                source_info,
            )
            raise Exception(
                f"HyGM must provide valid relationship mapping for {rel_name}"
            )

        if not from_pk:
            raise Exception(f"HyGM must provide primary key information for {rel_name}")

        from_label = (
            rel_def.start_node_labels[0] if rel_def.start_node_labels else from_table
        )
        to_label = rel_def.end_node_labels[0] if rel_def.end_node_labels else to_table

        select_sql = (
            f"SELECT {from_pk}, {fk_column} "
            f"FROM {from_table} "
            f"WHERE {fk_column} IS NOT NULL"
        )

        query = f"""
// Merge {rel_name} relationships (HyGM: {from_label} -> {to_label})
CALL migrate.mysql(
    '{select_sql}',
    {mysql_config_str}
)
YIELD row
MATCH (from_node:{from_label} {{{from_pk}: row.{from_pk}}})
MATCH (to_node:{to_label} {{{to_column}: row.{fk_column}}})
MERGE (from_node)-[:{rel_name}]->(to_node);"""

        return query

    def _generate_many_to_many_hygm_query(
        self,
        rel_name: str,
        rel_def,
        source_info: Dict[str, Any],
        mysql_config_str: str,
    ) -> str:
        """Generate many-to-many relationship query from HyGM mapping."""

        join_table = source_info.get("join_table")
        from_table = source_info.get("from_table")
        to_table = source_info.get("to_table")
        from_fk = source_info.get("join_from_column")
        to_fk = source_info.get("join_to_column")
        from_pk = source_info.get("from_column")
        to_pk = source_info.get("to_column")

        if not all([join_table, from_table, to_table, from_fk, to_fk, from_pk, to_pk]):
            logger.error(
                "Missing many-to-many relationship information for %s",
                rel_name,
            )
            raise Exception(
                "HyGM must provide complete many-to-many mapping for " f"{rel_name}"
            )

        from_label = (
            rel_def.start_node_labels[0] if rel_def.start_node_labels else from_table
        )
        to_label = rel_def.end_node_labels[0] if rel_def.end_node_labels else to_table

        select_sql = f"SELECT {from_fk}, {to_fk} " f"FROM {join_table}"

        query = f"""
// Merge {rel_name} relationships via {join_table}
// (HyGM: {from_label} <-> {to_label})
CALL migrate.mysql(
    '{select_sql}',
    {mysql_config_str}
)
YIELD row
MATCH (from:{from_label} {{{from_pk}: row.{from_fk}}})
MATCH (to:{to_label} {{{to_pk}: row.{to_fk}}})
MERGE (from)-[:{rel_name}]->(to);"""
        return query

    def _validate_post_migration(
        self,
        state: MigrationState,
    ) -> MigrationState:
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
            table_counts = structure.get("table_counts", {})

            # Default to all migrated tables when nothing specific is selected
            selected_tables = structure.get("selected_tables", [])
            if not selected_tables:
                # Use entity tables (exclude views and system tables)
                entity_tables = structure.get("entity_tables", {})
                selected_tables = list(entity_tables.keys())

            for table_name in selected_tables:
                if table_name in table_counts:
                    expected_nodes += table_counts[table_name]

            # Create expected data counts for the validator
            expected_data_counts = {
                "nodes": expected_nodes,
                "selected_tables": selected_tables,
            }

            # Run validation using existing connection and data counts
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

            if not state["errors"]:
                self._store_meta_graph(state)

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
            existing_meta_graph=None,
        )

        try:
            # Automatic mode compiles workflow without a checkpointer
            if self.modeling_mode == ModelingMode.AUTOMATIC:
                compiled_workflow = self.workflow.compile()
                final_state = compiled_workflow.invoke(initial_state)
            else:
                # Incremental mode enables a persistent checkpointer
                from langgraph.checkpoint.memory import MemorySaver

                memory = MemorySaver()
                compiled_workflow = self.workflow.compile(checkpointer=memory)

                # Provide required configuration for checkpointer
                config: RunnableConfig = {
                    "configurable": {"thread_id": "migration_thread_1"}
                }
                final_state = compiled_workflow.invoke(
                    initial_state,
                    config=config,
                )

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
