#!/usr/bin/env python3
# flake8: noqa
"""
SQL Database to Graph Migration Agent - Main Entry Point

This is the main entry point for the SQL database to graph migration agent.
Run with: uv run main.py
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add current directory to Python path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (  # noqa: E402
    MigrationEnvironmentError,
    DatabaseConnectionError,
    setup_and_validate_environment,
    probe_all_connections,
    probe_source_connection,
    print_environment_help,
    print_troubleshooting_help,
)
from core import SQLToMemgraphAgent  # noqa: E402
from core.hygm import GraphModelingStrategy, ModelingMode  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

MODE_CHOICES = {
    "automatic": ModelingMode.AUTOMATIC,
    "incremental": ModelingMode.INCREMENTAL,
}

STRATEGY_CHOICES = {
    "deterministic": GraphModelingStrategy.DETERMINISTIC,
    "llm": GraphModelingStrategy.LLM_POWERED,
    "llm_powered": GraphModelingStrategy.LLM_POWERED,
}

META_GRAPH_POLICIES = {"auto", "skip", "reset"}

LOG_LEVEL_CHOICES = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]

PROVIDER_CHOICES = ["openai", "anthropic", "gemini"]


def _lower_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.lower() if value else None


def _upper_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.upper() if value else None


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the migration agent."""

    env_mode = _lower_env("SQL2MG_MODE")
    env_strategy = _lower_env("SQL2MG_STRATEGY")
    env_meta_policy = _lower_env("SQL2MG_META_POLICY")
    env_log_level = _upper_env("SQL2MG_LOG_LEVEL")
    env_provider = _lower_env("LLM_PROVIDER")
    env_model = os.getenv("LLM_MODEL")

    parser = argparse.ArgumentParser(
        description="SQL database to graph migration agent",
    )

    parser.add_argument(
        "--mode",
        choices=sorted(MODE_CHOICES.keys()),
        default=env_mode,
        type=str.lower,
        help="Graph modeling mode (automatic|incremental). Overrides SQL2MG_MODE.",
    )

    parser.add_argument(
        "--strategy",
        choices=["deterministic", "llm"],
        default=env_strategy,
        type=str.lower,
        help="Graph modeling strategy (deterministic|llm). Overrides SQL2MG_STRATEGY.",
    )

    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=env_provider,
        type=str.lower,
        help=(
            "LLM provider (openai|anthropic|gemini). "
            "Overrides LLM_PROVIDER. Auto-detects if not specified."
        ),
    )

    parser.add_argument(
        "--model",
        default=env_model,
        help=(
            "LLM model name. Overrides LLM_MODEL. "
            "Uses provider default if not specified."
        ),
    )

    parser.add_argument(
        "--meta-graph",
        choices=sorted(META_GRAPH_POLICIES),
        default=env_meta_policy,
        type=str.lower,
        help=(
            "Meta graph policy: auto (default), skip stored metadata, or reset to "
            "ignore previous migrations. Overrides SQL2MG_META_POLICY."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=env_log_level,
        type=str.upper,
        help="Logging level for the agent. Overrides SQL2MG_LOG_LEVEL.",
    )

    parser.add_argument(
        "--mapping",
        default=None,
        metavar="PATH",
        help=(
            "Generate a mapping JSON file instead of running the migration. "
            "The file maps graph nodes/edges back to SQL tables and columns."
        ),
    )

    return parser.parse_args(argv)


def _configure_log_level(level_name: Optional[str]) -> None:
    """Configure global logging level if provided."""

    if not level_name:
        return

    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning("Unknown log level '%s'; falling back to INFO", level_name)
        numeric_level = logging.INFO

    logging.getLogger().setLevel(numeric_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)


def _resolve_mode(cli_mode: Optional[str]) -> Optional[ModelingMode]:
    if not cli_mode:
        return None
    resolved = MODE_CHOICES.get(cli_mode)
    if not resolved:
        logger.warning("Unrecognised mode '%s'; falling back to prompt", cli_mode)
    return resolved


def _resolve_strategy(cli_strategy: Optional[str]) -> Optional[GraphModelingStrategy]:
    if not cli_strategy:
        return None
    resolved = STRATEGY_CHOICES.get(cli_strategy)
    if not resolved:
        logger.warning(
            "Unrecognised strategy '%s'; falling back to prompt",
            cli_strategy,
        )
    return resolved


def print_banner() -> None:
    """Print application banner."""
    print("=" * 60)
    print("🚀 SQL Database to Graph Migration Agent")
    print("=" * 60)
    print("Intelligent database migration with LLM-powered analysis")
    print()


def get_graph_modeling_mode() -> ModelingMode:
    """
    Get user choice for graph modeling mode.

    Returns:
        ModelingMode: Selected modeling mode
    """
    print("Graph modeling mode:")
    print()
    print("  1. Automatic     - Generate graph model without prompts")
    print()
    print("  2. Incremental   - Review each table with end-of-session refinement")
    print()

    while True:
        try:
            choice = input("Select mode (1-2) or press Enter for automatic: ").strip()
            if not choice:
                return ModelingMode.AUTOMATIC  # Default to automatic

            if choice == "1":
                return ModelingMode.AUTOMATIC
            elif choice == "2":
                return ModelingMode.INCREMENTAL
            else:
                print("Invalid choice. Please select 1-2.")
        except ValueError:
            print("Invalid input. Please enter 1-2.")


def get_graph_modeling_strategy() -> GraphModelingStrategy:
    """
    Get user choice for graph modeling strategy.

    Returns:
        GraphModelingStrategy: Selected strategy
    """
    print("Graph modeling strategy:")
    print()
    print("  1. Deterministic - Rule-based graph model creation ")
    print()
    print("  2. AI - LLM-based graph model creation (full HyGM capabilities)")
    print()
    print()

    while True:
        try:
            choice = input(
                "Select strategy (1-2) or press Enter for deterministic: "
            ).strip()
            if not choice:
                return GraphModelingStrategy.DETERMINISTIC  # Default

            if choice == "1":
                return GraphModelingStrategy.DETERMINISTIC
            elif choice == "2":
                return GraphModelingStrategy.LLM_POWERED
            else:
                print("Invalid choice. Please select 1-2.")
        except ValueError:
            print("Invalid input. Please enter 1-2.")


def run_migration(
    source_db_config: Dict[str, Any],
    memgraph_config: Dict[str, Any],
    modeling_mode: ModelingMode,
    graph_modeling_strategy: GraphModelingStrategy,
    meta_graph_policy: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> tuple[Dict[str, Any], SQLToMemgraphAgent]:
    """
    Run the migration with the specified configuration.

    Args:
        source_db_config: Source database connection configuration
        memgraph_config: Memgraph connection configuration
        modeling_mode: Graph modeling mode (automatic or incremental)
        graph_modeling_strategy: Strategy for graph model creation
        meta_graph_policy: Meta graph handling policy (auto|skip|reset)
        llm_provider: LLM provider (openai|anthropic|gemini)
        llm_model: Specific LLM model name

    Returns:
        Tuple of (migration result dictionary, agent instance)
    """
    print("🔧 Creating migration agent...")

    if modeling_mode == ModelingMode.INCREMENTAL:
        mode_name = "incremental"
    else:
        mode_name = "automatic"
    strategy_name = graph_modeling_strategy.value
    print(f"🎯 Graph modeling: {mode_name} with {strategy_name} strategy")

    if llm_provider:
        print(f"🤖 LLM Provider: {llm_provider}")
    if llm_model:
        print(f"🎯 Model: {llm_model}")
    print()

    # Create agent with graph modeling settings
    agent = SQLToMemgraphAgent(
        modeling_mode=modeling_mode,
        graph_modeling_strategy=graph_modeling_strategy,
        meta_graph_policy=meta_graph_policy,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    print("🚀 Starting migration workflow...")
    print("This will:")
    print("  1. 🔍 Analyze source database schema")
    print("  2. 🎯 Generate graph model with HyGM")
    print("  3. 📝 Create indexes and constraints")
    print("  4. ⚙️  Generate migration queries")
    print("  5. 🔄 Execute migration to Memgraph")
    print("  6. ✅ Verify the migration results")
    print()

    # Handle incremental vs automatic mode
    if modeling_mode == ModelingMode.INCREMENTAL:
        print("🔄 Incremental mode: Review LLM-generated graph changes table by table")
        print("   then approve or tweak differences before refining the model")
        print()

    # Run the migration with the user's chosen settings
    result = agent.migrate(source_db_config, memgraph_config)
    return result, agent


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

    # Print post-migration validation results
    validation_report = result.get("validation_report")
    if validation_report:
        print("\n✅ Post-migration Validation:")
        if validation_report.get("success"):
            print("  🎯 Status: PASSED")
        else:
            print("  ⚠️  Status: Issues found")

        # Display validation score and metrics if available
        validation_score = validation_report.get("validation_score", 0)
        print(f"  📊 Validation Score: {int(validation_score)}/100")

        metrics = validation_report.get("metrics")
        if metrics:
            print(f"  📁 Tables: {metrics.tables_covered}/{metrics.tables_total}")
            print(
                f"  🏷️  Properties: {metrics.properties_covered}/{metrics.properties_total}"
            )
            print(
                f"  🔗 Relationships: {metrics.relationships_covered}/{metrics.relationships_total}"
            )
            print(f"  📇 Indexes: {metrics.indexes_covered}/{metrics.indexes_total}")
            print(
                f"  🔒 Constraints: {metrics.constraints_covered}/{metrics.constraints_total}"
            )

        # Show validation issues summary
        issues = validation_report.get("issues", [])
        if issues:
            critical_count = sum(
                1 for issue in issues if issue.get("severity") == "CRITICAL"
            )
            warning_count = sum(
                1 for issue in issues if issue.get("severity") == "WARNING"
            )
            info_count = sum(1 for issue in issues if issue.get("severity") == "INFO")

            print(
                f"  🚨 Issues: {critical_count} critical, {warning_count} warnings, {info_count} info"
            )

            # Show top critical issues
            critical_issues = [
                issue for issue in issues if issue.get("severity") == "CRITICAL"
            ]
            if critical_issues:
                print("  📋 Top Critical Issues:")
                for issue in critical_issues[:3]:
                    print(f"    - {issue.get('message', 'Unknown issue')}")
        else:
            print("  ✅ No validation issues found")

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


def graph_model_to_mapping(graph_model: Any) -> Dict[str, Any]:
    """
    Convert an internal GraphModel into the federated-GQL mapping format.

    The output JSON contains ``nodes`` and ``edges`` arrays that map graph
    labels / relationship types back to their source SQL tables and columns.
    """
    nodes = []
    for node in graph_model.nodes:
        entry: Dict[str, Any] = {
            "label": node.primary_label,
            "table": node.source.name if node.source else "",
            "id_column": (
                node.source.mapping.get("id_field", "") if node.source else ""
            ),
            "properties": {},
        }
        for prop in node.properties:
            # prop.source.field is "table.column"; we only need the column part
            if prop.source and prop.source.field:
                column = prop.source.field.split(".", 1)[-1]
            else:
                column = prop.key
            entry["properties"][prop.key] = column
        nodes.append(entry)

    edges = []
    for edge in graph_model.edges:
        source_mapping = edge.source.mapping if edge.source else {}
        # start_node / end_node are stored as "table.column"
        start_node_ref = source_mapping.get("start_node", "")
        end_node_ref = source_mapping.get("end_node", "")

        entry: Dict[str, Any] = {
            "rel_type": edge.edge_type,
            "table": edge.source.name if edge.source else "",
            "source_column": start_node_ref.split(".", 1)[-1] if start_node_ref else "",
            "target_column": end_node_ref.split(".", 1)[-1] if end_node_ref else "",
            "source_label": edge.start_node_labels[0] if edge.start_node_labels else "",
            "target_label": edge.end_node_labels[0] if edge.end_node_labels else "",
        }
        # Include edge properties when present
        if edge.properties:
            entry["properties"] = {}
            for prop in edge.properties:
                if prop.source and prop.source.field:
                    column = prop.source.field.split(".", 1)[-1]
                else:
                    column = prop.key
                entry["properties"][prop.key] = column
        edges.append(entry)

    return {"nodes": nodes, "edges": edges}


def generate_mapping(
    agent: SQLToMemgraphAgent,
    source_db_config: Dict[str, Any],
    mapping_path: str,
) -> None:
    """
    Run only schema analysis and graph modeling, then write the mapping file.

    This skips the full migration workflow entirely - no Memgraph connection
    is required.
    """
    from database.factory import DatabaseAnalyzerFactory
    from core.hygm import HyGM

    # 1. Connect and analyse the source database
    print("🔍 Analyzing source database schema...")
    config = source_db_config.copy()
    db_type = config.pop("database_type", "mysql")
    analyzer = DatabaseAnalyzerFactory.create_analyzer(
        database_type=db_type, **config
    )
    if not analyzer.connect():
        raise DatabaseConnectionError("Failed to connect to source database")

    db_structure = analyzer.get_database_structure()
    hygm_data = db_structure.to_hygm_format()
    analyzer.disconnect()
    print(f"  Found {len(hygm_data.get('entity_tables', {}))} entity tables")

    # 2. Build the graph model via HyGM
    print("🎯 Creating graph model...")
    graph_modeler = HyGM(
        llm=agent.llm,
        mode=agent.modeling_mode,
        strategy=agent.graph_modeling_strategy,
    )
    graph_model = graph_modeler.create_graph_model(
        hygm_data,
        domain_context="Database migration to graph database",
    )
    print(
        f"  {len(graph_model.nodes)} node types, "
        f"{len(graph_model.edges)} relationship types"
    )

    # 3. Write mapping file
    mapping = graph_model_to_mapping(graph_model)
    output = Path(mapping_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n📄 Mapping file written to {output}")


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the migration agent."""
    args = parse_cli_args(argv)

    _configure_log_level(args.log_level)

    print_banner()

    try:
        # Setup and validate environment
        print("🔧 Setting up environment...")
        source_db_config, memgraph_config = setup_and_validate_environment()
        print("✅ Environment validation completed")
        print()

        if args.mapping:
            # Mapping-only mode: connect to source DB, model, write file
            db_type = source_db_config.get("database_type", "mysql")
            db_name = source_db_config.get("database", "")
            db_host = source_db_config.get("host", "")
            print(f"🔌 Connecting to {db_type} database {db_name}@{db_host}...")
            source_ok, source_err = probe_source_connection(source_db_config)
            if not source_ok:
                raise DatabaseConnectionError(
                    f"Source database connection failed: {source_err}"
                )
            print(f"✅ Connected to {db_type} — ready for modeling")
            print()

            # Get user preferences
            graph_mode = _resolve_mode(args.mode) or get_graph_modeling_mode()
            graph_strategy = (
                _resolve_strategy(args.strategy) or get_graph_modeling_strategy()
            )

            meta_graph_policy = (args.meta_graph or "auto").lower()
            if meta_graph_policy not in META_GRAPH_POLICIES:
                logger.warning(
                    "Unrecognised meta graph policy '%s'; defaulting to auto",
                    meta_graph_policy,
                )
                meta_graph_policy = "auto"

            print("📄 Mapping mode: generating mapping file (no migration)...")
            print()
            agent = SQLToMemgraphAgent(
                modeling_mode=graph_mode,
                graph_modeling_strategy=graph_strategy,
                meta_graph_policy=meta_graph_policy,
                llm_provider=args.provider,
                llm_model=args.model,
            )
            generate_mapping(agent, source_db_config, args.mapping)
        else:
            # Full migration (original flow)
            # Probe database connections
            print("🔌 Testing database connections...")
            probe_all_connections(source_db_config, memgraph_config)
            print("✅ All connections verified")
            print()

            # Get user preferences
            graph_mode = _resolve_mode(args.mode) or get_graph_modeling_mode()
            graph_strategy = (
                _resolve_strategy(args.strategy) or get_graph_modeling_strategy()
            )

            meta_graph_policy = (args.meta_graph or "auto").lower()
            if meta_graph_policy not in META_GRAPH_POLICIES:
                logger.warning(
                    "Unrecognised meta graph policy '%s'; defaulting to auto",
                    meta_graph_policy,
                )
                meta_graph_policy = "auto"

            # Run migration
            result, agent = run_migration(
                source_db_config,
                memgraph_config,
                graph_mode,
                graph_strategy,
                meta_graph_policy,
                llm_provider=args.provider,
                llm_model=args.model,
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
