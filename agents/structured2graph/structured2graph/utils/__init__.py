"""
Utilities package for the SQL to graph migration agent.

This package contains reusable utility modules for environment management,
database probing, configuration management, and other common functionality.
"""

from .config import (
    MigrationConfig,
    get_available_presets,
    get_preset_config,
    merge_config_with_preset,
    print_config_summary,
)
from .environment import (
    DatabaseConnectionError,
    MigrationEnvironmentError,
    get_memgraph_config,
    get_optional_environment_variables,
    get_required_environment_variables,
    get_source_db_config,
    load_environment,
    print_environment_help,
    print_troubleshooting_help,
    probe_all_connections,
    probe_memgraph_connection,
    probe_source_connection,
    setup_and_validate_environment,
    validate_environment_variables,
    validate_llm_providers,
)

__all__ = [
    "DatabaseConnectionError",
    # Configuration utilities
    "MigrationConfig",
    # Environment utilities
    "MigrationEnvironmentError",
    "get_available_presets",
    "get_memgraph_config",
    "get_optional_environment_variables",
    "get_preset_config",
    "get_required_environment_variables",
    "get_source_db_config",
    "load_environment",
    "merge_config_with_preset",
    "print_config_summary",
    "print_environment_help",
    "print_troubleshooting_help",
    "probe_all_connections",
    "probe_memgraph_connection",
    "probe_source_connection",
    "setup_and_validate_environment",
    "validate_environment_variables",
    "validate_llm_providers",
]
