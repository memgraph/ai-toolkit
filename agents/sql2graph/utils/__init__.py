"""
Utilities package for the SQL to graph migration agent.

This package contains reusable utility modules for environment management,
database probing, configuration management, and other common functionality.
"""

from .environment import (
    MigrationEnvironmentError,
    DatabaseConnectionError,
    load_environment,
    get_required_environment_variables,
    get_optional_environment_variables,
    validate_environment_variables,
    get_source_db_config,
    get_memgraph_config,
    probe_source_connection,
    probe_memgraph_connection,
    validate_openai_api_key,
    setup_and_validate_environment,
    probe_all_connections,
    print_environment_help,
    print_troubleshooting_help,
)

from .config import (
    MigrationConfig,
    get_preset_config,
    merge_config_with_preset,
    print_config_summary,
    get_available_presets,
)

__all__ = [
    # Environment utilities
    "MigrationEnvironmentError",
    "DatabaseConnectionError",
    "load_environment",
    "get_required_environment_variables",
    "get_optional_environment_variables",
    "validate_environment_variables",
    "get_source_db_config",
    "get_memgraph_config",
    "probe_source_connection",
    "probe_memgraph_connection",
    "validate_openai_api_key",
    "setup_and_validate_environment",
    "probe_all_connections",
    "print_environment_help",
    "print_troubleshooting_help",
    # Configuration utilities
    "MigrationConfig",
    "get_preset_config",
    "merge_config_with_preset",
    "print_config_summary",
    "get_available_presets",
]
