"""
Configuration management utilities for the migration agent.

This module handles configuration parsing, validation, and default settings
for different environments and use cases.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MigrationConfig:
    """Configuration class for migration settings."""

    # MySQL settings
    mysql_host: str
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_port: int

    # Memgraph settings
    memgraph_url: str
    memgraph_username: str
    memgraph_password: str
    memgraph_database: str

    # OpenAI settings
    openai_api_key: str

    # Migration settings
    relationship_naming_strategy: str = "table_based"
    interactive_table_selection: bool = True

    @classmethod
    def from_environment(cls) -> "MigrationConfig":
        """Create configuration from environment variables."""
        return cls(
            # MySQL settings
            mysql_host=os.getenv("MYSQL_HOST", "host.docker.internal"),
            mysql_user=os.getenv("MYSQL_USER", "root"),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DATABASE", "sakila"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            # Memgraph settings
            memgraph_url=os.getenv("MEMGRAPH_URL", "bolt://localhost:7687"),
            memgraph_username=os.getenv("MEMGRAPH_USERNAME", ""),
            memgraph_password=os.getenv("MEMGRAPH_PASSWORD", ""),
            memgraph_database=os.getenv("MEMGRAPH_DATABASE", "memgraph"),
            # OpenAI settings
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            # Migration settings
            relationship_naming_strategy=os.getenv(
                "RELATIONSHIP_NAMING_STRATEGY", "table_based"
            ),
            interactive_table_selection=os.getenv(
                "INTERACTIVE_TABLE_SELECTION", "true"
            ).lower()
            == "true",
        )

    def to_mysql_config(self) -> Dict[str, str]:
        """Convert to MySQL configuration dictionary."""
        return {
            "host": self.mysql_host,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "database": self.mysql_database,
            "port": self.mysql_port,
        }

    def to_memgraph_config(self) -> Dict[str, str]:
        """Convert to Memgraph configuration dictionary."""
        return {
            "url": self.memgraph_url,
            "username": self.memgraph_username,
            "password": self.memgraph_password,
            "database": self.memgraph_database,
        }

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the configuration.

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []

        # Required fields
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")

        if not self.mysql_password:
            errors.append("MYSQL_PASSWORD is required")

        # Validate strategy
        valid_strategies = ["table_based", "llm"]
        if self.relationship_naming_strategy not in valid_strategies:
            errors.append(
                f"Invalid relationship_naming_strategy: "
                f"{self.relationship_naming_strategy}. "
                f"Must be one of: {valid_strategies}"
            )

        # Validate port
        if not 1 <= self.mysql_port <= 65535:
            errors.append(f"Invalid MySQL port: {self.mysql_port}")

        return len(errors) == 0, errors


def get_preset_config(preset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a preset configuration for common scenarios.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        Dictionary with preset configuration values or None if not found
    """
    presets = {
        "local_development": {
            "mysql_host": "localhost",
            "mysql_port": 3306,
            "mysql_user": "root",
            "mysql_database": "sakila",
            "memgraph_url": "bolt://localhost:7687",
            "relationship_naming_strategy": "table_based",
            "interactive_table_selection": True,
        },
        "docker_development": {
            "mysql_host": "host.docker.internal",
            "mysql_port": 3306,
            "mysql_user": "root",
            "mysql_database": "sakila",
            "memgraph_url": "bolt://localhost:7687",
            "relationship_naming_strategy": "table_based",
            "interactive_table_selection": True,
        },
        "production": {
            "mysql_host": "mysql-server",
            "mysql_port": 3306,
            "mysql_user": "migration_user",
            "memgraph_url": "bolt://memgraph-server:7687",
            "relationship_naming_strategy": "llm",
            "interactive_table_selection": False,
        },
    }

    return presets.get(preset_name)


def merge_config_with_preset(
    config: MigrationConfig, preset_name: str
) -> MigrationConfig:
    """
    Merge configuration with a preset, keeping existing values.

    Args:
        config: Existing configuration
        preset_name: Name of the preset to merge

    Returns:
        New configuration with preset values applied where not set
    """
    preset = get_preset_config(preset_name)
    if not preset:
        return config

    config_dict = config.__dict__.copy()

    # Apply preset values only for empty/default values
    for key, preset_value in preset.items():
        if hasattr(config, key):
            current_value = getattr(config, key)
            # Apply preset if current value is empty or default
            if (
                not current_value
                or (key == "mysql_host" and current_value == "host.docker.internal")
                or (key == "memgraph_url" and current_value == "bolt://localhost:7687")
            ):
                config_dict[key] = preset_value

    return MigrationConfig(**config_dict)


def print_config_summary(config: MigrationConfig) -> None:
    """Print a summary of the configuration."""
    print("🔧 Configuration Summary:")
    print("-" * 30)
    print(f"MySQL: {config.mysql_user}@{config.mysql_host}:{config.mysql_port}")
    print(f"Database: {config.mysql_database}")
    print(f"Memgraph: {config.memgraph_url}")
    print(f"Strategy: {config.relationship_naming_strategy}")
    print(f"Interactive: {config.interactive_table_selection}")
    print(f"OpenAI API: {'✅ Set' if config.openai_api_key else '❌ Missing'}")
    print()


def get_available_presets() -> list[str]:
    """Get a list of available preset names."""
    return ["local_development", "docker_development", "production"]
