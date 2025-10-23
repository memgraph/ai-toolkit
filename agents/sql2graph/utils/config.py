"""
Configuration management utilities for the migration agent.

This module handles configuration parsing, validation, and default settings
for different environments and use cases.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .environment import get_source_db_type


@dataclass
class MigrationConfig:
    """Configuration class for migration settings."""

    # Source database settings
    source_db_type: str
    source_db_host: str
    source_db_user: str
    source_db_password: str
    source_db_database: str
    source_db_port: int
    source_db_schema: Optional[str]

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
        db_type = get_source_db_type()
        if db_type == "postgresql":
            source_host = os.getenv("POSTGRES_HOST", "localhost")
            source_user = os.getenv("POSTGRES_USER", "postgres")
            source_password = os.getenv("POSTGRES_PASSWORD", "")
            source_database = os.getenv("POSTGRES_DATABASE", "postgres")
            source_port = int(os.getenv("POSTGRES_PORT", "5432"))
            source_schema = os.getenv("POSTGRES_SCHEMA", "public")
        else:
            source_host = os.getenv("MYSQL_HOST", "host.docker.internal")
            source_user = os.getenv("MYSQL_USER", "root")
            source_password = os.getenv("MYSQL_PASSWORD", "")
            source_database = os.getenv("MYSQL_DATABASE", "sakila")
            source_port = int(os.getenv("MYSQL_PORT", "3306"))
            source_schema = None

        return cls(
            source_db_type=db_type,
            source_db_host=source_host,
            source_db_user=source_user,
            source_db_password=source_password,
            source_db_database=source_database,
            source_db_port=source_port,
            source_db_schema=source_schema,
            memgraph_url=os.getenv("MEMGRAPH_URL", "bolt://localhost:7687"),
            memgraph_username=os.getenv("MEMGRAPH_USERNAME", ""),
            memgraph_password=os.getenv("MEMGRAPH_PASSWORD", ""),
            memgraph_database=os.getenv("MEMGRAPH_DATABASE", "memgraph"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            relationship_naming_strategy=os.getenv(
                "RELATIONSHIP_NAMING_STRATEGY", "table_based"
            ),
            interactive_table_selection=os.getenv(
                "INTERACTIVE_TABLE_SELECTION", "true"
            ).lower()
            == "true",
        )

    def to_source_db_config(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for analyzer creation."""
        config: Dict[str, Any] = {
            "database_type": self.source_db_type,
            "host": self.source_db_host,
            "user": self.source_db_user,
            "password": self.source_db_password,
            "database": self.source_db_database,
            "port": self.source_db_port,
        }
        if self.source_db_type == "postgresql" and self.source_db_schema:
            config["schema"] = self.source_db_schema
        return config

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
        errors: list[str] = []

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")

        if self.source_db_type == "postgresql":
            if not 1 <= self.source_db_port <= 65535:
                errors.append(f"Invalid PostgreSQL port: {self.source_db_port}")
        else:
            if not 1 <= self.source_db_port <= 65535:
                errors.append(f"Invalid MySQL port: {self.source_db_port}")

        valid_strategies = ["table_based", "llm"]
        if self.relationship_naming_strategy not in valid_strategies:
            errors.append(
                "Invalid relationship_naming_strategy: "
                f"{self.relationship_naming_strategy}. Must be one of: "
                f"{valid_strategies}"
            )

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
            "source_db_type": "mysql",
            "source_db_host": "localhost",
            "source_db_port": 3306,
            "source_db_user": "root",
            "source_db_database": "sakila",
            "memgraph_url": "bolt://localhost:7687",
            "relationship_naming_strategy": "table_based",
            "interactive_table_selection": True,
        },
        "docker_development": {
            "source_db_type": "mysql",
            "source_db_host": "host.docker.internal",
            "source_db_port": 3306,
            "source_db_user": "root",
            "source_db_database": "sakila",
            "memgraph_url": "bolt://localhost:7687",
            "relationship_naming_strategy": "table_based",
            "interactive_table_selection": True,
        },
        "production": {
            "source_db_type": "mysql",
            "source_db_host": "mysql-server",
            "source_db_port": 3306,
            "source_db_user": "migration_user",
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

    for key, preset_value in preset.items():
        if key not in config_dict:
            continue
        current_value = config_dict[key]
        if (
            not current_value
            or (key == "source_db_host" and current_value == "host.docker.internal")
            or (key == "memgraph_url" and current_value == "bolt://localhost:7687")
        ):
            config_dict[key] = preset_value

    return MigrationConfig(**config_dict)


def print_config_summary(config: MigrationConfig) -> None:
    """Print a summary of the configuration."""
    print("🔧 Configuration Summary:")
    print("-" * 30)
    source_details = (
        f"Source DB: {config.source_db_type}://"
        f"{config.source_db_user}@{config.source_db_host}:"
        f"{config.source_db_port}"
    )
    print(source_details)
    print(f"Database: {config.source_db_database}")
    if config.source_db_type == "postgresql" and config.source_db_schema:
        print(f"Schema: {config.source_db_schema}")
    print(f"Memgraph: {config.memgraph_url}")
    print(f"Strategy: {config.relationship_naming_strategy}")
    print(f"Interactive: {config.interactive_table_selection}")
    print(f"OpenAI API: {'✅ Set' if config.openai_api_key else '❌ Missing'}")
    print()


def get_available_presets() -> list[str]:
    """Get a list of available preset names."""
    return ["local_development", "docker_development", "production"]
