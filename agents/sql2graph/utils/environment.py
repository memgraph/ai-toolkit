"""
Environment and Database Configuration Utilities

This module handles environment variable validation, database connection
probing, and configuration setup for the SQL to graph migration agent.
"""

import os
import logging
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class MigrationEnvironmentError(Exception):
    """Custom exception for environment-related errors."""


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""


SUPPORTED_DATABASES = {"mysql", "postgresql"}


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_source_db_type() -> str:
    """Return the configured source database type with validation."""
    db_type = os.getenv("SOURCE_DB_TYPE", "mysql").strip().lower()
    if db_type not in SUPPORTED_DATABASES:
        logger.warning(
            "Unsupported SOURCE_DB_TYPE '%s'; defaulting to MySQL",
            db_type,
        )
        return "mysql"
    return db_type


def get_required_environment_variables() -> Dict[str, str]:
    """Get the required environment variables and their descriptions."""
    db_type = get_source_db_type()

    base_vars: Dict[str, str] = {
        "SOURCE_DB_TYPE": "Source database type (mysql|postgresql)",
        "MEMGRAPH_URL": ("Memgraph connection URL (default: bolt://localhost:7687)"),
    }

    # Note: LLM API keys are optional, at least one needed for LLM strategy
    # - OPENAI_API_KEY: OpenAI (GPT models)
    # - ANTHROPIC_API_KEY: Anthropic (Claude models)
    # - GOOGLE_API_KEY: Google (Gemini models)

    if db_type == "postgresql":
        base_vars.update(
            {
                "POSTGRES_HOST": "PostgreSQL host (default: localhost)",
                "POSTGRES_PORT": "PostgreSQL port (default: 5432)",
                "POSTGRES_USER": "PostgreSQL user (default: postgres)",
                "POSTGRES_PASSWORD": "PostgreSQL database password",
                "POSTGRES_DATABASE": ("PostgreSQL database name (default: postgres)"),
                "POSTGRES_SCHEMA": "PostgreSQL schema (default: public)",
            }
        )
    else:
        base_vars.update(
            {
                "MYSQL_HOST": "MySQL host (default: host.docker.internal)",
                "MYSQL_PORT": "MySQL port (default: 3306)",
                "MYSQL_USER": "MySQL user (default: root)",
                "MYSQL_PASSWORD": "MySQL database password",
                "MYSQL_DATABASE": "MySQL database name (default: sakila)",
            }
        )

    return base_vars


def get_optional_environment_variables() -> Dict[str, str]:
    """Get optional environment variables and their descriptions."""
    optional_vars = {
        "MEMGRAPH_USERNAME": "Memgraph username (default: empty)",
        "MEMGRAPH_PASSWORD": "Memgraph password (default: empty)",
        "MEMGRAPH_DATABASE": "Memgraph database name (default: memgraph)",
    }

    if get_source_db_type() == "postgresql":
        optional_vars.setdefault(
            "POSTGRES_SCHEMA", "PostgreSQL schema (default: public)"
        )

    return optional_vars


def validate_environment_variables() -> Tuple[bool, List[str]]:
    """
    Validate required environment variables.

    Returns:
        Tuple of (is_valid, missing_variables)
    """
    missing_vars: List[str] = []
    required_vars = get_required_environment_variables()

    db_type = get_source_db_type()

    # LLM API keys are no longer required - checked separately if using LLM strategy

    if db_type == "postgresql":
        if not os.getenv("POSTGRES_PASSWORD"):
            logger.warning(
                "POSTGRES_PASSWORD missing; attempting passwordless connection"
            )
    else:
        if not os.getenv("MYSQL_PASSWORD"):
            logger.warning("MYSQL_PASSWORD missing; attempting passwordless connection")

    return len(missing_vars) == 0, missing_vars


def get_source_db_config() -> Dict[str, Any]:
    """Get source database configuration from environment variables."""
    db_type = get_source_db_type()

    if db_type == "postgresql":
        return {
            "database_type": "postgresql",
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "database": os.getenv("POSTGRES_DATABASE", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "schema": os.getenv("POSTGRES_SCHEMA", "public"),
        }

    return {
        "database_type": "mysql",
        "host": os.getenv("MYSQL_HOST", "host.docker.internal"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "sakila"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
    }


def get_memgraph_config() -> Dict[str, str]:
    """
    Get Memgraph configuration from environment variables.

    Returns:
        Dictionary with Memgraph connection parameters.
    """
    return {
        "url": os.getenv("MEMGRAPH_URL", "bolt://localhost:7687"),
        "username": os.getenv("MEMGRAPH_USERNAME", ""),
        "password": os.getenv("MEMGRAPH_PASSWORD", ""),
        "database": os.getenv("MEMGRAPH_DATABASE", "memgraph"),
    }


def probe_source_connection(
    source_db_config: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Test source database connection using the configured analyzer."""
    try:
        import sys
        from pathlib import Path

        agents_root = Path(__file__).parent.parent
        if str(agents_root) not in sys.path:
            sys.path.insert(0, str(agents_root))

        from database.factory import DatabaseAnalyzerFactory

        config = source_db_config.copy()
        db_type = config.pop("database_type", "mysql")
        analyzer = DatabaseAnalyzerFactory.create_analyzer(db_type, **config)
        if analyzer.connect():
            analyzer.get_database_structure()
            analyzer.disconnect()
            return True, None
        return False, "Failed to establish connection"

    except ImportError as e:
        return False, f"Missing database dependencies: {e}"
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Connection error: {e}"


def probe_memgraph_connection(
    memgraph_config: Dict[str, str]
) -> Tuple[bool, Optional[str]]:
    """
    Test Memgraph database connection.

    Args:
        memgraph_config: Memgraph connection configuration

    Returns:
        Tuple of (is_connected, error_message)
    """
    try:
        from memgraph_toolbox.api.memgraph import Memgraph

        client = Memgraph(
            url=str(memgraph_config.get("url", "bolt://localhost:7687")),
            username=str(memgraph_config.get("username", "")),
            password=str(memgraph_config.get("password", "")),
            database=str(memgraph_config.get("database", "memgraph")),
        )

        client.query("MATCH (n) RETURN count(n) as node_count LIMIT 1")
        client.close()
        return True, None

    except ImportError as e:
        return False, f"Missing Memgraph dependencies: {e}"
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Connection error: {e}"


def validate_llm_providers() -> Tuple[bool, List[str], List[str]]:
    """
    Validate LLM provider API keys by making test requests.

    Returns:
        Tuple of (has_valid_provider, valid_providers, error_messages)
    """
    valid_providers = []
    errors = []

    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            llm.invoke("Test")
            valid_providers.append("OpenAI")
        except ImportError:
            errors.append("OpenAI: Missing dependencies (langchain-openai)")
        except Exception as e:  # pylint: disable=broad-except
            errors.append(f"OpenAI: {e}")

    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
            llm.invoke("Test")
            valid_providers.append("Anthropic")
        except ImportError:
            errors.append("Anthropic: Missing dependencies (langchain-anthropic)")
        except Exception as e:  # pylint: disable=broad-except
            errors.append(f"Anthropic: {e}")

    # Check Google (Gemini)
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
                google_api_key=google_key,
            )
            llm.invoke("Test")
            valid_providers.append("Gemini")
        except ImportError:
            errors.append("Gemini: Missing dependencies (langchain-google-genai)")
        except Exception as e:  # pylint: disable=broad-except
            errors.append(f"Gemini: {e}")

    has_valid = len(valid_providers) > 0
    return has_valid, valid_providers, errors


def setup_and_validate_environment() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Complete environment setup and validation.

    Returns:
        Tuple of (source_db_config, memgraph_config)

    Raises:
        MigrationEnvironmentError: If environment validation fails
        DatabaseConnectionError: If database connections fail
    """
    load_environment()

    is_valid, missing_vars = validate_environment_variables()
    if not is_valid:
        error_msg = "Missing required environment variables:\n"
        for var in missing_vars:
            error_msg += f"  - {var}\n"
        error_msg += (
            "\nPlease check your .env file and ensure all required variables " "are set"
        )
        raise MigrationEnvironmentError(error_msg)

    source_db_config = get_source_db_config()
    memgraph_config = get_memgraph_config()

    logger.info("Environment variables loaded successfully")
    return source_db_config, memgraph_config


def probe_all_connections(
    source_db_config: Dict[str, Any], memgraph_config: Dict[str, str]
) -> None:
    """
    Probe all database connections and validate API keys.

    Args:
        source_db_config: Source database connection configuration
        memgraph_config: Memgraph connection configuration

    Raises:
        DatabaseConnectionError: If any connection fails
    """
    errors: List[str] = []

    logger.info("Validating LLM provider API keys...")
    has_valid, valid_providers, llm_errors = validate_llm_providers()
    if has_valid:
        logger.info("✅ Valid LLM providers: %s", ", ".join(valid_providers))
    else:
        logger.warning("⚠️  No valid LLM providers found")

    for error in llm_errors:
        logger.warning("  %s", error)

    db_type = source_db_config.get("database_type", "mysql")
    logger.info("Testing %s connection...", db_type.capitalize())
    source_connected, source_error = probe_source_connection(source_db_config)
    if not source_connected:
        errors.append(f"{db_type}: {source_error}")
    else:
        logger.info(
            "✅ %s connection successful to %s@%s",
            db_type.capitalize(),
            source_db_config.get("database"),
            source_db_config.get("host"),
        )

    logger.info("Testing Memgraph connection...")
    memgraph_connected, memgraph_error = probe_memgraph_connection(memgraph_config)
    if not memgraph_connected:
        errors.append(f"Memgraph: {memgraph_error}")
    else:
        logger.info(
            "✅ Memgraph connection successful to %s",
            memgraph_config["url"],
        )

    if errors:
        error_msg = "Database connection failures:\n"
        for error in errors:
            error_msg += f"  - {error}\n"
        raise DatabaseConnectionError(error_msg)


def print_environment_help() -> None:
    """Print helpful environment setup information."""
    print("❌ Setup Error: Missing required environment variables")
    print("\nPlease ensure you have:")
    print("1. Created a .env file (copy from .env.example)")
    print("2. Set your OPENAI_API_KEY")
    print("3. Set SOURCE_DB_TYPE to mysql or postgresql")
    print("4. Provide the credentials for the selected source database")
    print("\nExample .env file:")
    print("OPENAI_API_KEY=your_openai_key_here")
    print("SOURCE_DB_TYPE=postgresql")
    print("POSTGRES_PASSWORD=your_postgres_password")
    print("POSTGRES_HOST=localhost")
    print("POSTGRES_USER=postgres")
    print("POSTGRES_DATABASE=your_database")
    print("MEMGRAPH_URL=bolt://localhost:7687")

    print("\nRequired environment variables:")
    for var, desc in get_required_environment_variables().items():
        print(f"  - {var}: {desc}")

    print("\nOptional environment variables:")
    for var, desc in get_optional_environment_variables().items():
        print(f"  - {var}: {desc}")


def print_troubleshooting_help() -> None:
    """Print troubleshooting information."""
    print("\nTroubleshooting steps:")
    print("1. Check your .env file exists and contains required variables")
    print("2. Verify your OpenAI API key is valid")
    print("3. Test the source database connection with dedicated probe scripts")
    print("4. Ensure Memgraph is running on the specified URL")
    print("5. Check network connectivity between services")
