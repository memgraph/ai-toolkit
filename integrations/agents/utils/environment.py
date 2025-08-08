"""
Environment and Database Configuration Utilities

This module handles environment variable validation, database connection
probing, and configuration setup for the MySQL to Memgraph migration agent.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class MigrationEnvironmentError(Exception):
    """Custom exception for environment-related errors."""


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_required_environment_variables() -> Dict[str, str]:
    """Get the required environment variables and their descriptions."""
    return {
        "OPENAI_API_KEY": "OpenAI API key for migration planning",
        "MEMGRAPH_URL": ("Memgraph connection URL " "(default: bolt://localhost:7687)"),
        "MYSQL_HOST": "MySQL host (default: host.docker.internal)",
        "MYSQL_PORT": "MySQL port (default: 3306)",
        "MYSQL_USER": "MySQL database user (default: root)",
        "MYSQL_PASSWORD": "MySQL database password",
        "MYSQL_DATABASE": "MySQL database name (default: sakila)",
    }


def get_optional_environment_variables() -> Dict[str, str]:
    """Get optional environment variables and their descriptions."""
    return {
        "MEMGRAPH_USERNAME": "Memgraph username (default: empty)",
        "MEMGRAPH_PASSWORD": "Memgraph password (default: empty)",
        "MEMGRAPH_DATABASE": "Memgraph database name (default: memgraph)",
    }


def validate_environment_variables() -> Tuple[bool, List[str]]:
    """
    Validate required environment variables.

    Returns:
        Tuple of (is_valid, missing_variables)
    """
    required_vars = get_required_environment_variables()
    missing_vars = []

    for var, description in required_vars.items():
        # Only OPENAI_API_KEY and MYSQL_PASSWORD are truly required
        if var in ["OPENAI_API_KEY", "MYSQL_PASSWORD"] and not os.getenv(var):
            missing_vars.append(f"{var} ({description})")

    return len(missing_vars) == 0, missing_vars


def get_mysql_config() -> Dict[str, str]:
    """
    Get MySQL configuration from environment variables.

    Returns:
        Dictionary with MySQL connection parameters.
    """
    return {
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


def probe_mysql_connection(mysql_config: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """
    Test MySQL database connection.

    Args:
        mysql_config: MySQL connection configuration

    Returns:
        Tuple of (is_connected, error_message)
    """
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path

        # Add agents root to path for absolute imports
        agents_root = Path(__file__).parent.parent
        if str(agents_root) not in sys.path:
            sys.path.insert(0, str(agents_root))

        from database.adapters.mysql import MySQLAnalyzer

        analyzer = MySQLAnalyzer(**mysql_config)
        if analyzer.connect():
            # Test basic query
            analyzer.get_database_structure()
            analyzer.disconnect()
            return True, None
        else:
            return False, "Failed to establish connection"

    except ImportError as e:
        return False, f"Missing MySQL dependencies: {e}"
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
        # Import here to avoid circular imports
        from memgraph_toolbox.api.memgraph import Memgraph

        client = Memgraph(
            url=memgraph_config.get("url"),
            username=memgraph_config.get("username"),
            password=memgraph_config.get("password"),
            database=memgraph_config.get("database"),
        )

        # Test basic query
        client.query("MATCH (n) RETURN count(n) as node_count LIMIT 1")
        client.close()
        return True, None

    except ImportError as e:
        return False, f"Missing Memgraph dependencies: {e}"
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Connection error: {e}"


def validate_openai_api_key() -> Tuple[bool, Optional[str]]:
    """
    Validate OpenAI API key by making a test request.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not set"

        # Import here to avoid circular imports
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.1, api_key=api_key, max_tokens=10
        )

        # Test with a minimal request
        llm.invoke("Test")
        return True, None

    except ImportError as e:
        return False, f"Missing OpenAI dependencies: {e}"
    except Exception as e:  # pylint: disable=broad-except
        return False, f"API key validation error: {e}"


def setup_and_validate_environment() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Complete environment setup and validation.

    Returns:
        Tuple of (mysql_config, memgraph_config)

    Raises:
        MigrationEnvironmentError: If environment validation fails
        DatabaseConnectionError: If database connections fail
    """
    # Load environment variables
    load_environment()

    # Validate required environment variables
    is_valid, missing_vars = validate_environment_variables()
    if not is_valid:
        error_msg = "Missing required environment variables:\n"
        for var in missing_vars:
            error_msg += f"  - {var}\n"
        error_msg += (
            "\nPlease check your .env file and ensure all " "required variables are set"
        )
        raise MigrationEnvironmentError(error_msg)

    # Get configurations
    mysql_config = get_mysql_config()
    memgraph_config = get_memgraph_config()

    logger.info("Environment variables loaded successfully")
    return mysql_config, memgraph_config


def probe_all_connections(
    mysql_config: Dict[str, str], memgraph_config: Dict[str, str]
) -> None:
    """
    Probe all database connections and validate API keys.

    Args:
        mysql_config: MySQL connection configuration
        memgraph_config: Memgraph connection configuration

    Raises:
        DatabaseConnectionError: If any connection fails
    """
    errors = []

    # Test OpenAI API key
    logger.info("Validating OpenAI API key...")
    openai_valid, openai_error = validate_openai_api_key()
    if not openai_valid:
        errors.append(f"OpenAI: {openai_error}")
    else:
        logger.info("✅ OpenAI API key validated successfully")

    # Test MySQL connection
    logger.info("Testing MySQL connection...")
    mysql_connected, mysql_error = probe_mysql_connection(mysql_config)
    if not mysql_connected:
        errors.append(f"MySQL: {mysql_error}")
    else:
        logger.info(
            "✅ MySQL connection successful to %s@%s",
            mysql_config["database"],
            mysql_config["host"],
        )

    # Test Memgraph connection
    logger.info("Testing Memgraph connection...")
    memgraph_connected, memgraph_error = probe_memgraph_connection(memgraph_config)
    if not memgraph_connected:
        errors.append(f"Memgraph: {memgraph_error}")
    else:
        logger.info("✅ Memgraph connection successful to %s", memgraph_config["url"])

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
    print("3. Set your MYSQL_PASSWORD")
    print("\nExample .env file:")
    print("OPENAI_API_KEY=your_openai_key_here")
    print("MYSQL_PASSWORD=your_mysql_password")
    print("MYSQL_HOST=localhost")
    print("MYSQL_USER=root")
    print("MYSQL_DATABASE=sakila")
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
    print("3. Test MySQL connection with: uv run mysql_troubleshoot.py")
    print("4. Ensure Memgraph is running on the specified URL")
    print("5. Check network connectivity between services")
