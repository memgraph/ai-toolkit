"""Pytest configuration for MCP Memgraph tests."""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables before tests run."""
    # Set MCP_TRANSPORT to stdio for tests (if not already set)
    if "MCP_TRANSPORT" not in os.environ:
        os.environ["MCP_TRANSPORT"] = "stdio"

    # Ensure MCP_HOST is not set for stdio transport
    # (or set to empty string to avoid validation errors)
    if "MCP_HOST" not in os.environ:
        os.environ["MCP_HOST"] = ""

    yield

    # Cleanup after all tests (optional)
