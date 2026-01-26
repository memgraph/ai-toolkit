#!/usr/bin/env python3
"""
Quick setup script for GraphRAG with Memgraph.

This script helps verify your environment is correctly configured
for building GraphRAG applications with Memgraph.

Usage:
    python setup_check.py
"""

import sys
import os


def check_python_version():
    """Check Python version is 3.10+."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_packages():
    """Check required packages are installed."""
    packages = {
        "memgraph_toolbox": "memgraph-toolbox",
        "unstructured": "unstructured",
        "lightrag": "lightrag",
        "openai": "openai",
    }

    all_installed = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (pip install {package_name})")
            all_installed = False

    # Check optional packages
    optional = {
        "lightrag_memgraph": "lightrag-memgraph",
        "unstructured2graph": "unstructured2graph",
    }

    for import_name, package_name in optional.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"○ {package_name} (optional, pip install {package_name})")

    return all_installed


def check_env_vars():
    """Check required environment variables."""
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["MEMGRAPH_HOST", "MEMGRAPH_PORT"]

    all_set = True
    for var in required_vars:
        if os.environ.get(var):
            print(f"✓ {var} is set")
        else:
            print(f"✗ {var} is not set")
            all_set = False

    for var in optional_vars:
        if os.environ.get(var):
            print(f"✓ {var} = {os.environ.get(var)}")
        else:
            print(f"○ {var} not set (using defaults)")

    return all_set


def check_memgraph_connection():
    """Check connection to Memgraph."""
    try:
        from memgraph_toolbox.api.memgraph import Memgraph

        host = os.environ.get("MEMGRAPH_HOST", "localhost")
        port = os.environ.get("MEMGRAPH_PORT", "7687")
        url = f"bolt://{host}:{port}"

        mg = Memgraph(url=url)
        result = mg.query("RETURN 1 AS test")

        if result and result[0]["test"] == 1:
            print(f"✓ Connected to Memgraph at {url}")

            # Check for MAGE algorithms
            try:
                mg.query("CALL mg.procedures() YIELD name RETURN name LIMIT 1")
                print("✓ MAGE algorithms available")
            except:
                print("○ MAGE algorithms not detected (some features may be limited)")

            return True
    except Exception as e:
        print(f"✗ Cannot connect to Memgraph: {e}")
        print("  Start Memgraph: docker run -p 7687:7687 memgraph/memgraph-mage")
        return False


def check_vector_search():
    """Check if vector search is available."""
    try:
        from memgraph_toolbox.api.memgraph import Memgraph

        host = os.environ.get("MEMGRAPH_HOST", "localhost")
        port = os.environ.get("MEMGRAPH_PORT", "7687")
        url = f"bolt://{host}:{port}"

        mg = Memgraph(url=url)

        # Check for embeddings module
        try:
            mg.query(
                "CALL mg.procedures() YIELD name WHERE name STARTS WITH 'embeddings' RETURN name LIMIT 1"
            )
            print("✓ Embeddings module available")
        except:
            print("○ Embeddings module not detected")

        # Check for vector_search module
        try:
            mg.query(
                "CALL mg.procedures() YIELD name WHERE name STARTS WITH 'vector_search' RETURN name LIMIT 1"
            )
            print("✓ Vector search module available")
        except:
            print("○ Vector search module not detected")

        return True
    except:
        return False


def main():
    """Run all checks."""
    print("=" * 50)
    print("GraphRAG with Memgraph - Environment Check")
    print("=" * 50)
    print()

    print("Python Version:")
    print("-" * 30)
    python_ok = check_python_version()
    print()

    print("Required Packages:")
    print("-" * 30)
    packages_ok = check_packages()
    print()

    print("Environment Variables:")
    print("-" * 30)
    env_ok = check_env_vars()
    print()

    print("Memgraph Connection:")
    print("-" * 30)
    memgraph_ok = check_memgraph_connection()
    print()

    if memgraph_ok:
        print("Vector Search Capabilities:")
        print("-" * 30)
        check_vector_search()
        print()

    print("=" * 50)
    if python_ok and packages_ok and env_ok and memgraph_ok:
        print("✓ All checks passed! Ready for GraphRAG.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
