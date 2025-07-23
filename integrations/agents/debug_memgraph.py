#!/usr/bin/env python3
"""
Debug script to examine and test individual Memgraph migration queries
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "memgraph-toolbox" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "langchain-memgraph"))

from dotenv import load_dotenv
from main import MySQLToMemgraphAgent
from memgraph_toolbox.api.memgraph import Memgraph

# Load environment variables
load_dotenv()


def debug_memgraph_queries():
    """Debug Memgraph query execution issues"""

    # MySQL config
    mysql_config = {
        "host": os.getenv("MYSQL_HOST", "host.docker.internal"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "password"),
        "database": os.getenv("MYSQL_DATABASE", "sakila"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
    }

    # Memgraph config
    memgraph_config = {
        "url": os.getenv("MEMGRAPH_URL", "bolt://localhost:7687"),
        "username": os.getenv("MEMGRAPH_USER", ""),
        "password": os.getenv("MEMGRAPH_PASSWORD", ""),
        "database": os.getenv("MEMGRAPH_DATABASE", "memgraph"),
    }

    try:
        print("🔍 DEBUGGING MEMGRAPH MIGRATION QUERIES")
        print("=" * 50)

        # Test basic Memgraph connection
        print("1. Testing Memgraph connection...")
        client = Memgraph(
            url=memgraph_config["url"],
            username=memgraph_config["username"],
            password=memgraph_config["password"],
            database=memgraph_config["database"],
        )

        # Test basic query
        result = client.query("RETURN 1 as test")
        print(f"   ✅ Basic connection works: {result}")

        # Test database clearing queries
        print("\n2. Testing database clearing queries...")

        # First, let's see what's in the database
        try:
            node_count = client.query("MATCH (n) RETURN count(n) as count")
            print(
                f"   Current nodes in database: {node_count[0]['count'] if node_count else 0}"
            )
        except Exception as e:
            print(f"   ⚠️ Could not count nodes: {e}")

        # Test the problematic clearing queries
        clearing_queries = [
            "MATCH (n) DETACH DELETE n",
            "DROP CONSTRAINT ON (n) ASSERT exists(n.id)",
        ]

        for i, query in enumerate(clearing_queries):
            try:
                print(f"   Testing clearing query {i+1}: {query}")
                result = client.query(query)
                print(f"   ✅ Success: {result}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

        # Generate migration queries to test
        print("\n3. Generating migration queries...")
        agent = MySQLToMemgraphAgent(
            relationship_naming_strategy="table_based",
            interactive_table_selection=False,
        )

        # Create initial state
        initial_state = {
            "mysql_config": mysql_config,
            "memgraph_config": memgraph_config,
            "database_structure": {},
            "migration_queries": [],
            "migration_plan": "",
            "current_step": "Initializing",
            "errors": [],
            "completed_tables": [],
            "total_tables": 0,
        }

        # Run through schema analysis and query generation
        state = agent._analyze_mysql_schema(initial_state)
        state = agent._select_tables_for_migration(state)
        state = agent._generate_cypher_queries(state)

        queries = state["migration_queries"]
        print(f"   Generated {len(queries)} queries")

        # Test first few queries individually
        print("\n4. Testing individual migration queries...")

        # Test MySQL connection through migrate module
        mysql_config_str = agent._get_mysql_config_for_migrate(mysql_config)
        test_migrate_query = f"""
        CALL migrate.mysql('SELECT 1 as test_col LIMIT 1', {mysql_config_str})
        YIELD row
        RETURN row.test_col as result
        """

        try:
            print("   Testing migrate.mysql() connection...")
            result = client.query(test_migrate_query)
            print(f"   ✅ migrate.mysql() works: {result}")
        except Exception as e:
            print(f"   ❌ migrate.mysql() failed: {e}")
            return

        # Test first node creation query
        if queries:
            print(f"\n   Testing first migration query...")
            first_query = queries[0]
            print(f"   Query: {first_query[:200]}...")

            try:
                result = client.query(first_query)
                print(f"   ✅ First query succeeded: {result}")

                # Check if nodes were created
                node_count = client.query("MATCH (n) RETURN count(n) as count")
                print(
                    f"   Nodes after first query: {node_count[0]['count'] if node_count else 0}"
                )

            except Exception as e:
                print(f"   ❌ First query failed: {e}")
                print(f"   Full query was:")
                print(first_query)

        # Test a simple node creation without migrate module
        print("\n5. Testing simple node creation (without migrate)...")
        simple_query = "CREATE (n:TestNode {id: 1, name: 'test'}) RETURN n"
        try:
            result = client.query(simple_query)
            print(f"   ✅ Simple node creation works: {result}")
        except Exception as e:
            print(f"   ❌ Simple node creation failed: {e}")

        # Clean up test node
        try:
            client.query("MATCH (n:TestNode) DELETE n")
        except:
            pass

    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "client" in locals():
            try:
                client.close()
            except:
                pass


if __name__ == "__main__":
    debug_memgraph_queries()
