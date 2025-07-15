# TODO(antejavor): This should become a test script for MySQL connection issues

#!/usr/bin/env python3
"""
MySQL Connection Troubleshooting Script
This script helps diagnose MySQL connection issues for the migration agent.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_mysql_import():
    """Test if MySQL connector can be imported."""
    print("Testing MySQL connector import...")
    try:
        import mysql.connector

        print("✓ mysql.connector imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import mysql.connector: {e}")
        print("Solution: Install mysql-connector-python")
        print("Run: pip install mysql-connector-python")
        return False


def check_environment_variables():
    """Check if required environment variables are set."""
    print("\nChecking environment variables...")

    env_vars = {
        "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
        "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
        "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE", "sakila"),
        "MYSQL_PORT": os.getenv("MYSQL_PORT", "3306"),
    }

    missing_vars = []
    for var, value in env_vars.items():
        if value is None or value == "":
            missing_vars.append(var)
            print(f"✗ {var}: Not set")
        else:
            # Mask password for security
            display_value = "*" * len(value) if "PASSWORD" in var else value
            print(f"✓ {var}: {display_value}")

    if missing_vars:
        print(f"\nMissing variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        return False
    return True


def test_basic_connection():
    """Test basic MySQL connection."""
    print("\nTesting MySQL connection...")

    try:
        import mysql.connector
        from mysql.connector import Error

        config = {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
        }

        print(
            f"Attempting to connect to {config['host']}:{config['port']} as {config['user']}"
        )

        connection = mysql.connector.connect(**config)

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✓ Successfully connected to MySQL Server version {db_info}")

            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            current_db = cursor.fetchone()
            print(f"✓ Current database: {current_db[0] if current_db[0] else 'None'}")

            cursor.close()
            connection.close()
            return True

    except Error as e:
        print(f"✗ MySQL connection failed: {e}")

        # Provide specific troubleshooting based on error
        error_str = str(e).lower()
        if "access denied" in error_str:
            print("\nTroubleshooting: Access Denied")
            print("- Check username and password in .env file")
            print("- Verify user has permission to connect to MySQL")
            print("- Try connecting with mysql client: mysql -u root -p")
        elif "can't connect" in error_str or "connection refused" in error_str:
            print("\nTroubleshooting: Connection Refused")
            print("- Check if MySQL server is running")
            print("- Verify the host and port are correct")
            print("- Check if firewall is blocking the connection")
            print("- Try: brew services start mysql (on macOS)")
            print("- Or: sudo systemctl start mysql (on Linux)")
        elif "unknown database" in error_str:
            print("\nTroubleshooting: Database Not Found")
            print("- Check if the database name is correct in .env")
            print("- Create the database if it doesn't exist")

        return False

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_sakila_database():
    """Test connection to Sakila database specifically."""
    print("\nTesting Sakila database connection...")

    try:
        import mysql.connector

        config = {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE", "sakila"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
        }

        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()

        # Check if Sakila tables exist
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]

        expected_tables = ["actor", "film", "customer", "rental", "inventory"]
        found_tables = [t for t in expected_tables if t in tables]

        print(f"✓ Connected to {config['database']} database")
        print(f"✓ Found {len(tables)} tables total")
        print(f"✓ Sakila tables found: {found_tables}")

        if len(found_tables) < 3:
            print(
                "⚠️  Warning: Few Sakila tables found. Database might not be properly imported."
            )
            print("Consider importing Sakila schema and data:")
            print("1. Download from: https://dev.mysql.com/doc/index-other.html")
            print("2. Import: mysql -u root -p < sakila-schema.sql")
            print("3. Import: mysql -u root -p < sakila-data.sql")

        # Test a simple query
        cursor.execute("SELECT COUNT(*) FROM actor")
        actor_count = cursor.fetchone()[0]
        print(f"✓ Actor table has {actor_count} records")

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print(f"✗ Sakila database test failed: {e}")
        return False


def test_database_analyzer():
    """Test the custom database analyzer."""
    print("\nTesting database analyzer...")

    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))

        from database_analyzer import MySQLAnalyzer

        config = {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE", "sakila"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
        }

        analyzer = MySQLAnalyzer(**config)

        if analyzer.connect():
            print("✓ Database analyzer connected successfully")

            tables = analyzer.get_tables()
            print(
                f"✓ Found {len(tables)} tables: {tables[:5]}{'...' if len(tables) > 5 else ''}"
            )

            if tables:
                schema = analyzer.get_table_schema(tables[0])
                print(f"✓ Successfully retrieved schema for '{tables[0]}' table")

            analyzer.disconnect()
            return True
        else:
            print("✗ Database analyzer failed to connect")
            return False

    except ImportError as e:
        print(f"✗ Failed to import database_analyzer: {e}")
        return False
    except Exception as e:
        print(f"✗ Database analyzer test failed: {e}")
        return False


def provide_setup_instructions():
    """Provide setup instructions for MySQL."""
    print("\n" + "=" * 60)
    print("MYSQL SETUP INSTRUCTIONS")
    print("=" * 60)

    print("\n1. Install MySQL (if not installed):")
    print("   macOS: brew install mysql")
    print("   Ubuntu: sudo apt install mysql-server")
    print("   Windows: Download from https://dev.mysql.com/downloads/")

    print("\n2. Start MySQL service:")
    print("   macOS: brew services start mysql")
    print("   Linux: sudo systemctl start mysql")
    print("   Windows: Start MySQL service from Services panel")

    print("\n3. Set up MySQL user (if needed):")
    print("   mysql -u root -p")
    print("   CREATE USER 'your_user'@'localhost' IDENTIFIED BY 'your_password';")
    print("   GRANT ALL PRIVILEGES ON *.* TO 'your_user'@'localhost';")
    print("   FLUSH PRIVILEGES;")

    print("\n4. Download and import Sakila database:")
    print("   Download: https://dev.mysql.com/doc/index-other.html")
    print("   Import schema: mysql -u root -p < sakila-schema.sql")
    print("   Import data: mysql -u root -p < sakila-data.sql")

    print("\n5. Update .env file with your credentials:")
    print("   MYSQL_HOST=localhost")
    print("   MYSQL_USER=root")
    print("   MYSQL_PASSWORD=your_password")
    print("   MYSQL_DATABASE=sakila")
    print("   MYSQL_PORT=3306")


def main():
    """Run all MySQL connection tests."""
    print("MySQL Connection Troubleshooting")
    print("=" * 40)

    tests = [
        ("MySQL Import Test", test_mysql_import),
        ("Environment Variables Check", check_environment_variables),
        ("Basic Connection Test", test_basic_connection),
        ("Sakila Database Test", test_sakila_database),
        ("Database Analyzer Test", test_database_analyzer),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        else:
            # If a test fails, stop and provide help
            break

    print(f"\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All MySQL tests passed! Your setup is ready.")
    else:
        print("⚠️  Some tests failed. See troubleshooting advice above.")
        provide_setup_instructions()


if __name__ == "__main__":
    main()
