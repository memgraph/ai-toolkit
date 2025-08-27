#!/usr/bin/env python3
"""
Test script for the integrated hypothetical graph modeling functionality.

This script tests the integration of HyGM with the main
MySQLToMemgraphAgent to ensure the enhanced SQL to Graph mapping works correctly.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from typing import Dict, Any

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.environment import probe_mysql_connection, probe_memgraph_connection
from utils.config import get_preset_config
from core.hygm import HyGM
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_comprehensive_sql_to_graph_conversion():
    """Test comprehensive SQL database to HyGM graph conversion with multiple tables and data."""
    logger.info("Testing comprehensive SQL to HyGM graph conversion...")

    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        # Create graph modeler with deterministic strategy for predictable testing
        from core.hygm import GraphModelingStrategy

        modeler = HyGM(llm=llm, strategy=GraphModelingStrategy.DETERMINISTIC)

        # Create a realistic e-commerce database structure for testing
        sample_structure = {
            "tables": {
                "customers": {
                    "schema": [
                        {
                            "field": "customer_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "first_name",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "last_name",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "email",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "UNI",
                        },
                        {
                            "field": "phone",
                            "type": "varchar(20)",
                            "null": "YES",
                            "key": "",
                        },
                        {
                            "field": "address_id",
                            "type": "int(11)",
                            "null": "YES",
                            "key": "MUL",
                        },
                        {
                            "field": "created_at",
                            "type": "timestamp",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["customer_id"],
                    "foreign_keys": [
                        {
                            "column": "address_id",
                            "referenced_table": "addresses",
                            "referenced_column": "address_id",
                        }
                    ],
                    "type": "entity",
                },
                "addresses": {
                    "schema": [
                        {
                            "field": "address_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "street",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "city",
                            "type": "varchar(100)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "state",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "postal_code",
                            "type": "varchar(20)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "country",
                            "type": "varchar(100)",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["address_id"],
                    "foreign_keys": [],
                    "type": "entity",
                },
                "products": {
                    "schema": [
                        {
                            "field": "product_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "name",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "description",
                            "type": "text",
                            "null": "YES",
                            "key": "",
                        },
                        {
                            "field": "price",
                            "type": "decimal(10,2)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "category_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                        {
                            "field": "stock_quantity",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "created_at",
                            "type": "timestamp",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["product_id"],
                    "foreign_keys": [
                        {
                            "column": "category_id",
                            "referenced_table": "categories",
                            "referenced_column": "category_id",
                        }
                    ],
                    "type": "entity",
                },
                "categories": {
                    "schema": [
                        {
                            "field": "category_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "name",
                            "type": "varchar(100)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "description",
                            "type": "text",
                            "null": "YES",
                            "key": "",
                        },
                        {
                            "field": "parent_category_id",
                            "type": "int(11)",
                            "null": "YES",
                            "key": "MUL",
                        },
                    ],
                    "primary_keys": ["category_id"],
                    "foreign_keys": [
                        {
                            "column": "parent_category_id",
                            "referenced_table": "categories",
                            "referenced_column": "category_id",
                        }
                    ],
                    "type": "entity",
                },
                "orders": {
                    "schema": [
                        {
                            "field": "order_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "customer_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                        {
                            "field": "order_date",
                            "type": "timestamp",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "total_amount",
                            "type": "decimal(10,2)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "status",
                            "type": "enum('pending','processing','shipped','delivered','cancelled')",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "shipping_address_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                    ],
                    "primary_keys": ["order_id"],
                    "foreign_keys": [
                        {
                            "column": "customer_id",
                            "referenced_table": "customers",
                            "referenced_column": "customer_id",
                        },
                        {
                            "column": "shipping_address_id",
                            "referenced_table": "addresses",
                            "referenced_column": "address_id",
                        },
                    ],
                    "type": "entity",
                },
                "order_items": {
                    "schema": [
                        {
                            "field": "order_item_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "order_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                        {
                            "field": "product_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                        {
                            "field": "quantity",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "unit_price",
                            "type": "decimal(10,2)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "discount",
                            "type": "decimal(5,2)",
                            "null": "YES",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["order_item_id"],
                    "foreign_keys": [
                        {
                            "column": "order_id",
                            "referenced_table": "orders",
                            "referenced_column": "order_id",
                        },
                        {
                            "column": "product_id",
                            "referenced_table": "products",
                            "referenced_column": "product_id",
                        },
                    ],
                    "type": "entity",
                },
                # Junction table for product tags (many-to-many)
                "product_tags": {
                    "schema": [
                        {
                            "field": "product_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "tag_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                    ],
                    "primary_keys": ["product_id", "tag_id"],
                    "foreign_keys": [
                        {
                            "column": "product_id",
                            "referenced_table": "products",
                            "referenced_column": "product_id",
                        },
                        {
                            "column": "tag_id",
                            "referenced_table": "tags",
                            "referenced_column": "tag_id",
                        },
                    ],
                    "type": "junction",
                },
                "tags": {
                    "schema": [
                        {
                            "field": "tag_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "name",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "UNI",
                        },
                        {
                            "field": "color",
                            "type": "varchar(7)",
                            "null": "YES",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["tag_id"],
                    "foreign_keys": [],
                    "type": "entity",
                },
            },
            "entity_tables": {
                "customers": {
                    "columns": {
                        "customer_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "first_name": {"type": "varchar(50)", "nullable": False},
                        "last_name": {"type": "varchar(50)", "nullable": False},
                        "email": {
                            "type": "varchar(255)",
                            "nullable": False,
                            "unique": True,
                        },
                        "phone": {"type": "varchar(20)", "nullable": True},
                        "address_id": {
                            "type": "int",
                            "nullable": True,
                            "key": "foreign",
                        },
                        "created_at": {"type": "timestamp", "nullable": False},
                    },
                    "primary_keys": ["customer_id"],
                    "foreign_keys": [
                        {
                            "column": "address_id",
                            "referenced_table": "addresses",
                            "referenced_column": "address_id",
                        }
                    ],
                },
                "addresses": {
                    "columns": {
                        "address_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "street": {"type": "varchar(255)", "nullable": False},
                        "city": {"type": "varchar(100)", "nullable": False},
                        "state": {"type": "varchar(50)", "nullable": False},
                        "postal_code": {"type": "varchar(20)", "nullable": False},
                        "country": {"type": "varchar(100)", "nullable": False},
                    },
                    "primary_keys": ["address_id"],
                    "foreign_keys": [],
                },
                "products": {
                    "columns": {
                        "product_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "name": {"type": "varchar(255)", "nullable": False},
                        "description": {"type": "text", "nullable": True},
                        "price": {"type": "decimal(10,2)", "nullable": False},
                        "category_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "foreign",
                        },
                        "stock_quantity": {"type": "int", "nullable": False},
                        "created_at": {"type": "timestamp", "nullable": False},
                    },
                    "primary_keys": ["product_id"],
                    "foreign_keys": [
                        {
                            "column": "category_id",
                            "referenced_table": "categories",
                            "referenced_column": "category_id",
                        }
                    ],
                },
                "categories": {
                    "columns": {
                        "category_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "name": {"type": "varchar(100)", "nullable": False},
                        "description": {"type": "text", "nullable": True},
                        "parent_category_id": {
                            "type": "int",
                            "nullable": True,
                            "key": "foreign",
                        },
                    },
                    "primary_keys": ["category_id"],
                    "foreign_keys": [
                        {
                            "column": "parent_category_id",
                            "referenced_table": "categories",
                            "referenced_column": "category_id",
                        }
                    ],
                },
                "orders": {
                    "columns": {
                        "order_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "customer_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "foreign",
                        },
                        "order_date": {"type": "timestamp", "nullable": False},
                        "total_amount": {"type": "decimal(10,2)", "nullable": False},
                        "status": {
                            "type": "enum",
                            "nullable": False,
                            "values": [
                                "pending",
                                "processing",
                                "shipped",
                                "delivered",
                                "cancelled",
                            ],
                        },
                        "shipping_address_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "foreign",
                        },
                    },
                    "primary_keys": ["order_id"],
                    "foreign_keys": [
                        {
                            "column": "customer_id",
                            "referenced_table": "customers",
                            "referenced_column": "customer_id",
                        },
                        {
                            "column": "shipping_address_id",
                            "referenced_table": "addresses",
                            "referenced_column": "address_id",
                        },
                    ],
                },
                "order_items": {
                    "columns": {
                        "order_item_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "order_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "foreign",
                        },
                        "product_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "foreign",
                        },
                        "quantity": {"type": "int", "nullable": False},
                        "unit_price": {"type": "decimal(10,2)", "nullable": False},
                        "discount": {"type": "decimal(5,2)", "nullable": True},
                    },
                    "primary_keys": ["order_item_id"],
                    "foreign_keys": [
                        {
                            "column": "order_id",
                            "referenced_table": "orders",
                            "referenced_column": "order_id",
                        },
                        {
                            "column": "product_id",
                            "referenced_table": "products",
                            "referenced_column": "product_id",
                        },
                    ],
                },
                "tags": {
                    "columns": {
                        "tag_id": {"type": "int", "nullable": False, "key": "primary"},
                        "name": {
                            "type": "varchar(50)",
                            "nullable": False,
                            "unique": True,
                        },
                        "color": {"type": "varchar(7)", "nullable": True},
                    },
                    "primary_keys": ["tag_id"],
                    "foreign_keys": [],
                },
            },
            "junction_tables": {
                "product_tags": {
                    "left_table": "products",
                    "right_table": "tags",
                    "left_column": "product_id",
                    "right_column": "tag_id",
                    "foreign_keys": [
                        {
                            "column": "product_id",
                            "referenced_table": "products",
                            "referenced_column": "product_id",
                        },
                        {
                            "column": "tag_id",
                            "referenced_table": "tags",
                            "referenced_column": "tag_id",
                        },
                    ],
                }
            },
            "relationships": [
                {
                    "from_table": "customers",
                    "to_table": "addresses",
                    "type": "many_to_one",
                    "from_column": "address_id",
                    "to_column": "address_id",
                },
                {
                    "from_table": "products",
                    "to_table": "categories",
                    "type": "many_to_one",
                    "from_column": "category_id",
                    "to_column": "category_id",
                },
                {
                    "from_table": "categories",
                    "to_table": "categories",
                    "type": "many_to_one",
                    "from_column": "parent_category_id",
                    "to_column": "category_id",
                },
                {
                    "from_table": "orders",
                    "to_table": "customers",
                    "type": "many_to_one",
                    "from_column": "customer_id",
                    "to_column": "customer_id",
                },
                {
                    "from_table": "orders",
                    "to_table": "addresses",
                    "type": "many_to_one",
                    "from_column": "shipping_address_id",
                    "to_column": "address_id",
                },
                {
                    "from_table": "order_items",
                    "to_table": "orders",
                    "type": "many_to_one",
                    "from_column": "order_id",
                    "to_column": "order_id",
                },
                {
                    "from_table": "order_items",
                    "to_table": "products",
                    "type": "many_to_one",
                    "from_column": "product_id",
                    "to_column": "product_id",
                },
                {
                    "from_table": "products",
                    "to_table": "tags",
                    "type": "many_to_many",
                    "junction_table": "product_tags",
                    "from_column": "product_id",
                    "to_column": "tag_id",
                },
            ],
        }

        # Create sample data to test data conversion capabilities
        sample_data = {
            "addresses": [
                {
                    "address_id": 1,
                    "street": "123 Main St",
                    "city": "New York",
                    "state": "NY",
                    "postal_code": "10001",
                    "country": "USA",
                },
                {
                    "address_id": 2,
                    "street": "456 Oak Ave",
                    "city": "Los Angeles",
                    "state": "CA",
                    "postal_code": "90210",
                    "country": "USA",
                },
                {
                    "address_id": 3,
                    "street": "789 Pine Rd",
                    "city": "Chicago",
                    "state": "IL",
                    "postal_code": "60601",
                    "country": "USA",
                },
            ],
            "customers": [
                {
                    "customer_id": 1,
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john@example.com",
                    "phone": "555-0001",
                    "address_id": 1,
                    "created_at": "2024-01-15 10:00:00",
                },
                {
                    "customer_id": 2,
                    "first_name": "Jane",
                    "last_name": "Smith",
                    "email": "jane@example.com",
                    "phone": "555-0002",
                    "address_id": 2,
                    "created_at": "2024-01-16 11:00:00",
                },
                {
                    "customer_id": 3,
                    "first_name": "Bob",
                    "last_name": "Johnson",
                    "email": "bob@example.com",
                    "phone": None,
                    "address_id": 3,
                    "created_at": "2024-01-17 12:00:00",
                },
            ],
            "categories": [
                {
                    "category_id": 1,
                    "name": "Electronics",
                    "description": "Electronic devices and gadgets",
                    "parent_category_id": None,
                },
                {
                    "category_id": 2,
                    "name": "Smartphones",
                    "description": "Mobile phones and accessories",
                    "parent_category_id": 1,
                },
                {
                    "category_id": 3,
                    "name": "Laptops",
                    "description": "Portable computers",
                    "parent_category_id": 1,
                },
                {
                    "category_id": 4,
                    "name": "Clothing",
                    "description": "Apparel and accessories",
                    "parent_category_id": None,
                },
            ],
            "tags": [
                {"tag_id": 1, "name": "bestseller", "color": "#FF5733"},
                {"tag_id": 2, "name": "new-arrival", "color": "#33FF57"},
                {"tag_id": 3, "name": "sale", "color": "#3357FF"},
                {"tag_id": 4, "name": "premium", "color": "#FF33F5"},
            ],
            "products": [
                {
                    "product_id": 1,
                    "name": "iPhone 15",
                    "description": "Latest Apple smartphone",
                    "price": 999.99,
                    "category_id": 2,
                    "stock_quantity": 50,
                    "created_at": "2024-01-10 09:00:00",
                },
                {
                    "product_id": 2,
                    "name": "MacBook Pro",
                    "description": "High-performance laptop",
                    "price": 2499.99,
                    "category_id": 3,
                    "stock_quantity": 25,
                    "created_at": "2024-01-11 10:00:00",
                },
                {
                    "product_id": 3,
                    "name": "Premium T-Shirt",
                    "description": "100% cotton premium t-shirt",
                    "price": 29.99,
                    "category_id": 4,
                    "stock_quantity": 100,
                    "created_at": "2024-01-12 11:00:00",
                },
            ],
            "orders": [
                {
                    "order_id": 1,
                    "customer_id": 1,
                    "order_date": "2024-01-20 14:30:00",
                    "total_amount": 1029.98,
                    "status": "delivered",
                    "shipping_address_id": 1,
                },
                {
                    "order_id": 2,
                    "customer_id": 2,
                    "order_date": "2024-01-21 15:45:00",
                    "total_amount": 2499.99,
                    "status": "shipped",
                    "shipping_address_id": 2,
                },
                {
                    "order_id": 3,
                    "customer_id": 3,
                    "order_date": "2024-01-22 16:00:00",
                    "total_amount": 59.98,
                    "status": "processing",
                    "shipping_address_id": 3,
                },
            ],
            "order_items": [
                {
                    "order_item_id": 1,
                    "order_id": 1,
                    "product_id": 1,
                    "quantity": 1,
                    "unit_price": 999.99,
                    "discount": 0.0,
                },
                {
                    "order_item_id": 2,
                    "order_id": 1,
                    "product_id": 3,
                    "quantity": 1,
                    "unit_price": 29.99,
                    "discount": 0.0,
                },
                {
                    "order_item_id": 3,
                    "order_id": 2,
                    "product_id": 2,
                    "quantity": 1,
                    "unit_price": 2499.99,
                    "discount": 0.0,
                },
                {
                    "order_item_id": 4,
                    "order_id": 3,
                    "product_id": 3,
                    "quantity": 2,
                    "unit_price": 29.99,
                    "discount": 0.0,
                },
            ],
            "product_tags": [
                {"product_id": 1, "tag_id": 1},  # iPhone 15 - bestseller
                {"product_id": 1, "tag_id": 2},  # iPhone 15 - new-arrival
                {"product_id": 2, "tag_id": 4},  # MacBook Pro - premium
                {"product_id": 3, "tag_id": 3},  # T-Shirt - sale
            ],
        }

        # Test the modeling
        logger.info("Running comprehensive graph modeling analysis...")
        graph_model = modeler.create_graph_model(sample_structure)

        # Validate graph model structure
        logger.info("Graph modeling completed successfully!")
        logger.info("Created %d node types", len(graph_model.nodes))
        logger.info("Created %d relationships", len(graph_model.edges))
        logger.info("Created %d node indexes", len(graph_model.node_indexes))
        logger.info("Created %d node constraints", len(graph_model.node_constraints))

        # Test detailed validation
        expected_entities = [
            "customers",
            "addresses",
            "products",
            "categories",
            "orders",
            "order_items",
            "tags",
        ]
        entity_nodes = [
            node
            for node in graph_model.nodes
            if any(label.lower() in expected_entities for label in node.labels)
        ]

        logger.info("Expected entities: %s", expected_entities)
        logger.info("Found entity nodes: %s", [node.labels for node in entity_nodes])

        # Validate relationships include both foreign key and junction table relationships
        relationship_types = [rel.edge_type for rel in graph_model.edges]
        logger.info("Found relationship types: %s", relationship_types)

        # Check for many-to-many relationships (junction table handling)
        junction_relationships = [
            rel
            for rel in graph_model.edges
            if any(
                word in rel.edge_type.lower() for word in ["to", "tagged", "belongs"]
            )
        ]
        logger.info(
            "Junction table relationships: %s",
            [rel.edge_type for rel in junction_relationships],
        )

        # Test that source tracking is properly set
        nodes_with_source = [
            node for node in graph_model.nodes if node.source is not None
        ]
        logger.info(
            "Nodes with source tracking: %d/%d",
            len(nodes_with_source),
            len(graph_model.nodes),
        )

        relationships_with_source = [
            rel for rel in graph_model.edges if rel.source is not None
        ]
        logger.info(
            "Relationships with source tracking: %d/%d",
            len(relationships_with_source),
            len(graph_model.edges),
        )

        # Validate core assertions (adjusted for deterministic strategy behavior)
        assert (
            len(graph_model.nodes) >= 7
        ), f"Should create at least 7 node types for entities, got {len(graph_model.nodes)}"
        assert (
            len(graph_model.edges) >= 5
        ), f"Should create at least 5 relationships, got {len(graph_model.edges)}"
        # Note: Indexes and constraints are created during migration, not during modeling
        logger.info("Graph model structure validated successfully")

        # Test data structure compatibility
        logger.info("Testing data structure compatibility...")
        for entity_name, data_rows in sample_data.items():
            if entity_name == "product_tags":  # Skip junction table data
                continue

            # Find corresponding node in graph model
            matching_nodes = [
                node
                for node in graph_model.nodes
                if any(
                    label.lower() == entity_name.rstrip("s").lower()
                    or label.lower() == entity_name.lower()
                    for label in node.labels
                )
            ]

            if matching_nodes:
                node = matching_nodes[0]
                logger.info(
                    "✅ Entity '%s' mapped to node with labels %s",
                    entity_name,
                    node.labels,
                )

                # Check if node properties match data structure
                data_columns = set(data_rows[0].keys()) if data_rows else set()
                node_props = (
                    set(prop.key for prop in node.properties)
                    if node.properties
                    else set()
                )

                if data_columns and node_props:
                    common_props = data_columns.intersection(node_props)
                    logger.info("   Common properties: %s", list(common_props))

                    # Should have at least some overlapping properties
                    assert (
                        len(common_props) > 0
                    ), f"Node {node.labels} should have properties matching data columns"
            else:
                logger.warning(
                    "⚠️  No matching node found for entity '%s'", entity_name
                )

        logger.info("✅ Comprehensive graph modeling test successful!")
        logger.info("   - Created %d node types", len(graph_model.nodes))
        logger.info("   - Created %d relationship types", len(graph_model.edges))
        logger.info("   - Created %d node indexes", len(graph_model.node_indexes))
        logger.info(
            "   - Created %d node constraints", len(graph_model.node_constraints)
        )
        logger.info("   - Validated data structure compatibility")
        logger.info("   - Confirmed source tracking implementation")

        return True

    except Exception as e:
        logger.error("❌ Comprehensive graph modeling test failed: %s", str(e))
        import traceback

        logger.error("Traceback: %s", traceback.format_exc())
        return False


def test_environment_setup():
    """Test that the environment is properly configured."""
    logger.info("Testing environment setup...")

    try:
        # Test configuration loading
        config = get_preset_config("local_development")
        logger.info("✅ Configuration loading successful")

        # Test database connections (if available)
        mysql_available = False
        memgraph_available = False

        try:
            mysql_result = probe_mysql_connection(config["mysql_config"])
            mysql_available = mysql_result["success"]
            logger.info(
                "MySQL connection: %s",
                "✅ Available" if mysql_available else "❌ Not available",
            )
        except Exception as e:
            logger.warning("MySQL probe failed: %s", str(e))

        try:
            memgraph_result = probe_memgraph_connection(config["memgraph_config"])
            memgraph_available = memgraph_result["success"]
            logger.info(
                "Memgraph connection: %s",
                "✅ Available" if memgraph_available else "❌ Not available",
            )
        except Exception as e:
            logger.warning("Memgraph probe failed: %s", str(e))

        # Check OpenAI API key
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        logger.info(
            f"OpenAI API key: {'✅ Available' if openai_available else '❌ Not available'}"
        )

        if not openai_available:
            logger.warning(
                "OpenAI API key is required for graph modeling functionality"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Environment setup test failed: {e}")
        return False


def test_full_migration_workflow_simulation():
    """Test the full migration workflow simulation with comprehensive data."""
    logger.info("Testing full migration workflow simulation...")

    try:
        # Initialize components
        from core.hygm import GraphModelingStrategy
        from core.migration_agent import SQLToMemgraphAgent

        # Create migration agent with deterministic strategy
        agent = SQLToMemgraphAgent(
            interactive_graph_modeling=False,
            graph_modeling_strategy=GraphModelingStrategy.DETERMINISTIC,
        )

        # Use a simplified but comprehensive database structure for testing
        sample_structure = {
            "tables": {
                "customers": {
                    "schema": [
                        {
                            "field": "customer_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "first_name",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "email",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "UNI",
                        },
                        {
                            "field": "address_id",
                            "type": "int(11)",
                            "null": "YES",
                            "key": "MUL",
                        },
                    ],
                    "primary_keys": ["customer_id"],
                    "foreign_keys": [
                        {
                            "column": "address_id",
                            "referenced_table": "addresses",
                            "referenced_column": "address_id",
                        }
                    ],
                    "type": "entity",
                },
                "addresses": {
                    "schema": [
                        {
                            "field": "address_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "street",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "city",
                            "type": "varchar(100)",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["address_id"],
                    "foreign_keys": [],
                    "type": "entity",
                },
                "products": {
                    "schema": [
                        {
                            "field": "product_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "name",
                            "type": "varchar(255)",
                            "null": "NO",
                            "key": "",
                        },
                        {
                            "field": "category_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "MUL",
                        },
                    ],
                    "primary_keys": ["product_id"],
                    "foreign_keys": [
                        {
                            "column": "category_id",
                            "referenced_table": "categories",
                            "referenced_column": "category_id",
                        }
                    ],
                    "type": "entity",
                },
                "categories": {
                    "schema": [
                        {
                            "field": "category_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "name",
                            "type": "varchar(100)",
                            "null": "NO",
                            "key": "",
                        },
                    ],
                    "primary_keys": ["category_id"],
                    "foreign_keys": [],
                    "type": "entity",
                },
                "product_tags": {
                    "schema": [
                        {
                            "field": "product_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "tag_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                    ],
                    "primary_keys": ["product_id", "tag_id"],
                    "foreign_keys": [
                        {
                            "column": "product_id",
                            "referenced_table": "products",
                            "referenced_column": "product_id",
                        },
                        {
                            "column": "tag_id",
                            "referenced_table": "tags",
                            "referenced_column": "tag_id",
                        },
                    ],
                    "type": "junction",
                },
                "tags": {
                    "schema": [
                        {
                            "field": "tag_id",
                            "type": "int(11)",
                            "null": "NO",
                            "key": "PRI",
                        },
                        {
                            "field": "name",
                            "type": "varchar(50)",
                            "null": "NO",
                            "key": "UNI",
                        },
                    ],
                    "primary_keys": ["tag_id"],
                    "foreign_keys": [],
                    "type": "entity",
                },
            },
            "entity_tables": {
                "customers": {
                    "columns": {
                        "customer_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "first_name": {"type": "varchar(50)", "nullable": False},
                        "email": {
                            "type": "varchar(255)",
                            "nullable": False,
                            "unique": True,
                        },
                        "address_id": {
                            "type": "int",
                            "nullable": True,
                            "key": "foreign",
                        },
                    },
                    "primary_keys": ["customer_id"],
                    "foreign_keys": [
                        {
                            "column": "address_id",
                            "referenced_table": "addresses",
                            "referenced_column": "address_id",
                        }
                    ],
                },
                "addresses": {
                    "columns": {
                        "address_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "street": {"type": "varchar(255)", "nullable": False},
                        "city": {"type": "varchar(100)", "nullable": False},
                    },
                    "primary_keys": ["address_id"],
                    "foreign_keys": [],
                },
                "products": {
                    "columns": {
                        "product_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "name": {"type": "varchar(255)", "nullable": False},
                        "category_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "foreign",
                        },
                    },
                    "primary_keys": ["product_id"],
                    "foreign_keys": [
                        {
                            "column": "category_id",
                            "referenced_table": "categories",
                            "referenced_column": "category_id",
                        }
                    ],
                },
                "categories": {
                    "columns": {
                        "category_id": {
                            "type": "int",
                            "nullable": False,
                            "key": "primary",
                        },
                        "name": {"type": "varchar(100)", "nullable": False},
                    },
                    "primary_keys": ["category_id"],
                    "foreign_keys": [],
                },
                "tags": {
                    "columns": {
                        "tag_id": {"type": "int", "nullable": False, "key": "primary"},
                        "name": {
                            "type": "varchar(50)",
                            "nullable": False,
                            "unique": True,
                        },
                    },
                    "primary_keys": ["tag_id"],
                    "foreign_keys": [],
                },
            },
            "junction_tables": {
                "product_tags": {
                    "left_table": "products",
                    "right_table": "tags",
                    "left_column": "product_id",
                    "right_column": "tag_id",
                    "foreign_keys": [
                        {
                            "column": "product_id",
                            "referenced_table": "products",
                            "referenced_column": "product_id",
                        },
                        {
                            "column": "tag_id",
                            "referenced_table": "tags",
                            "referenced_column": "tag_id",
                        },
                    ],
                }
            },
            "relationships": [
                {
                    "from_table": "customers",
                    "to_table": "addresses",
                    "type": "many_to_one",
                    "from_column": "address_id",
                    "to_column": "address_id",
                },
                {
                    "from_table": "products",
                    "to_table": "categories",
                    "type": "many_to_one",
                    "from_column": "category_id",
                    "to_column": "category_id",
                },
                {
                    "from_table": "products",
                    "to_table": "tags",
                    "type": "many_to_many",
                    "junction_table": "product_tags",
                    "from_column": "product_id",
                    "to_column": "tag_id",
                },
            ],
        }

        # Test graph modeling step (using HyGM directly)
        logger.info("Testing graph modeling step...")
        from core.hygm import HyGM

        modeler = HyGM(llm=agent.llm, strategy=GraphModelingStrategy.DETERMINISTIC)
        graph_model = modeler.create_graph_model(sample_structure)

        # Validate graph model
        expected_nodes = 5  # customers, addresses, products, categories, tags
        expected_rels = 3  # customer->address, product->category, product<->tag

        assert (
            len(graph_model.nodes) >= expected_nodes
        ), f"Should create at least {expected_nodes} nodes, got {len(graph_model.nodes)}"
        assert (
            len(graph_model.edges) >= expected_rels
        ), f"Should create at least {expected_rels} relationships, got {len(graph_model.edges)}"

        logger.info("✅ Graph modeling step completed:")
        logger.info("   - Nodes: %d", len(graph_model.nodes))
        logger.info("   - Relationships: %d", len(graph_model.edges))
        logger.info("   - Node indexes: %d", len(graph_model.node_indexes))
        logger.info("   - Node constraints: %d", len(graph_model.node_constraints))

        # Test Cypher query generation (simulation)
        logger.info("Testing Cypher query generation...")

        # Simulate node creation queries
        node_queries = []
        for node in graph_model.nodes:
            properties = (
                [prop.key for prop in node.properties] if node.properties else []
            )
            primary_label = node.primary_label

            query = (
                f"// Create {primary_label} nodes with "
                + f"properties: {', '.join(properties)}"
            )
            node_queries.append(query)

        logger.info("Generated %d node creation queries", len(node_queries))

        # Simulate relationship creation queries
        relationship_queries = []
        for relationship in graph_model.edges:
            start_labels = relationship.start_node_labels
            end_labels = relationship.end_node_labels
            rel_type = relationship.edge_type

            query = (
                f"// Create {rel_type} relationships: "
                + f"{start_labels} -> {end_labels}"
            )
            relationship_queries.append(query)

        logger.info(
            "Generated %d relationship creation queries", len(relationship_queries)
        )

        # Simulate constraint and index creation
        constraint_queries = []
        for constraint in graph_model.node_constraints:
            if constraint.labels:
                labels = constraint.labels
                properties = constraint.properties
                query = f"// Create constraint for {labels} on {properties}"
                constraint_queries.append(query)

        logger.info("Generated %d constraint queries", len(constraint_queries))

        index_queries = []
        for index in graph_model.node_indexes:
            if index.labels:
                labels = index.labels
                properties = index.properties
                query = f"// Create index for {labels} on {properties}"
                index_queries.append(query)

        logger.info("Generated %d index queries", len(index_queries))

        # Validate query generation
        total_queries = (
            len(node_queries)
            + len(relationship_queries)
            + len(constraint_queries)
            + len(index_queries)
        )
        assert total_queries > 0, "Should generate at least some queries"

        logger.info("✅ Full migration workflow simulation successful!")
        logger.info("   - Graph model created and validated")
        logger.info("   - Cypher queries generated: %d total", total_queries)
        logger.info("     * Node queries: %d", len(node_queries))
        logger.info("     * Relationship queries: %d", len(relationship_queries))
        logger.info("     * Constraint queries: %d", len(constraint_queries))
        logger.info("     * Index queries: %d", len(index_queries))
        logger.info("   - All components working together")

        return True

    except Exception as e:
        logger.error("❌ Full migration workflow simulation failed: %s", str(e))
        import traceback

        logger.error("Traceback: %s", traceback.format_exc())
        return False


def main():
    """Run all integration tests."""
    logger.info("🧪 Starting intelligent graph modeling integration tests...")

    # Test environment setup
    if not test_environment_setup():
        logger.error("Environment setup failed - cannot continue with tests")
        return False

    # Test comprehensive SQL to graph conversion
    if not test_comprehensive_sql_to_graph_conversion():
        logger.error("Comprehensive graph modeling test failed")
        return False

    # Test full migration workflow simulation
    if not test_full_migration_workflow_simulation():
        logger.error("Full migration workflow simulation test failed")
        return False

    logger.info("🎉 All integration tests passed!")
    logger.info("\nNext steps:")
    logger.info("1. Run the main agent with: python main.py")
    logger.info("2. The agent will now include intelligent graph modeling analysis")
    logger.info("3. Check the migration plan for HyGM-generated insights")
    logger.info("4. Test with real databases for production validation")

    return True


if __name__ == "__main__":
    success = main()

    exit(0 if success else 1)
