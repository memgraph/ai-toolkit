#!/usr/bin/env python3
"""
Example demonstrating how to use the GraphModel.to_schema_format() method
to convert a graph model into the comprehensive schema format.
"""

import json
from core.graph_modeling import GraphModel, GraphNode, GraphRelationship


def create_sample_graph_model():
    """Create a sample GraphModel for demonstration."""

    # Create sample nodes
    user_node = GraphNode(
        name="users",
        label="User",
        properties=["id", "username", "email", "created_at", "status"],
        primary_key="id",
        indexes=["username", "email"],
        constraints=["UNIQUE(id)", "UNIQUE(email)"],
        source_table="users",
    )

    product_node = GraphNode(
        name="products",
        label="Product",
        properties=["id", "name", "price", "category", "in_stock"],
        primary_key="id",
        indexes=["category"],
        constraints=["UNIQUE(id)"],
        source_table="products",
    )

    # Create sample relationship
    owns_relationship = GraphRelationship(
        name="PURCHASED",
        type="many_to_many",
        from_node="users",
        to_node="products",
        properties=["quantity", "purchase_date", "total_price"],
        directionality="directed",
        source_info={"from_table": "users", "to_table": "products"},
    )

    # Create the GraphModel
    return GraphModel(
        nodes=[user_node, product_node],
        relationships=[owns_relationship],
    )


def create_sample_data():
    """Create sample data for schema conversion."""
    return {
        "users": [
            {
                "id": 1,
                "username": "john_doe",
                "email": "john@example.com",
                "created_at": "2023-01-15 10:30:00",
                "status": "active",
            },
            {
                "id": 2,
                "username": "jane_smith",
                "email": "jane@example.com",
                "created_at": "2023-02-20 14:22:00",
                "status": "active",
            },
            {
                "id": 3,
                "username": "bob_wilson",
                "email": "bob@example.com",
                "created_at": "2023-03-10 09:15:00",
                "status": "inactive",
            },
        ],
        "products": [
            {
                "id": 101,
                "name": "Laptop Pro",
                "price": 1299.99,
                "category": "electronics",
                "in_stock": True,
            },
            {
                "id": 102,
                "name": "Wireless Mouse",
                "price": 49.99,
                "category": "electronics",
                "in_stock": True,
            },
            {
                "id": 103,
                "name": "Office Chair",
                "price": 299.99,
                "category": "furniture",
                "in_stock": False,
            },
        ],
    }


def main():
    """Demonstrate the schema conversion."""
    print("GraphModel Schema Conversion Example")
    print("=" * 40)

    # Create sample graph model
    graph_model = create_sample_graph_model()
    sample_data = create_sample_data()

    # Convert to schema format
    schema = graph_model.to_schema_format(sample_data)

    # Pretty print the result
    print("\nGenerated Schema:")
    print(json.dumps(schema, indent=2))

    # Show some key insights
    print(f"\nSchema Summary:")
    print(f"- Nodes: {len(schema['nodes'])}")
    print(f"- Edges: {len(schema['edges'])}")
    print(f"- Node Indexes: {len(schema['node_indexes'])}")
    print(f"- Node Constraints: {len(schema['node_constraints'])}")
    print(f"- Enums detected: {len(schema['enums'])}")

    # Show property type detection
    print(f"\nProperty Type Detection:")
    for node in schema["nodes"]:
        print(f"\n{node['labels'][0]} node properties:")
        for prop in node["properties"]:
            types = [t["type"] for t in prop["types"]]
            print(f"  - {prop['key']}: {', '.join(types)}")


if __name__ == "__main__":
    main()
