#!/usr/bin/env python3
"""Example usage of LightRAG Memgraph integration."""

import asyncio
from lightrag_memgraph import initialize_lightrag_memgraph


async def main():
    """Example main function demonstrating LightRAG Memgraph usage."""
    print("Memgraph LightRAG Wrapper")

    # Initialize LightRAG with Memgraph storage
    rag = await initialize_lightrag_memgraph(working_dir="./lightrag_storage")

    # Example usage (uncomment to test):
    await rag.insert_text("This is a sample document about artificial intelligence.")
    result = await rag.query_hybrid("What is this document about?")
    print(f"Query result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
