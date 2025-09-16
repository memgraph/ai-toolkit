# LightRAG Memgraph Integration

A Python library that integrates LightRAG with Memgraph for knowledge graph-based Retrieval-Augmented Generation (RAG).

## Features

- **Knowledge Graph Storage**: Uses Memgraph as the graph storage backend for LightRAG
- **Multiple Search Modes**: Supports both naive and hybrid search strategies
- **Async Support**: Fully asynchronous API for better performance
- **Easy Integration**: Simple interface for text insertion and querying

## Installation

This library requires Python 3.8 or higher and uses uv for dependency management.

```bash
# Install the library
uv add lightrag-memgraph

# Or install in development mode
uv pip install -e .
```

## Dependencies

- `lightrag-hku[api]` - Core LightRAG functionality
- `openai` - OpenAI API integration for embeddings and LLM

## Usage

### Basic Setup

```python
import asyncio
from lightrag_memgraph import initialize_lightrag_memgraph

async def main():
    # Initialize LightRAG with Memgraph storage
    rag = await initialize_lightrag_memgraph(working_dir="./my_rag_storage")
    
    # Insert some text
    await rag.insert_text("Your document content here...")
    
    # Query the knowledge graph
    result = await rag.query_hybrid("What is this document about?")
    print(result)

asyncio.run(main())
```

### Using the Class Directly

```python
import asyncio
from lightrag_memgraph import LightRAGMemgraph

async def main():
    # Create instance
    rag = LightRAGMemgraph(working_dir="./my_rag_storage", log_level="INFO")
    
    # Initialize
    await rag.initialize()
    
    # Insert text
    await rag.insert_text("Document content...")
    
    # Search
    naive_result = await rag.query_naive("Simple query")
    hybrid_result = await rag.query_hybrid("Complex query")
    
    print(f"Naive: {naive_result}")
    print(f"Hybrid: {hybrid_result}")

asyncio.run(main())
```

### Using Convenience Functions

```python
import asyncio
from lightrag_memgraph import initialize_lightrag_memgraph, naive_search, hybrid_search, insert_text

async def main():
    rag = await initialize_lightrag_memgraph()
    
    # Insert text
    await insert_text(rag, "Your content here...")
    
    # Search using convenience functions
    result1 = await naive_search(rag, "Query 1")
    result2 = await hybrid_search(rag, "Query 2")

asyncio.run(main())
```

## Configuration

### Environment Variables

Make sure to set your OpenAI API key and Memgraph connection details:

```bash
export OPENAI_API_KEY="your-api-key-here"
export MEMGRAPH_URI="bolt://localhost:7687"
```

### Memgraph Configuration

The library uses Memgraph as the graph storage backend. Ensure you have a Memgraph instance running and accessible. The connection details are configured through environment variables:

- `MEMGRAPH_URI`: Memgraph connection URI (default: `bolt://localhost:7687`)
- `MEMGRAPH_USER`: Memgraph username (optional)
- `MEMGRAPH_PASSWORD`: Memgraph password (optional)

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## API Reference

### LightRAGMemgraph

Main class for LightRAG with Memgraph integration.

#### Methods

- `__init__(working_dir: str, log_level: str)`: Initialize the instance
- `initialize()`: Initialize LightRAG with Memgraph storage
- `query_naive(query: str)`: Perform naive search
- `query_hybrid(query: str)`: Perform hybrid search  
- `insert_text(text: str)`: Insert text into the knowledge graph

### Convenience Functions

- `initialize_lightrag_memgraph(working_dir: str)`: Initialize and return a ready-to-use instance
- `naive_search(rag, query: str)`: Perform naive search
- `hybrid_search(rag, query: str)`: Perform hybrid search
- `insert_text(rag, text: str)`: Insert text

## Notes

- This is an experimental integration and may have limitations
- MemgraphStorage configuration may need to be adjusted based on your Memgraph setup
- Both `initialize_storages()` and `initialize_pipeline_status()` are called automatically during initialization

## License

This project is part of the Memgraph AI Toolkit.
