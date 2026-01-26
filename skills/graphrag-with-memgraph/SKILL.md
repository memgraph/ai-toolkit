---
name: graphrag-with-memgraph
description: Build Graph Retrieval-Augmented Generation (GraphRAG) applications using Memgraph. Covers document ingestion, entity extraction, knowledge graph construction, and hybrid retrieval combining vector similarity with graph traversal. Use when building knowledge bases, document Q&A systems, semantic search with graph context, or converting unstructured documents into queryable knowledge graphs.
license: Apache-2.0
metadata:
  author: memgraph
  version: "1.0"
  repository: https://github.com/memgraph/ai-toolkit
---

# GraphRAG with Memgraph

Build powerful Graph Retrieval-Augmented Generation applications that combine vector similarity search with graph traversal for enhanced context retrieval.

## When to Use This Skill

Use this skill when:

- Converting unstructured documents (PDFs, URLs, HTML, Markdown) into a knowledge graph
- Building Q&A systems that need relationship-aware context retrieval
- Implementing semantic search that leverages graph connections
- Creating knowledge bases with entity extraction and linking

Do NOT use this skill for:

- Simple vector-only RAG (use standard RAG approaches)
- Structured data already in tabular format (consider sql2graph)
- Real-time streaming data ingestion

## Architecture Overview

```
Documents (PDF, URL, MD, DOCX)
         │
         ▼
┌─────────────────────┐
│   unstructured      │  ← Parse & chunk documents
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   LightRAG          │  ← Extract entities & relationships
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Memgraph          │  ← Store graph + vectors
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Hybrid Retrieval  │  ← Vector search + graph traversal
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   LLM Answer        │  ← Generate response
└─────────────────────┘
```

## Prerequisites

1. **Memgraph instance** running with MAGE algorithms
2. **OpenAI API key** (or compatible LLM provider)
3. **Python 3.10+**

## Step 1: Install Dependencies

```bash
# Install the Memgraph AI Toolkit packages
pip install memgraph-toolbox
pip install unstructured2graph
pip install lightrag-memgraph

# For full document support (PDF, DOCX, etc.)
pip install "unstructured2graph[all-docs]"
```

## Step 2: Set Up Environment

```bash
export OPENAI_API_KEY="your-api-key"
export MEMGRAPH_HOST="localhost"
export MEMGRAPH_PORT="7687"
```

## Step 3: Document Ingestion Pipeline

### 3.1 Initialize Connections

```python
import asyncio
from memgraph_toolbox.api.memgraph import Memgraph
from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import from_unstructured, create_index

# Connect to Memgraph
memgraph = Memgraph(
    url="bolt://localhost:7687",
    username="",
    password=""
)

# Create required indexes BEFORE ingestion
create_index(memgraph, "Chunk", "hash")
```

### 3.2 Ingest Documents

```python
async def ingest_documents(sources: list[str]):
    """
    Ingest documents into Memgraph knowledge graph.

    Args:
        sources: List of file paths or URLs
    """
    # Initialize LightRAG wrapper for entity extraction
    lightrag = MemgraphLightRAGWrapper(
        log_level="WARNING",
        disable_embeddings=True  # Use Memgraph's embeddings instead
    )
    await lightrag.initialize(working_dir="./lightrag_storage")

    # Process documents
    await from_unstructured(
        sources=sources,
        memgraph=memgraph,
        lightrag_wrapper=lightrag,
        only_chunks=False,    # Extract entities, not just chunks
        link_chunks=True,     # Create NEXT relationships between chunks
    )

    await lightrag.afinalize()

# Example usage
asyncio.run(ingest_documents([
    "https://example.com/document.pdf",
    "./local_docs/report.md",
    "./data/whitepaper.pdf"
]))
```

### 3.3 Compute Embeddings and Create Vector Index

```python
from unstructured2graph import compute_embeddings, create_vector_search_index

# Generate embeddings for all chunks
compute_embeddings(memgraph, "Chunk")

# Create vector search index
create_vector_search_index(memgraph, "Chunk", "embedding")
```

## Step 4: Hybrid Retrieval (The GraphRAG Query)

This is the core of GraphRAG: combining vector similarity with graph traversal.

```python
def graphrag_retrieve(question: str, vector_limit: int = 5, graph_limit: int = 5) -> list[str]:
    """
    Retrieve context using hybrid vector + graph search.

    1. Vector search finds semantically similar chunks
    2. Graph traversal expands to connected entities and chunks
    3. Results are ranked by graph centrality (degree)
    """
    query = f"""
    // Step 1: Get embedding for the question
    CALL embeddings.text(['{question}']) YIELD embeddings, success

    // Step 2: Vector similarity search
    CALL vector_search.search('vs_name', {vector_limit}, embeddings[0])
    YIELD distance, node, similarity

    // Step 3: Graph traversal - find connected chunks via BFS
    MATCH (node)-[r*bfs]-(dst:Chunk)

    // Step 4: Rank by connectivity (more connections = more important)
    WITH DISTINCT dst, degree(dst) AS degree
    ORDER BY degree DESC

    RETURN dst LIMIT {graph_limit};
    """

    retrieved = []
    for row in memgraph.query(query):
        chunk = row["dst"]
        if "text" in chunk:
            retrieved.append(chunk["text"])
        elif "description" in chunk:
            retrieved.append(chunk["description"])

    return retrieved
```

## Step 5: Generate Answer

```python
from openai import OpenAI

def answer_question(question: str) -> str:
    """Complete GraphRAG pipeline: retrieve + generate."""

    # Retrieve context
    chunks = graphrag_retrieve(question)

    if not chunks:
        return "I don't have enough information to answer that question."

    context = "\n\n".join(chunks)

    # Generate answer
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based only on the provided context. If the context doesn't contain enough information, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content
```

## Complete Example

```python
import asyncio
from memgraph_toolbox.api.memgraph import Memgraph
from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import (
    from_unstructured,
    create_index,
    compute_embeddings,
    create_vector_search_index
)
from openai import OpenAI

async def build_knowledge_graph(sources: list[str]):
    """Ingest documents and build the knowledge graph."""
    memgraph = Memgraph()

    # Clear existing data (optional, for fresh start)
    memgraph.query("MATCH (n) DETACH DELETE n;")

    # Create indexes
    create_index(memgraph, "Chunk", "hash")

    # Initialize LightRAG
    lightrag = MemgraphLightRAGWrapper(disable_embeddings=True)
    await lightrag.initialize(working_dir="./lightrag_storage")

    # Ingest
    await from_unstructured(
        sources=sources,
        memgraph=memgraph,
        lightrag_wrapper=lightrag,
        link_chunks=True
    )
    await lightrag.afinalize()

    # Compute embeddings and create vector index
    compute_embeddings(memgraph, "Chunk")
    create_vector_search_index(memgraph, "Chunk", "embedding")

    return memgraph

def query_knowledge_graph(memgraph: Memgraph, question: str) -> str:
    """Query the knowledge graph with GraphRAG."""

    # Hybrid retrieval
    results = memgraph.query(f"""
        CALL embeddings.text(['{question}']) YIELD embeddings
        CALL vector_search.search('vs_name', 5, embeddings[0])
        YIELD node, similarity
        MATCH (node)-[*bfs]-(related:Chunk)
        WITH DISTINCT related, degree(related) AS importance
        ORDER BY importance DESC
        RETURN related.text AS text LIMIT 5
    """)

    context = "\n\n".join([r["text"] for r in results if r["text"]])

    # Generate answer
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content

# Usage
async def main():
    sources = [
        "https://memgraph.com/docs/getting-started",
        "./my_documents/report.pdf"
    ]

    memgraph = await build_knowledge_graph(sources)

    answer = query_knowledge_graph(
        memgraph,
        "What are the key features of Memgraph?"
    )
    print(answer)

asyncio.run(main())
```

## Alternative: LangChain Integration

If you're building with LangChain, use the `langchain-memgraph` integration:

```python
from langchain.chat_models import init_chat_model
from langchain_memgraph import MemgraphToolkit
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langgraph.prebuilt import create_react_agent

# Initialize
llm = init_chat_model("gpt-4o", model_provider="openai")
graph = MemgraphLangChain(url="bolt://localhost:7687")

# Create toolkit with all Memgraph tools
toolkit = MemgraphToolkit(db=graph.db, llm=llm)

# Create agent
agent = create_react_agent(llm, toolkit.get_tools())

# Query
response = agent.invoke({
    "messages": [{"role": "user", "content": "What entities are in the graph?"}]
})
```

## Graph Schema Created

After ingestion, your Memgraph database will contain:

| Node Label | Description        | Key Properties                        |
| ---------- | ------------------ | ------------------------------------- |
| `Chunk`    | Document chunks    | `text`, `hash`, `embedding`, `source` |
| `Entity`   | Extracted entities | `name`, `description`, `type`         |

| Relationship   | Description                   |
| -------------- | ----------------------------- |
| `NEXT`         | Sequential chunk order        |
| `MENTIONED_IN` | Entity → Chunk reference      |
| `RELATES_TO`   | Entity → Entity relationships |

## Retrieval Patterns

### Pattern 1: Direct Vector Search

```cypher
CALL embeddings.text(['your question']) YIELD embeddings
CALL vector_search.search('vs_name', 10, embeddings[0])
YIELD node, similarity
RETURN node.text, similarity
ORDER BY similarity DESC
```

### Pattern 2: Vector + 1-Hop Expansion

```cypher
CALL embeddings.text(['your question']) YIELD embeddings
CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node
MATCH (node)-[:MENTIONED_IN|RELATES_TO]-(related)
RETURN DISTINCT related
```

### Pattern 3: Vector + BFS Traversal (Recommended)

```cypher
CALL embeddings.text(['your question']) YIELD embeddings
CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node
MATCH (node)-[*bfs]-(dst:Chunk)
WITH DISTINCT dst, degree(dst) AS importance
ORDER BY importance DESC
RETURN dst LIMIT 10
```

### Pattern 4: Entity-Centric Retrieval

```cypher
MATCH (e:Entity)
WHERE e.name CONTAINS 'specific_term'
MATCH (e)-[:MENTIONED_IN]->(chunk:Chunk)
RETURN chunk.text
```

## Troubleshooting

| Issue                    | Solution                                                                     |
| ------------------------ | ---------------------------------------------------------------------------- |
| "Vector index not found" | Run `create_vector_search_index()` after computing embeddings                |
| Empty retrieval results  | Check that documents were ingested: `MATCH (n:Chunk) RETURN count(n)`        |
| Slow queries             | Create label indexes: `CREATE INDEX ON :Chunk;`                              |
| LightRAG errors          | Ensure `OPENAI_API_KEY` is set for entity extraction                         |
| Connection refused       | Verify Memgraph is running: `docker run -p 7687:7687 memgraph/memgraph-mage` |

## Performance Tips

1. **Batch ingestion**: Process documents in batches of 10-20 for large collections
2. **Index early**: Create indexes before ingestion, not after
3. **Limit traversal depth**: Use `*bfs..3` instead of unbounded `*bfs` for large graphs
4. **Cache embeddings**: Store question embeddings if asking similar questions repeatedly
