# Retrieval Strategies Reference

Choosing the right retrieval strategy for your GraphRAG application.

## Strategy Comparison

| Strategy        | Best For                  | Latency | Context Quality |
| --------------- | ------------------------- | ------- | --------------- |
| Vector Only     | Simple semantic search    | Low     | Medium          |
| Vector + 1-Hop  | Entity-aware retrieval    | Medium  | High            |
| Vector + BFS    | Deep relationship context | High    | Very High       |
| Entity-First    | Known entity queries      | Low     | High            |
| Hybrid Weighted | General purpose GraphRAG  | Medium  | Very High       |

## Strategy 1: Vector Only

Use when you need fast, simple semantic search without graph context.

```python
def vector_only_retrieve(memgraph, question: str, limit: int = 10) -> list[str]:
    """Pure vector similarity search."""
    results = memgraph.query(f"""
        CALL embeddings.text(['{question}']) YIELD embeddings
        CALL vector_search.search('vs_name', {limit}, embeddings[0])
        YIELD node, similarity
        RETURN node.text AS text, similarity
        ORDER BY similarity DESC
    """)
    return [r["text"] for r in results]
```

**Pros:**

- Fastest retrieval
- Simple to implement
- Good for focused questions

**Cons:**

- Misses related context
- No relationship awareness
- Limited to indexed content

## Strategy 2: Vector + 1-Hop Expansion

Use when you want immediate neighbors of similar chunks.

```python
def vector_one_hop_retrieve(memgraph, question: str) -> list[str]:
    """Vector search + immediate graph neighbors."""
    results = memgraph.query(f"""
        CALL embeddings.text(['{question}']) YIELD embeddings
        CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node

        // Get the matched chunk
        WITH node

        // Expand to immediate neighbors
        OPTIONAL MATCH (node)-[:MENTIONED_IN|NEXT]-(neighbor)

        // Collect both original and neighbors
        WITH node, collect(DISTINCT neighbor) AS neighbors

        RETURN node.text AS matched_text,
               [n IN neighbors | n.text] AS neighbor_texts
    """)

    texts = []
    for r in results:
        texts.append(r["matched_text"])
        texts.extend([t for t in r["neighbor_texts"] if t])
    return texts
```

**Pros:**

- Adds immediate context
- Reasonable latency
- Good entity awareness

**Cons:**

- Limited depth
- May miss important distant connections

## Strategy 3: Vector + BFS Traversal

Use when you need comprehensive context from the entire connected subgraph.

```python
def vector_bfs_retrieve(memgraph, question: str, max_depth: int = 3) -> list[str]:
    """Vector search + breadth-first graph traversal."""
    results = memgraph.query(f"""
        CALL embeddings.text(['{question}']) YIELD embeddings
        CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node

        // BFS traversal to find connected chunks
        MATCH (node)-[*bfs..{max_depth}]-(dst:Chunk)

        // Rank by importance (connectivity)
        WITH DISTINCT dst, degree(dst) AS importance
        ORDER BY importance DESC

        RETURN dst.text AS text LIMIT 10
    """)
    return [r["text"] for r in results]
```

**Pros:**

- Rich context from graph structure
- Finds non-obvious connections
- Best for complex questions

**Cons:**

- Higher latency
- May include less relevant content
- Needs depth limiting

## Strategy 4: Entity-First Retrieval

Use when the question references specific named entities.

```python
def entity_first_retrieve(memgraph, question: str, entities: list[str]) -> list[str]:
    """Start from known entities, expand to chunks."""
    entity_list = ", ".join([f"'{e}'" for e in entities])

    results = memgraph.query(f"""
        // Find matching entities
        MATCH (e:Entity)
        WHERE e.name IN [{entity_list}]

        // Get chunks mentioning these entities
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)

        // Also get related entities and their chunks
        OPTIONAL MATCH (e)-[:RELATES_TO]-(related:Entity)-[:MENTIONED_IN]->(rc:Chunk)

        WITH collect(DISTINCT c) + collect(DISTINCT rc) AS all_chunks
        UNWIND all_chunks AS chunk

        RETURN DISTINCT chunk.text AS text
    """)
    return [r["text"] for r in results]
```

**Pros:**

- Precise for entity-focused questions
- Fast when entities are known
- No vector search needed

**Cons:**

- Requires entity extraction from question
- Misses semantic matches
- Limited to indexed entities

## Strategy 5: Hybrid Weighted (Recommended)

Combines vector similarity with graph centrality for balanced retrieval.

```python
def hybrid_weighted_retrieve(
    memgraph,
    question: str,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4
) -> list[str]:
    """Combine vector similarity with graph importance."""
    results = memgraph.query(f"""
        CALL embeddings.text(['{question}']) YIELD embeddings
        CALL vector_search.search('vs_name', 10, embeddings[0]) YIELD node, similarity

        // Calculate graph importance
        WITH node, similarity, degree(node) AS graph_score

        // Normalize and combine scores
        WITH node,
             similarity * {vector_weight} +
             (toFloat(graph_score) / 100.0) * {graph_weight} AS combined_score

        ORDER BY combined_score DESC
        RETURN node.text AS text, combined_score
        LIMIT 10
    """)
    return [r["text"] for r in results]
```

**Pros:**

- Balances relevance and importance
- Tunable weights
- Good general-purpose strategy

**Cons:**

- More complex scoring
- Requires weight tuning

## Choosing a Strategy

```
                    ┌─────────────────┐
                    │ What type of    │
                    │ question?       │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌──────▼──────┐    ┌──────▼──────┐
    │ Simple    │     │ Entity-     │    │ Complex/    │
    │ semantic  │     │ specific    │    │ Exploratory │
    └─────┬─────┘     └──────┬──────┘    └──────┬──────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌────────────┐    ┌────────────┐
    │ Vector    │     │ Entity-    │    │ Vector +   │
    │ Only      │     │ First      │    │ BFS        │
    └───────────┘     └────────────┘    └────────────┘
```

## Performance Benchmarks (Approximate)

| Strategy               | 1K Chunks | 10K Chunks | 100K Chunks |
| ---------------------- | --------- | ---------- | ----------- |
| Vector Only            | 10ms      | 20ms       | 50ms        |
| Vector + 1-Hop         | 20ms      | 50ms       | 150ms       |
| Vector + BFS (depth 3) | 50ms      | 200ms      | 500ms       |
| Entity-First           | 5ms       | 15ms       | 40ms        |
| Hybrid Weighted        | 30ms      | 80ms       | 200ms       |

_Benchmarks are illustrative. Actual performance depends on hardware, graph density, and query complexity._

## Combining Strategies

For production systems, consider combining strategies:

```python
async def adaptive_retrieve(memgraph, question: str) -> list[str]:
    """Choose strategy based on question characteristics."""

    # Extract entities from question (using LLM or NER)
    entities = extract_entities(question)

    if entities:
        # Entity-specific question
        return entity_first_retrieve(memgraph, question, entities)
    elif is_exploratory_question(question):
        # Complex question needing deep context
        return vector_bfs_retrieve(memgraph, question, max_depth=3)
    else:
        # Default: balanced hybrid approach
        return hybrid_weighted_retrieve(memgraph, question)
```
