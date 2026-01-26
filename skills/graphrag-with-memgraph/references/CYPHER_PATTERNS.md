# Cypher Patterns for GraphRAG

Detailed Cypher query patterns for graph-enhanced retrieval in Memgraph.

## Vector Search Basics

### Create Vector Index

```cypher
-- Create index with 384 dimensions (default for sentence-transformers)
CREATE VECTOR INDEX vs_name
ON :Chunk(embedding)
WITH CONFIG {'dimension': 384, 'capacity': 10000};
```

### Search by Vector

```cypher
CALL vector_search.search('vs_name', 10, $query_vector)
YIELD node, distance, similarity
RETURN node.text, similarity
ORDER BY similarity DESC;
```

### Generate Embeddings for Text

```cypher
-- Single text
CALL embeddings.text(['What is GraphRAG?']) YIELD embeddings
RETURN embeddings[0];

-- Multiple texts
CALL embeddings.text(['query1', 'query2', 'query3']) YIELD embeddings
RETURN embeddings;
```

## Graph Traversal Patterns

### BFS from Vector Results

```cypher
-- Find all connected chunks within 3 hops
CALL embeddings.text([$question]) YIELD embeddings
CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node
MATCH path = (node)-[*bfs..3]-(related:Chunk)
RETURN DISTINCT related.text;
```

### Weighted by Connectivity

```cypher
-- Rank results by number of connections (importance)
CALL embeddings.text([$question]) YIELD embeddings
CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node
MATCH (node)-[*bfs]-(dst:Chunk)
WITH DISTINCT dst, degree(dst) AS importance
ORDER BY importance DESC
RETURN dst.text LIMIT 10;
```

### Follow Specific Relationship Types

```cypher
-- Only follow MENTIONED_IN and RELATES_TO
CALL embeddings.text([$question]) YIELD embeddings
CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node
MATCH (node)-[:MENTIONED_IN|RELATES_TO*1..2]-(related)
RETURN DISTINCT related;
```

## Entity-Based Retrieval

### Find Entities by Name

```cypher
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($search_term)
RETURN e.name, e.description, e.type;
```

### Entity to Chunks

```cypher
-- Find all chunks mentioning an entity
MATCH (e:Entity {name: $entity_name})-[:MENTIONED_IN]->(c:Chunk)
RETURN c.text, c.source;
```

### Related Entities

```cypher
-- Find entities related to a given entity
MATCH (e:Entity {name: $entity_name})-[:RELATES_TO]-(related:Entity)
RETURN related.name, related.type;
```

### Entity Co-occurrence

```cypher
-- Find entities that appear together in the same chunk
MATCH (e1:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity)
WHERE e1 <> e2
RETURN e1.name, e2.name, count(c) AS co_occurrences
ORDER BY co_occurrences DESC;
```

## Hybrid Patterns (Vector + Graph)

### Vector Search with Entity Context

```cypher
-- Get chunks + their mentioned entities
CALL embeddings.text([$question]) YIELD embeddings
CALL vector_search.search('vs_name', 5, embeddings[0]) YIELD node, similarity
OPTIONAL MATCH (entity:Entity)-[:MENTIONED_IN]->(node)
RETURN node.text, similarity, collect(entity.name) AS entities;
```

### Multi-Hop Context Expansion

```cypher
-- Expand context through entity relationships
CALL embeddings.text([$question]) YIELD embeddings
CALL vector_search.search('vs_name', 3, embeddings[0]) YIELD node
MATCH (node)<-[:MENTIONED_IN]-(e:Entity)-[:RELATES_TO]-(related:Entity)-[:MENTIONED_IN]->(other:Chunk)
WHERE other <> node
RETURN DISTINCT other.text;
```

### Sequential Context (Previous/Next Chunks)

```cypher
-- Get surrounding chunks for context
CALL embeddings.text([$question]) YIELD embeddings
CALL vector_search.search('vs_name', 3, embeddings[0]) YIELD node
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(node)
OPTIONAL MATCH (node)-[:NEXT]->(next:Chunk)
RETURN prev.text AS previous, node.text AS matched, next.text AS next;
```

## Aggregation and Summarization

### Count Entities by Type

```cypher
MATCH (e:Entity)
RETURN e.type, count(e) AS count
ORDER BY count DESC;
```

### Most Connected Entities

```cypher
MATCH (e:Entity)
RETURN e.name, e.type, degree(e) AS connections
ORDER BY connections DESC
LIMIT 10;
```

### Chunks per Source

```cypher
MATCH (c:Chunk)
RETURN c.source, count(c) AS chunk_count
ORDER BY chunk_count DESC;
```

## Index Management

### Create Property Index

```cypher
CREATE INDEX ON :Chunk(hash);
CREATE INDEX ON :Entity(name);
```

### Create Label Index

```cypher
CREATE INDEX ON :Chunk;
CREATE INDEX ON :Entity;
```

### List All Indexes

```cypher
SHOW INDEX INFO;
```

### Drop Index

```cypher
DROP INDEX ON :Chunk(hash);
```

## Parameterized Queries (Python)

Use parameterized queries for safety and performance:

```python
# Safe parameterized query
results = memgraph.query(
    """
    CALL embeddings.text([$question]) YIELD embeddings
    CALL vector_search.search('vs_name', $limit, embeddings[0]) YIELD node
    RETURN node.text
    """,
    params={
        "question": user_question,
        "limit": 10
    }
)
```

## Performance Optimization

### Use EXPLAIN to Analyze

```cypher
EXPLAIN MATCH (c:Chunk)-[:MENTIONED_IN]-(e:Entity)
WHERE e.name = 'Memgraph'
RETURN c.text;
```

### Use PROFILE for Execution Stats

```cypher
PROFILE MATCH (c:Chunk)
WHERE c.text CONTAINS 'graph'
RETURN c.text;
```

### Limit Early

```cypher
-- Good: Limit before expensive operations
MATCH (c:Chunk)
WITH c LIMIT 100
MATCH (c)<-[:MENTIONED_IN]-(e:Entity)
RETURN c, collect(e);

-- Bad: Limit after expensive join
MATCH (c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
RETURN c, collect(e)
LIMIT 100;
```
