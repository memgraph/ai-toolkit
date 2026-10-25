# Sessions Graph

Store durable memories and retrieve context from a user's past agent sessions in Memgraph.

Sessions Graph is the memory component of [Context Graph](../README.md). It links users to sessions, stores memory files, extracts entities and relationships, and retrieves dated evidence for the agent's next answer.

**Using a coding agent?** The [Context Graph installer](../README.md#getting-started) wires this package automatically. The examples below are for direct Python use.

## Installation

Requires Memgraph 3.6 or newer. Use the MAGE image for vector recall.

```bash
pip install sessions-graph
```

Optional features:

```bash
pip install 'sessions-graph[memgraph-context-graph]'  # runtime connector
pip install 'sessions-graph[reconciliation]'         # local entity extraction
```

Capture, extraction, embeddings, and recall need no LLM API key. An optional key enables ontology derivation.

## Quick start

```python
from sessions_graph import SessionsGraph

graph = SessionsGraph()  # connects via MEMGRAPH_URL / MEMGRAPH_USER / MEMGRAPH_PASSWORD env vars
graph.setup()  # creates constraints and the text index (run once)

# Write a memory
mem = graph.save_memory(
    user_id="alice",
    content="Prefers Python over TypeScript",
    session_id="s-abc123",  # optional — links memory to a session for provenance
)

# Retrieve all memories for a user
memories = graph.get_memories("alice")

# Search memories by content (full-text, powered by Tantivy)
results = graph.search_memories("alice", "Python")

# Update or delete
graph.update_memory(mem.memory_id, "Prefers Python, especially for data tooling")
graph.delete_memory(mem.memory_id)
```

## Memory files

Agents can organize curated memory as files under `/memories`, shared across sessions. The file API supports viewing, creating, editing, deleting, and renaming, with version history and checks against stale edits.

```python
store = graph.memory_store("alice", session_id="s-abc123")
store.create("/memories/feedback/testing.md", "Prefer real Memgraph in tests.")
print(store.view("/memories/feedback/testing.md"))
```

See [memory files and the Python API](docs/python-api.md) for path rules, version history, text search, and method signatures. Coding agents use the same store through the [MCP memory tool](../memgraph-context-graph/README.md#memory-context-graph-as-the-harnesss-memory).

## Recall

`recall(user_id, question)` returns what one user's own past sessions hold
about a question: the evidence, not an answer. The caller's model answers
from it.

```python
recalled = graph.recall("alice", "Which database did we pick for the cache?")
print(recalled.render(today="2026-10-05"))  # header with reading rules, then the rows
recalled.to_json()  # source rows and exchange groups as data
```

Recall combines six searches over the user's history:

| Lane | Finds |
|---|---|
| `memories` | Curated memory files, by vector and full-text search |
| `turns` | Conversation messages nearest the question by vector |
| `text` | Messages matching the question by full-text search |
| `entities` | Relevant entities and their facts |
| `facts` | Extracted facts nearest the question by vector |
| `user_facts` | Facts grouped by relevant relation types, for questions that need a complete set |

Saved memory files come first. By default, recall then groups the related source messages into exchanges that fit a 6,000-token row budget per space, and uses extracted facts to rank them rather than showing them as separate claims. Supplemental search adds complementary problem-resolution exchanges. Set `packing="turns"` to restore separate turn and fact rows, or `supplemental_search=False` to turn off supplemental discovery. Without MAGE, recall falls back to text search and reports that in its output.

Exchange dependencies are included in the standard install; offline deployments must first populate the tokenizer cache, as described in [exchange recall](../eval/recall-exchanges.md). On the matched 400-question retrieval comparison, exchanges used 38.1% fewer context tokens than historical recall but took 0.859 s versus 0.744 s for optimized turns. A general quality improvement has not been established; the report records question-type regressions and the remaining validation.

Tune search widths and packing with `RecallConfig.from_mapping(...)` or the config file's `[recall]` section; existing config files that omit the new keys adopt the defaults. Coding agents call the same read path through the MCP `recall` tool; see [recall settings](../memgraph-context-graph/docs/memory.md#recall-memory-for-the-harnesss-model) and the [benchmark results](../eval/README.md#official-numbers).

## Session reconciliation

Reconciliation turns session messages and tool activity into entities and relationships with local GLiNER2. It processes memory files separately so edits can replace what an earlier version produced.

```bash
sessions-graph reconcile --pending
sessions-graph reconcile --session s-abc123
```

Extraction runs outside hooks. Turn and session ends queue work; optional automatic processing starts on session end. See the [reconciliation guide](docs/reconciliation.md#session-reconciliation) for scheduling, configuration, and programmatic use.

## Ontology versions

Each user has an adopted ontology describing the entity and relation types extraction should produce. You can load a YAML schema or optionally let an LLM propose extensions, which are checked against the existing model before adoption.

```bash
sessions-graph ontology load --file coding.yaml
sessions-graph ontology show
```

See [ontology versions](docs/reconciliation.md#ontology-versions) for the default model, pinned types, derivation, and adoption checks.

## Embeddings for recall

MAGE computes vectors inside Memgraph. Turn ends schedule message embedding; reconciliation embeds the entities and facts it creates. Catch up pending work with:

```bash
sessions-graph embed --pending
```

The default model is `BAAI/bge-small-en-v1.5` (384 dimensions). It downloads inside Memgraph on first use and needs at least 2 GiB of memory. See [embeddings for recall](docs/reconciliation.md#embeddings-for-recall) for model changes and failure handling.

## Integration with Memgraph Context Graph

SDK applications get capture, the memory index, and tools through the [SDK integration](../memgraph-context-graph/docs/sdk-memory.md).

For custom event wiring, register `SessionsGraphConnector` with an `AgentLink`; see the [Python API guide](docs/python-api.md#integration-with-memgraph-context-graph). Hook integrations read configuration from the Context Graph config file; direct `SessionsGraph()` use reads the connection environment variables shown above.

## Graph schema

```text
(:User)-[:HAD_SESSION]->(:Session)
(:User)-[:HAS_MEMORY]->(:Memory)
(:Session)-[:PRODUCED_MEMORY]->(:Memory)
(:Memory)-[:ABOUT]->(:Project)
(:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)
(:Memory)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(:Entity)
```

Sessions Graph owns users, memories, and session provenance. Other components share the Session node. Actions belong to [Actions Graph](../actions-graph/); chunks and extracted entities come from [unstructured2graph](../../unstructured2graph/).

## API reference

The [Python API guide](docs/python-api.md#api-reference) lists memory, session, and reconciler methods.

## Development

From the repository root:

```bash
./scripts/dev-memgraph.sh test sessions-graph
```
