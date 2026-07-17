# Memgraph Toolbox

The **Memgraph Toolbox** is a collection of tools designed to interact with a
Memgraph database. These tools provide functionality for querying, analyzing,
and managing data within Memgraph, making it easier to work with graph data.
They are made to be easily called from other frameworks such as
**MCP**, **LangChain** or **LlamaIndex**.

## Available Tools

Below is a list of tools included in the toolbox, along with their descriptions:

1. `CypherTool` - Executes arbitrary [Cypher queries](https://memgraph.com/docs/querying) on a Memgraph database.
2. `SearchSchemaTool` - Searches the entire graph [schema](https://memgraph.com/docs/querying/schema) (nodes, relationships and enums) by a regex pattern.
3. `NodeSchemaTool` - Returns the full [schema](https://memgraph.com/docs/querying/schema) definition of a node by its labels, including properties, indexes, constraints, and connected relationships.
4. `RelationshipSchemaTool` - Returns the full [schema](https://memgraph.com/docs/querying/schema) definition of a relationship by its type and connected node labels, including properties and indexes.
5. `EnumSchemaTool` - Returns the [schema](https://memgraph.com/docs/querying/schema) definition of an enum by its name, including its values.

## Usage

Each tool is implemented as a Python class inheriting from `BaseTool`. To use a
tool:

1. Instantiate the tool with a `Memgraph` database connection.
2. Call the `call` method with the required arguments.

Example:

```python
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.memgraph_toolbox import MemgraphToolbox

# Connect to Memgraph
db = Memgraph(url="bolt://localhost:7687", username="", password="")

# Show available tools
toolbox = MemgraphToolbox(db)
for tool in toolbox.get_all_tools():
    print(f"Tool Name: {tool.name}, Description: {tool.description}")

# Run a Cypher query
cypher_tool = toolbox.get_tool("run_cypher_query")
result = cypher_tool.call({"query": "MATCH (n) RETURN n LIMIT 5"})
print(result)

# Get the schema definition of a node
node_schema_tool = toolbox.get_tool("get_node_schema")
result = node_schema_tool.call({"node_labels": ["Person"]})
print(result)
```

## Installation

Install the base package:

```bash
pip install memgraph-toolbox
```

### Optional dependencies

For the MCP prompt client (litellm + mcp):

```bash
pip install 'memgraph-toolbox[client]'
```

For evaluation metrics (deepeval, sentence-transformers, torch):

```bash
pip install 'memgraph-toolbox[evaluations]'
```

For running tests:

```bash
pip install 'memgraph-toolbox[test]'
```

For all optional dependencies (e.g., CI):

```bash
pip install 'memgraph-toolbox[client,evaluations,test]'
```

## Requirements

- Python 3.10+
- Running [Memgraph instance](https://memgraph.com/docs/getting-started)
- _(Optional)_ Memgraph [MAGE library](https://memgraph.com/docs/advanced-algorithms/install-mage) (for certain algorithms like pagerank and betweenness_centrality)

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to
improve the toolbox.

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.
