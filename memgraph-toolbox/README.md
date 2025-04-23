# Memgraph Toolbox

The **Memgraph Toolbox** is a collection of tools designed to interact with a
Memgraph database. These tools provide functionality for querying, analyzing,
and managing data within Memgraph, making it easier to work with graph data.
They are made to be easily called from other framework implementations such as
**MCP**, **Langchain** or **Lamaindex**.

## Available Tools

Below is a list of tools included in the toolbox, along with their descriptions:

1. `ShowTriggersTool` - Shows trigger information from a Memgraph database.
2. `ShowStorageInfoTool` - Shows storage information from a Memgraph database.
3. `ShowSchemaInfoTool` - Shows schema information from a Memgraph database.
4. `PageRankTool` - Calculates PageRank on a graph in Memgraph.
5. `ShowIndexInfoTool` - Shows index information from a Memgraph database.
6. `CypherTool` - Executes arbitrary Cypher queries on a Memgraph database.
7. `ShowConstraintInfoTool` - Shows constraint information from a Memgraph database.
8. `ShowConfigTool` - Shows configuration information from a Memgraph database.
9. `BetweennessCentralityTool` - Calculates betweenness centrality for nodes in a graph.

## Usage

Each tool is implemented as a Python class inheriting from `BaseTool`. To use a
tool:

1. Instantiate the tool with a `Memgraph` database connection.
2. Call the `call` method with the required arguments.

Example:

```python
from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.api.memgraph import Memgraph

# Connect to Memgraph
db = Memgraph()

# Use the ShowTriggersTool
tool = ShowTriggersTool(db)
triggers = tool.call({})
print(triggers)
```

## Requirements

- Python 3.10+
- Memgraph database
- Memgraph Mage (for certain tools like `pagerank` and `run_betweenness_centrality`)

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to
improve the toolbox.

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.
