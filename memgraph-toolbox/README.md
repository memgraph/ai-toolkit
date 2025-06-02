# Memgraph Toolbox

The **Memgraph Toolbox** is a collection of tools designed to interact with a
Memgraph database. These tools provide functionality for querying, analyzing,
and managing data within Memgraph, making it easier to work with graph data.
They are made to be easily called from other frameworks such as
**MCP**, **LangChain** or **LlamaIndex**.

## Available Tools

Below is a list of tools included in the toolbox, along with their descriptions:

1. `ShowTriggersTool` - Shows [trigger](https://memgraph.com/docs/fundamentals/triggers) information from a Memgraph database.
2. `ShowStorageInfoTool` - Shows storage information from a Memgraph database.
3. `ShowSchemaInfoTool` - Shows [schema](https://memgraph.com/docs/querying/schema) information from a Memgraph database.
4. `PageRankTool` - Calculates [PageRank](https://memgraph.com/docs/advanced-algorithms/available-algorithms/pagerank) on a graph in Memgraph.
5. `BetweennessCentralityTool` - Calculates [betweenness centrality](https://memgraph.com/docs/advanced-algorithms/available-algorithms/betweenness_centrality) for nodes in a graph.
6. `ShowIndexInfoTool` - Shows [index](https://memgraph.com/docs/fundamentals/indexes) information from a Memgraph database.
7. `CypherTool` - Executes arbitrary [Cypher queries](https://memgraph.com/docs/querying) on a Memgraph database.
8. `ShowConstraintInfoTool` - Shows [constraint](https://memgraph.com/docs/fundamentals/constraints) information from a Memgraph database.
9. `ShowConfigTool` - Shows [configuration](https://memgraph.com/docs/database-management/configuration) information from a Memgraph database.

## Usage

Each tool is implemented as a Python class inheriting from `BaseTool`. To use a
tool:

1. Instantiate the tool with a `Memgraph` database connection.
2. Call the `call` method with the required arguments.

Example:

```python
from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.memgraph_toolkit import MemgraphToolkit

# Connect to Memgraph
db = Memgraph(url="bolt://localhost:7687", username="", password="")

# Show available tools
toolbox = MemgraphToolkit(db)
for tool in toolbox.get_all_tools():
    print(f"Tool Name: {tool.name}, Description: {tool.description}")

# Use the ShowTriggersTool
tool = ShowTriggersTool(db)
triggers = tool.call({})
print(triggers)
```

## Requirements

- Python 3.10+
- Running [Memgraph instance](https://memgraph.com/docs/getting-started)
- Memgraph [MAGE library](https://memgraph.com/docs/advanced-algorithms/install-mage) (for certain tools like `pagerank` and `run_betweenness_centrality`)

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to
improve the toolbox.

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.
