# 🚀 Memgraph MCP Server

Memgraph MCP Server is a lightweight server implementation of the Model Context Protocol (MCP) designed to connect Memgraph with LLMs.

![mcp-server](./mcp-server.png)

## Run Memgraph MCP server

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
2. Install [Claude for Desktop](https://claude.ai/download).
3. Add the Memgraph server to Claude config

You can do it in the UI, by opening your Claude desktop app navigate to `Settings`, under the `Developer` section, click on `Edit Config` and add the
following content:

```
{
    "mcpServers": {
      "mpc-memgraph": {
        "command": "uv",
        "args": [
            "run",
            "--with",
            "mcp-memgraph",
            "--python",
            "3.13",
            "mcp-memgraph"
        ]
     }
   }
}
```

Or you can open the config file in your favorite text editor. The location of the config file depends on your operating system:

**MacOS/Linux**

```
~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows**

```
%APPDATA%/Claude/claude_desktop_config.json
```

> [!NOTE]  
> You may need to put the full path to the uv executable in the command field. You can get this by running `which uv` on MacOS/Linux or `where uv` on Windows. Make sure you pass in the absolute path to your server.

### Chat with the database

1. Run Memgraph MAGE:
   ```
   docker run -p 7687:7687 memgraph/memgraph-mage --schema-info-enabled=True
   ```
   The `--schema-info-enabled` configuration setting is set to `True` to allow LLM to run `SHOW SCHEMA INFO` query.
2. Open Claude Desktop and see the Memgraph tools and resources listed. Try it out! (You can load dummy data from [Memgraph Lab](https://memgraph.com/docs/data-visualization) Datasets)

## 🔧Tools

The Memgraph MCP Server exposes the following tools over MCP. Each tool runs a Memgraph‐toolbox operation and returns a list of records (dictionaries).

### run_query(query: str)

Run any arbitrary Cypher query against the connected Memgraph database.  
Parameters:

- `query`: A valid Cypher query string.

### get_configuration()

Fetch the current Memgraph configuration settings.  
Equivalent to running `SHOW CONFIGURATION`.

### get_index()

Retrieve information about existing indexes.  
Equivalent to running `SHOW INDEX INFO`.

### get_constraint()

Retrieve information about existing constraints.  
Equivalent to running `SHOW CONSTRAINT INFO`.

### get_schema()

Fetch the graph schema (labels, relationships, property keys).  
Equivalent to running `SHOW SCHEMA INFO`.

### get_storage()

Retrieve storage usage metrics for nodes, relationships, and properties.  
Equivalent to running `SHOW STORAGE INFO`.

### get_triggers()

List all database triggers.  
Equivalent to running `SHOW TRIGGERS`.

### get_betweenness_centrality()

Compute betweenness centrality on the entire graph.  
Uses `BetweennessCentralityTool` under the hood.

### get_page_rank()

Compute PageRank scores for all nodes.  
Uses `PageRankTool` under the hood.

## 🗺️ Roadmap

The Memgraph MCP Server is just at its beginnings. We're actively working on expanding its capabilities and making it even easier to integrate Memgraph into modern AI workflows.
