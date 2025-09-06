# 🚀 Memgraph MCP Server

Memgraph MCP Server is a lightweight server implementation of the Model Context Protocol (MCP) designed to connect Memgraph with LLMs.

![mcp-server](./mcp-server.png)

## Run Memgraph MCP server with Claude

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

### get_node_neighborhood(node_id: str, max_distance: int = 1, limit: int = 100)

Find nodes within a specified distance from a given node.
Parameters:

- `node_id`: The ID of the starting node to find neighborhood around
- `max_distance`: Maximum distance (hops) to search from the starting node. Default is 1
- `limit`: Maximum number of nodes to return. Default is 100

Uses `NodeNeighborhoodTool` under the hood.

### search_node_vectors(index_name: str, query_vector: List[float], limit: int = 10)

Perform vector similarity search on nodes in Memgraph using cosine similarity.
Parameters:

- `index_name`: Name of the index to use for the vector search
- `query_vector`: Query vector to search for similarity
- `limit`: Number of similar nodes to return. Default is 10

Uses `NodeVectorSearchTool` under the hood.

## 🐳 Run Memgraph MCP server with Docker

### Building Memgraph MCP image

To build the Docker image using your local `memgraph-toolbox` code, run from the root of the monorepo:

```bash
cd /path/to/ai-toolkit
docker build -f integrations/mcp-memgraph/Dockerfile -t mcp-memgraph:latest .
```

This will include your local `memgraph-toolbox` and install it inside the image.

### Running the Docker image

#### 1. Streamable HTTP mode (recommended for most users)

To connect to local Memgraph containers, by default the MCP server will be available at `http://localhost:8000/mcp/`:

```bash
docker run --rm mcp-memgraph:latest
```

#### 2. Stdio mode (for integration with MCP stdio clients)

Configure your MCP host to run the docker command and utilize stdio:

```bash
docker run --rm -i -e MCP_TRANSPORT=stdio mcp-memgraph:latest
```

> 📄 Note: By default, the server will connect to a Memgraph instance running on localhost docker network `bolt://host.docker.internal:7687`. If you have a Memgraph instance running on a different host or port, you can specify it using environment variables.

#### 3. Custom Memgraph connection (external instance, no host network)

To avoid using host networking, or to connect to an external Memgraph instance:

```bash
docker run --rm \
  -p 8000:8000 \
  -e MEMGRAPH_URL=bolt://memgraph:7687 \
  -e MEMGRAPH_USER=myuser \
  -e MEMGRAPH_PASSWORD=password \
  mcp-memgraph:latest
```


## ⚙️ Configuration

### Environment Variables

The following environment variables can be used to configure the Memgraph MCP Server, whether running with Docker or directly (e.g., with `uv` or `python`).

- `MEMGRAPH_URL`: The Bolt URL of the Memgraph instance to connect to. Default: `bolt://host.docker.internal:7687`
    - The default value allows you to connect to a Memgraph instance running on your host machine from within the Docker container.
- `MEMGRAPH_USER`: The username for authentication. Default: `memgraph`
- `MEMGRAPH_PASSWORD`: The password for authentication. Default: empty
- `MEMGRAPH_DATABASE`: The database name to connect to. Default: `memgraph`
- `MCP_TRANSPORT`: The transport protocol to use. Options: `http` (default), `stdio`

You can set these environment variables in your shell, in your Docker run command, or in your deployment environment. 

### Connecting from VS Code (HTTP server)

If you are using VS Code MCP extension or similar, your configuration for an HTTP server would look like:

```json
{
    "servers": {
        "mcp-memgraph-http": {
            "url": "http://localhost:8000/mcp/"
        }
    }
}
```

> **Note:** The URL must end with `/mcp/`.

---

#### Running the Docker image in Visual Studio Code using stdio

You can also run the server using stdio for integration with MCP stdio clients:

1. Open Visual Studio Code, open Command Palette (Ctrl+Shift+P or Cmd+Shift+P on Mac), and select `MCP: Add server...`.
2. Choose `Command (stdio)`
3. Enter `docker` as the command to run.
4. For Server ID enter `mcp-memgraph`.
5. Choose "User" (adds to user-space `settings.json`) or "Workspace" (adds to `.vscode/mcp.json`).

When the settings open, enhance the args as follows:

```json
{
    "servers": {
        "mcp-memgraph": {
            "type": "stdio",
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "-e", "MCP_TRANSPORT=stdio",
                "mcp-memgraph:latest"
            ]
        }
    }
}
```

To connect to a remote Memgraph instance with authentication, add environment variables to the `args` list:

```json
{
    "servers": {
        "mcp-memgraph": {
            "type": "stdio",
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "-e", "MCP_TRANSPORT=stdio",
                "-e", "MEMGRAPH_URL=bolt://memgraph:7687",
                "-e", "MEMGRAPH_USER=myuser",
                "-e", "MEMGRAPH_PASSWORD=mypassword",
                "mcp-memgraph:latest"
            ]
        }
    }
}
```

---

Open GitHub Copilot in Agent mode and you'll be able to interact with the Memgraph MCP server.
