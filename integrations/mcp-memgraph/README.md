# Memgraph MCP Server

## Run server

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and create `venv` with `uv venv`. Activate virtual environment with `.venv\Scripts\activate`. 
2. Install dependencies: `uv add "mcp[cli]" httpx`
2. Run Memgraph MCP server: `uv run server.py`.
3. Install [Claude for Desktop](https://claude.ai/download).
4. Add the Memgraph server to Claude config: 

**MacOS/Linux**
```
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows**

```
code $env:AppData\Claude\claude_desktop_config.json
```

Example config:
```
{
    "mcpServers": {
      "mpc-memgraph": {
        "command": "/Users/katelatte/.local/bin/uv",
        "args": [
            "--directory",
            "/Users/katelatte/projects/mcp-memgraph",
            "run",
            "server.py"
        ]
     }
   }
}
```
> [!NOTE]  
> You may need to put the full path to the uv executable in the command field. You can get this by running `which uv` on MacOS/Linux or `where uv` on Windows. Make sure you pass in the absolute path to your server.

5. Open Claude Desktop and see the Memgraph tool listed. Try it out! (Have data loaded in Memgraph)

## Tools

### run_query()
Run a Cypher query against Memgraph

## Resources

### get_schema()
Get Memgraph schema information (run SHOW SCHEMA INFO; query) -> prerequisite: `--schema-info-enabled=true`.

## Roadmap

We just started working on the Memgraph MCP Server. Our goal is to provide a wider set of tools in the near future.
